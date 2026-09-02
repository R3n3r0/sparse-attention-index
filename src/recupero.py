"""Quanto costa indirizzare la memoria a 1 bit, invece che leggerla tutta?

LA TESI, E DA DOVE VIENE.

Due misure del 18-19 agosto 2026 (TARGET_395.md §7.18) delimitano il problema
su questa macchina:

  - a 131k token il 70% dei byte letti a ogni passo di decodifica e' KV, non
    pesi. Il contesto E' il costo.
  - quantizzare la KV non aiuta: la decompressione costa piu' della banda che
    risparmia (-42% a 64k). Comprimere l'archivio e' la strada sbagliata.

La strada che resta e' non leggerlo quasi tutto. E qui entra l'unico risultato
netto prodotto dal progetto (§7.8): **l'indirizzamento sopravvive alla
quantizzazione estrema, i valori no.** Chiavi a 1 bit costano 1/16 della
memoria e perdono 3,7 punti di accuratezza di indirizzamento su 32k voci.

Da cui la tesi, che questo banco deve confermare o demolire:

    Spendere bit nell'INDIRIZZO, non nell'ARCHIVIO. Un indice a 1 bit si
    legge quasi gratis, si valuta con il calcolo che a batch 1 e' inutilizzato
    per 57x, e serve solo a scegliere i pochi valori in piena precisione che
    vale la pena leggere davvero.

COSA MISURA QUESTO BANCO, E COSA NO.

Misura **qualita' contro frazione di KV letta**, non velocita'. E' deliberato:
un guadagno di velocita' che distrugge la qualita' non interessa, e un kernel
veloce si scrive solo dopo aver saputo quale politica di selezione regge.
Separare le due domande e' anche l'unico modo di non ripetere l'errore del
§7.2, dove sei leve venivano confrontate senza controllo appaiato.

LA STRUTTURA, che e' quella del deployment: **prefill pieno, decodifica
selettiva.** Il documento si legge una volta per intero (e a quel punto e'
gia' pagato); sono i passi di generazione successivi a rileggere la KV a ogni
token, ed e' li' che si sceglie cosa leggere.

I CONTROLLI APPAIATI. Cinque politiche a parita' esatta di budget:

    pieno      nessuna restrizione                       -> tetto assoluto
    casuale    pagine a caso                             -> pavimento
    oracolo    top-k sui punteggi VERI di attention      -> tetto di QUALUNQUE
                                                            metodo di selezione
    quest      estremi min/max per pagina, in fp16       -> riferimento di
                                                            letteratura (ICML 2024)
    bit1       chiavi a 1 bit, punteggio di Hamming      -> la tesi

Il numero che conta non e' "bit1 e' vicino a pieno" — a budget alto lo sono
tutti. E' **quanto della distanza fra `casuale` e `oracolo` viene recuperata
da `bit1`**, e come si confronta con `quest` a parita' di byte di indice.

Uso:
    python src/recupero.py --modello Qwen/Qwen3-4B --ctx 8192
    python src/recupero.py --budget 0.01,0.02,0.05,0.1,0.25
"""

import argparse
import math

import torch

PAGINA = 16          # token per pagina, come in Quest
SINK = 4             # primi token sempre letti (attention sink, StreamingLLM)
BLOCCO = 512         # token per blocco di prefill (vedi il commento in main)
LOCALE = 128         # ultimi token sempre letti: la finestra scorrevole e'
                     # gia' nota come fortissima (TARGET_395.md §7.13), quindi
                     # negarla renderebbe il banco facile in modo disonesto


class Scelta:
    """Configurazione globale letta dall'attention sostituita.

    Globale e non passata per argomento perche' l'interfaccia di transformers
    non offre un canale per parametri propri: si registra una funzione con
    firma fissa. Il banco imposta questi campi prima di ogni giro.
    """
    politica = "pieno"
    budget = 1.0         # frazione di pagine "centrali" da leggere
    letti = 0            # contatore: token di KV effettivamente lettI
    totali = 0           # contatore: token di KV disponibili


def _quantizza(v, bit, giu):
    """Quantizza un vettore per pagina a `bit` bit, con scala e offset propri.

    `giu=True` arrotonda verso il basso, `False` verso l'alto. La direzione
    non e' un dettaglio estetico: il punteggio di Quest e' un LIMITE
    SUPERIORE su max(q·k) nella pagina, e un limite arrotondato dalla parte
    sbagliata smette di essere un limite — una pagina importante potrebbe
    finire sotto la soglia e non venire mai letta. Arrotondando il massimo in
    su e il minimo in giu' il limite resta valido, solo un po' piu' largo.
    """
    lo = v.amin(dim=-1, keepdim=True)
    hi = v.amax(dim=-1, keepdim=True)
    passo = (hi - lo).clamp_min(1e-6) / (2 ** bit - 1)
    q = (v - lo) / passo
    q = torch.floor(q) if giu else torch.ceil(q)
    return lo + q.clamp(0, 2 ** bit - 1) * passo


def _punteggi_pagina(q, k, politica):
    """Punteggio per pagina, per testa KV. q: [B,Hq,L,D]  k: [B,Hkv,N,D]

    Restituisce [B, Hkv, n_pagine]. Il punteggio e' aggregato sulle query del
    gruppo GQA con un massimo: basta che UNA query del gruppo voglia la
    pagina, perche' la pagina va letta una volta sola per tutto il gruppo.
    """
    B, Hkv, N, D = k.shape
    Hq = q.shape[1]
    g = Hq // Hkv
    np_ = N // PAGINA
    kp = k[:, :, :np_ * PAGINA].reshape(B, Hkv, np_, PAGINA, D)
    qg = q.reshape(B, Hkv, g, -1, D)                    # [B,Hkv,g,L,D]

    # Gli assi del punteggio per token sono [b,h,g,l,p,s]: si riduce su g
    # (le query del gruppo GQA), l (le query del blocco) e s (i token dentro
    # la pagina), tenendo p. Sbagliare quali assi ridurre non da' errore, da'
    # un tensore della forma giusta e del contenuto sbagliato — cioe' un
    # risultato plausibile e falso.
    RIDUCI = (2, 3, 5)

    if politica == "oracolo":
        # I punteggi veri di attention: il meglio che qualunque selettore possa
        # fare. Non e' implementabile (per calcolarli servirebbe leggere tutta
        # la KV) ma e' il tetto contro cui si misura ogni euristica.
        s = torch.einsum("bhgld,bhpsd->bhglps", qg, kp)
        return s.amax(dim=RIDUCI)

    if politica.startswith("quest") and politica != "quest":
        # LA TESI, SPOSTATA DOVE HA SENSO. `bit1` perde perche' approssima il
        # punteggio di ogni token e poi ne prende il massimo: l'errore per
        # token domina il massimo. Quest invece non approssima, DELIMITA — il
        # suo punteggio e' un limite superiore su max(q·k) dentro la pagina.
        # Delimitare batte approssimare, e centrare o scalare i segni non
        # cambia questo (misurato: bit1c 8-21%, bit1s -8-5% del divario).
        #
        # Ma il limite di Quest si paga in fp16, e ai budget piccoli l'INDICE
        # domina il conto: a budget 0,01 sono 6,25% di indice contro 2,7% di
        # pagine lette. Quindi la domanda giusta non e' "si puo' sostituire il
        # metodo con 1 bit" — e' "quanti bit servono al LIMITE".
        #
        # L'arrotondamento e' DIREZIONALE: il massimo verso l'alto, il minimo
        # verso il basso. Cosi' il limite resta un limite anche quantizzato, e
        # una pagina che conta non puo' essere scartata per un errore di
        # arrotondamento.
        bit = int(politica[5:])
        kmin, kmax = kp.amin(dim=3), kp.amax(dim=3)      # [B,Hkv,np,D]
        kmin = _quantizza(kmin, bit, giu=True)
        kmax = _quantizza(kmax, bit, giu=False)
        s = torch.maximum(torch.einsum("bhgld,bhpd->bhglp", qg, kmin),
                          torch.einsum("bhgld,bhpd->bhglp", qg, kmax))
        return s.amax(dim=(2, 3))

    if politica == "quest":
        # Limite superiore di q·k dentro la pagina: per ogni dimensione si
        # prende l'estremo che il segno della query favorisce. Indice: due
        # vettori fp16 per pagina.
        kmin = kp.amin(dim=3)                            # [B,Hkv,np,D]
        kmax = kp.amax(dim=3)
        s = torch.maximum(torch.einsum("bhgld,bhpd->bhglp", qg, kmin),
                          torch.einsum("bhgld,bhpd->bhglp", qg, kmax))
        return s.amax(dim=(2, 3))

    if politica == "bit1":
        # LA TESI. L'INDICE e' a 1 bit per dimensione (il segno della chiave);
        # la query resta in piena precisione, perche' la query non si
        # memorizza — si calcola sul momento. Indice: D bit per token, cioe'
        # 1/32 della KV in fp16.
        s = torch.einsum("bhgld,bhpsd->bhglps", qg, torch.sign(kp))
        return s.amax(dim=RIDUCI)

    if politica in ("bit1c", "bit1s"):
        # LA TESI, RIPARATA. `bit1` grezzo perde contro Quest di dieci volte
        # in KL (misurato, 24 documenti). L'ipotesi sul perche': il risultato
        # a 1 bit del §7.8 veniva da codici CASUALI a media nulla e modulo
        # unitario. Le chiavi di attention vere non sono ne' l'una ne'
        # l'altra cosa: hanno una componente comune sistematica e norme molto
        # diverse fra token. Il segno grezzo butta via proprio la scala, che
        # e' quello che Quest conserva con i suoi estremi per pagina.
        #
        #   bit1c  si sottrae la media per dimensione, calcolabile una volta
        #          durante il prefill. Costo aggiuntivo: un vettore per testa.
        #   bit1s  in piu' una scala per token, alpha_t = media(|k_t - mu|),
        #          cioe' la quantizzazione a 1 bit fatta come si deve.
        #          Costo: 16 bit per token, contro i D=128 bit dei segni.
        mu = k.mean(dim=2, keepdim=True)                 # [B,Hkv,1,D]
        c = kp - mu.unsqueeze(3)
        b = torch.sign(c)
        if politica == "bit1s":
            b = b * c.abs().mean(dim=-1, keepdim=True)   # scala per token
        s = torch.einsum("bhgld,bhpsd->bhglps", qg, b)
        return s.amax(dim=RIDUCI)

    if politica == "hamming":
        # Variante piu' estrema: anche la query ridotta al segno. Serve a
        # separare "l'indice puo' essere a 1 bit" da "tutto il confronto puo'
        # essere a 1 bit" — la seconda e' molto piu' economica ma non e' la
        # tesi, e vale sapere quanto costa in piu'.
        s = torch.einsum("bhgld,bhpsd->bhglps", torch.sign(qg), torch.sign(kp))
        return s.amax(dim=RIDUCI)

    if politica == "casuale":
        return torch.rand(B, Hkv, np_, device=k.device)

    raise ValueError(politica)


def _ripeti_kv(x, n):
    """Espande le teste KV su quelle di query (GQA). n=1 per MHA puro."""
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def _eager(module, query, key, value, attention_mask, scaling, dropout=0.0):
    """Attention esplicita, scritta qui invece di importarla dal modello.

    Prima importavo `eager_attention_forward` da `modeling_qwen3`, il che
    legava tutto il banco a UNA architettura. La formula e' identica in tutte
    le famiglie di transformer, e riscriverla in sei righe rende il metodo
    applicabile a qualunque modello — che e' il punto: una tecnica che vale
    su un modello solo non e' una tecnica.

    Serve la versione esplicita, e non SDPA, perche' al passo selettivo va
    aggiunta una maschera arbitraria (le pagine scelte). A L=1 la matrice dei
    punteggi e' 1xN, quindi costa poco.
    """
    g = query.shape[1] // key.shape[1]
    k, v = _ripeti_kv(key, g), _ripeti_kv(value, g)
    w = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[..., :k.shape[-2]]
    w = torch.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout:
        w = torch.nn.functional.dropout(w, p=dropout, training=module.training)
    return torch.matmul(w, v).transpose(1, 2).contiguous(), w


def attenzione_selettiva(module, query, key, value, attention_mask,
                         scaling, dropout=0.0, **kw):
    """Attention con selezione di pagine, registrata dentro transformers.

    Sul PREFILL (molte query in parallelo) non seleziona niente: il documento
    si legge una volta e a quel punto e' pagato. Seleziona in DECODIFICA, che
    e' dove ogni token rilegge tutta la KV ed e' dove sta il costo.

    Il prefill passa da SDPA e non da eager, e non e' un'ottimizzazione: eager
    materializza la matrice dei punteggi, che a 32.768 token sono 69 GB per
    strato. Con SDPA il prefill e' esatto e sta in memoria a qualunque
    lunghezza. Eager resta solo per il passo selettivo, dove la matrice e'
    1xN — cioe' dove serve poter aggiungere una maschera arbitraria.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    L, N = query.shape[2], key.shape[2]

    # LA DISTINZIONE E' L == 1, NON UNA SOGLIA. Prima qui c'era `L <= 64` per
    # dire "decodifica": e l'ultimo blocco di un prefill da 812 token con
    # blocchi da 128 ne ha 44, quindi finiva nel ramo di decodifica. Li' la
    # maschera arriva vuota — legittimo per un singolo token, che vede solo
    # passato — e quei 44 token si vedevano l'un l'altro anche in avanti.
    # Nessun errore, nessun NaN: solo una cache non causale e risultati
    # plausibili. Il selftest lo becca, la soglia arbitraria no.
    if L > 1:
        # Prefill, eventualmente a blocchi. La maschera va costruita se
        # transformers non la fornisce: `is_causal=True` di PyTorch allinea
        # la causalita' in alto a sinistra, mentre a blocchi le L query
        # stanno IN FONDO alle N chiavi e serve l'allineamento in basso a
        # destra. Costruirla esplicitamente vale per ogni L e ogni N.
        if attention_mask is None:
            iq = torch.arange(L, device=query.device).view(-1, 1) + (N - L)
            ik = torch.arange(N, device=query.device).view(1, -1)
            attention_mask = torch.zeros(1, 1, L, N, dtype=query.dtype,
                                         device=query.device)
            attention_mask.masked_fill_(ik > iq, torch.finfo(query.dtype).min)
        return sdpa_attention_forward(module, query, key, value,
                                      attention_mask, dropout=dropout,
                                      scaling=scaling, is_causal=False, **kw)

    selettiva = (Scelta.politica != "pieno" and N > SINK + LOCALE + PAGINA)
    if not selettiva:
        if Scelta.politica != "pieno":
            Scelta.letti += N * key.shape[1]
            Scelta.totali += N * key.shape[1]
        return _eager(module, query, key, value, attention_mask, scaling, dropout)

    B, Hkv = key.shape[0], key.shape[1]
    # Regione "centrale": quello che non e' ne' sink ne' finestra locale. Solo
    # li' si sceglie; sink e finestra si leggono sempre.
    i0, i1 = SINK, N - LOCALE
    centro_k = key[:, :, i0:i1]
    n_pag = centro_k.shape[2] // PAGINA
    if n_pag < 2:
        return _eager(module, query, key, value, attention_mask, scaling, dropout)

    if Scelta.politica.startswith("sparq"):
        # SPARQ (Ribar et al., arXiv:2312.04985), il riferimento token-level.
        # Non sceglie pagine: usa le r componenti di PIU' GRANDE MODULO della
        # query per stimare i punteggi di TUTTI i token leggendo solo quelle r
        # dimensioni delle chiavi, poi tiene i migliori token. Il suo indice
        # costa quindi r/(2D) della KV: con r=16 e D=128 e' 1/16, la stessa
        # spesa dell'indice fp16 di Quest — il che rende il confronto diretto.
        #
        # E' incluso perche' senza un secondo metodo della letteratura sullo
        # STESSO banco il confronto resterebbe qualitativo, ed e' la critica
        # piu' probabile.
        # "sparq16" = r componenti in fp16;  "sparq16q4" = le stesse r
        # componenti quantizzate a 4 bit. La domanda dei 4 bit non riguarda le
        # PAGINE, riguarda l'INDICE — e anche SparQ ne ha uno. Se regge anche
        # qui, il risultato non migliora il metodo perdente ma quello
        # vincente, e vale su due meccanismi indipendenti.
        testa = Scelta.politica[5:]
        if "q" in testa:
            r, bit_q = (int(z) for z in testa.split("q"))
        else:
            r, bit_q = int(testa), 0
        D = centro_k.shape[-1]
        g = query.shape[1] // Hkv
        qg = query.reshape(B, Hkv, g, -1, D)
        # le r dimensioni scelte sono per gruppo GQA: la pagina, qui il token,
        # si legge una volta sola per tutto il gruppo
        peso = qg.abs().amax(dim=(2, 3))                     # [B,Hkv,D]
        dims = peso.topk(min(r, D), dim=-1).indices          # [B,Hkv,r]
        kr = centro_k.gather(3, dims.unsqueeze(2).expand(-1, -1, centro_k.shape[2], -1))
        if bit_q:
            # quantizzazione per token sulle sole r componenti conservate
            lo = kr.amin(dim=-1, keepdim=True)
            hi = kr.amax(dim=-1, keepdim=True)
            passo = (hi - lo).clamp_min(1e-6) / (2 ** bit_q - 1)
            kr = lo + torch.round((kr - lo) / passo).clamp(0, 2 ** bit_q - 1) * passo
        qr = qg.gather(4, dims.unsqueeze(2).unsqueeze(3).expand(-1, -1, g, qg.shape[3], -1))
        pt = torch.einsum("bhgld,bhnd->bhgln", qr, kr).amax(dim=(2, 3))  # [B,Hkv,n]
        n_tok = centro_k.shape[2]
        k_tok = max(1, int(round(Scelta.budget * n_tok)))
        scelti = pt.topk(k_tok, dim=-1).indices
        piena = torch.zeros(B, Hkv, N, dtype=torch.bool, device=key.device)
        piena[:, :, :i0] = True
        piena[:, :, i1:] = True
        piena[:, :, i0:i1].scatter_(-1, scelti, True)
        Scelta.letti += int(piena.sum().item())
        Scelta.totali += B * Hkv * N
        m = piena.repeat_interleave(g, dim=1).unsqueeze(2)
        blocco = torch.zeros_like(m, dtype=query.dtype)
        blocco.masked_fill_(~m, torch.finfo(query.dtype).min)
        if attention_mask is not None:
            blocco = blocco + attention_mask[:, :, :, :N]
        return _eager(module, query, key, value, blocco, scaling, dropout)

    s = _punteggi_pagina(query, centro_k, Scelta.politica)   # [B,Hkv,n_pag]
    k_sel = max(1, int(round(Scelta.budget * n_pag)))
    top = s.topk(k_sel, dim=-1).indices                      # [B,Hkv,k_sel]

    # Maschera additiva: -inf su tutto il centro tranne le pagine scelte.
    tieni = torch.zeros(B, Hkv, n_pag, dtype=torch.bool, device=key.device)
    tieni.scatter_(-1, top, True)
    tieni = tieni.repeat_interleave(PAGINA, dim=-1)          # [B,Hkv,n_pag*P]
    piena = torch.ones(B, Hkv, N, dtype=torch.bool, device=key.device)
    piena[:, :, i0:i0 + n_pag * PAGINA] = tieni

    Scelta.letti += int(piena.sum().item())
    Scelta.totali += B * Hkv * N

    g = query.shape[1] // Hkv
    m = piena.repeat_interleave(g, dim=1).unsqueeze(2)       # [B,Hq,1,N]
    blocco = torch.zeros_like(m, dtype=query.dtype)
    blocco.masked_fill_(~m, torch.finfo(query.dtype).min)
    if attention_mask is not None:
        blocco = blocco + attention_mask[:, :, :, :N]
    return _eager(module, query, key, value, blocco, scaling, dropout)


# ---------------------------------------------------------------------------
# Il banco: quanto la selezione SPOSTA il modello, su testo naturale.
#
# PERCHE' NON IL PASSKEY. Ci ho provato per primo, ed e' il banco che usa la
# letteratura. Ma su Qwen3-4B il compito e' al limite: con lo stesso formato e
# testi lunghi quasi uguali (4.143 contro 4.863 token) il modello a attention
# PIENA a volte ritrova la chiave e a volte no. Un controllo che fallisce da
# solo non puo' fare da riferimento per nessun confronto — misurerebbe la
# fortuna del prompt, non la politica di selezione. (Vale anche la pena
# ricordarlo: col template di chat il modello risponde "questo testo sembra
# ripetitivo e senza senso" e fallisce sempre. Il formato conta.)
#
# LA METRICA QUI e' la divergenza KL fra la distribuzione del token successivo
# con attention PIENA e quella con attention SELETTIVA, sullo stesso identico
# stato. Proprieta' che servono:
#
#   - non richiede che il modello sappia fare un compito: misura di quanto lo
#     si e' cambiato, che e' esattamente la domanda;
#   - e' continua, quindi discrimina anche dove un compito binario saturerebbe;
#   - e' appaiata per costruzione: stesso documento, stessa cache, stesso
#     passo. La differenza puo' venire solo dalla politica.
#
# L'ASSE DEI BYTE E' CONTATO CON L'INDICE DENTRO, e non e' un dettaglio
# pignolo: e' il punto della tesi. Leggere una pagina scelta costa K e V; ma
# per sceglierla bisogna leggere l'indice di TUTTE le pagine. Rispetto alla KV
# in fp16 (4D byte per token):
#
#   bit1 / hamming   D/8 byte per token  ->  1/32
#   quest            due vettori fp16 ogni 16 token  ->  1/16
#   oracolo          richiede le chiavi vere, 2D byte  ->  1/2  (non
#                    implementabile: e' un tetto, non un metodo)
#   casuale          niente
#
# Un metodo che legge il 10% delle pagine ma ha un indice da 1/16 costa 16,3%,
# non 10%. A budget piccoli l'indice DOMINA, ed e' li' che 1 bit deve vincere.

import random

def costo_indice(politica, D):
    """Byte di indice per token, come frazione della KV in fp16 (4D byte).

    E' l'asse onesto del confronto: per scegliere quali pagine leggere
    bisogna leggere l'indice di TUTTE le pagine, sempre. A budget piccoli
    l'indice e' il termine dominante — Quest in fp16 costa il 6,3% di indice
    contro il 2,7% di pagine lette a budget 0,01 — quindi confrontare i
    metodi sulla sola frazione di pagine e' fuorviante.
    """
    if politica in ("casuale", "pieno"):
        return 0.0
    if politica == "oracolo":
        return 0.5                      # servono le chiavi vere: 2D byte
    if politica in ("bit1", "hamming", "bit1c"):
        return (D / 8) / (4 * D)        # un segno per dimensione
    if politica == "bit1s":
        return (D / 8 + 2) / (4 * D)    # piu' una scala fp16 per token
    if politica.startswith("sparq"):
        # r componenti delle chiavi per ogni token. In fp16 sono r*2 byte; a
        # `bit` bit sono r*bit/8 piu' scala e offset (4 byte), su 4D di KV.
        testa = politica[5:]
        if "q" in testa:
            r, bq = (int(z) for z in testa.split("q"))
            return (r * bq / 8 + 4) / (4 * D)
        return (int(testa) * 2) / (4 * D)
    if politica == "quest":
        return (2 * D * 2) / (PAGINA * 4 * D)
    if politica.startswith("quest"):
        bit = int(politica[5:])
        # due vettori quantizzati per pagina, piu' scala e offset di ciascuno
        return (2 * D * bit / 8 + 8) / (PAGINA * 4 * D)
    raise ValueError(politica)


def contesti(tok, n_doc, n_tok, seme=0):
    """Documenti di testo naturale lunghi n_tok token, da WikiText-103."""
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    par = [t for t in d["text"] if len(t) > 400]
    rng = random.Random(seme)
    fuori = []
    for _ in range(n_doc):
        pezzi, ids = [], []
        while len(ids) < n_tok:
            pezzi.append(par[rng.randrange(len(par))])
            ids = tok("".join(pezzi), return_tensors="pt").input_ids[0]
        fuori.append(ids[:n_tok].unsqueeze(0))
    return fuori


@torch.no_grad()
def coda_valutata(model, cache, ingressi, veri):
    """Decodifica gli ultimi token su una COPIA della cache, e li valuta.

    La copia e' cio' che rende il confronto appaiato invece che ripetuto: ogni
    politica parte dallo stesso identico stato.

    Restituisce (nll_media, log_probabilita_per_passo). La NLL sui token VERI
    del documento e' la metrica che serve davvero: e' la perplexity, cioe'
    quanto il modello peggiora nel suo compito. La KL contro l'attention piena
    dice invece di quanto lo si e' spostato — utile, ma indiretta, e nessuno
    la accetta come misura di qualita'.

    Si valutano molte posizioni per documento e non una sola: la selezione
    cambia a ogni passo (la query e' diversa), quindi una posizione sola
    misura un colpo di fortuna.
    """
    import copy
    c = copy.deepcopy(cache)
    nll, lps = 0.0, []
    for i in range(ingressi.shape[1]):
        r = model(ingressi[:, i:i + 1], past_key_values=c, use_cache=True)
        lp = torch.log_softmax(r.logits[:, -1].float(), -1)
        nll += -lp[0, veri[0, i]].item()
        lps.append(lp)
    return nll / ingressi.shape[1], lps


def selftest(model, tok, dev):
    """Il prefill a blocchi deve dare gli stessi logit di quello in un colpo.

    E' il controllo che mancava. Senza, un errore di allineamento della
    maschera causale non da' nessun errore: da' una cache corrotta, e i
    risultati restano plausibili. Se ne accorge solo chi nota che la
    selezione CASUALE batte l'ORACOLO — cioe' per fortuna.
    """
    from transformers import DynamicCache
    ids = tok("Nel mezzo del cammin di nostra vita " * 90,
              return_tensors="pt").input_ids.to(dev)
    n = ids.shape[1]
    Scelta.politica = "pieno"
    with torch.no_grad():
        c1 = DynamicCache()
        a = model(ids, past_key_values=c1, use_cache=True).logits[:, -1].float()
        c2 = DynamicCache()
        for j in range(0, n, 128):          # blocchi piccoli apposta
            b = model(ids[:, j:j + 128], past_key_values=c2,
                      use_cache=True).logits[:, -1].float()
    # Il criterio NON e' lo scarto massimo sui logit: in bf16 quello vale
    # gia' ~1,0 su logit di ~18 per il solo accumulo diverso su 36 strati, e
    # boccerebbe un prefill corretto. Il criterio e' la KL fra le due
    # distribuzioni piu' l'accordo sul token piu' probabile: se la maschera e'
    # disallineata la KL esplode di ordini di grandezza, se e' solo aritmetica
    # resta minuscola.
    la = torch.log_softmax(a, -1)
    lb = torch.log_softmax(b, -1)
    kl = torch.sum(la.exp() * (la - lb)).item()
    stesso = int(a.argmax(-1) == b.argmax(-1))
    d = (a - b).abs().max().item()
    print(f"  selftest prefill a blocchi ({n} token, blocchi da 128):")
    print(f"    KL {kl:.2e}   stesso token piu' probabile: {'si' if stesso else 'NO'}"
          f"   (scarto max sui logit {d:.3f}, atteso in bf16)")
    if kl > 1e-2 or not stesso:
        # NON si abortisce: si RIPIEGA. Il prefill a blocchi e' un espediente
        # per non esaurire la memoria a contesti lunghi, non parte del metodo.
        # Se un modello non lo tollera — Phi-3.5 perche' la rotazione dipende
        # dalla lunghezza, BitNet perche' quantizza le attivazioni con una
        # scala calcolata a ogni passaggio — la risposta giusta e' leggere il
        # documento in un colpo solo, non fermarsi. Cosi' il banco si adatta
        # al modello invece di richiedere un caso particolare per famiglia.
        print("    NON riproduce: prefill in un colpo solo\n")
        return False
    print("    OK\n")
    return True


def main():
    # L'INDICE HA DUE LEVE, non una. Il suo costo e' circa bit/(4*PAGINA),
    # quindi dimezzare i bit e raddoppiare la pagina risparmiano gli stessi
    # byte. Ma non costano la stessa qualita': una pagina piu' grande rende il
    # limite piu' largo (min/max su piu' token) e fa scegliere peggio. Senza
    # confrontare le due leve a parita' di byte, il risultato sui 4 bit non e'
    # difendibile — sarebbe solo un modo fra tanti di ottenere lo stesso conto.
    global PAGINA

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modello", default="Qwen/Qwen3-4B")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--doc", type=int, default=16, help="documenti misurati")
    p.add_argument("--budget", default="0.01,0.02,0.05,0.10,0.25,0.50")
    p.add_argument("--politiche", default="casuale,hamming,bit1,quest,oracolo")
    p.add_argument("--pagina", type=int, default=PAGINA,
                   help="token per pagina: l'ALTRA leva sul costo dell'indice")
    p.add_argument("--blocco", type=int, default=BLOCCO,
                   help="token per blocco di prefill; 0 = un colpo solo")
    p.add_argument("--posizioni", type=int, default=32,
                   help="token finali valutati per documento")
    p.add_argument("--seme", type=int, default=0)
    p.add_argument("--csv", default="results/recupero.csv")
    a = p.parse_args()

    from transformers import (AttentionInterface, AutoModelForCausalLM,
                              AutoTokenizer, DynamicCache)
    AttentionInterface.register("selettiva", attenzione_selettiva)
    PAGINA = a.pagina

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.modello)
    model = AutoModelForCausalLM.from_pretrained(
        a.modello, dtype=torch.bfloat16,
        attn_implementation="selettiva").to(dev).eval()
    c = model.config
    print(f"{a.modello}: {c.num_hidden_layers} strati, "
          f"{c.num_attention_heads}/{c.num_key_value_heads} teste q/kv")
    d_testa = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads

    # ROTAZIONE DIPENDENTE DALLA LUNGHEZZA: il prefill a blocchi non e' lecito.
    # Phi-3.5 (LongRoPE) sceglie fattori di rotazione DIVERSI a seconda della
    # lunghezza della sequenza: coi blocchi i primi usano il fattore "corto" e
    # gli ultimi quello "lungo", e la cache finisce con rotazioni miste. Non
    # da' nessun errore — da' perplexity 27.000 invece di 15, cioe' un modello
    # rotto che sembra funzionare. Il selftest a 812 token non lo vede, perche'
    # sotto la soglia entrambi i percorsi usano lo stesso fattore.
    rp = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {}
    if isinstance(rp, dict) and (rp.get("original_max_position_embeddings")
                                 or str(rp.get("rope_type", rp.get("type", ""))).lower()
                                 in ("longrope", "su", "dynamic")):
        if a.blocco:
            print("  rotazione dipendente dalla lunghezza: prefill in un colpo "
                  "solo (blocchi non leciti)\n")
            a.blocco = 0

    budget = [float(x) for x in a.budget.split(",")]
    politiche = a.politiche.split(",")
    if a.blocco and not selftest(model, tok, dev):
        a.blocco = 0

    docs = contesti(tok, a.doc, a.ctx, a.seme)
    print(f"{len(docs)} documenti da {a.ctx:,} token, pagine da {PAGINA} token\n")

    import csv as _csv
    import os
    os.makedirs("results", exist_ok=True)
    nuovo = not os.path.exists(a.csv)
    fh = open(a.csv, "a", newline="", buffering=1)
    w = _csv.writer(fh)
    if nuovo:
        w.writerow(["modello", "ctx", "pagina", "politica", "budget", "doc",
                    "ppl", "ppl_piena", "rapporto_ppl", "kl",
                    "frazione_pagine", "frazione_byte"])

    acc = {}
    K = a.posizioni
    for i, ids in enumerate(docs):
        ids = ids.to(dev)
        Scelta.politica = "pieno"
        base = DynamicCache()
        # Prefill A BLOCCHI. Su gfx1151 SDPA non ha un kernel a memoria
        # lineare e ricade sul percorso matematico anche dichiarando
        # is_causal: a 32.768 token in un colpo solo chiede 128 GB e va in
        # OOM. A blocchi il picco e' BLOCCO x N invece di N x N, quindi la
        # lunghezza del contesto non e' piu' vincolata dalla memoria. Il
        # risultato e' identico: e' la stessa attention, calcolata in piu'
        # chiamate sulla stessa cache.
        fine = ids.shape[1] - K - 1
        with torch.no_grad():
            passo = a.blocco or fine
            for j in range(0, fine, passo):
                model(ids[:, j:min(j + passo, fine)],
                      past_key_values=base, use_cache=True)
        ingressi, veri = ids[:, fine:fine + K], ids[:, fine + 1:fine + 1 + K]
        nll_p, lps_p = coda_valutata(model, base, ingressi, veri)

        for pol in politiche:
            for b in budget:
                Scelta.politica, Scelta.budget = pol, b
                Scelta.letti = Scelta.totali = 0
                nll_s, lps_s = coda_valutata(model, base, ingressi, veri)
                fr = Scelta.letti / max(Scelta.totali, 1)
                Scelta.politica = "pieno"
                kl = sum(torch.sum(p.exp() * (p - q)).item()
                         for p, q in zip(lps_p, lps_s)) / K
                byte = fr + costo_indice(pol, d_testa)
                # Il rapporto fra le perplexity e' la degradazione relativa, ed
                # e' cio' che si confronta fra documenti: le perplexity assolute
                # variano molto da un documento all'altro, il rapporto no.
                rapp = math.exp(nll_s - nll_p)
                acc.setdefault((pol, b), []).append((rapp, fr, byte, kl, nll_s))
                w.writerow([a.modello, a.ctx, PAGINA, pol, b, i, f"{math.exp(nll_s):.4f}",
                            f"{math.exp(nll_p):.4f}", f"{rapp:.5f}", f"{kl:.6f}",
                            f"{fr:.4f}", f"{byte:.4f}"])
        print(f"  documento {i + 1}/{len(docs)}  "
              f"(ppl piena {math.exp(nll_p):.2f})", flush=True)

    fh.close()

    def med(v, j):
        return sum(x[j] for x in v) / len(v)

    print(f"\n  {len(docs)} documenti x {K} posizioni = "
          f"{len(docs) * K:,} token valutati per configurazione\n")
    print(f"  {'politica':<10} {'budget':>7} {'pagine':>8} {'byte':>8} "
          f"{'ppl / ppl piena':>16} {'KL':>9}")
    for pol in politiche:
        for b in budget:
            v = acc.get((pol, b))
            if not v:
                continue
            print(f"  {pol:<10} {b:7.2f} {100 * med(v, 1):7.1f}% "
                  f"{100 * med(v, 2):7.1f}% {med(v, 0):15.4f} "
                  f"{med(v, 3):9.4f}")

    # LA FRONTIERA, ora sulla PERPLEXITY e non sulla KL. Confrontare a parita'
    # di budget e' fuorviante perche' ogni metodo paga un indice diverso: si
    # mettono tutti i punti su un unico asse di byte e si guarda chi sta sotto.
    print("\n  Frontiera: nessun altro punto legge meno byte E degrada meno")
    punti = sorted((med(v, 2), med(v, 0), pol, b) for (pol, b), v in acc.items())
    migliore = float("inf")
    for byte, rapp, pol, b in punti:
        if rapp < migliore - 1e-9:
            migliore = rapp
            print(f"    {100 * byte:6.1f}% byte   ppl x{rapp:.4f}   "
                  f"{pol} @ {b:.2f}")
    print(f"\n  -> {a.csv}")


if __name__ == "__main__":
    main()
