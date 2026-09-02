"""I byte risparmiati diventano secondi? La misura che manca.

PERCHE' QUESTA E' LA DOMANDA CHE DECIDE TUTTO.

`recupero.py` misura QUALITA' contro BYTE LETTI, ed e' misurato bene: su tre
architetture, leggendo l'8% dei byte della KV, la perplexity peggiora dell'1-3%
(RECUPERO_SELETTIVO.md). Ma i byte non sono secondi. Un guadagno teorico di
banda diventa velocita' solo se esiste il codice che legge davvero di meno.

E su questa macchina non e' scontato: la KV quantizzata a 8 bit dimezza i byte
e **costa il 42% di velocita'** a 64k token, perche' il costo di decomprimere
supera la banda risparmiata (TARGET_395.md §7.18). Lo stesso puo' succedere
qui: scegliere le pagine costa calcolo, e raccoglierle costa copie.

TRE MISURE, in ordine di quanto sono ottimistiche:

    pieno       decodifica normale, tutta la KV               riferimento
    tetto       la KV e' fisicamente piu' corta della frazione scelta.
                Non e' implementabile — nessuno sa in anticipo quali token
                servono — ma e' il MASSIMO che qualunque implementazione
                possa dare. Se il tetto non paga, il kernel non serve scriverlo.
    raccolta    la selezione vera: si valutano gli indici, si RACCOLGONO le
                sole pagine scelte in un tensore compatto e si fa l'attention
                solo su quelle. E' quanto si ottiene davvero senza scendere
                a scrivere un kernel fuso.

Lo scarto fra `tetto` e `raccolta` e' il costo dell'implementazione, ed e' il
numero che dice se serve un kernel o se basta questo.

Uso:
    python src/velocita.py --ctx 32768 --frazione 0.05
"""

import argparse
import time

import torch

PAGINA = 16
SINK = 4
LOCALE = 128


class Modo:
    attivo = "pieno"        # pieno | raccolta | sparso
    budget = 0.05
    letti = 0
    totali = 0


def _ripeti_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def _attn(query, key, value, scaling, mask=None):
    g = query.shape[1] // key.shape[1]
    k, v = _ripeti_kv(key, g), _ripeti_kv(value, g)
    w = torch.matmul(query, k.transpose(2, 3)) * scaling
    if mask is not None:
        w = w + mask
    w = torch.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(w, v).transpose(1, 2).contiguous(), None


def attenzione(module, query, key, value, attention_mask, scaling,
               dropout=0.0, **kw):
    """Selezione con RACCOLTA vera: le pagine non scelte non vengono lette.

    E' la differenza che conta rispetto a `recupero.py`, dove la selezione e'
    una maschera: mascherare misura la qualita' ma legge comunque tutto, quindi
    non puo' misurare la velocita'. Qui si costruiscono gli indici dei soli
    token scelti e si usa `gather`, che legge solo quelli.
    """
    L, N = query.shape[2], key.shape[2]
    if Modo.attivo == "pieno" or L > 1 or N <= SINK + LOCALE + 2 * PAGINA:
        return _attn(query, key, value, scaling, attention_mask)

    if Modo.attivo in ("sparso", "contiguo"):
        # STESSI BYTE, LAYOUT DIVERSO. La contabilita' dei byte non distingue
        # fra leggere 32 pagine contigue da 16 token e leggere 512 token
        # sparpagliati: sono gli stessi byte. La memoria pero' non si legge a
        # byte, si legge a blocchi — quindi due metodi che "costano uguale"
        # sulla carta possono non costare uguale in secondi. E' l'unico modo
        # di sapere se il vantaggio di SparQ sull'asse dei byte sopravvive
        # all'hardware. Qui gli indici sono CASUALI e non scelti: si sta
        # misurando il costo del layout, non la qualita' della scelta.
        B, Hkv, _, D = key.shape
        i0, i1 = SINK, N - LOCALE
        n_c = i1 - i0
        quanti = max(1, int(round(Modo.budget * n_c)))
        if Modo.attivo == "sparso":
            idx = torch.randint(i0, i1, (B, Hkv, quanti), device=key.device)
        else:
            # stessi token, ma raccolti in pagine contigue da PAGINA
            n_p = max(1, quanti // PAGINA)
            p0 = torch.randint(0, max(1, n_c // PAGINA), (B, Hkv, n_p),
                               device=key.device)
            off = torch.arange(PAGINA, device=key.device)
            idx = (p0.unsqueeze(-1) * PAGINA + off).flatten(-2) + i0
            idx = idx.clamp(i0, i1 - 1)
        coda = torch.cat([torch.arange(0, i0, device=key.device),
                          torch.arange(i1, N, device=key.device)])
        idx = torch.cat([idx, coda.expand(B, Hkv, -1)], dim=-1)
        Modo.letti += idx.shape[-1] * Hkv
        Modo.totali += N * Hkv
        e = idx.unsqueeze(-1).expand(-1, -1, -1, D)
        return _attn(query, key.gather(2, e), value.gather(2, e), scaling)

    B, Hkv, _, D = key.shape
    g = query.shape[1] // Hkv
    i0, i1 = SINK, N - LOCALE
    centro = key[:, :, i0:i1]
    n_pag = centro.shape[2] // PAGINA
    kp = centro[:, :, :n_pag * PAGINA].reshape(B, Hkv, n_pag, PAGINA, D)
    qg = query.reshape(B, Hkv, g, L, D)

    # Punteggio di pagina: limite superiore su max(q·k), indice a 4 bit.
    kmin, kmax = kp.amin(dim=3), kp.amax(dim=3)
    lo, hi = kmin.amin(-1, keepdim=True), kmin.amax(-1, keepdim=True)
    passo = (hi - lo).clamp_min(1e-6) / 15
    kmin = lo + torch.floor((kmin - lo) / passo).clamp(0, 15) * passo
    lo, hi = kmax.amin(-1, keepdim=True), kmax.amax(-1, keepdim=True)
    passo = (hi - lo).clamp_min(1e-6) / 15
    kmax = lo + torch.ceil((kmax - lo) / passo).clamp(0, 15) * passo
    s = torch.maximum(torch.einsum("bhgld,bhpd->bhglp", qg, kmin),
                      torch.einsum("bhgld,bhpd->bhglp", qg, kmax)).amax(dim=(2, 3))

    k_sel = max(1, int(round(Modo.budget * n_pag)))
    top = s.topk(k_sel, dim=-1).indices                       # [B,Hkv,k_sel]

    # Indici dei token: pagine scelte + sink + finestra locale.
    off = torch.arange(PAGINA, device=key.device)
    idx = (top.unsqueeze(-1) * PAGINA + off).flatten(-2) + i0  # [B,Hkv,k_sel*P]
    coda = torch.cat([torch.arange(0, i0, device=key.device),
                      torch.arange(i1, N, device=key.device)])
    idx = torch.cat([idx, coda.expand(B, Hkv, -1)], dim=-1)
    Modo.letti += idx.shape[-1] * Hkv
    Modo.totali += N * Hkv

    e = idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return _attn(query, key.gather(2, e), value.gather(2, e), scaling)


@torch.no_grad()
def decodifica(model, cache, primo, passi=24):
    import copy
    c = copy.deepcopy(cache)
    cur = primo
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(passi):
        r = model(cur, past_key_values=c, use_cache=True)
        cur = r.logits[:, -1].argmax(-1, keepdim=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return passi / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modello", default="Qwen/Qwen3-4B")
    p.add_argument("--ctx", default="8192,32768,65536")
    p.add_argument("--frazione", type=float, default=0.05)
    p.add_argument("--csv", default="results/velocita.csv")
    a = p.parse_args()

    from transformers import (AttentionInterface, AutoModelForCausalLM,
                              DynamicCache)
    AttentionInterface.register("velocita", attenzione)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        a.modello, dtype=torch.bfloat16,
        attn_implementation="velocita").to(dev).eval()
    c = model.config
    d_t = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    kv_tok = c.num_hidden_layers * c.num_key_value_heads * 2 * d_t * 2
    pesi = sum(x.numel() for x in model.parameters()) * 2
    print(f"{a.modello}: {pesi / 1e9:.2f} GB di pesi, "
          f"{kv_tok / 1024:.0f} KiB di KV per token\n")

    import csv as _csv
    import os
    os.makedirs("results", exist_ok=True)
    nuovo = not os.path.exists(a.csv)
    fh = open(a.csv, "a", newline="", buffering=1)
    w = _csv.writer(fh)
    if nuovo:
        w.writerow(["modello", "ctx", "modo", "tok_s", "guadagno",
                    "frazione_letta", "kv_gb", "pesi_gb"])

    Modo.budget = a.frazione
    print(f"  {'contesto':>9} {'KV':>8} {'pieno':>9} {'tetto':>9} "
          f"{'pagine':>9} {'sparso':>9}   {'pag/spar':>8}")
    for n in [int(x) for x in a.ctx.split(",")]:
        ids = torch.randint(0, 1000, (1, n), device=dev)
        Modo.attivo = "pieno"
        base = DynamicCache()
        with torch.no_grad():
            for j in range(0, n - 1, 512):
                model(ids[:, j:min(j + 512, n - 1)],
                      past_key_values=base, use_cache=True)
        coda = ids[:, -1:]
        v_pieno = decodifica(model, base, coda)

        # TETTO: la KV e' fisicamente accorciata alla frazione scelta. Non e'
        # implementabile, e' il limite superiore di qualunque implementazione.
        corto = DynamicCache()
        m = max(SINK + LOCALE + 2 * PAGINA, int(n * a.frazione))
        with torch.no_grad():
            for j in range(0, m - 1, 512):
                model(ids[:, j:min(j + 512, m - 1)],
                      past_key_values=corto, use_cache=True)
        v_tetto = decodifica(model, corto, coda)

        Modo.attivo = "raccolta"
        Modo.letti = Modo.totali = 0
        v_racc = decodifica(model, base, coda)
        fr = Modo.letti / max(Modo.totali, 1)

        # LAYOUT A CONFRONTO: entrambi con indici CASUALI e nessuno scoring,
        # cosi' l'unica differenza e' la contiguita'. Nella prima versione
        # confrontavo il percorso a pagine (che valuta l'indice a 4 bit) con
        # quello sparso (che non valuta niente): misuravo il costo della
        # SCELTA, non quello del LAYOUT, e il segno usciva rovesciato.
        Modo.attivo = "contiguo"
        Modo.letti = Modo.totali = 0
        v_cont = decodifica(model, base, coda)
        Modo.attivo = "sparso"
        Modo.letti = Modo.totali = 0
        v_spar = decodifica(model, base, coda)
        fr_s = Modo.letti / max(Modo.totali, 1)
        Modo.attivo = "pieno"

        print(f"  {n:9,} {kv_tok * n / 1e9:7.2f}G {v_pieno:9.1f} {v_tetto:9.1f} "
              f"{v_cont:9.1f} {v_spar:9.1f}   {v_cont / v_spar:6.2f}x")
        for nome, v in (("pieno", v_pieno), ("tetto", v_tetto),
                        ("raccolta", v_racc), ("contiguo", v_cont),
                        ("sparso", v_spar)):
            w.writerow([a.modello, n, nome, f"{v:.2f}",
                        f"{v / v_pieno:.3f}", f"{fr:.4f}",
                        f"{kv_tok * n / 1e9:.3f}", f"{pesi / 1e9:.3f}"])
        del corto
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    fh.close()
    print(f"\n  frazione di KV effettivamente letta dalla raccolta: {100 * fr:.1f}%")
    print(f"  -> {a.csv}")
    print("\n  'pagine' e 'sparso' leggono gli STESSI byte con layout diverso:")
    print("  32 pagine contigue da 16 token contro 512 token sparpagliati.")
    print("  Il rapporto e' quanto vale la contiguita' su questo hardware, e")
    print("  non compare in nessuna contabilita' dei byte.")
    print("\n  Il TETTO e' il massimo ottenibile: se non paga, il kernel non")
    print("  serve scriverlo. Lo scarto fra tetto e raccolta e' il costo")
    print("  dell'implementazione in PyTorch, cioe' quanto un kernel fuso")
    print("  potrebbe ancora recuperare.")


if __name__ == "__main__":
    main()
