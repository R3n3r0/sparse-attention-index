"""Il tetto di velocita' di un modello su questa macchina, e quanto ne resta.

A batch 1 la decodifica e' un problema di LETTURA: per ogni token si rileggono
tutti i pesi attivi. Quindi

    tok/s_max = banda_effettiva / byte_attivi_per_token

Questo script calcola quel tetto e, dato un valore misurato, dice che
frazione se ne sta usando. Serve a distinguere due diagnosi che si
confondono sempre:

  - "vado piano perche' il modello e' grosso"     -> vicino al tetto, e'
                                                     fisica, serve un modello
                                                     piu' piccolo o piu' sparso;
  - "vado piano perche' lo stack e' inefficiente" -> lontano dal tetto, e'
                                                     ingegneria, e si recupera
                                                     senza toccare la qualita'.

Banda misurata su questa macchina (src/roofline.py, 12 agosto 2026):
lettura pura 237,4 GB/s, copia 211,2 GB/s. Il GEMV a batch 1 ne tocca solo
il 26,6 GB/s, cioe' l'11%: quel divario e' il primo posto dove guardare.

Uso:
    python src/tetto.py --gguf ~/models/Qwen3-30B-A3B-Q6_K.gguf
    python src/tetto.py --params 30e9 --attivi 3e9 --bit 6.6 --misurato 45
"""

import argparse
import os
import struct

BANDA_GB = 237.4          # lettura pura misurata, non di targa


def leggi_gguf(path):
    """Metadati e SOMMA VERA dei tensori, divisi fra densi e per-esperto.

    Prima versione: stimavo la frazione attiva come n_usati/n_esperti piu' un
    forfait del 15% per la parte densa. Su Qwen3-30B-A3B dava 21% di pesi
    attivi contro il 10% vero (il nome "A3B" dice 3B attivi su 30) — cioe' un
    tetto sbagliato di due volte. Un tetto sbagliato e' peggio di nessun
    tetto: fa sembrare vicino alla fisica uno stack che invece ha margine.

    Qui i tensori si contano davvero. In llama.cpp i pesi per-esperto hanno
    "_exps" nel nome (ffn_gate_exps, ffn_up_exps, ffn_down_exps); tutto il
    resto e' letto a ogni token.
    """
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path} non e' un GGUF")
        ver, n_ten, n_kv = struct.unpack("<IQQ", f.read(20))
        meta = {}
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            chiave = f.read(klen).decode("utf-8", "replace")
            tipo = struct.unpack("<I", f.read(4))[0]
            val = _leggi_valore(f, tipo)
            if any(s in chiave for s in ("expert", "block_count", "length",
                                         "head_count", "context")):
                meta[chiave] = val
        densi = esperti = 0
        for _ in range(n_ten):
            nlen = struct.unpack("<Q", f.read(8))[0]
            nome = f.read(nlen).decode("utf-8", "replace")
            n_dim = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{n_dim}Q", f.read(8 * n_dim))
            f.read(4)                              # tipo del tensore
            f.read(8)                              # offset
            n_el = 1
            for d in dims:
                n_el *= d
            if "_exps" in nome:
                esperti += n_el
            else:
                densi += n_el
        meta["_el_densi"], meta["_el_esperti"] = densi, esperti
    meta["_byte_su_disco"] = os.path.getsize(path)
    meta["_versione"], meta["_n_tensori"] = ver, n_ten
    return meta


_FISSI = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
          6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _leggi_valore(f, tipo):
    if tipo in _FISSI:
        fmt = _FISSI[tipo]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if tipo == 8:                                   # stringa
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    if tipo == 9:                                   # array
        t_el, n = struct.unpack("<IQ", f.read(12))
        return [_leggi_valore(f, t_el) for _ in range(n)]
    raise SystemExit(f"tipo GGUF {tipo} non gestito")


def riporta(nome, byte_totali, byte_attivi, misurato=None):
    tetto = BANDA_GB * 1e9 / byte_attivi
    print(f"\n{nome}")
    print(f"  peso su disco          {byte_totali / 1e9:8.1f} GB")
    print(f"  letti per token        {byte_attivi / 1e9:8.1f} GB"
          f"   ({100 * byte_attivi / byte_totali:.0f}% del totale)")
    print(f"  TETTO a {BANDA_GB:.0f} GB/s      {tetto:8.1f} tok/s")
    if misurato:
        print(f"  misurato               {misurato:8.1f} tok/s"
              f"   -> {100 * misurato / tetto:.0f}% del tetto")
        if misurato / tetto < 0.5:
            print("  DIAGNOSI: lontano dal tetto. Il limite non e' la fisica "
                  "della macchina\n            ma lo stack: c'e' margine senza "
                  "toccare il modello.")
        else:
            print("  DIAGNOSI: vicino al tetto. Per andare piu' veloce serve "
                  "leggere MENO\n            byte per token: piu' sparsita', "
                  "meno bit, o un modello minore.")
    return tetto


def byte_kv(meta, bit_kv=16.0):
    """Byte di KV cache per token di contesto.

    A batch 1 e con flash attention, ogni token generato rilegge TUTTA la KV
    del prefisso. Quindi questo numero, moltiplicato per la profondita', si
    somma ai pesi attivi nel conto della banda: e' il motivo per cui la
    velocita' cala col contesto anche quando il modello non cambia.

    Senza flash attention il conto non vale: la matrice dei punteggi viene
    materializzata e il traffico cresce molto piu' in fretta (misurato su
    questa macchina: 2,0 tok/s contro 21,7 a 65.536 token, cioe' 10,9x).
    """
    n_l = next(v for k, v in meta.items() if k.endswith("block_count"))
    n_kv = next(v for k, v in meta.items() if k.endswith("head_count_kv"))
    d_k = next(v for k, v in meta.items() if k.endswith("attention.key_length"))
    d_v = next(v for k, v in meta.items() if k.endswith("attention.value_length"))
    return n_l * n_kv * (d_k + d_v) * bit_kv / 8


def tabella_contesto(nome, attivi, meta, profondita, bit_kv, misurati=None):
    """Tetto di banda in funzione della profondita' del contesto."""
    per_tok = byte_kv(meta, bit_kv)
    nativo = next((v for k, v in meta.items() if k.endswith("context_length")), 0)
    print(f"\n{nome}  —  KV a {bit_kv:.0f} bit: {per_tok / 1024:.0f} KiB/token"
          f"   (finestra nativa {nativo:,})")
    print(f"  {'profondita':>11} {'KV GB':>7} {'letti GB':>9} {'TETTO':>8}"
          f"{'  misurato   %tetto' if misurati else ''}")
    for i, d in enumerate(profondita):
        kv = per_tok * d
        byte = attivi + kv
        tetto = BANDA_GB * 1e9 / byte
        riga = (f"  {d:11,} {kv / 1e9:7.2f} {byte / 1e9:9.2f} "
                f"{tetto:8.1f}")
        if misurati and i < len(misurati) and misurati[i]:
            m = misurati[i]
            riga += f"  {m:9.1f} {100 * m / tetto:7.0f}%"
        print(riga + ("   <- oltre la finestra nativa" if nativo and d > nativo else ""))


CALCOLO_TFLOPS = 26.9        # calcolo di picco misurato (src/roofline.py)


def prefill(meta, attivi_el, n, secondi=None):
    """Il costo del prefill e' dominato dall'attention, e cresce col quadrato.

    Serve a distinguere due diagnosi che sul contesto lungo si confondono:
    "il prefill e' lento perche' lo stack e' inefficiente" (si ripara) oppure
    "perche' l'attention e' quadratica" (non si ripara senza cambiare
    l'attention). A 131.072 token su Qwen3-30B-A3B l'attention e' l'88% delle
    FLOP del prefill: e' fisica, e nessuna manopola di llama.cpp la tocca.

    Attenzione al fattore che e' facile dimenticare: le FLOP di attention
    vanno moltiplicate per il NUMERO DI STRATI. Senza quel fattore il conto
    sottostima di 48x e fa sembrare che il prefill giri al 3% del picco
    (cioe' che ci sia un 30x da recuperare) invece che al 25%.
    """
    n_l = next(v for k, v in meta.items() if k.endswith("block_count"))
    n_h = next(v for k, v in meta.items() if k.endswith("attention.head_count"))
    d_k = next(v for k, v in meta.items() if k.endswith("attention.key_length"))
    f_pesi = 2 * attivi_el * n
    # QK^T e AV, per strato, sommate su tutte le posizioni: 4 * n^2/2 * n_h * d_k
    f_attn = 2 * n * n * n_h * d_k * n_l
    tot = f_pesi + f_attn
    print(f"\n  prefill di {n:,} token")
    print(f"    FLOP sui pesi      {f_pesi / 1e12:9.1f} T   ({100 * f_pesi / tot:.0f}%)")
    print(f"    FLOP di attention  {f_attn / 1e12:9.1f} T   ({100 * f_attn / tot:.0f}%)")
    print(f"    a {CALCOLO_TFLOPS:.1f} TFLOP/s di picco    {tot / 1e12 / CALCOLO_TFLOPS:6.0f} s")
    if secondi:
        eff = (tot / 1e12 / secondi) / CALCOLO_TFLOPS
        print(f"    misurato           {secondi:9.0f} s   "
              f"-> {tot / 1e12 / secondi:.1f} TFLOP/s, {100 * eff:.0f}% del picco")
    return tot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", help="percorso di un file .gguf")
    p.add_argument("--params", type=float, help="parametri totali (es. 30e9)")
    p.add_argument("--attivi", type=float, help="parametri ATTIVI per token")
    p.add_argument("--bit", type=float, default=16.0, help="bit per peso")
    p.add_argument("--misurato", type=float, help="tok/s osservati")
    p.add_argument("--contesti", help="profondita' separate da virgola, es. 0,4096,65536")
    p.add_argument("--bit-kv", type=float, default=16.0, help="bit per elemento di KV")
    p.add_argument("--misurati", help="tok/s osservati alle stesse profondita'")
    p.add_argument("--prefill", type=int, help="token di prefill da analizzare")
    p.add_argument("--prefill-secondi", type=float, help="secondi osservati")
    a = p.parse_args()

    if a.gguf:
        m = leggi_gguf(a.gguf)
        tot = m["_byte_su_disco"]
        n_exp = next((v for k, v in m.items() if k.endswith("expert_count")), 0)
        n_uti = next((v for k, v in m.items() if k.endswith("expert_used_count")), 0)
        nome = os.path.basename(a.gguf)
        densi, esp = m["_el_densi"], m["_el_esperti"]
        if n_exp and n_uti and esp:
            # elementi letti per token = tutti i densi + la quota di esperti
            el_attivi = densi + esp * n_uti / n_exp
            attivi = tot * el_attivi / (densi + esp)
            print(f"\n  [MoE: {n_uti}/{n_exp} esperti. Parametri: "
                  f"{densi / 1e9:.2f}G densi + {esp / 1e9:.2f}G in esperti"
                  f"  ->  {el_attivi / 1e9:.2f}G attivi per token]")
        else:
            attivi = tot
        riporta(nome, tot, attivi, a.misurato)
        if a.contesti:
            prof = [int(x) for x in a.contesti.split(",")]
            mis = [float(x) for x in a.misurati.split(",")] if a.misurati else None
            tabella_contesto(nome, attivi, m, prof, a.bit_kv, mis)
        if a.prefill:
            prefill(m, el_attivi if n_exp and n_uti and esp else densi + esp,
                    a.prefill, a.prefill_secondi)
    elif a.params and a.attivi:
        byte = a.bit / 8
        riporta("modello", a.params * byte, a.attivi * byte, a.misurato)
    else:
        raise SystemExit("serve --gguf oppure --params e --attivi")


if __name__ == "__main__":
    main()
