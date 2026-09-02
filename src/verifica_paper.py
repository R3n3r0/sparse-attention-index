"""Ogni numero del paper corrisponde ancora ai CSV?

PERCHE' ESISTE.

Le tabelle del paper sono state riscritte tre volte mentre i risultati
cambiavano: la metrica e' passata da KL a perplexity, il prefill non causale ha
invalidato una misura intera, e l'arrivo di SparQ ha ribaltato l'inquadramento.
Ogni riscrittura e' un'occasione per lasciare indietro un numero vecchio.

Un numero stantio in un paper non e' un errore di battitura: e' la prima cosa
che un revisore controlla se ha accesso al codice, ed e' quella che fa dubitare
di tutto il resto. Questo script riestrae le cifre dai CSV e le confronta con
quelle scritte in `paper/main.tex`.

Uso:
    python src/verifica_paper.py
"""

import csv
import statistics as st
import sys


def carica(percorso, filtri=None):
    """Carica un CSV di risultati, filtrando su colonne arbitrarie.

    Il filtro non e' un lusso: piu' file contengono piu' condizioni nella
    stessa tabella — `pagine.csv` ha quattro dimensioni di pagina,
    `indici2.csv` ha due modelli. Mediarle insieme produce un numero che non
    compare da nessuna parte e fa sembrare sbagliato un paper corretto. E'
    successo due volte scrivendo questo script, una per ciascun file, il che e'
    il motivo per cui il filtro ora e' generale invece che rattoppato.
    """
    d = {}
    try:
        righe = list(csv.DictReader(open(percorso)))
    except FileNotFoundError:
        return None
    for x in righe:
        if filtri and any(x.get(k) != v for k, v in filtri.items()):
            continue
        chiave = (x["politica"], float(x["budget"]))
        d.setdefault(chiave, []).append(
            (float(x["rapporto_ppl"]), float(x["frazione_byte"])))
    return {k: (st.mean(a for a, _ in v), st.mean(b for _, b in v))
            for k, v in d.items()}


# (descrizione, file, politica, budget, campo, valore atteso nel paper, tolleranza)
ATTESI = [
    ("knee Qwen3-4B 8k, 4 bit",  "results/ppl8k.csv",   "quest4",   0.01, "ppl", 1.169, 0.004),
    ("knee Qwen3-4B 8k, fp16",   "results/ppl8k.csv",   "quest",    0.01, "ppl", 1.165, 0.004),
    ("knee Qwen3-4B 32k, 4 bit", "results/ppl32k.csv",  "quest4",   0.01, "ppl", 1.199, 0.004),
    ("knee SmolLM2 8k, 4 bit",   "results/ppl_smol.csv","quest4",   0.01, "ppl", 1.301, 0.004),
    ("knee Phi-3.5 8k, 4 bit",   "results/ppl_phi.csv", "quest4",   0.01, "ppl", 1.105, 0.004),
    ("BitNet 5%",                "results/ppl_bitnet.csv","quest4", 0.05, "ppl", 1.113, 0.004),
    ("Qwen3-4B 4k 5% (controllo)","results/ppl_qwen4k.csv","quest4",0.05, "ppl", 1.012, 0.004),
    ("trasferimento: bound 4 bit","results/indici.csv",  "quest4",   0.02, "ppl", 1.078, 0.004),
    ("trasferimento: bound fp16", "results/indici.csv",  "quest",    0.02, "ppl", 1.074, 0.004),
    ("trasferimento: SparQ 2 bit","results/indici.csv",  "sparq16q2",0.02, "ppl", 0.943, 0.004),
    ("trasferimento: SparQ 4 bit","results/indici.csv",  "sparq16q4",0.02, "ppl", 0.926, 0.004),
    ("trasferimento: SparQ fp16", "results/indici.csv",  "sparq16",  0.02, "ppl", 0.926, 0.004),
    ("testa a testa: SparQ4 byte","results/indici.csv",  "sparq16q4",0.02, "byte", 0.059, 0.002),
    ("testa a testa: SparQ16 byte","results/indici.csv", "sparq16",  0.02, "byte", 0.098, 0.002),
    ("leve: pagina 16 + 4 bit",  "results/pagine.csv?pagina=16", "quest4",   0.02, "ppl", 1.011, 0.006),
    ("leve: pagina 64 + fp16",   "results/pagine.csv?pagina=64", "quest",    0.02, "ppl", 1.080, 0.006),
    ("trasf. Phi, 4 bit",        "results/indici2.csv?modello=microsoft/Phi-3.5-mini-instruct",
     "sparq16q4", 0.02, "ppl", 0.981, 0.004),
    ("trasf. Phi, fp16",         "results/indici2.csv?modello=microsoft/Phi-3.5-mini-instruct",
     "sparq16",   0.02, "ppl", 0.980, 0.004),
    ("trasf. SmolLM2, 4 bit",    "results/indici2.csv?modello=HuggingFaceTB/SmolLM2-135M",
     "sparq16q4", 0.02, "ppl", 1.006, 0.004),
    ("trasf. SmolLM2, fp16",     "results/indici2.csv?modello=HuggingFaceTB/SmolLM2-135M",
     "sparq16",   0.02, "ppl", 1.009, 0.004),
]

# Il compito a valle ha una metrica diversa (richiamo), quindi controlli a parte.
ATTESI_COMPITO = [
    ("compito: 4 bit @0,5%",  "quest4",    0.005, 1.00, 0.026),
    ("compito: fp16 @0,5%",   "quest",     0.005, 0.93, 0.072),
    ("compito: SparQ4 @0,5%", "sparq16q4", 0.005, 1.00, 0.032),
    ("compito: SparQ16 @0,5%","sparq16",   0.005, 1.00, 0.071),
    ("compito: casuale @5%",  "casuale",   0.05,  0.00, 0.054),
]


def main():
    problemi = 0
    print(f"  {'controllo':<32} {'nel paper':>10} {'nei dati':>10}  esito")
    cache = {}
    for desc, f, pol, b, campo, atteso, toll in ATTESI:
        if f not in cache:
            nome, _, q = f.partition("?")
            filtri = dict(p.split("=", 1) for p in q.split("&")) if q else None
            cache[f] = carica(nome, filtri)
        d = cache[f]
        if d is None:
            print(f"  {desc:<32} {'':>10} {'':>10}  FILE ASSENTE: {f}")
            problemi += 1
            continue
        if (pol, b) not in d:
            print(f"  {desc:<32} {atteso:10.3f} {'—':>10}  MANCA {pol}@{b}")
            problemi += 1
            continue
        v = d[(pol, b)][0 if campo == "ppl" else 1]
        ok = abs(v - atteso) <= toll
        print(f"  {desc:<32} {atteso:10.3f} {v:10.3f}  {'ok' if ok else 'DIVERSO'}")
        problemi += 0 if ok else 1

    try:
        c = {}
        for x in csv.DictReader(open("results/compito.csv")):
            c.setdefault((x["politica"], float(x["budget"])), []).append(
                (int(x["ok"]), float(x["frazione_byte"])))
        for desc, pol, b, ric, by in ATTESI_COMPITO:
            v = c[(pol, b)]
            r = sum(x[0] for x in v) / len(v)
            bb = st.mean(x[1] for x in v)
            ok = abs(r - ric) <= 0.04 and abs(bb - by) <= 0.002
            print(f"  {desc:<32} {100 * ric:9.0f}% {100 * r:9.0f}%  "
                  f"{'ok' if ok else 'DIVERSO'}")
            problemi += 0 if ok else 1
    except (FileNotFoundError, KeyError) as e:
        print(f"  compito a valle: non verificabile ({e})")
        problemi += 1

    print()
    if problemi:
        print(f"  {problemi} numeri non corrispondono: correggere il paper "
              f"PRIMA di sottomettere.")
        sys.exit(1)
    print("  Tutti i numeri verificati corrispondono ai dati.")


if __name__ == "__main__":
    main()
