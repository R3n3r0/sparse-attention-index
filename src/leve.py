"""Le due leve sul costo dell'indice sono equivalenti? Confronto a parita' di byte.

L'OBIEZIONE, CHE E' LA PRIMA CHE FAREBBE UN REVISORE.

Il costo dell'indice e' circa

    indice / KV  =  bit / (4 · dimensione_pagina)

quindi **dimezzare i bit e raddoppiare la pagina risparmiano gli stessi byte**.
Se pagine da 64 token con indice fp16 reggono quanto pagine da 16 con indice a
4 bit, il risultato sui 4 bit non e' un contributo: e' uno fra i modi di
ottenere lo stesso conto, e il piu' semplice vince.

PREVISIONE, DICHIARATA PRIMA DELLA MISURA. Le due leve non dovrebbero essere
equivalenti. Quantizzare a 4 bit allarga il limite ma **conserva la
risoluzione della scelta**; ingrandire la pagina la distrugge — con pagine
quattro volte piu' grandi si sceglie a blocchi quattro volte piu' grossolani, e
ogni pagina scelta trascina dentro token inutili. Quindi i 4 bit dovrebbero
vincere, e il divario crescere ai budget stretti.

Se la previsione e' sbagliata il risultato si ridimensiona, e va detto.

Uso:
    python src/leve.py                      # legge results/pagine.csv
"""

import argparse
import csv
import statistics as st


def carica(percorso):
    d = {}
    for x in csv.DictReader(open(percorso)):
        chiave = (int(x["pagina"]), x["politica"], float(x["budget"]))
        d.setdefault(chiave, []).append(
            (float(x["rapporto_ppl"]), float(x["frazione_byte"])))
    return {k: (st.mean(a for a, _ in v), st.mean(b for _, b in v))
            for k, v in d.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/pagine.csv")
    a = p.parse_args()
    d = carica(a.csv)

    pagine = sorted({k[0] for k in d})
    budget = sorted({k[2] for k in d})
    print(f"\n  Perplexity in piu', per dimensione di pagina e precisione "
          f"dell'indice\n")
    intest = "  ".join(f"{p:>4}t" for p in pagine)
    print(f"  {'budget':>7} {'indice':>7}   {intest}")
    for b in budget:
        for pol, et in (("quest4", "4 bit"), ("quest", "fp16")):
            celle = []
            for pg in pagine:
                v = d.get((pg, pol, b))
                celle.append(f"{100 * (v[0] - 1):+5.1f}%" if v else "    —")
            print(f"  {b:7.2f} {et:>7}   " + "  ".join(celle))
        print()

    # IL CONFRONTO CHE DECIDE: coppie che costano gli STESSI byte per strade
    # diverse. Raddoppiare la pagina e dimezzare i bit sono la stessa spesa.
    print("  A PARITA' DI BYTE, quale leva conviene\n")
    print(f"  {'budget':>7}   {'pagina 16 + 4 bit':>22}   "
          f"{'pagina 64 + fp16':>22}   verdetto")
    for b in budget:
        A = d.get((16, "quest4", b))
        B = d.get((64, "quest", b))
        if not A or not B:
            continue
        va, vb = 100 * (A[0] - 1), 100 * (B[0] - 1)
        chi = ("4 bit" if va < vb - 0.15 else
               "pagina" if vb < va - 0.15 else "pari")
        print(f"  {b:7.2f}   {va:+9.1f}% a {100 * A[1]:5.1f}% byte   "
              f"{vb:+9.1f}% a {100 * B[1]:5.1f}% byte   {chi}")

    print("\n  Se vince 'pagina', il contributo sui 4 bit e' ridondante:")
    print("  bastava usare pagine piu' grandi, che non richiede quantizzare.")


if __name__ == "__main__":
    main()
