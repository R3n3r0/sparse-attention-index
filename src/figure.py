"""Le due figure del paper, generate dai CSV invece che a mano.

Generarle dai dati e non disegnarle serve a una cosa sola: se un numero cambia,
la figura cambia con lui. Le tabelle di questo paper sono state riscritte tre
volte mentre i risultati cambiavano, e una figura disegnata a mano sarebbe
rimasta indietro senza che nessuno se ne accorgesse.

Figura 1 — la frontiera. Il risultato centrale: perplexity contro byte letti.
Ogni curva e' un metodo, ogni punto un budget. Piu' a sinistra e' meglio (si
legge meno), piu' in basso e' meglio (si perde meno qualita'). La distanza
ORIZZONTALE fra la curva a 4 bit e quella in piena precisione e' il risultato:
stessa qualita', molti meno byte.

Figura 2 — il ginocchio. Perplexity contro bit dell'indice, per i due
meccanismi. Mostra dove ciascuno si rompe, e che i due punti di rottura sono
diversi: il limite per pagina cede a 2 bit, il punteggio per componenti no.

Uso:
    python src/figure.py
"""

import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Grafica sobria: niente cornice a destra e in alto, niente griglia pesante,
# caratteri della stessa famiglia del testo.
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})

BUDGET = [0.01, 0.02, 0.05, 0.10]


def carica(percorso, filtro=None):
    d = {}
    for x in csv.DictReader(open(percorso)):
        if filtro and any(x.get(k) != v for k, v in filtro.items()):
            continue
        d.setdefault((x["politica"], float(x["budget"])), []).append(
            (float(x["rapporto_ppl"]), float(x["frazione_byte"])))
    return {k: (st.mean(a for a, _ in v), st.mean(b for _, b in v))
            for k, v in d.items()}


def serie(d, pol, budget=BUDGET):
    x, y = [], []
    for b in budget:
        if (pol, b) in d:
            p, by = d[(pol, b)]
            x.append(100 * by)
            y.append(p)
    return x, y


def frontiera(uscita):
    d = carica("results/indici.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    stili = [
        ("casuale",   "random",                  "#999999", "--", "o", 3.5, 1.1),
        ("quest",     "bound, fp16 index",       "#4A5457", "--", "s", 4.0, 1.3),
        ("quest4",    "bound, 4-bit index",      "#14655C", "-",  "s", 5.0, 2.0),
        ("sparq16",   "SparQ, fp16 index",       "#9B4A34", "--", "^", 4.0, 1.3),
        ("sparq16q4", "SparQ, 4-bit index",      "#C2571F", "-",  "^", 5.0, 2.0),
    ]
    for pol, et, col, ls, mk, ms, lw in stili:
        x, y = serie(d, pol)
        if x:
            ax.plot(x, y, ls, color=col, marker=mk, markersize=ms,
                    linewidth=lw, label=et)
    ax.axhline(1.0, color="#B0B0B0", linewidth=0.6, zorder=0)
    ax.text(19.4, 1.004, "dense attention", fontsize=7.5, color="#888888",
            ha="right", va="bottom")

    # la freccia che mostra il risultato: stessa qualita', meno byte
    if ("quest4", 0.02) in d and ("quest", 0.02) in d:
        p4, b4 = d[("quest4", 0.02)]
        pf, bf = d[("quest", 0.02)]
        ax.annotate("", xy=(100 * b4, p4), xytext=(100 * bf, pf),
                    arrowprops=dict(arrowstyle="->", color="#14655C",
                                    linewidth=1.0, shrinkA=4, shrinkB=4))
        ax.text((100 * b4 + 100 * bf) / 2, p4 + 0.018,
                r"same quality, $1.86\times$ fewer bytes",
                fontsize=7.5, color="#14655C", ha="center")

    ax.set_xlabel("bytes of the KV cache read per token (index included), \\%")
    ax.set_ylabel("perplexity relative to dense")
    ax.set_xlim(0, 20)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(uscita)
    print(f"  {uscita}")


def ginocchio(uscita):
    d = carica("results/indici.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bit = [2, 4, 8, 16]
    limite = [d.get((f"quest{b}", 0.02), d.get(("quest", 0.02)))[0]
              for b in (2, 4, 8)] + [d[("quest", 0.02)][0]]
    comp = [d[(f"sparq16q{b}", 0.02)][0] for b in (2, 4, 8)] + \
           [d[("sparq16", 0.02)][0]]
    ax.plot(bit, limite, "-s", color="#14655C", markersize=5, linewidth=2.0,
            label="per-page min/max bound")
    ax.plot(bit, comp, "-^", color="#C2571F", markersize=5, linewidth=2.0,
            label="query-component scoring")
    ax.axvline(4, color="#B0B0B0", linewidth=0.6, linestyle=":", zorder=0)
    ax.text(4.15, ax.get_ylim()[1], "4 bits", fontsize=7.5, color="#888888",
            va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks(bit)
    ax.set_xticklabels(["2", "4", "8", "fp16"])
    ax.set_xlabel("bits per index dimension")
    ax.set_ylabel("perplexity relative to dense")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(uscita)
    print(f"  {uscita}")


if __name__ == "__main__":
    import os
    os.makedirs("paper", exist_ok=True)
    frontiera("paper/frontiera.pdf")
    ginocchio("paper/ginocchio.pdf")
