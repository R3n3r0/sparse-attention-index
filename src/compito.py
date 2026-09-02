"""Il compito a valle: la selezione conserva l'INFORMAZIONE, non solo la fluenza?

PERCHE' LA PERPLEXITY NON BASTA.

`recupero.py` misura di quanto peggiora la predizione del token successivo. E'
la metrica standard del settore, ma ha un difetto noto per questa domanda: e'
dominata dai token vicini, che la finestra locale serve comunque. Un metodo che
butta via tutto il contesto lontano puo' avere una perplexity quasi intatta e
avere perso ogni informazione che stava li'.

Il richiamo di una chiave nascosta separa le due cose: o il modello la ritrova
o no, e la chiave sta a una profondita' che solo la selezione puo' raggiungere.
E' anche il compito che riportano Quest, SparQ e InfLLM, quindi i numeri sono
confrontabili.

IL CONTROLLO CHE RENDE LEGGIBILE TUTTO. Per ogni pagliaio si verifica PRIMA che
il modello ad attention piena trovi la chiave. Se non la trova, il pagliaio si
scarta: un banco in cui il riferimento fallisce misura la fortuna del prompt,
non la politica. (Vale la pena ricordarlo: con il template di chat Qwen3-4B
risponde "questo testo sembra ripetitivo e senza senso" e fallisce sempre; in
completamento grezzo ritrova la chiave. Il formato non e' un dettaglio.)

Il prefill e' denso e si fa UNA volta per pagliaio: tutte le politiche partono
dallo stesso identico stato, e l'unica differenza e' quanta cache rileggono a
ogni token generato.

Uso:
    python src/compito.py --ctx 32768 --profondita 0.1,0.3,0.5,0.7,0.9
"""

import argparse
import copy
import random

import torch

from recupero import BLOCCO, Scelta, attenzione_selettiva, costo_indice

RIEMPIMENTO = ("The grass is green. The sky is blue. The sun is yellow. "
               "Here we go. There and back again.\n")
PREAMBOLO = ("There is an important piece of info hidden inside a lot of "
             "irrelevant text. Find it and memorize it.\n\n")
DOMANDA = "\nWhat is the pass key? The pass key is"


def pagliaio(n, frazione, segreto):
    corpo = [RIEMPIMENTO] * n
    corpo.insert(min(int(frazione * n), n),
                 f"The pass key is {segreto}. Remember it. "
                 f"{segreto} is the pass key.\n")
    return PREAMBOLO + "".join(corpo) + DOMANDA


@torch.no_grad()
def genera(model, cache, primo, n=14):
    c = copy.deepcopy(cache)
    out, cur = [], primo
    for _ in range(n):
        r = model(cur, past_key_values=c, use_cache=True)
        cur = r.logits[:, -1].argmax(-1, keepdim=True)
        out.append(cur.item())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modello", default="Qwen/Qwen3-4B")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--profondita", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--semi", type=int, default=3)
    p.add_argument("--budget", default="0.01,0.02,0.05")
    p.add_argument("--politiche", default="casuale,quest4,quest,sparq16q4,sparq16")
    p.add_argument("--csv", default="results/compito.csv")
    a = p.parse_args()

    from transformers import (AttentionInterface, AutoModelForCausalLM,
                              AutoTokenizer, DynamicCache)
    AttentionInterface.register("selettiva", attenzione_selettiva)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.modello)
    model = AutoModelForCausalLM.from_pretrained(
        a.modello, dtype=torch.bfloat16,
        attn_implementation="selettiva").to(dev).eval()
    c = model.config
    d_t = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads

    prof = [float(x) for x in a.profondita.split(",")]
    budget = [float(x) for x in a.budget.split(",")]
    politiche = a.politiche.split(",")
    n_riemp = max(8, a.ctx // 24)

    import csv as _csv
    import os
    os.makedirs("results", exist_ok=True)
    nuovo = not os.path.exists(a.csv)
    fh = open(a.csv, "a", newline="", buffering=1)
    w = _csv.writer(fh)
    if nuovo:
        w.writerow(["modello", "ctx", "politica", "budget", "seme",
                    "profondita", "segreto", "ok", "frazione_byte"])

    acc, scartati = {}, 0
    for seme in range(a.semi):
        rng = random.Random(2000 + seme)
        for f in prof:
            seg = rng.randrange(10 ** 7, 10 ** 8)
            ids = tok(pagliaio(n_riemp, f, seg),
                      return_tensors="pt").input_ids.to(dev)
            N = ids.shape[1]
            Scelta.politica = "pieno"
            base = DynamicCache()
            with torch.no_grad():
                for j in range(0, N - 1, BLOCCO):
                    model(ids[:, j:min(j + BLOCCO, N - 1)],
                          past_key_values=base, use_cache=True)
            coda = ids[:, -1:]

            testo = tok.decode(genera(model, base, coda))
            if str(seg) not in testo.replace(",", ""):
                scartati += 1
                print(f"  seme {seme} prof {f:.1f}: controllo FALLITO, scartato",
                      flush=True)
                continue
            print(f"  seme {seme} prof {f:.1f}: {N:,} token, controllo ok",
                  flush=True)

            for pol in politiche:
                for b in budget:
                    Scelta.politica, Scelta.budget = pol, b
                    Scelta.letti = Scelta.totali = 0
                    t = tok.decode(genera(model, base, coda))
                    fr = Scelta.letti / max(Scelta.totali, 1)
                    Scelta.politica = "pieno"
                    ok = str(seg) in t.replace(",", "")
                    byte = fr + costo_indice(pol, d_t)
                    acc.setdefault((pol, b), []).append((ok, byte))
                    w.writerow([a.modello, N, pol, b, seme, f, seg,
                                int(ok), f"{byte:.4f}"])
    fh.close()

    n_prove = len(next(iter(acc.values()))) if acc else 0
    print(f"\n  {n_prove} pagliai validi ({scartati} scartati dal controllo)\n")
    print(f"  {'politica':<11} {'budget':>7} {'byte':>8} {'richiamo':>9}")
    for pol in politiche:
        for b in budget:
            v = acc.get((pol, b))
            if not v:
                continue
            print(f"  {pol:<11} {b:7.2f} {100 * sum(x[1] for x in v) / len(v):7.1f}% "
                  f"{100 * sum(x[0] for x in v) / len(v):8.0f}%")
    print(f"\n  -> {a.csv}")
    print("\n  Il richiamo separa cio' che la perplexity confonde: un metodo")
    print("  puo' avere perplexity quasi intatta e aver perso l'informazione")
    print("  lontana, perche' la perplexity la fanno i token vicini.")


if __name__ == "__main__":
    main()
