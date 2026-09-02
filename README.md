# The Index Is the Hidden Cost of Sparse Attention

Benchmark code for measuring what the *index* of a sparse-attention method
costs, and how many bits it actually needs.

Sparse KV-cache retrieval reads only the entries a query needs — but choosing
them requires reading an **index over everything**, at every decoding step.
That term is rarely accounted for, and in the aggressive-sparsity regime that
makes these methods attractive it dominates the byte budget.

Everything here is training-free and applies to any pretrained transformer.

## Requirements

```
torch  transformers  datasets  matplotlib
```

Developed with PyTorch 2.11 (ROCm) on a Radeon 8060S. Nothing is
hardware-specific except the wall-clock benchmark.

## The benchmarks

**`recupero.py`** — quality against bytes read. Prefill is dense and shared;
selection acts during decoding. Comparisons are paired by construction: the
document is prefilled once and the cache is copied per policy, so the only
difference between arms is which entries are re-read.

```
python src/recupero.py --modello Qwen/Qwen3-4B --ctx 8192 --doc 24 \
  --posizioni 32 --budget 0.01,0.02,0.05,0.10 \
  --politiche casuale,quest2,quest4,quest8,quest,sparq16q4,sparq16,oracolo
```

Selection policies, with what each costs to store and read as an index:

| policy | what it does | index cost |
|---|---|---|
| `casuale` | random pages — the floor | 0 |
| `quest{n}` | per-page min/max bound, index quantized to *n* bits | ≈ *n*/(4·page) |
| `quest` | the same bound at `fp16` | 1/16 |
| `sparq{r}` | top-*r* query components, scored per token | *r*/(2·D) |
| `sparq{r}q{n}` | the same, components quantized to *n* bits | ≈ *rn*/(32·D) |
| `bit1`, `hamming`, `bit1c`, `bit1s` | 1-bit approximations of the score | 1/32 |
| `oracolo` | top-*k* by true attention scores | 1/2 |

`oracolo` is a strong reference, **not** an upper bound: ranking pages by their
maximum score does not minimise output distortion, and several bounded policies
beat it at generous budgets.

**`compito.py`** — passkey retrieval. Perplexity is dominated by nearby tokens,
so a method can discard all distant context and keep it nearly intact; the task
separates the two. Every haystack is first verified against dense attention and
discarded if the reference fails.

```
python src/compito.py --ctx 32768 --profondita 0.1,0.3,0.5,0.7,0.9 --semi 3
```

**`velocita.py`** — wall-clock. Unlike `recupero.py`, which masks and therefore
still reads everything, this gathers the selected entries so unselected ones
are genuinely not read. Also compares contiguous against scattered layout at
matched byte counts, which byte accounting cannot distinguish.

```
python src/velocita.py --ctx 8192,32768,65536 --frazione 0.05
```

**`leve.py`** — the two levers on index cost, compared at matched bytes: fewer
bits versus larger pages. They save identical bytes by construction; they do
not cost identical quality.

**`tetto.py`** — bandwidth-ceiling and index-cost accounting from a GGUF file.

**`figure.py`**, **`verifica_paper.py`** — figure generation and a consistency
check, both reading the result CSVs produced by the benchmarks above.

## Two silent failure modes

Both produce plausible numbers with no error, and both are guarded against in
the code:

1. **An attention implementation registered through `transformers` receives no
   causal mask.** An explicit implementation must build one; omitting it makes
   prefill bidirectional. This surfaced only because random selection appeared
   to beat exact selection, which is impossible.

2. **Chunked prefill is unsound** for models whose rotary embedding depends on
   sequence length (Phi-3.5's LongRoPE) and for models that recompute
   activation quantization scales per forward pass (BitNet). `recupero.py` runs
   a self-test and falls back to single-shot prefill when the chunked path does
   not reproduce it.

## Note on language

Docstrings and comments are in Italian; identifiers and this README are in
English. The docstrings carry the reasoning behind each design choice,
including the measurements that ruled alternatives out.

## Licence

MIT.
