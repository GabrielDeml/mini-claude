# mini-claude — Design Doc v0.3

**Status:** Draft. v0.1 → v0.2: revised after review (four Fable reviewers, adversarial verification, completeness critic). v0.2 → v0.3: planned for **Colab Pro** (L4 default GPU, compute-unit budget).
**Date:** 2026-11-24
**Scope:** how we train a small decoder-only transformer language model *from scratch* on Google Colab GPUs: what we build, in what order, and how we know each step worked. No code exists yet. This doc is the plan the code follows.

---

## 1. Goals and non-goals

### Goals

1. **Understand every line.** Write the model, tokenizer pipeline, data loader, training loop and sampler ourselves in plain PyTorch. No `transformers.Trainer`, no pretrained weights.
2. **Train a real model end to end on Colab.** First a character-level model (minutes), then a ~30M-parameter model that writes coherent short stories (~1–3 h on a Colab Pro L4 GPU), then a tiny instruction-tuned version you can chat with.
3. **Make runs reproducible and resumable.** Colab disconnects. Every run must survive a disconnect and resume from its last checkpoint.
4. **Measure, don't guess.** Every milestone has a numeric pass/fail check (loss target, throughput, test suite). Where no trustworthy reference exists, the threshold is marked *provisional* and is fixed by an earlier measurement.
5. **Leave a path to "mini Claude".** The design extends to broader pretraining and chat fine-tuning without a rewrite.

### Non-goals (v1)

- Matching any production model's quality. A 30M model trained for a few GPU-hours is a toy. The point is learning the full pipeline.
- Multi-GPU / distributed training. Colab gives one GPU.
- RLHF / preference training (future work only).
- Serving or a UI beyond a command-line / notebook sampler and a small chat REPL.

---

## 2. Primer: how training a language model works

Everything later in the doc implements these ideas. Terms in *italics* are also in the glossary (Appendix B).

### 2.1 The task: next-token prediction

A language model reads a sequence of tokens `t_1 … t_n` and, at every position `i`, outputs a probability distribution over the vocabulary for the *next* token `t_{i+1}`. Training text gives the right answer for free at every position, so one sequence of length `T` provides `T` training examples at once.

### 2.2 Tokens

Text becomes integers via a *tokenizer*.

- **Character-level:** each character gets an id (vocab ≈ 65 for Shakespeare). Trivial and bug-free.
- **Byte-pair encoding (BPE):** start with every byte (256 possible values) as its own token. Repeatedly find the most frequent adjacent pair of tokens in the training text and merge it into a new token. Stop when the vocabulary reaches the target size (we use 8,192). Encoding applies the learned merges in the same order. In practice the library first splits text into word-like chunks, so merges never cross word boundaries. Result: common chunks like `" once"` or `"ing"` become single tokens, and text becomes ~3–4× fewer tokens than characters. That means more text fits in the model's context and each step covers more text.

### 2.3 The model: a decoder-only transformer

Notation: `d_model` = the width of the vector that represents each token; `n_layers` = number of stacked blocks; `T` = context length (tokens the model sees at once).

1. **Embedding.** Each token id indexes a learned table and becomes a vector of size `d_model`.
2. **`n_layers` identical blocks**, each doing two things, each wrapped in a *residual connection* (`x = x + f(norm(x))`) so signal and gradients flow through deep stacks:
   - **Causal self-attention** — how tokens share information:
     - Every position computes three vectors from its own vector with learned linear maps: a **query** `q` ("what am I looking for?"), a **key** `k` ("what do I contain?") and a **value** `v` ("what do I pass on if selected?").
     - Position `i` scores every position `j` with `score(i, j) = q_i · k_j / √head_dim`.
     - **Causal mask:** scores for `j > i` (the future) are set to −∞, so position `i` can only use tokens it would actually have seen at generation time. Without this the model would cheat by reading the answer.
     - Softmax over `j` turns scores into weights that sum to 1; the output at `i` is the weighted sum of the `v_j`.
     - **Multi-head:** this runs `n_heads` times in parallel on slices of size `head_dim` (`n_heads × head_dim = d_model`), so different heads can track different relationships. Outputs are concatenated and projected back to `d_model`.
     - **Position:** the score above depends only on token *content*, so on its own the model cannot tell order or distance. We use **RoPE** (rotary position embeddings): `q` and `k` are rotated by an angle proportional to their position, which makes `q_i · k_j` depend on the offset `i − j`.
   - **MLP** — a per-position feed-forward network that transforms each vector independently (where much of the "knowledge" is stored).
3. **Output head.** A final normalization and a linear layer turn each vector into `vocab_size` scores (*logits*); softmax turns logits into probabilities.

### 2.4 The loss

*Cross-entropy:* `loss = −mean(log p(correct next token))`, measured in *nats*. Lower is better.
- A model guessing uniformly over `V` tokens scores `ln V` (≈ 9.01 for `V = 8192`, ≈ 4.17 for `V = 65`). A freshly initialized model starts at about this value.
- `exp(loss)` is *perplexity*: roughly "how many tokens the model is choosing between".
- Loss per token depends on the tokenizer (bigger tokens carry more information each). To compare tokenizers we convert to *bits per byte* (§6.5).

### 2.5 Learning

A *parameter* is one learned number in the model (an entry in a weight matrix). The *gradient* of a parameter says how much the loss would change if that parameter were nudged up. Training repeats:

1. Sample a *batch* of token windows from the training data.
2. **Forward pass:** compute logits and loss.
3. **Backward pass:** autograd computes the gradient for every parameter.
4. **Optimizer step:** AdamW moves each parameter a little against its gradient. The step size is the *learning rate* (LR).
5. Periodically measure loss on held-out **validation** data that the model never trains on. If training loss keeps falling but validation loss rises, the model is memorizing its training set (*overfitting*).

### 2.6 Generation

Feed a prompt, take the distribution at the last position, sample a token, append it, repeat. *Temperature* scales how random sampling is (0 = always pick the most likely token); *top-k* restricts sampling to the `k` most likely tokens.

### 2.7 Compute rules of thumb

- Training costs about `6 · N · D` floating-point operations (FLOPs), where `N` = parameters and `D` = training tokens (2 per parameter per token for the forward pass, 4 for the backward pass), plus a smaller attention term (§7).
- The "Chinchilla" heuristic says that *for a fixed compute budget*, the best split is roughly `D ≈ 20 · N` tokens. It is a guide for sizing runs, not a stopping rule: training a fixed model on more tokens still helps, just less efficiently.

### 2.8 The knobs of the training loop

- **Batch size / tokens per step.** How many tokens contribute to one optimizer step. Larger batches give less noisy gradients but fewer steps for the same data. *Gradient accumulation* runs several smaller "micro-batches" and adds up their gradients before one step, to get a big batch that doesn't fit in memory at once. It does not make training faster.
- **Steps, epochs.** One *step* = one optimizer update. One *epoch* = one full pass over the training data.
- **Learning-rate schedule.** *Warmup:* start the LR near 0 and ramp it up over the first few percent of steps, because Adam's running statistics are unreliable at the start and a full-size LR can blow up. *Cosine decay:* then lower the LR smoothly toward a small floor; smaller late steps let the model settle into a better minimum.
- **Weight decay.** Each step shrinks weights slightly toward zero; a mild regularizer that keeps weights from growing without bound.
- **Gradient clipping.** If the overall gradient norm exceeds a threshold (1.0), scale it down. Protects against one bad batch causing a huge update.
- **Dropout.** During training only, randomly zero a fraction of activations so the model can't rely on any single path. Helps when data is small and seen many times; unnecessary when every token is seen about once.
- **Mixed precision.** Do the big matrix multiplications in 16-bit floats (fast on GPU tensor cores) while keeping the master copy of weights in 32-bit. `fp16` has a small range, so tiny gradients can round to zero; a *GradScaler* multiplies the loss by a large factor before the backward pass and divides it back out afterwards. `bf16` has fp32's range and needs no scaler, but only newer GPUs support it natively.

---

## 3. Staged plan

One codebase grows through these stages. Each reuses everything before it.

| Stage | Data | Tokenizer | Model | Where | Expected time | Purpose |
|---|---|---|---|---|---|---|
| 0 | Tiny Shakespeare slices; synthetic | char | debug ~0.1M; mac ~0.9M | Mac CPU/MPS | seconds – few min | Unit tests; first real learning curve on the laptop |
| 1 | Tiny Shakespeare (~1.1M chars) | char (V=65) | ~10.6M | Colab L4 | ~10–15 min | Validates the whole loop on GPU; first look at overfitting |
| 2 | TinyStories V2 (GPT-4-generated stories) | our BPE (V=8192) | ~30M | Colab L4 | ~1–3 h (1 session; fixed by M1 measurement) | **Main v1 deliverable:** coherent stories |
| 2.5 | TinyStories-Instruct or synthetic prompts | same + chat tokens | 30M fine-tune | Colab L4 | ~15–30 min | First "mini Claude" moment: instruction following + chat REPL |
| 3 | TinyStories | same | 5M / 15M (30M reused) | Colab L4 | ~4–6 GPU-hours | Mini scaling study; LR sweep |
| 4 (future) | FineWeb-Edu sample + chat data | BPE (V≈32k) | ~125M | Colab A100 (L4 fallback) | ~4–6 h per run on A100 | General text, then chat fine-tuning |

**Why TinyStories for v1:** it is a dataset of simple children's stories generated by GPT-4, built specifically so that very small models can learn grammar, consistency and simple reasoning. A 30M model trained on web text produces word salad; a 30M TinyStories model produces readable stories. It is the fastest way to see a from-scratch model clearly "work". V2 is the GPT-4-only version (higher quality than V1, which mixed in GPT-3.5 stories).

---

## 4. System overview

```
 ┌──────────── Mac (local) ──────────────┐        ┌─────────────── Colab (GPU) ────────────────┐
 │ edit code, run CPU tests + mini runs   │        │ notebooks/colab_train.ipynb                 │
 │ git commit → PR → GitHub (CI: pytest)  ├──────▶ │  1. check GPU, print versions               │
 │                                        │        │  2. mount Google Drive   (user approves)    │
 │ Claude drives Colab via Chrome ────────┼──────▶ │  3. git clone/pull, pip install             │
 └────────────────────────────────────────┘        │  4. data: copy .bin Drive → /content        │
                                                   │  5. pytest -m gpu                           │
                                                   │  6. launch training (detached process)      │
                                                   │  7. monitor cell (tails metrics)            │
                                                   │  8. plot / sample (after or during run)     │
                                                   │  checkpoints + logs → local, then → Drive   │
                                                   └─────────────────────────────────────────────┘
```

Principles:
- **The repo is the source of truth.** The notebook is a thin launcher. We never edit model code inside Colab; changes go through git and review.
- **Google Drive is the persistent disk; `/content` is the fast disk.** Colab's local disk is wiped when the runtime is recycled. Data is copied Drive → `/content` at session start; checkpoints are written locally first, then copied to Drive (§6.4).
- **Same code on Mac and Colab.** Device, precision and `torch.compile` are chosen at runtime from the hardware.

---

## 5. Environment and repository

### 5.1 Layout

```
mini-claude/
├── README.md
├── pyproject.toml             # [tool.setuptools] packages = ["minic"]; requires-python >= 3.11
├── .github/workflows/test.yml # CPU pytest on every PR
├── docs/
│   ├── DESIGN.md              # this file
│   └── RESULTS.md             # one entry per milestone (§11)
├── minic/                     # the package ("mini-claude")
│   ├── config.py              # dataclasses: ModelConfig, TrainConfig, DataConfig, Config; loader + CLI overrides
│   ├── model.py               # RMSNorm, RoPE, Attention, SwiGLU MLP, Block, GPT, generate()
│   ├── tokenizer.py           # CharTokenizer, BPETokenizer (train/load/encode/decode/to_json)
│   ├── data.py                # prepare (download → tokenize → .bin + meta.json), batch sampler
│   ├── train.py               # training loop, LR schedule, eval, checkpoint/resume, logging
│   ├── sample.py              # generate from a checkpoint (CLI + function)
│   ├── chat.py                # Stage 2.5 chat REPL
│   ├── evaluate.py            # loss estimates, full-val pass, bits-per-byte, baselines
│   └── hardware.py            # device / dtype / compile selection, peak-FLOPs table, MFU, benchmark
├── configs/                   # debug, shakespeare_char_mac, shakespeare_char, tinystories_{5m,15m,30m}, sft_30m
├── notebooks/
│   ├── colab_spike.ipynb      # M0a plumbing spike (throwaway)
│   └── colab_train.ipynb      # thin launcher
└── tests/                     # CPU tests; GPU tests marked @pytest.mark.gpu
```

### 5.2 Environment decisions

- **Python:** `requires-python = ">=3.11"`. Mac uses a `uv` virtualenv pinned to the same major.minor as the Colab image (recorded at M0a).
- **Dependencies:** lower bounds only (`torch>=2.4`, `numpy`, `tokenizers`, `huggingface_hub`, `matplotlib`, `tqdm`); `pytest` in a `dev` extra. On Colab: `pip install -e . --no-deps` plus an explicit `pip install tokenizers huggingface_hub`, so Colab's preinstalled CUDA build of PyTorch is never replaced.
- **Config format:** each file in `configs/` is Python that defines `config = Config(model=..., train=..., data=...)`, loaded with `runpy`. Run-level flags are plain (`--config`, `--run_name`, `--fresh`, `--init_from`, `--benchmark`); any config field is overridden with a dotted key, e.g. `--train.peak_lr=5e-4`, `--train.compile=false`, `--train.persist_dir=...`. The fully resolved config is written to `runs/<name>/config.json` and embedded in every checkpoint.
- **Repo visibility:** stays public so Colab can clone without credentials. If it ever goes private, the user (not Claude) stores a GitHub token in Colab Secrets.

---

## 6. Component design

### 6.1 Tokenizer (`minic/tokenizer.py`)

**Interface (both tokenizers):** `encode(text, allow_special=False) → list[int]`, `decode(ids) → str`, `vocab_size`, `eot_id`, `to_json() → str`, `from_json(str)`. `allow_special=False` means text such as the literal string `"<|endoftext|>"` in a user prompt is encoded as ordinary text, never as a control token. Data prep and `chat.py` insert special-token ids directly. *Implementation note:* the `tokenizers` library always matches registered special tokens in input text, so after training we save `tokenizer.json` with its `added_tokens` list emptied (the strings stay in the vocab, so decoding still works), and our wrapper implements `allow_special=True` itself by splitting text on the five special strings and inserting their ids.

**Stage 0–1 — character-level.** Vocab = sorted set of characters in the training text; `eot_id = None` (Shakespeare is one document). Encoding a character outside the alphabet raises an error.

**Stage 2+ — byte-level BPE, V = 8192**, trained with the Hugging Face `tokenizers` library (Rust; trains on a few hundred MB in about a minute):
- ByteLevel pre-tokenizer with `add_prefix_space=False`, ByteLevel decoder, no normalizer.
- `initial_alphabet = ByteLevel.alphabet()` so all 256 byte values are in the vocab; together with byte-level encoding this guarantees any UTF-8 string can be encoded (no unknown tokens), even characters absent from the training sample.
- Decode with `skip_special_tokens=False` so round-trips are exact.
- Trained on ~200 MB of the train split only (never validation).
- **Special tokens at fixed ids 0–4:** `<|endoftext|>` (document separator), `<|user|>`, `<|assistant|>`, `<|system|>`, `<|end|>`. The chat tokens are used in Stage 2.5, so reserving them now avoids retraining the tokenizer and re-tokenizing the dataset there. The Stage 4 tokenizer (new data, V≈32k) keeps the same special tokens and ids.
- Why V = 8192: TinyStories uses simple vocabulary, so a small vocab still compresses well, and the embedding table (`V × d_model`) stays a modest share (14%) of a 30M model.

**Where the tokenizer lives:** `tokenizer.json` is saved next to the data on Drive, copied into each run directory, and **embedded as a JSON string inside every checkpoint** with its SHA-256. `sample.py` and `chat.py` rebuild the tokenizer from the checkpoint, so a checkpoint is self-contained on any machine. `train.py` refuses to start if the tokenizer hash in `meta.json` does not match the tokenizer it loaded.

**Optional exercise (off the critical path):** a pure-Python BPE trainer, tested against the library on small inputs.

### 6.2 Data pipeline (`minic/data.py`)

**Prepare (once per dataset).** Prepare is CPU-only work: run it on the Mac or a Colab CPU runtime when convenient; running it once on the GPU runtime is acceptable. Write to local disk, then copy finished files to Drive once (never write incrementally to Drive).

1. **Download.**
   - Tiny Shakespeare: direct download of `input.txt`; hold out the last 10% as validation.
   - TinyStories V2: `huggingface_hub.hf_hub_download(repo_id="roneneldan/TinyStories", repo_type="dataset", filename=...)` for `TinyStoriesV2-GPT4-train.txt` (~2.2 GB) and `TinyStoriesV2-GPT4-valid.txt` (~22 MB). Record the resolved revision hash. **Do not** use `datasets.load_dataset("roneneldan/TinyStories")`: its default configuration is the V1 data.
2. **Split documents** on the literal `<|endoftext|>` separator lines in the raw files; strip whitespace; drop empty stories.
3. **Train the BPE** (Stage 2 only) from an iterator over the first ~200 MB of train stories.
4. **Tokenize in streaming shards** (~50k stories at a time with `encode_batch`), append `eot_id` after each story, and append each shard as `uint16` to `train.bin` / `val.bin` (valid because V ≤ 65,535). Never hold the whole text or all token ids in memory; peak RAM stays under ~2 GB on a Colab VM.
5. **Write `meta.json` last** (so a half-finished prepare is never mistaken for a complete one), containing per split: token count, UTF-8 byte count, story count, tokens-per-story percentiles; plus vocab size, tokenizer SHA-256, dataset revision, unigram entropy of `train.bin` (a baseline, §9.3), git commit.
6. **Spot check** (M3): decode three random windows of `train.bin` and read them.

Size: ~0.5B tokens × 2 bytes ≈ 1 GB.

**Batches (every training step).**
- `np.memmap` the `.bin` file (never loaded fully into RAM) from local `/content`.
- Offsets come from a **stateless generator**: `np.random.default_rng([seed, step, micro_step])`. Each step's batch is a pure function of the step number, so resume needs no data-loader state and a resumed run sees exactly the batches it would have seen.
- Take `B` windows of `T + 1` tokens; `x = window[:-1]`, `y = window[1:]`; convert to int64; on CUDA, pin memory and copy with `non_blocking=True`.
- Windows may span a story boundary; that is standard practice and the `<|endoftext|>` token teaches the model where stories end. No cross-document attention masking in v1.

### 6.3 Model (`minic/model.py`)

**What changed since GPT-2 (2019):** GPT-2 used LayerNorm, a learned table of position vectors added to the embeddings, a GELU MLP with `4 · d_model` hidden units, and biases everywhere. Today's small open models (Llama-style) use the choices below. Each is small to implement, so we learn the current recipe.

| Component | Choice | Why |
|---|---|---|
| Normalization | **RMSNorm**, pre-norm (before attention and MLP) + final norm; computed in fp32 internally (`x.float() … .type_as(x)`) | Simpler than LayerNorm (no mean subtraction, no bias) with the same quality. Pre-norm keeps gradients well-behaved in deep stacks; the original post-norm transformer needed long, careful warmup. We still use a short warmup (§2.8). |
| Position | **RoPE** on q and k, base θ = 10,000 | No learned position table; scores depend on relative offset (§2.3) |
| Attention | Multi-head, `head_dim = d_model / n_heads` (64 in all GPU configs, 32 in debug/mac), causal, via `F.scaled_dot_product_attention(is_causal=True)` | Uses fused kernels (FlashAttention on Ampere+, memory-efficient kernel on T4) without writing CUDA |
| MLP | **SwiGLU**: `W2(silu(W1 x) ⊙ W3 x)`, `d_ff ≈ 8/3 · d_model`, rounded up to a multiple of 64 | Better quality per parameter than a GELU MLP. The `8/3` keeps parameter count equal to a `4·d_model` GELU MLP, since SwiGLU has three matrices instead of two. |
| Biases | None in linear layers | Fewer params, no quality loss |
| Embeddings | Input embedding **tied** to the output head (same matrix) | Saves `V × d_model` params (14% of the 30M model) |
| Init | Normal(0, 0.02); `Wo` and `W2` scaled by `1/√(2·n_layers)` | Keeps residual-stream variance stable with depth (GPT-2 recipe) |
| Dropout | 0 for TinyStories (~1 epoch); 0.2 for Shakespeare (tiny data, many epochs). When > 0, applied (a) to attention weights as `dropout_p = self.dropout if self.training else 0.0` — **SDPA does not look at `model.eval()`**, so this guard is mandatory — and (b) with `nn.Dropout` on the attention and MLP outputs before the residual add, and on the embedding output | |

**Forward:** `forward(idx: Long[B, T], targets: Long[B, T] | None = None, last_only: bool = False) → (logits, loss)`. Logits for all positions by default; `last_only=True` (used by `generate`) computes the head only for the last position. Loss uses `logits.float()` so cross-entropy is computed in fp32.

**Generation:** `generate(idx, max_new_tokens, temperature=0.8, top_k=200, stop_id=None, generator=None)`:
- each step feeds only the last `T` tokens (`idx[:, -T:]`), so prompts plus output can exceed the context;
- `top_k` is clamped to `min(top_k, vocab_size)`; `None` disables it; `temperature=0` means greedy argmax;
- stops early when `stop_id` is produced (`<|endoftext|>` for stories, `<|end|>` for chat);
- runs under `torch.no_grad()` in eval mode and restores the previous mode afterwards;
- recomputes the full context each step (no KV cache). Simple and fast enough at these sizes; a KV cache is an optional later exercise.

**Configurations.** Parameter count = `n_layers × (4·d² + 3·d·d_ff + 2·d) + d + V·d` (embedding counted once because it is tied). A unit test asserts the code matches this formula. Config names are nominal.

| Config | V | d_model | n_layers | n_heads | d_ff | T | Params total | Non-embedding |
|---|---|---|---|---|---|---|---|---|
| debug | 65 | 64 | 2 | 2 | 192 | 64 | 0.11M | 0.11M |
| shakespeare_char_mac | 65 | 128 | 4 | 4 | 384 | 64 | 0.86M | 0.85M |
| shakespeare_char | 65 | 384 | 6 | 6 | 1024 | 256 | 10.65M | 10.62M |
| tinystories_5m | 8192 | 256 | 4 | 4 | 704 | 512 | 5.31M | 3.21M |
| tinystories_15m | 8192 | 384 | 8 | 6 | 1024 | 512 | 17.31M | 14.16M |
| **tinystories_30m** | 8192 | 512 | 8 | 8 | 1408 | 512 | **29.89M** | 25.70M |

### 6.4 Training loop (`minic/train.py`)

```
load + resolve config → build raw_model; model = compile(raw_model) if enabled
optimizer, scaler = build(...)  → resume from checkpoint if present (§6.4 resume rules)
for step in range(start_step, max_steps):
    lr = schedule(step); set lr on optimizer
    for micro in range(grad_accum):
        x, y = get_batch("train", step, micro)            # stateless RNG
        with autocast(device_type, dtype, enabled=(dtype != float32)):
            _, loss = model(x, y)
        scaler.scale(loss / grad_accum).backward()
    scaler.unscale_(opt); grad_norm = clip_grad_norm_(params, 1.0)
    if not finite(loss): save debug info, abort (never checkpoint a NaN)
    if not finite(grad_norm): fp16 overflow → scaler.step skips this step, scaler.update halves the scale;
        count and log it; abort only after 5 consecutive skips or scale < 1 (bf16/fp32: abort immediately)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    every log_interval:  log step, loss (EMA), lr, grad_norm (last and max since last log), scaler scale, tokens/s, MFU, GPU mem
    every eval_interval: eval mode + no_grad on raw_model: train & val loss estimates, fixed-prompt samples → train mode
    every ckpt_minutes (wall-clock) and at the end: checkpoint
```

**Optimizer:** AdamW, ε = 1e-8, fused implementation on CUDA. Weight decay 0.1 on every parameter with ≥ 2 dimensions (all linear weights and the tied embedding/output matrix, as in nanoGPT and llama2.c); no decay on RMSNorm gains. A unit test asserts the two groups.

**LR schedule:** linear warmup → cosine decay to 10% of peak. For TinyStories configs, warmup is a *fraction of total steps* (4%, minimum 100 steps), so it stays sensible when tokens/step changes. Stage 0–1 configs set it explicitly (§7.1: nanoGPT's 100 steps; debug uses a constant LR). Because cosine decay is tied to `max_steps`, the total length of a run must be fixed before launch; a finished run cannot simply be "continued longer".

**Tokens per step and accumulation:** tokens per optimizer step is a *tuned hyperparameter*. Accumulation is only needed when the chosen batch doesn't fit in memory; at a fixed token budget, a 4× larger batch means 4× fewer updates. The M3.5 pilot compares 32,768 tokens/step (micro-batch 64 × 512, no accumulation, ~18k steps over 600M tokens) against 131,072 (× 4 accumulation, ~4.6k steps) and keeps whichever gives lower loss per token. Configs set a default micro-batch (§7.1); `hardware.py` only halves it (doubling `grad_accum`) if the first step runs out of memory. `grad_accum = tokens_per_step / (micro_batch × T)` must be an integer.

**Gradient clipping:** global norm 1.0. Grad-norm spikes are the earliest warning of instability.

**Mixed precision.** Selection rule in `hardware.py`: `major, _ = torch.cuda.get_device_capability()`. If `major ≥ 8`: bf16 autocast, TF32 on, no scaler. Otherwise fp16 autocast with `torch.amp.GradScaler("cuda")`. (Do **not** use `torch.cuda.is_bf16_supported()`: it returns True on a T4 via slow emulation.) The scaler is always constructed with `enabled=(dtype == float16)`, so the loop is identical on every device. Master weights and optimizer state stay fp32.

| GPU | Capability | Autocast | GradScaler | Peak dense 16-bit TFLOP/s (for MFU) |
|---|---|---|---|---|
| T4 | 7.5 Turing | fp16 | yes | 65 |
| **L4 (our default)** | 8.9 Ada | bf16 | no | 121 |
| A100 | 8.0 Ampere | bf16 | no | 312 |
| Mac MPS / CPU | — | fp32 | no | — (tests and small runs only) |

Unknown GPUs log tokens/s without MFU. Notebook cell 1 prints the chosen dtype.

**`torch.compile`:** keep `raw_model` and `model = torch.compile(raw_model)`. Only the training forward/backward uses the compiled `model`. Evaluation, sampling and `state_dict()` use `raw_model` (avoids recompiling for every new sequence length during generation, and checkpoints load with or without compile). Compile is on by default on CUDA for runs longer than ~15 minutes, off for debug runs, LR probes, MPS and CPU. Compile time is logged separately, and the first ~10 steps are excluded from tokens/s and MFU.

**Checkpointing** (critical on Colab; the Drive mount is a network filesystem with asynchronous upload, not a local disk):
- **Contents:** a dict of tensors and plain Python types, loadable with `torch.load(weights_only=True)`: model state, optimizer state, scaler state, step, resolved config (as a dict) and its hash, tokenizer JSON + SHA-256, torch CPU and CUDA RNG state (only matters with dropout), best val loss, cumulative training seconds, tokens seen, git commit, torch version. No NumPy RNG state is needed thanks to stateless batch sampling.
- **Write path:** `torch.save` to local `/content/runs/<name>/ckpt.pt.tmp` → `os.replace` to `ckpt.pt` locally → copy to Drive as `ckpt.pt.tmp` → rename the existing Drive `ckpt.pt` to `ckpt_prev.pt` → rename `ckpt.pt.tmp` to `ckpt.pt` → write a small `ckpt.json` (step, size, SHA-256) last. If the Drive copy fails, log the error and keep training.
- **Honest guarantee:** the *previous* checkpoint stays intact. A checkpoint written in the last minute or two before a runtime is reclaimed may not have reached Drive. On resume, try `ckpt.pt`; if loading fails or the hash disagrees with `ckpt.json`, fall back to `ckpt_prev.pt`, and log which file was used. `train.py` never unmounts (it is Colab-agnostic); notebook cell 9 calls `drive.flush_and_unmount()` once, after the last run of the session.
- **Cadence:** by wall-clock, every ~15 minutes, plus at the end. Independent of `eval_interval`.
- **Weights-only exports:** `best.pt` (lowest val loss, ~120 MB fp32 for 30M) only for multi-epoch runs (Shakespeare) where the final model is not the best; single-pass runs export `final.pt` once at the end. A fp16 export (~60 MB) is what the user downloads to the Mac for local sampling. Exports drop optimizer, scaler and RNG state but keep the config and tokenizer JSON, so `sample.py` works from them.
- **Drive budget:** full checkpoint ≈ 12 bytes/param ≈ 360 MB for 30M (weights + two Adam moments), × 2 generations, + 1 GB data. Replaced files may sit in Drive's Trash (for 30 days) and still count against quota. M0a checks this. If so, a 15-minute cadence adds ~1.4 GB of Trash per training hour (~1.5–4.5 GB for a 1–3 h M4 run), so the Drive copy drops to every 30 minutes and the user empties Trash once after the run. Free Drive is 15 GB shared with Gmail and Photos. Startup logs free space and warns below 5 GB.

**Resume rules:**
- A run has a **local dir** (`/content/runs/<name>`) and a **persistent dir** (Drive `…/mini-claude/runs/<name>`); on the Mac they are the same. On start, if the local dir has no checkpoint but the persistent dir does, first copy `ckpt.pt`, `ckpt_prev.pt`, `ckpt.json`, `metrics.jsonl` and `samples.jsonl` to the local dir, so the log history is continued rather than overwritten. Then resume automatically. Starting over under an existing name requires `--fresh`.
- If the saved GradScaler state is empty (checkpoint written under bf16 or on CPU) and the current scaler is enabled, skip loading it.
- Compare the checkpoint's config hash with the current resolved config; abort on any difference outside an allow-list (`local_dir`, `persist_dir`, `ckpt_minutes`, `compile`, micro-batch/accumulation at equal tokens-per-step).
- Warn loudly (and log) if the git commit differs from the checkpoint's.
- Append a `{"event": "resume", "from_step": k, ...}` record to the metrics log.

**Logging:** one compact line per `log_interval` to stdout (no tqdm progress bars: they are unreadable in captured logs), and the same record as JSON to `runs/<name>/metrics.jsonl`. Each record carries `step`, `session_id`, `git_commit`, `elapsed_train_s` (cumulative across sessions), `tokens_seen`. After a resume some steps are logged twice; readers keep the last record per step. Fixed-prompt samples go to `samples.jsonl`. Logs live locally and are copied to Drive with each checkpoint. No third-party tracker in v1.

### 6.5 Evaluation (`minic/evaluate.py`)

All evaluation runs on `raw_model` in `eval()` mode under `torch.no_grad()`, then restores train mode. It uses its own `torch.Generator` / NumPy generator seeded from `eval_seed`, so it never consumes or reseeds the training RNG.

- **Periodic loss estimates:** train and val loss, each averaged over the same fixed set of `eval_batches` windows every time. These two numbers (not the noisy, dropout-affected per-step training loss) are what we compare for overfitting.
- **Milestone number:** one full pass over `val.bin` in non-overlapping windows of length `T`. This is the number recorded in `RESULTS.md`.
- **Bits per byte (bpb):** `val_loss_nats × (val_tokens / val_bytes) / ln 2`, with token and byte counts for the val split from `meta.json` (`<|endoftext|>` counts as a token and adds no bytes). bpb lets us compare tokenizers and vocab sizes on the *same* dataset. Shakespeare and TinyStories numbers are never compared with each other.
- **Baselines** (from `meta.json`): `ln V` (uniform guessing) and the unigram entropy of the training tokens (a model that ignores context entirely). A working model must be far below both.
- **Fixed-prompt samples** at each eval, fixed seed: `"Once upon a time, there was a little"`, `"Tom and his dog went to the park."` (Stage 1 uses `"ROMEO:"`).
- **Throughput:** tokens/s and *MFU* (model FLOPs utilization = achieved FLOP/s ÷ GPU peak). Training FLOPs per token = `6N + 12 · n_layers · T · d_model` (the second term is attention's `QKᵀ` and weighted-sum of `V`, which `6N` doesn't count).
- Later (Stage 3+): grade samples with an LLM judge for grammar, consistency and creativity, as in the TinyStories paper.

### 6.6 Sampling and chat (`minic/sample.py`, `minic/chat.py`)

- `sample(checkpoint, prompt, n, max_new_tokens, temperature, top_k, seed)`: loads config and tokenizer **from the checkpoint**; works from a checkpoint copied into an empty directory on any machine.
- `chat.py` (Stage 2.5): a ~30-line REPL that wraps each user turn as `<|user|>…<|end|><|assistant|>` and generates until `<|end|>`.

### 6.7 Instruction fine-tuning (Stage 2.5)

Starting from the M4 checkpoint, a short supervised fine-tune (SFT) on prompt → story pairs formatted as `<|user|>Write a story about {topic} that uses the words {A}, {B}, {C}.<|end|><|assistant|>{story}<|end|>`, built from TinyStories-Instruct (`roneneldan/TinyStoriesInstruct`, which pairs stories with word/feature/summary prompts) or generated from TinyStories itself. Differences from pretraining: lower peak LR (~10% of pretraining), ~1 epoch, and **loss only on assistant tokens** (targets for the prompt are set to `-100`, the ignore index). This exercises the whole chat pipeline (template, special tokens, loss masking, stop token, REPL) cheaply (~1 CU), long before Stage 4.

Mechanics: `train.py --init_from <final.pt>` loads weights only (fresh optimizer and scaler, step 0, new run name), so the resume guard does not apply. SFT data is stored one example per row, padded to `T`, in a separate file; targets are `-100` for prompt and padding positions (the random-window `.bin` format cannot carry masks). Config `sft_30m`: 32,768 tokens/step, peak LR 10% of the M4 value, 4% warmup, cosine to 10%, one pass over the SFT set, dropout 0.

---

## 7. Training configurations and compute budget

### 7.1 Training hyperparameters

| | debug | shakespeare_char_mac | shakespeare_char | tinystories_5m / 15m | tinystories_30m |
|---|---|---|---|---|---|
| micro-batch × T | 8 × 64 | 12 × 64 | 64 × 256 | 64 × 512 | 64 × 512 |
| tokens/step | 512 | 768 | 16,384 | from pilot | from pilot (32,768 or 131,072) |
| max steps | 300 | 2,000 | 5,000 | same tokens as 30M run (§10 M5) | 600M tokens / tokens-per-step |
| peak LR | 3e-3 constant | 1e-3 | 1e-3 | M5 sweep (default 1e-3) | from pilot (default 5e-4) |
| min LR / warmup | — / 0 | 1e-4 / 100 | 1e-4 / 100 | 10% / 4% of steps | 10% / 4% of steps |
| betas | (0.9, 0.95) | (0.9, 0.99) | (0.9, 0.99) | (0.9, 0.95) | (0.9, 0.95) |
| weight decay | 0 | 0.1 | 0.1 | 0.1 | 0.1 |
| dropout | 0 | 0 | 0.2 | 0 | 0 |
| log / eval interval | 10 / 50 | 50 / 250 | 10 / 250 | 10 / every ~32M tokens | 10 / every ~32M tokens (250 steps at 131k tokens/step, 1,000 at 32k) |
| eval_batches | 4 | 20 | 100 | 50 | 50 |
| checkpoint | end | end | every 15 min | every 15 min | every 15 min |
| compile | off | off | on | on (off for probes) | on (off for probes) |

Stage 1 and the Mac config follow nanoGPT's published Shakespeare settings, so its reference losses apply. Seed 1337 everywhere.

### 7.2 Compute budget

**Hardware: Colab Pro.** Pro is metered in *compute units* (CU): 100 per month, used up at a rate that depends on the GPU. Google does not publish the rates and changes them; community measurements in 2026 are roughly **T4 ≈ 1.1–1.2 CU/h, L4 ≈ 1.5–1.7 CU/h, A100 40 GB ≈ 5.3–5.4 CU/h, A100 80 GB ≈ 7–7.5 CU/h**. M0a and M1 record the real rates from Colab's resources panel. Pro does **not** include background execution (that is Pro+), and no GPU type is guaranteed.

| GPU | Memory | Peak 16-bit (dense) | Planning MFU | 30M-model tokens/s | Role |
|---|---|---|---|---|---|
| T4 | 16 GB | 65 TFLOP/s (fp16) | 10–15% | ~32–47k | M0a spike, fp16-path tests, fallback |
| **L4** | 24 GB | 121 TFLOP/s (bf16) | 10–25% | ~59–148k | **Default for Stages 1–3**: bf16 (no GradScaler), cheap per hour, usually available |
| A100 40 GB | 40 GB | 312 TFLOP/s (bf16) | 15–30% | ~230–460k | Fast iteration when available; Stage 4 |

CU rates roughly track peak FLOP/s (per CU/h: T4 ≈ 56, L4 ≈ 75, A100 ≈ 58 TFLOP/s), so per run the three GPUs cost a similar number of CU (below), the L4 slightly least at equal MFU. The L4 is the default because of that, because it runs bf16, and because it is usually available. M1 measures tokens/s on each GPU we can get and picks the one with the best tokens per CU.

**Stage 2, tinystories_30m:**
- FLOPs/token ≈ `6 × 29.9M + 12 × 8 × 512 × 512` ≈ 179M + 25M ≈ **205 MFLOP/token**.
- Tokens: 600M (≈ `20 × N`). TinyStories V2 is roughly half a billion tokens with our tokenizer (M3 measures it), so ≈ 1–1.3 epochs.
- Total ≈ 600M × 205M ≈ **1.2 × 10¹⁷ FLOPs**.
- Time on **L4**: 15% MFU → ~89k tokens/s → **~1.9 h** (≈ 3 CU); 10% → ~2.8 h; 25% → ~1.1 h. On an A100: ~22–44 min (≈ 2–4 CU). On a T4: ~3.5–5.3 h (≈ 4–6.5 CU). One session.
- **Go / no-go (M1):** if the benchmark projects 600M tokens at ≤ 5 CU and ≤ 4 h (L4: ≥ ~57k tokens/s at 1.7 CU/h; T4: ≥ 42k; A100: ≥ ~180k), run as planned. Otherwise follow the fallback ladder: (1) switch to the GPU with the best measured tokens per CU, (2) switch to tinystories_15m (still produces coherent stories), (3) reduce to ~400M tokens. These times are compute only; evals at the §7.1 interval add ~3%. Reducing the context to 256 saves little (attention is only ~12% of FLOPs).
- Memory: weights + grads + AdamW ≈ 16 bytes/param ≈ 0.5 GB; activations at 64 × 512 × 8 layers fit easily in 16 GB (T4) or 24 GB (L4).

**Stage 1, shakespeare_char:** 5,000 steps × 16,384 tokens ≈ 82M tokens (≈ 80 epochs of a 1M-character text, so it overfits by design and needs dropout). ≈ 6 × 10¹⁵ FLOPs: minutes of compute, ~10–15 min wall-clock on an L4 with compile and evals (~10–20 min on a T4).

**Stage 4 (reference only):** ~125M params × 2.5B tokens ≈ 2 × 10¹⁸ FLOPs → ~4–6 h on an A100 at ~40% MFU (≈ 21–32 CU per run on the 40 GB A100, a fifth to a third of a month's Pro allowance); ~18–30 h on an L4 at 15–25% MFU (≈ 28–52 CU). A100 availability is not guaranteed. The Stage 4 design doc must include a CU budget and an L4 fallback using the throughput measured in M1–M5.

**v1 compute-unit budget (Stages 0–3):**

| Milestone | GPU | GPU-hours | CU |
|---|---|---|---|
| M0a spike | T4 | ~0.5 | ~0.6 |
| M1 smoke test + benchmark | L4 (+ short T4 and A100 if available) | ~1.5 | ~3 |
| M2 Shakespeare | L4 | ~0.3 | ~0.5 |
| M3 data prep | Mac or CPU runtime | — | ~0 |
| M3.5 pilot | L4 | ~0.5–1 | ~1–2 |
| M4 TinyStories-30M | L4 | ~1–3 | ~2–5 |
| M4.5 instruct fine-tune | L4 | ~0.5 | ~1 |
| M5 scaling + LR sweep | L4 | ~4–6 | ~7–10 |
| **Total (×2 margin for reruns)** | | | **~30–45** |

v1 fits in one month's 100 CU with room for a longer M4 run (e.g. 2–3 epochs) if the results justify it. **Spending rule:** before connecting any GPU runtime, Claude states in chat the GPU, the planned runs, the estimated CU and the CU balance shown in the resources panel, and waits for a go-ahead. A separate explicit go-ahead is needed for any single run over ~5 CU, any GPU dearer than the L4 (A100, H100, G4), any rerun of a failed run, and once cumulative v1 spend passes 45 CU.

**Storage:** data ~1 GB per tokenized dataset + ~0.7 GB per active run (two full checkpoint generations) + weight exports. Delete full checkpoints of finished runs (keep exports).

---

## 8. Colab operations (Claude drives the browser)

### 8.1 Notebook cells (`notebooks/colab_train.ipynb`)

1. **Check:** `!nvidia-smi`; print Python, torch, CUDA versions, GPU capability, chosen dtype.
2. **Mount Drive:** `drive.mount('/content/drive')`. *The user approves this prompt.*
3. **Code:** if `/content/mini-claude` exists, `git -C … pull --ff-only`, else `git clone https://github.com/GabrielDeml/mini-claude`; `%cd /content/mini-claude` (a `!cd` would not persist); `pip install -q -e . --no-deps && pip install -q tokenizers huggingface_hub`; print `git rev-parse HEAD`.
4. **Data:** copy `train.bin`, `val.bin`, `meta.json`, `tokenizer.json` from Drive to `/content/data`. Fails with a clear message if `meta.json` is missing (prepare is a separate, explicit cell).
5. **GPU tests:** `!pytest -m gpu -q` (~1 min).
6. **Launch (detached):** `!nohup python -m minic.train --config configs/tinystories_30m.py --run_name ts30m-v1 > /content/train.log 2>&1 & echo $! > /content/train.pid`. Training runs as a background process, so the kernel stays free. Refuses to launch if the PID in `/content/train.pid` is still alive.
7. **Monitor:** loops every 60 s, printing the last `metrics.jsonl` record and whether the training PID is alive. Keeps a cell executing while training is alive and gives Claude one compact line to read; **exits when the training PID is no longer alive**. Interrupting it does not stop training.
8. **Plot / sample:** plot `metrics.jsonl` (last record per step); sample from the latest local `ckpt.pt` (or `final.pt` / `best.pt` once it exists). Can run mid-training after interrupting the monitor.
9. **Finish session:** `drive.flush_and_unmount()`, then `from google.colab import runtime; runtime.unassign()` to release the GPU. A connected runtime consumes compute units even when idle (confirm `unassign` works at M0a).

### 8.2 Run protocol

- **Claude** opens the notebook from GitHub (`colab.research.google.com/github/GabrielDeml/mini-claude/blob/main/notebooks/colab_train.ipynb`) in the user's Chrome, sets *Runtime → Change runtime type* to the GPU chosen for that run (L4 by default), runs cells, and reads the monitor cell's output.
- **Polling:** during a run, Claude checks the monitor output every ~10–15 minutes using scheduled wake-ups (not one long blocking wait) and records step, val loss, tokens/s. If the training process has died, Claude reads the tail of `/content/train.log` before doing anything else.
- **User-only steps** (Claude never does these): Google sign-in; the Drive access prompt **on every fresh runtime**; the "notebook not authored by Google / Run anyway" warning; any "are you still there?" or CAPTCHA prompt; storing any credential in Colab Secrets; any terms-of-service or consent dialog; any Colab Pro / compute-unit purchase or upgrade prompt; creating or verifying a Kaggle account; emptying Drive Trash or deleting files on Drive. Claude asks in chat before *Disconnect and delete runtime* (M0a, M1) and before passing `--fresh`.
- **Disconnects pause the run until the user is available.** Nothing is lost beyond the last checkpoint. Recovery: Claude reconnects. If `/content/train.pid` names a live process (only the browser disconnected), Claude re-runs only cell 7. Otherwise (fresh runtime) Claude runs cell 1, the user approves cell 2, and Claude runs cells 3–7; training resumes automatically from the checkpoint.
- Colab Pro has no background execution (that is Pro+): the notebook tab must stay open in the user's Chrome while a run is going, and fully unattended runs are not expected. Claude keeps the tab open and the monitor cell running.
- **Release the runtime.** Within one polling interval of a run finishing or dying, Claude runs cell 8 if needed and then cell 9. Releasing a runtime with no live training process needs no confirmation; Claude asks before deleting a runtime whose training process is still alive.

### 8.3 Constraints we design around

- Session length and idle limits are not published, even on Pro → resumable runs, 15-minute checkpoints, runs sized to one session.
- Local disk is ephemeral → everything that matters reaches Drive.
- GPU type is not guaranteed → `hardware.py` adapts dtype and compile; configs specify tokens per step, and the micro-batch is a default that may be halved on out-of-memory.
- Compute units can run out before the monthly renewal → track CU per run in `RESULTS.md`; once the balance is zero Colab reverts to free-tier limits (T4 when available, no CU); last resort, Kaggle Notebooks (free weekly GPU quota) with the same repo; only `--train.persist_dir` and `--data.dir` differ.
- The preferred GPU may be unavailable → `hardware.py` adapts; any of T4/L4/A100 can run every v1 config.

---

## 9. Testing and verification

### 9.1 CPU unit tests (Mac and GitHub Actions; `pytest`; < 1 min)

| Test | What it catches |
|---|---|
| Output shapes for `(B, T)` input; loss is a scalar; `last_only=True` returns `(B, 1, V)` | wiring bugs |
| **Causality** (eval mode, fp32): for `j ∈ {1, T//2, T−1}`, replace token `j` with a different id; logits at positions `< j` equal (atol 1e-6) **and** logits at `≥ j` differ (max abs diff > 1e-3) | the mask or RoPE leaking the future (the most dangerous silent bug), or a model that ignores input |
| Parameter count equals the §6.3 formula for every config | wrong sizes, accidental untied weights |
| Weight-decay groups: no-decay group is exactly the RMSNorm gains; the tied matrix appears once | optimizer grouping bugs |
| **Initial loss** in `[ln V − 0.02, ln V + 0.25]` (expected ≈ `ln V + 0.02²·d_model/2` + a small tied-embedding term: ≈ 4.2 debug, ≈ 4.3 shakespeare_char, ≈ 9.1 tinystories_30m) | bad init, wrong loss reduction |
| **Overfit one batch:** debug config, B=8, T=64, dropout 0, constant LR 3e-3, weight decay 0, 300 steps → loss < 0.1 (thresholds confirmed on the first local run) | broken backprop / optimizer / loss |
| **Dropout and eval mode:** with dropout 0.2, two `eval()` forwards give identical logits; two `train()` forwards differ | SDPA dropout active at eval time |
| RoPE: `q_i · k_j` after rotation depends only on `i − j` | RoPE implementation errors |
| `generate`: default args on V=65 (top_k clamp), `max_new_tokens > T` (context crop), output length, ids < V, same seed → same output, stops on `stop_id`, restores train mode | generation edge cases |
| Char tokenizer round-trips any string over its own alphabet and raises outside it. BPE round-trips arbitrary UTF-8 (emoji, CJK, tabs, newlines, leading spaces, `<|endoftext|>` as literal text with `allow_special=False`) | tokenizer bugs |
| Data: `y[:, :-1] == x[:, 1:]`; values < V; dtype int64; same `(seed, step, micro_step)` → same batch; raw separator text never appears in the token stream, only as `eot_id` | target off-by-one, prep bugs |
| LR schedule at step 0, end of warmup, middle, end | schedule off-by-one |
| **Checkpoint round-trip** (CPU, fp32, compile off, dropout 0.1): training `2k` steps straight vs `k` steps, save, load in a new subprocess, `k` more → per-step losses for steps `k+1…2k` and final weights are exactly equal (`torch.equal`) | missing checkpoint state |
| Resume guards: config change aborts; `--fresh` required to restart a name | silently resuming the wrong run |
| `sample.py` works from a checkpoint copied into an empty temp dir | checkpoint not self-contained |

### 9.2 GPU tests (`pytest -m gpu`, on the Colab runtime at M1 and before every long run; ~1 min)

- SDPA attention output equals a naive masked-softmax implementation (fp32).
- Causality under fp16/bf16 autocast (with a looser tolerance).
- Compiled vs uncompiled logits are close.
- 50 autocast steps at peak LR in the runtime's selected dtype (bf16 on L4/A100, fp16 on T4), and again with fp16 + GradScaler forced on: finite loss, non-zero grads, scaler scale not collapsing (fp16).
- 20-step train → save → resume in the selected dtype and with fp16 + GradScaler forced, compile on and off; loss after resume within tolerance (not bitwise; CUDA kernels are not deterministic).
- A checkpoint from a compiled model loads into an uncompiled one.

### 9.3 Reading the loss curve

Landmarks: step-0 loss ≈ `ln V` (+ ~0.1); the unigram entropy (a model that ignores context); for Shakespeare char, ~2.5 nats is roughly "knows letter pairs" and is reached within a few hundred steps, ~1.5 takes a few thousand.

| Symptom | Likely cause | First check |
|---|---|---|
| Stuck at `ln V` | LR ≈ 0, gradients not flowing, targets wrong | LR log, grad-norm, data test |
| Stuck near unigram entropy | model not using context: mask or position bug | causality and RoPE tests |
| Loss far too low very early, or step-0 loss well **below** `ln V` | future leak or `x`/`y` misalignment | causality and data tests |
| NaN / inf | fp16 overflow or LR too high | scaler scale and grad-norm before the event |
| Spiky loss, grad-norm spikes | LR too high, warmup too short | lower peak LR |
| Train falls, val rises | overfitting (expected for Shakespeare; a bug or split problem for TinyStories) | dropout, `best.pt`, data split |
| Jump after resume | state not restored, different code/config | resume event record, commit and config hash |

Samples improve in a predictable order: random characters (char model) or random word pieces (BPE model) → common words → grammatical sentences → consistent stories.

---

## 10. Milestones

| # | Milestone | Done when |
|---|---|---|
| **M0a** | Colab plumbing spike (half a day, before model work) | Claude opens `colab_spike.ipynb` from GitHub in Chrome, selects a T4 (cheapest), runs `nvidia-smi` and reads the output; the user approves the Drive prompt on each of the two runtimes; a 400 MB dummy file is written to Drive via tmp + rename, the runtime is deleted, and a fresh runtime reads it back intact (write speed recorded); a detached background process keeps running while another cell executes and while no cell runs for 10 min; check whether replaced Drive files land in Trash; record the image's Python/torch/CUDA versions; record the CU/hour shown for the T4 in the resources panel and the list of GPU types the runtime menu offers (L4 and A100 rates are recorded at M1 when those runtimes are first used); confirm `runtime.unassign()` releases the runtime. |
| **M0** | Model + tests + first local run (split into small PRs: model, then tokenizer/data, then train/sample) | All §9.1 tests pass on Mac and in CI; Fable review addressed; `shakespeare_char_mac` trains end to end on Mac CPU/MPS through the real `train.py` in a few minutes to val loss < 2.0 nats/char (nanoGPT's comparable CPU run reaches ~1.88); plot and sample produced; kill-and-resume done locally. |
| **M1** | Colab smoke test + benchmark | From a fresh L4 runtime: full `pytest -m gpu` passes (and once on a short T4 session to cover the fp16 + GradScaler path); a ~10-minute shakespeare_char run (checkpoint every ~1 min) is killed via *Disconnect and delete runtime* after ≥ 2 checkpoints, and on a fresh runtime resumes at the checkpoint's step with matching LR (and scaler scale on fp16) and next val loss within ~0.05 of the last pre-kill value; tokens/s recorded for ~200 steps each of fp32, bf16 autocast, and bf16 + compile (loss curves agree within noise); `--benchmark` on the 30M config (random tokens, ~50 post-compile steps) records tokens/s, MFU and tokens per CU on each GPU available (A100 only after the §7.2 go-ahead) → GPU choice and go/no-go per §7.2. |
| **M2** | Shakespeare char model | Best val loss (`best.pt`) ≤ 1.5 nats/char (nanoGPT's best for this setup ≈ 1.47); the plot shows the overfitting turn (val loss bottoms out and rises while train keeps falling); samples look like Shakespeare-formatted text. |
| **M3** | Tokenizer + TinyStories data | BPE trained (V=8192); round-trip tests pass on 1,000 dataset stories; `train.bin`/`val.bin`/`meta.json`/`tokenizer.json` on Drive; token/byte counts, unigram entropy and prep time recorded; spot check read; §7.2 budget updated with the real token count. |
| **M3.5** | Pilot (~0.5–1 GPU-hour on an L4) | (1) tinystories_5m runs end to end for ~10 min (shakes out the BPE data path on GPU). (2) 30M config at 32,768 tokens/step, ~300 steps each at peak LR {5e-4, 1e-3, 2e-3}, 100-step warmup then constant (no cosine), compile off: choose the largest LR with stable grad-norm and scaler and loss within ~2% of the best. (3) 32,768 vs 131,072 tokens/step, each for ~25M tokens at the LR chosen in (2): keep the lower loss per token. (4) With the chosen LR and tokens/step, run the first 10M tokens under the exact M4 config (same `max_steps`, so the same warmup) and record the val-loss estimate at 10M tokens as the M4 early gate. |
| **M4** | **TinyStories-30M** (main v1 deliverable) | Planned run completes with no NaN; at 10M tokens seen, the val-loss estimate is within 0.05 of the pilot gate (else stop and investigate); full-pass val loss ≤ 1.5 nats/token *(provisional, restated in bpb at M3; for scale, llama2.c's 15M and 42M models, trained far longer, report 1.07 and 0.85 with a 32k vocab)* and far below the unigram baseline; final train–val gap < 0.05; ≥ 8 of 10 fixed-seed samples (5 per fixed prompt) are grammatical, multi-sentence, with consistent characters. |
| **M4.5** | Tiny instruct fine-tune + chat REPL | SFT run completes; for 10 held-out prompts asking for a story using 3 given words, ≥ 8 responses are coherent stories containing all 3 words and stop at `<|end|>`; `chat.py` works on the Mac from the exported checkpoint. |
| **M5** | Scaling study + LR sweep | 5M and 15M trained on the same token budget as the M4 run (M4 is the 30M point), warmup as a fraction of steps, LR swept per size; val bpb decreases monotonically 5M → 15M → 30M; plot val bpb vs non-embedding params and vs FLOPs. Optional second series at 20 tokens/param. |
| **M6** | Stage 4 design doc | Separate doc for general pretraining + chat SFT, with a Colab Pro compute-unit budget built from M1–M5 throughput. |

---

## 11. Development process

- Small PRs to `main`, at most one milestone (often part of one) per PR. CI runs the CPU tests on every PR.
- **Every code change is reviewed by Fable agents** before merge: maths (shapes, masks, loss, schedule), PyTorch pitfalls (dtype, device, compile, eval mode), test coverage. Findings are verified before they are acted on.
- CPU tests must pass before anything runs on Colab, and `pytest -m gpu` must pass on the runtime before any long run, so GPU time goes to training rather than debugging.
- **Written to be read:** tensor operations in `model.py` carry shape comments (e.g. `# (B, n_heads, T, head_dim)`), and each module starts with a short header pointing at the doc section it implements.
- **`docs/RESULTS.md`:** the PR that closes each milestone appends an entry: config, git commit, hardware, tokens/s, MFU, final train/val loss and bpb, the loss plot, three sample excerpts.
- Every run records its git commit and full config in the checkpoint and the metrics log.
- **Secrets:** never pasted into notebooks or committed. Any token (HF, W&B, GitHub) goes into Colab Secrets, entered by the user.

---

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Browser-driven Colab or Drive behaves differently than assumed | Late discovery after building everything | M0a spike before model work |
| Colab disconnects / quota limits | Lost training time | 15-min checkpoints with previous-copy fallback; auto-resume; runs sized to one session |
| Drive re-auth needed on every fresh runtime | Run paused until the user is back | Documented user step; nothing lost beyond last checkpoint |
| Presence / CAPTCHA prompts | Run paused | Claude stops and notifies the user; never bypassed |
| Drive quota or Trash fills up | I/O errors mid-run | Storage budget (§7.2); weights-only exports; delete finished runs' full checkpoints; user empties Trash; startup free-space check |
| GPU slower than estimated | Run costs more CU or more hours than budgeted | M1 benchmark + go/no-go + fallback ladder (§7.2) |
| Untested hyperparameters on the long run | Wasted hours | M3.5 pilot; M4 early gate at 10M tokens |
| fp16 instability (T4 fallback only; L4/A100 use bf16) | NaN loss | GradScaler; loss and RMSNorm in fp32; clipping; abort on non-finite loss or repeated skipped steps; monitor scaler scale |
| Silent mask / RoPE / dropout-in-eval bug | Model "learns" but behaves wrongly | §9.1 causality, RoPE and eval-mode tests |
| Off-by-one in targets | Suspiciously low loss | Data test; step-0 loss below `ln V` symptom |
| GPU-only code paths untested | Crash or NaN hours in | `pytest -m gpu` before every long run |
| Resuming under changed code/config | Corrupted run | Config-hash guard, commit warning, `--fresh` |
| `pip install` replaces Colab's torch | Long install, CUDA mismatch | `--no-deps` install; lower-bound deps only |
| `torch.compile` problems | Crash / slow start | `--train.compile=false`; compile-agnostic checkpoints |
| Compute units run out / preferred GPU unavailable | Schedule slip | CU budget (§7.2) tracked per run; Claude asks before costly runs; free-tier T4 at zero balance, or Kaggle |
| Browser tab closed mid-run (no background execution on Pro) | Runtime disconnects | Tab stays open; checkpoints + resume; monitor cell keeps the session active |
| Expecting real chat from a 30M TinyStories model | Disappointment | Stated up front; Stage 2.5 gives a story-instruction "mini Claude"; general chat needs Stage 4 |

---

## 13. Future work (Stage 4 sketch)

1. **General pretraining:** ~125M model on a few billion tokens of FineWeb-Edu, V≈32k tokenizer (same special tokens), on a Colab A100 in one session (L4 fallback: several resumed sessions). Expected quality at best roughly GPT-2-small class. Projects such as Karpathy's nanochat are a useful reference for this full pipeline at small scale.
2. **Chat SFT** on general conversation data, reusing the Stage 2.5 code path.
3. **Inference:** KV cache, nicer chat REPL.
4. **Optional:** preference tuning (DPO); small benchmarks (e.g. HellaSwag); other optimizers (e.g. Muon); architecture ablations as exercises.

---

## 14. Decisions and open questions

**Decided:**
- No W&B or other tracker in v1 (local JSONL + matplotlib); can be added later behind a flag with a user-entered key.
- The pure-Python BPE trainer is an optional exercise, off the critical path.
- **Colab tier: Colab Pro** (100 CU/month, no background execution). L4 is the default GPU; budget in §7.2.

**Open:**
1. Drive free space available for the project (§6.4 warns below 5 GB; the M4 run needs ~3 GB plus any Trash churn).

---

## Appendix A: Data and licences

- **Tiny Shakespeare:** public-domain text (via the char-rnn/nanoGPT `input.txt`).
- **TinyStories:** `roneneldan/TinyStories` on Hugging Face; CDLA-Sharing-1.0 per its dataset card at time of writing (re-check at M3).
- **TinyStories-Instruct:** `roneneldan/TinyStoriesInstruct` (`TinyStories-Instruct-train.txt`, ~2.7 GB; `-valid.txt`, ~27 MB); its dataset card is nearly empty, so check the licence at M4.5.
- Dataset revisions are pinned in `meta.json`.

## Appendix B: Glossary

- **Autocast:** PyTorch context that runs selected operations (mainly matmuls) in 16-bit while keeping the rest in fp32.
- **Batch / micro-batch:** the token windows used for one optimizer step / for one forward-backward pass within it.
- **bf16 / fp16:** 16-bit float formats; bf16 has fp32's range with less precision, fp16 more precision with a much smaller range.
- **Bits per byte (bpb):** loss converted to bits per byte of original text; tokenizer-independent.
- **Cosine decay / warmup:** LR schedule shapes (§2.8).
- **Dropout:** randomly zeroing activations during training (§2.8).
- **Epoch:** one full pass over the training data.
- **Gradient accumulation:** summing gradients over several micro-batches before one optimizer step.
- **Gradient clipping:** scaling the gradient down when its norm exceeds a threshold.
- **GradScaler:** scales the fp16 loss up before backward so small gradients don't underflow, then unscales before the step.
- **KV cache:** storing past keys/values during generation so each new token costs one position instead of the whole context.
- **memmap:** reading a file as an array on demand without loading it into RAM.
- **MFU:** model FLOPs utilization: achieved training FLOP/s ÷ the GPU's peak.
- **Nats:** natural-log units of loss (`ln`); bits use `log2` (1 nat ≈ 1.44 bits).
- **Perplexity:** `exp(loss)`.
- **Pre-norm:** normalizing the input of each sublayer rather than its output.
- **SDPA:** `scaled_dot_product_attention`, PyTorch's fused attention function.
- **SFT:** supervised fine-tuning on prompt/response pairs.
- **TF32:** a reduced-precision fp32 matmul mode on Ampere+ GPUs; faster, practically lossless for training.
- **`torch.compile`:** JIT-compiles the model into fused GPU kernels for speed.
- **Weight decay:** shrinking weights slightly every step (§2.8).
- **Weight tying:** using the same matrix for the input embedding and the output head.

## Appendix C: References

- Vaswani et al., 2017 — *Attention Is All You Need*.
- Radford et al., 2019 — GPT-2.
- Kaplan et al., 2020; Hoffmann et al., 2022 — scaling laws; `6ND`; Chinchilla.
- Chowdhery et al., 2022 — PaLM (Appendix B: FLOPs-per-token and MFU).
- Su et al., 2021 — RoFormer (RoPE). Shazeer, 2020 — GLU variants (SwiGLU). Zhang & Sennrich, 2019 — RMSNorm.
- Eldan & Li, 2023 — *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
- Karpathy — nanoGPT, llama2.c, nanochat (reference implementations; we write our own and use theirs to sanity-check numbers).
