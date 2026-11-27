# HANDOFF

Update at the end of every session.

## 2026-11-27
**Done:** `model.py` adapted (`GPT(n_in, A)`, paper size: embed 64, 2 layers, 4 heads, block_size 16, dropout 0 unverified) and commented. `train.py` rewritten: tensors from `data.py` and the tokenizer, `get_batch`, AdamW with warmup then linear decay (paper lr 2.5e-3, batch 256, 20 epochs), `test_loss`, `exact_match` (greedy generation, whole answer must match). **k1 reaches 100% exact-match on the test set** (n_train 10,000, n_test 2,000, seed 0), about 30 s on MPS.
**Workaround:** `nll_loss` segfaults on MPS with torch 2.0.0 on macOS 26, so the loss in `model.py` is computed on CPU (`logits.cpu()`). Real fix is `pip install -U torch` (not done).
**Next (Sat 11/28):** run split, k2 and direct (change `fmt` at the top of `train.py`), then 3 to 5 seeds per format and the error vs m plot, then parameter counts (body vs output layer). Sun: convergence check, write-up draft, README. Mon 11/30: finish, push, email Chaudhari.
**Decisions:** paper lr for every format, no lr check or tuning; loss on answer tokens via `ignore_index` (-100); deadline moved to Mon 11/30.
**Open:** size of D (n_train 10,000 and n_test 2,000 are placeholders); confirm dropout against the paper's Appendix A.3; `.DS_Store` untracked.

## 2026-11-26
**Done:** pushed repo to GitHub (main), restructured into `cot_arithmetic/`, `tests/`, `docs/`. `data.py`: fixed-D disjoint train/test sampler, prompt tokens, answer tokens for split, k1, k2, direct. `tokenizer.py`: per-format answer vocab at ids 0..A-1 (A = 7, 10, 100, 1999), shared 12-token prompt vocab at ids A..A+11, `encode_example` returns (x, y) with -100 on prompt targets so `F.cross_entropy` does answer-only loss. `tests/test_tokenizer.py` passes for all four formats (`python3 -m tests.test_tokenizer`).
**Broken:** `model.py` and `train.py` are the nanoGPT Shakespeare code copied verbatim; they do not run in this repo yet (globals, `input.txt`, char vocab).
**Next:** adapt `model.py` (embedding `n_in` rows, output layer `A`, embed 64, 2 layers, 4 heads, small block_size), `train.py` (batching from `data.py` and the tokenizer, AdamW lr 2.5e-3, batch 256, 10% warmup then linear decay, 20 epochs, exact-match eval by greedy generation). Then a k1 run until test exact-match is over 95%, then the lr check and the sweep of all 4 formats.
**Decisions:** reversed output; fixed D, hyperparameters and prompt tokenization; per-format answer vocab; 3-digit operands so the direct vocab stays small; separate ids for prompt and answer tokens so the output layer is exactly an m-way choice; `ignore_index` (-100) instead of a mask.
**Decided 2026-11-26:** use the paper's learning rate (2.5e-3, 10% warmup, linear decay) for every format, no tuning, so no separate lr check.
**Open:** size of D (n_train, n_test); the redundant "thinking" variant (parked); `.DS_Store` is untracked (add to `.gitignore`?).

## 2026-11-24
**Done:** read the paper, designed the experiment (see CLAUDE.md), set up the repo.
**Broken:** nothing yet.
**Next (Wed 11/25):** data generator for all 4 formats, a tokenizer per format, answer-only loss, k1 training run. Done when k1 gets over 95% exact-match on the test set.
**Decisions:** reversed output; fixed D, hyperparameters and prompt tokenization; per-format answer vocab; 3-digit operands so the direct vocab stays small.
**Open:** tune learning rate once or per format; the redundant "thinking" variant (parked).
