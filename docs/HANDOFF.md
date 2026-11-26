# HANDOFF

Update at the end of every session.

## 2026-11-26
**Done:** pushed repo to GitHub (main), restructured into `cot_arithmetic/`, `tests/`, `docs/`. `data.py`: fixed-D disjoint train/test sampler, prompt tokens, answer tokens for split, k1, k2, direct. `tokenizer.py`: per-format answer vocab at ids 0..A-1 (A = 7, 10, 100, 1999), shared 12-token prompt vocab at ids A..A+11, `encode_example` returns (x, y) with -100 on prompt targets so `F.cross_entropy` does answer-only loss. `tests/test_tokenizer.py` passes for all four formats (`python3 -m tests.test_tokenizer`).
**Broken:** `model.py` and `train.py` are the nanoGPT Shakespeare code copied verbatim; they do not run in this repo yet (globals, `input.txt`, char vocab).
**Next:** adapt `model.py` (embedding `n_in` rows, output layer `A`, embed 64, 2 layers, 4 heads, small block_size), `train.py` (batching from `data.py` and the tokenizer, AdamW lr 2.5e-3, batch 256, 10% warmup then linear decay, 20 epochs, exact-match eval by greedy generation). Then a k1 run until test exact-match is over 95%, then the lr check and the sweep of all 4 formats.
**Decisions:** reversed output; fixed D, hyperparameters and prompt tokenization; per-format answer vocab; 3-digit operands so the direct vocab stays small; separate ids for prompt and answer tokens so the output layer is exactly an m-way choice; `ignore_index` (-100) instead of a mask.
**Open:** size of D (n_train, n_test); tune learning rate once or per format; the redundant "thinking" variant (parked); `.DS_Store` is untracked (add to `.gitignore`?).

## 2026-11-24
**Done:** read the paper, designed the experiment (see CLAUDE.md), set up the repo.
**Broken:** nothing yet.
**Next (Wed 11/25):** data generator for all 4 formats, a tokenizer per format, answer-only loss, k1 training run. Done when k1 gets over 95% exact-match on the test set.
**Decisions:** reversed output; fixed D, hyperparameters and prompt tokenization; per-format answer vocab; 3-digit operands so the direct vocab stays small.
**Open:** tune learning rate once or per format; the redundant "thinking" variant (parked).
