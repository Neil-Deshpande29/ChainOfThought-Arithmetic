# HANDOFF

Update at the end of every session.

## 2026-11-24
**Done:** read the paper, designed the experiment (see CLAUDE.md), set up the repo.
**Broken:** nothing yet.
**Next (Wed 11/25):** data generator for all 4 formats, a tokenizer per format, answer-only loss, k1 training run. Done when k1 gets over 95% exact-match on the test set.
**Decisions:** reversed output; fixed D, hyperparameters and prompt tokenization; per-format answer vocab; 3-digit operands so the direct vocab stays small.
**Open:** tune learning rate once or per format; the redundant "thinking" variant (parked).
