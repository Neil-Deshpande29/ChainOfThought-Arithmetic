# Arithmetic transformer: chain-of-thought degree sweep

Starter project for Prof. Pratik Chaudhari, based on Nadgir, Balasubramanian and Chaudhari, "How does Chain of Thought decompose complex tasks?" (ICML 2026). Due Sun Nov 29, 2026.

## Mode: LEARNING
Neil writes the core code: data generator, tokenizers, model, training loop, evaluation.
Claude tutors. Rules for Claude in this repo:
- Do not write core code or hand over full solutions unless Neil explicitly asks for code.
- Explain the concept, ask a guiding question, give a hint, then let Neil implement.
- Review code Neil writes: point out bugs by location and say what's wrong conceptually, and let him fix it.
- Boilerplate is fine to write when asked: plotting, CLI args, config loading, CI, README formatting.
- Keep explanations short and direct. No em dashes.

## The experiment
Train one small GPT per answer format on 3-digit addition and compare test error. The lowest-error format brackets m*, the paper's optimal degree e^(d/2).

| Format | Steps (n) | Choices per step (m) |
|---|---|---|
| split | 8 | 2 ("0-4 or 5-9"), then 5 |
| k1 | 4 | 10 (one digit) |
| k2 | 2 | 100 (two-digit chunk) |
| direct | 1 | whole sum as one token (~2,000 used) |

Format rules:
- Operands padded to 3 digits, sum padded to 4 digits and written reversed (ones place first). Example: 37+48 -> prompt `037+048=`, k1 answer `5 8 0 0`.
- The prompt is tokenized digit by digit in every format (keeps d fixed).
- Only the answer vocabulary changes per format, so each step really is an m-way choice.
- Loss on answer tokens only.

Held fixed across formats: the same train/test problems (fixed D), the same hyperparameters, the same model body.
Starting hyperparameters (paper, Appendix A.3): GPT-2 with embedding size 64, 2 layers, 4 heads; AdamW lr 2.5e-3, batch 256, 10% warmup then linear decay, 20 epochs. Train until test error plateaus. Report body and output-layer parameters separately.

The generator takes an `op` argument (only "+" for now). Subtraction and multiplication come later, to compare d across operations.

## Layout
- `cot_arithmetic/data.py`: problem sampling and per-format formatting
- `cot_arithmetic/tokenizer.py`: per-format vocabularies
- `cot_arithmetic/model.py`: GPT
- `cot_arithmetic/train.py`: training loop and eval (exact-match accuracy)
- `tests/`: tests (run from the repo root with `python3 -m tests.test_tokenizer`)
- `results/`: CSVs and plots
- `docs/HANDOFF.md`: updated at the end of every session

## Run
(fill in once train.py exists)
