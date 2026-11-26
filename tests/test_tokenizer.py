import random

from cot_arithmetic.data import FORMATS, make_example
from cot_arithmetic.tokenizer import Tokenizer

EXPECTED_A = {"split": 7, "k1": 10, "k2": 100, "direct": 1999}

rng = random.Random(0)
pairs = [(0, 0), (0, 1), (999, 999), (37, 48)] + [(rng.randrange(1000), rng.randrange(1000)) for _ in range(500)]

for fmt in FORMATS:
    tok = Tokenizer(fmt)

    # sizes
    assert tok.A == EXPECTED_A[fmt], (fmt, tok.A)
    assert tok.n_in == tok.A + 12, (fmt, tok.n_in)
    assert len(tok.itos) == tok.A + 12, (fmt, len(tok.itos))

    # id layout: answers at 0..A-1, prompt at A..A+11
    assert sorted(tok.answer_stoi.values()) == list(range(tok.A))
    assert sorted(tok.prompt_stoi.values()) == list(range(tok.A, tok.A + 12))
    for t, i in tok.answer_stoi.items():
        assert tok.itos[i] == t
    for t, i in tok.prompt_stoi.items():
        assert tok.itos[i] == t

    # round trips and id ranges on real examples
    for a, b in pairs:
        p, ans = make_example(a, b, fmt)
        p_ids, ans_ids = tok.encode_prompt(p), tok.encode_answer(ans)
        assert tok.decode_prompt(p_ids) == p
        assert tok.decode_answer(ans_ids) == ans
        assert all(i >= tok.A for i in p_ids)
        assert all(0 <= i < tok.A for i in ans_ids)

    print(f"{fmt}: ok (A={tok.A}, n_in={tok.n_in})")
