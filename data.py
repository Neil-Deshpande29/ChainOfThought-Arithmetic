import random

FORMATS = ("split", "k1", "k2", "direct")


def sample_problems(n_train, n_test, op="+", seed=0):
    """Fixed D: disjoint train/test operand pairs, sampled without replacement
    from all 1000 x 1000 pairs. Same call (same seed) gives the same D for every format."""
    if op != "+":
        raise NotImplementedError("only '+' for now")
    idx = random.Random(seed).sample(range(10**6), n_train + n_test)
    pairs = [(i // 1000, i % 1000) for i in idx]
    return pairs[:n_train], pairs[n_train:]


def answer(a, b, op="+"):
    if op != "+":
        raise NotImplementedError("only '+' for now")
    return a + b


def prompt_tokens(a, b, op="+"):
    """Digit by digit in every format, e.g. 037+048="""
    return list(f"{a:03d}{op}{b:03d}=")


def answer_tokens(a, b, fmt, op="+"):
    """Answer as a list of string tokens. Sum padded to 4 digits, then reversed (ones first)."""
    total = answer(a, b, op)
    s = f"{total:04d}"[::-1]
    if fmt == "k1":
        return list(s)  # 4 steps, 10 choices
    if fmt == "k2":
        return [s[0:2], s[2:4]]  # 2 steps, 100 choices
    if fmt == "split":
        out = []  # 8 steps: half (2 choices), then offset in half (5 choices)
        for ch in s:
            d = int(ch)
            out += ["0-4" if d < 5 else "5-9", str(d % 5)]
        return out
    if fmt == "direct":
        return [str(total)]  # 1 step, whole sum as one token (0..1998)
    raise ValueError(f"unknown format: {fmt}")


def make_example(a, b, fmt, op="+"):
    """(prompt_tokens, answer_tokens). Loss mask over answer tokens is applied downstream."""
    return prompt_tokens(a, b, op), answer_tokens(a, b, fmt, op)
