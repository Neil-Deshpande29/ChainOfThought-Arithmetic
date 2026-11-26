
FORMATS = ("split", "k1", "k2", "direct")

def answer_vocab(fmt):
    """Return a list of answer tokens for a given format."""
    if fmt == "k1":
        return [str(i) for i in range(10)]  # 10 choices
    if fmt == "k2":
        return [f"{i:02d}" for i in range(100)]  # 100 choices
    if fmt == "split":
        return ["0-4", "5-9"] + [str(i) for i in range(5)]  # 7 choices
    if fmt == "direct":
        return [str(i) for i in range(1999)]  # 1999 choices
    raise ValueError(f"unknown format: {fmt}")

class Tokenizer:
    """Tokenizer for the addition task. Maps tokens to indices and vice versa."""

    def __init__(self, fmt):
        if fmt not in FORMATS:
            raise ValueError(f"unknown format: {fmt}")
        self.fmt = fmt
        self.prompt_vocab = [str(i) for i in range(10)] + ["+", "="] 
        self.answer_vocab = answer_vocab(fmt)
        self.A = len(self.answer_vocab)
        self.n_in = self.A + len(self.prompt_vocab)
        self.answer_stoi = {tok: i for i, tok in enumerate(self.answer_vocab)}
        self.prompt_stoi = {tok: i for i, tok in enumerate(self.prompt_vocab, self.A)}
        self.itos = {**({i: tok for i, tok in enumerate(self.answer_vocab)}), 
                     **({i + self.A: tok for i, tok in enumerate(self.prompt_vocab)})}
        

    def encode_prompt(self, tokens):
        """Encode the prompt tokens as a list of indices."""
        return [self.prompt_stoi[tok] for tok in tokens]

    def encode_answer(self, tokens):
        """Encode the answer tokens as a list of indices."""
        return [self.answer_stoi[tok] for tok in tokens]

    def decode_prompt(self, indices):
        """Decode a list of indices back to prompt tokens."""
        return [self.itos[i] for i in indices]

    def decode_answer(self, indices):
        """Decode a list of indices back to answer tokens."""
        return [self.itos[i] for i in indices]
    
    
    def encode_example(self, prompt_tokens, answer_tokens):
        """Encode a single example (prompt_tokens, answer_tokens) as two lists of indices (x, y) for training."""
        encoded = self.encode_prompt(prompt_tokens) + self.encode_answer(answer_tokens)
        x = encoded[:-1]  # all but last token
        y = [-100]*(len(prompt_tokens) - 1) + self.encode_answer(answer_tokens)   # -100 for prompt tokens, then answer tokens, and shifted forward by 1
        return x, y


