import json
import re
from pathlib import Path

CLASSIC_PATTERN = re.compile(r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
BLOCK_SEPARATOR = "."
PAD_TOKEN = "<"


def load_token_list(path):
    token_path = Path(path)
    if token_path.suffix.lower() == ".json":
        loaded = json.loads(token_path.read_text())
        if isinstance(loaded, dict):
            tokens = list(loaded.keys())
        elif isinstance(loaded, list):
            tokens = loaded
        else:
            raise ValueError(f"Unsupported JSON token format in {path}")
    else:
        tokens = [line.strip() for line in token_path.read_text().splitlines() if line.strip()]

    return tokens


def tokenize_smiles(smiles, tokenization_mode="classic"):
    smiles = smiles.strip()
    if tokenization_mode == "block":
        tokens = []
        for index, fragment in enumerate(smiles.split(BLOCK_SEPARATOR)):
            if index > 0:
                tokens.append(BLOCK_SEPARATOR)
            fragment = fragment.strip()
            if fragment:
                tokens.append(fragment)
        return tokens

    return CLASSIC_PATTERN.findall(smiles)


def pad_tokens(tokens, max_len):
    if len(tokens) >= max_len:
        return tokens[:max_len]

    return tokens + [PAD_TOKEN] * (max_len - len(tokens))


def tokenize_and_pad(smiles, max_len, tokenization_mode="classic"):
    return pad_tokens(tokenize_smiles(smiles, tokenization_mode), max_len)


def build_vocab(sequences, tokenization_mode="classic", vocab_path=None, extra_tokens=None):
    if vocab_path:
        tokens = load_token_list(vocab_path)
    else:
        tokens = []
        for sequence in sequences:
            tokens.extend(tokenize_smiles(sequence, tokenization_mode))

    if extra_tokens:
        tokens.extend(extra_tokens)

    tokens.append(PAD_TOKEN)
    if tokenization_mode == "block":
        tokens.append(BLOCK_SEPARATOR)

    return sorted(set(tokens))


def max_token_length(sequences, tokenization_mode="classic"):
    return max(len(tokenize_smiles(sequence, tokenization_mode)) for sequence in sequences)