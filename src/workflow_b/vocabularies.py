
TRADITIONAL_ENVIRONMENT_VOCAB = [
    "person", "man", "woman", "child",
    "horse", "donkey", "cow", "sheep", "dog", "cat",
    "cart", "wagon", "basket", "tool",
    "tree", "plant", "crop",
    "house", "building", "church", "fence",
    "table", "chair", "food", "bread", "fruit", "vegetables"
]

def build_prompt(vocab: list[str]) -> str:
    return ", ".join(vocab)

