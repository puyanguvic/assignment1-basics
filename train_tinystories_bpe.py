import pickle
from pathlib import Path

from cs336_basics.bpe import train_bpe

def main():
    vocab, merges = train_bpe(
        input_path = "data/TinyStoriesV2-GPT4-train.txt",
        vocab_size = 10_000,
        special_tokens = ["<|endoftext|>"],
        num_processes = 120
    )

    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    with (output / "tinystories_bpe.pkl").open("wb") as f:
        pickle.dump((vocab, merges), f)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Saved to: {output / 'tinystories_bpe.pkl'}")


if __name__ == "__main__":
    main()