from fenn import Fenn
from fenn.utils import set_seed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from huggingface_hub import hf_hub_download

from modules import CharDataset, CharLM

app = Fenn()

@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"])

    # ========================================
    # Data — TinyShakespeare (~1 MB, auto-downloaded on first run)
    # Full works of Shakespeare concatenated into a single text file.
    # We build a character-level vocabulary from the raw text and create
    # sliding-window (input, target) sequence pairs for next-char prediction.
    # Replace open(...) with any plain-text file to train on a different corpus.
    # ========================================
    text_path = hf_hub_download(
        repo_id="Trelis/tiny-shakespeare", filename="input.txt", repo_type="dataset"
    )
    text = open(text_path, encoding="utf-8").read()

    dataset   = CharDataset(text, seq_len=args["data"]["seq_len"])
    vocab_size = dataset.vocab_size
    print(f"Corpus: {len(text):,} chars | Vocab: {vocab_size} unique chars | "
          f"Sequences: {len(dataset):,}")

    n_val   = int(len(dataset) * args["test"]["size"])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args["train"]["seed"]))
    # ========================================

    train_loader = DataLoader(train_ds, batch_size=args["train"]["batch"],
                              shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args["test"]["batch"],
                              shuffle=False, pin_memory=True)

    # ========================================
    # Model — embedding + stacked LSTM + linear head.
    # Increase embed_dim / hidden_dim for better quality at the cost of speed.
    # ========================================
    model = CharLM(
        vocab_size=vocab_size,
        embed_dim=args["model"]["embed_dim"],
        hidden_dim=args["model"]["hidden_dim"],
        num_layers=args["model"]["num_layers"],
        dropout=args["model"]["dropout"],
    ).to(device)
    # ========================================

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(args["train"]["lr"]))

    for epoch in range(1, args["train"]["epochs"] + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(X)                               # (B, T, V)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()
            train_loss += loss.item()

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits, _ = model(X)
                val_loss += loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1)).item()

        print(f"Epoch {epoch:>2}/{args['train']['epochs']}  "
              f"train_loss={train_loss / len(train_loader):.4f}  "
              f"val_loss={val_loss / len(val_loader):.4f}")

    # ========================================
    # Generate — seed the model with a prompt, sample autoregressively.
    # Lower temperature → more repetitive but coherent.
    # Higher temperature → more creative but potentially incoherent.
    # ========================================
    prompt = args["generate"]["prompt"]
    prompt_idx = [dataset.stoi.get(c, 0) for c in prompt]
    generated  = model.generate(
        prompt_idx=prompt_idx,
        max_new_tokens=args["generate"]["max_new_tokens"],
        temperature=args["generate"]["temperature"],
        device=device,
    )
    print("\n--- Generated Text ---")
    print("".join(dataset.itos[i] for i in generated))
    # ========================================

if __name__ == "__main__":
    app.run()
