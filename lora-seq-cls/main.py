from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn.trainers import LoRATrainer

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

from modules import SentimentDataset

app = Fenn()

@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"])

    # ========================================
    # Data — SST-2 (binary sentiment: 0=negative, 1=positive)
    # The full training split has ~67k examples; we use a 2000-sample
    # subset to keep the demo fast. Remove the slice to train on all data.
    # ========================================
    # SST-2: labeled splits are "train" and "validation".
    # We use "validation" as the held-out test set.
    raw = load_dataset("sst2")

    train_texts  = raw["train"]["sentence"]#[:2000]
    train_labels = raw["train"]["label"]#[:2000]

    test_texts  = raw["validation"]["sentence"]
    test_labels = raw["validation"]["label"]
    # ========================================

    model_name = args["model"]["name"]
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, args["model"]["max_length"])
    test_dataset  = SentimentDataset(test_texts,  test_labels,  tokenizer, args["model"]["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=args["train"]["batch"], shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args["test"]["batch"],  shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=args["model"]["num_labels"],
    )

    optimizer = optim.AdamW(model.parameters(), lr=float(args["train"]["lr"]))

    # DistilBERT attention layers that benefit most from LoRA adaptation.
    # Swap target_modules for other architectures (e.g. ["query", "value"] for BERT).
    trainer = LoRATrainer(
        model=model,
        optim=optimizer,
        task_type="SEQ_CLS",
        r=args["lora"]["r"],
        lora_alpha=args["lora"]["alpha"],
        lora_dropout=float(args["lora"]["dropout"]),
        target_modules=["q_lin", "v_lin"],
        device=device,
    )

    trainer.fit(train_loader=train_loader, epochs=args["train"]["epochs"])
    predictions = trainer.predict(test_loader)

    print(f"Test Accuracy: {accuracy_score(test_labels, predictions):.4f}")

if __name__ == "__main__":
    app.run()
