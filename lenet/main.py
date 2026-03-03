from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn.trainers import Trainer

from modules import LeNet

from sklearn.metrics import accuracy_score

app = Fenn()

@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"])  # set seed for reproducibility

    # ========================================
    # Data Transforms
    # ========================================
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    # ========================================
    # TODO: REPLACE WITH YOUR ACTUAL DATA
    # Using CIFAR-10 as an example dataset
    # ========================================
    train_dataset = datasets.CIFAR10(
        root=args["dataset"]["dir"],
        train=True,
        download=True,
        transform=train_transform
    )

    # Get test set and split into validation (50%) and test (50%)
    full_test_dataset = datasets.CIFAR10(
        root=args["dataset"]["dir"],
        train=False,
        download=True,
        transform=test_transform
    )

    # Split test set: 5000 val + 5000 test (from 10000 total)
    val_size = 5000
    test_size = len(full_test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(
        full_test_dataset,
        [val_size, test_size]
    )
    # ========================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["train"]["batch"],
        shuffle=True,
        pin_memory=True
    )

    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["test"]["batch"],
        shuffle=False,
        pin_memory=True
    )

    # Test loader (for final evaluation)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args["test"]["batch"],
        shuffle=False,
        pin_memory=True
    )

    model = LeNet(
        in_channels=args["model"]["in_channels"],
        num_classes=args["model"]["num_classes"]
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(args["train"]["lr"]))

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optim=optimizer,
        epochs=args["train"]["epochs"],
        num_classes=args["model"]["num_classes"],
        device=device,
        checkpoint_dir="./checkpoints",
        save_best=True,
        early_stopping_patience=5
    )

    model = trainer.fit(train_loader=train_loader, val_loader=val_loader, val_epoch=5)

    # ========================================
    # Evaluation
    # ========================================
    predictions = []
    grounds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.detach().cpu().tolist())
            grounds.extend(labels.detach().cpu().tolist())

    print(f"Accuracy: {accuracy_score(grounds, predictions):.4f}")

if __name__ == "__main__":
    app.run()
