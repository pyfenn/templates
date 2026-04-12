from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn.trainers import ClassificationTrainer

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from sklearn.metrics import accuracy_score

app = Fenn()

@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"])

    # ========================================
    # Data — CIFAR-10 (auto-downloaded on first run)
    # Standard ImageNet normalisation so pretrained weights transfer correctly.
    # Replace CIFAR10 with any torchvision dataset or your own ImageFolder.
    # ========================================
    _mean = (0.485, 0.456, 0.406)
    _std  = (0.229, 0.224, 0.225)

    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(_mean, _std),
    ])

    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(_mean, _std),
    ])

    train_dataset = datasets.CIFAR10(
        root=args["dataset"]["dir"], train=True,  download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=args["dataset"]["dir"], train=False, download=True, transform=test_transform
    )
    # ========================================

    train_loader = DataLoader(
        train_dataset, batch_size=args["train"]["batch"], shuffle=True,  pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,  batch_size=args["test"]["batch"],  shuffle=False, pin_memory=True
    )

    # ========================================
    # Model — any timm backbone, head replaced for num_classes.
    # freeze_backbone=true: only the classifier head is trained (fast).
    # freeze_backbone=false: all weights are updated (full fine-tuning).
    # ========================================
    model = timm.create_model(
        args["model"]["name"],
        pretrained=args["model"]["pretrained"],
        num_classes=args["model"]["num_classes"],
    )

    if args["model"]["freeze_backbone"]:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    # ========================================

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(args["train"]["lr"]),
        weight_decay=float(args["train"]["weight_decay"]),
    )

    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        optim=optimizer,
        num_classes=args["model"]["num_classes"],
        device=device,
    )

    trainer.fit(train_loader=train_loader, epochs=args["train"]["epochs"])

    y_test      = [label for _, labels in test_loader for label in labels.tolist()]
    predictions = trainer.predict(test_loader)

    print(f"Test Accuracy: {accuracy_score(y_test, predictions):.4f}")

if __name__ == "__main__":
    app.run()
