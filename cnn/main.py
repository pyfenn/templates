try:
    from torchvision import datasets, transforms
except ImportError as e:
    raise RuntimeError(
        "Torchvision is required for this feature. "
        "Install it yourself (GPU/CPU) or use 'pip install smle[torchvision]'."
    ) from e

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

except ImportError as e:
    raise RuntimeError(
        "Torch is required for this feature. "
        "Install it yourself (GPU/CPU) or use 'pip install smle[torch]'."
    ) from e


from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn.trainers import Trainer
from modules import CNN
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
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
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

    test_dataset = datasets.CIFAR10(
        root=args["dataset"]["dir"],
        train=False,
        download=True,
        transform=test_transform
    )
    # ========================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["train"]["batch"],
        shuffle=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args["test"]["batch"],
        shuffle=False,
        pin_memory=True
    )

    model = CNN(
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
        device=device
    )

    model = trainer.fit(train_loader=train_loader)

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
