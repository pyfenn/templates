try:
    import timm
except ImportError as e:
    raise RuntimeError(
        "timm is required for this template. "
        "Install it yourself via 'pip install timm'."
    ) from e

try:
    from torchvision.datasets import CIFAR10
    from torchvision import transforms as T
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


from sklearn.metrics import accuracy_score
from pathlib import Path

from fenn import entrypoint
from fenn.utils import set_seed
from fenn.trainer import Trainer

@entrypoint
def main(args):

    set_seed(args["training"]["seed"])
    export_dir = Path(args["training"]["export_dir"])
    data_dir = Path(args["dataset"]["dir"])

    device = args["training"]["device"]

    base_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=base_transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=base_transform)

    train_loader = DataLoader(train_dataset, batch_size=args["training"]["train_batch"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args["training"]["test_batch"], shuffle=False)

    # ========================================
    # To create a new model:
    # pretrainer = False
    # num_classes = 10
    # ========================================

    model = timm.create_model('resnet50', pretrained=True)

    # ========================================
    # True - Full fine-tuning
    # False - Head fine-tuning
    # ========================================

    for param in model.parameters():
        param.requires_grad = False

    # ========================================

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(args["training"]["learning_rate"]))

    trainer = Trainer(model, loss_fn, optimizer, device=device)
    model = trainer.fit(
        train_loader=train_loader,
        epochs=args["training"]["epochs"],
        export_dir=export_dir
    )

    model.eval()

    y_true = []
    y_preds = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device)

        outputs = model(imgs)

        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_preds)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
