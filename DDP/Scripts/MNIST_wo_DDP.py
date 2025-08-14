# MNIST_wo_DDP.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def get_device():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        current = torch.cuda.current_device()
        print(f"[Device] Detected {count} CUDA device(s):")
        for i, n in enumerate(names):
            print(f"  - cuda:{i}: {n}")
        print(f"[Device] Using cuda:{current}")
        return torch.device(f"cuda:{current}")
    else:
        print("[Device] CUDA not available — using CPU")
        return torch.device("cpu")

class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_mnist(epochs: int = 5, batch_size: int = 64, num_workers: int = 4):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    device = get_device()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST('.', download=True, train=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    model = CNN_MNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()

        total = 0
        loss_sum = 0.0
        correct = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bsz = xb.size(0)
            total      += bsz
            loss_sum   += loss.item() * bsz
            correct    += (logits.argmax(1) == yb).sum().item()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        epoch_time = time.time() - start

        avg_loss = loss_sum / total
        acc = correct / total
        print(f"Epoch {epoch:02d}/{epochs} | loss: {avg_loss:.4f} | acc: {acc:.4f} | time: {epoch_time:.2f}s")

if __name__ == "__main__":
    train_mnist(epochs=5, batch_size=64)
