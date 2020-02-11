import torch
import torchvision
from torchvision import transforms

def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    train_set = torchvision.datasets.CIFAR10('dataset', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10('dataset', train=False, transform=transform, download=True)

    valid_set, test_set = torch.utils.data.random_split(test_set, [5000, 5000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader