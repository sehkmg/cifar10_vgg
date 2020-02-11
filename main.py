import argparse
import torch
from tqdm import tqdm

from dataset import load_cifar10
from model import VGG11
from utils import set_seed

def main(args):
    set_seed(args.dataset_seed)
    train_loader, valid_loader, test_loader = load_cifar10(args.batch_size)
    set_seed(args.seed)

    model = VGG11().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs+1):
        print('Epoch: {}'.format(epoch))

        # training
        train_loss = 0
        correct = 0
        total = 0
        for img, label in tqdm(train_loader, total=len(train_loader)):
            img = img.cuda()
            label = label.cuda()

            pred = model(img)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred_label = pred.max(-1)[1]
            correct += (pred_label == label).sum().item()
            total += pred_label.shape[0]

        train_loss /= len(train_loader)
        train_error = 1 - correct / total

        with torch.no_grad():
            # validation
            valid_loss = 0
            correct = 0
            total = 0
            for img, label in valid_loader:
                img = img.cuda()
                label = label.cuda()

                pred = model(img)
                loss = criterion(pred, label)

                valid_loss += loss.item()

                pred_label = pred.max(-1)[1]
                correct += (pred_label == label).sum().item()
                total += pred_label.shape[0]

            valid_loss /= len(valid_loader)
            valid_error = 1 - correct / total

            # testing
            test_loss = 0
            correct = 0
            total = 0
            for img, label in test_loader:
                img = img.cuda()
                label = label.cuda()

                pred = model(img)
                loss = criterion(pred, label)

                test_loss += loss.item()
                pred_label = pred.max(-1)[1]
                correct += (pred_label == label).sum().item()
                total += pred_label.shape[0]

            test_loss /= len(valid_loader)
            test_error = 1 - correct / total
        
        # save loss and error
        with open('results/loss.txt', 'a') as f:
            f.write('{}\t{}\t{}\n'.format(train_loss, valid_loss, test_loss))

        with open('results/error.txt', 'a') as f:
            f.write('{}\t{}\t{}\n'.format(train_error, valid_error, test_error))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_seed',
        type=int,
        default=1
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128
    )
    args = parser.parse_args()

    main(args)
