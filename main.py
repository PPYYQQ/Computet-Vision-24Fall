import torch
import torch.nn.functional as F
from models import VGG, ResNet, ResNext
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.optim as optim
import wandb
import copy
import torch.nn as nn

def train(model, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # create dataset, data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomResizedCrop(size=(32, 32), antialias=True),
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # create dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform= transform)

    # create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)

    # create scheduler 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)

    model = model.to(f'cuda:{args.device}')
    model.train()
    for epoch in range(args.epochs):
        acumulate_loss = 0.0
        for idx, data in enumerate(train_loader):
            inputs, labels = data
            if args.model != 'cnn':
                inputs = inputs.view(inputs.size(0), -1).to(f'cuda:{args.device}')
            else:
                inputs = inputs.to(f'cuda:{args.device}')
            labels = labels.to(f'cuda:{args.device}')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = args.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acumulate_loss += loss.item()
            if idx % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {idx + 1}] loss: {acumulate_loss / 100:.3f}')
                wandb.log({"train_loss": acumulate_loss / 100})
                acumulate_loss = 0.0
            
        if scheduler != 'None':
            scheduler.step()
            wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
        else:
            wandb.log({"learning_rate": args.lr})

        test(model, args)
        print(args.best_acc)
    torch.save(args.best_model.state_dict(), f'{args.model}_model_state.pth')

def test(model, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    # create testing dataset
    # create dataloader
    # test
        # forward
    if args.run == 'test':
        model.load_state_dict(torch.load(f'{args.model}_model_state.pth', weights_only=True))
        model.to(f'cuda:{args.device}')

    # Create testing dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # create dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    # create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # create testing dataset
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    
    # create dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # Training loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            if args.model != 'cnn':
                images = images.view(images.size(0), -1).to(f'cuda:{args.device}')
            else:
                images = images.to(f'cuda:{args.device}')
            labels = labels.to(f'cuda:{args.device}')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 Train images: {100 * correct / total:.2f}%')
    wandb.log({"train_accuracy": 100 * correct / total})

    # Testing loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if args.model != 'cnn':
                images = images.view(images.size(0), -1).to(f'cuda:{args.device}')
            else:
                images = images.to(f'cuda:{args.device}')
            labels = labels.to(f'cuda:{args.device}')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
    wandb.log({"test_accuracy": 100 * correct / total})

    if 100 * correct / total > args.best_acc:
        args.best_model = copy.deepcopy(model)  
        args.best_model.load_state_dict(copy.deepcopy(model.state_dict()))  
        args.best_acc = 100 * correct / total


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument()
    args = parser.parse_args()

    # Hyperparameter
    args.lr = 5e-3
    momentum = 0.9
    args.epochs = 40
    args.criterion = nn.CrossEntropyLoss()
    args.best_model = None
    args.best_acc = 0
    args.step_size = 10
    args.gamma = 0.1
    args.T_max = 50
    args.batch_size = 2048

    if args.model == "VGG":
        model = VGG(args)
    elif args.model == "ResNet":
        model = ResNet(args)
    elif args.model == "ResNext":
        model = ResNext(args)

    # train / test
    if args.run == 'train':
        train(model, args)
    elif args.run == 'test':
        test(model, args)
    else: 
        raise AssertionError