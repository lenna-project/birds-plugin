import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
from torchvision.models.mobilenet import MobileNet_V2_Weights
import time
from tqdm import tqdm

USE_CUDA = True

print("Cuda availability:", torch.cuda.is_available())

img_height, img_width = 224, 224
batch_size = 256
num_labels = 525

data_dir = './100-bird-species/'

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(img_height),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets

print("Loading datasets")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'valid', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
               for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

print("Dataset sizes:", dataset_sizes)

# Define the model
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_labels)

# Define the directory to save the checkpoints
checkpoint_dir = './checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
model = model.to(device)

# Define the loss function, optimizer, and the number of epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 200

print("Starting training")
print("Device:", device)
print("Model:", model)
print("Criterion:", criterion)
print("Optimizer:", optimizer)
print("Epochs:", num_epochs)
print("Checkpoint directory:", checkpoint_dir)

# Load the model, optimizer, and other parameters from checkpoint
start_epoch = 0
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

if os.path.isfile(checkpoint_path):
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
else:
    print("No checkpoint found at '{}'. Starting from scratch".format(checkpoint_path))

# Initialize the progress bar
epoch_progress = tqdm(range(start_epoch, num_epochs), desc="Epoch Progress", position=0, leave=True)

# Training loop
for epoch in epoch_progress:
    print(f'Epoch {epoch}/{num_epochs - 1}')
    epoch_start = time.time()
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        dataloader_progress = tqdm(dataloaders[phase], desc=f"{phase} Progress", position=1, leave=False)

        for inputs, labels in dataloader_progress:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            dataloader_progress.set_description(f"{phase} Loss: {loss.item():.4f}")

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    epoch_end = time.time()  # End time for the epoch
    time_elapsed = epoch_end - epoch_start

    # Update epoch progress description for elapsed time
    epoch_progress.set_description(f"Epoch {epoch} Completed in {time_elapsed:.2f}s")

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'running_loss': running_loss,
        'running_corrects': running_corrects
    }, checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'running_loss': running_loss,
        'running_corrects': running_corrects
    }, checkpoint_path)

print('Training complete')