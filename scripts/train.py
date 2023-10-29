import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.mobilenet import MobileNet_V2_Weights
from torchvision.models.efficientnet import EfficientNet_B2_Weights
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

USE_CUDA = True

MODEL_TYPE = 'MobileNetV2'
# MODEL_TYPE = 'EfficientNetB2'

NUM_EPOCHS = 45

print("Cuda availability:", torch.cuda.is_available())

cuda = torch.device('cuda')     # Default HIP device
print("cuda: ", torch.cuda.get_device_name(device=cuda))

img_height, img_width = 224, 224
batch_size = 256 + 128
num_labels = 525

data_dir = './100-bird-species/'

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(img_height),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.47853944, 0.4732864, 0.47434163])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.47853944, 0.4732864, 0.47434163])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.47853944, 0.4732864, 0.47434163])
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
model = None

if MODEL_TYPE == 'MobileNetV2':
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
elif MODEL_TYPE == 'EfficientNetB2':
    model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

# Replace the final layer (classifier) for the new task
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_labels)

# Unfreeze the parameters of the classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# Define the directory to save the checkpoints
checkpoint_dir = './checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

tensorboard_dir = './tensorboard/'

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
model = model.to(device)

# Define the loss function, optimizer, and the number of epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001, momentum=0.89, alpha=0.85, eps=0.004, weight_decay=0.00004)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)


print("Starting training")
print("Device:", device)
print("Model:", model)
print("Criterion:", criterion)
print("Optimizer:", optimizer)
print("Epochs:", NUM_EPOCHS)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

print("Checkpoint directory:", checkpoint_dir)
print("Tensorboard directory:", tensorboard_dir)
print("run `tensorboard --logdir tensorboard` to view tensorboard")

# Load the model, optimizer, and other parameters from checkpoint
start_epoch = 0
checkpoint_path = os.path.join(checkpoint_dir, f'{MODEL_TYPE}-checkpoint.pth')

if os.path.isfile(checkpoint_path):
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
else:
    print("No checkpoint found at '{}'. Starting from scratch".format(checkpoint_path))

tensorboard_writer = SummaryWriter(tensorboard_dir)

# Initialize the progress bar
epoch_progress = tqdm(range(start_epoch, NUM_EPOCHS), desc="Epoch Progress", position=0, leave=True)

# Training loop
for epoch in epoch_progress:
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    epoch_start = time.time()
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        total_samples = 0
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

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels)
            total_samples += batch_size

            # Optional: Update loss description for each batch
            dataloader_progress.set_description(f"{phase} Batch Loss: {running_loss/total_samples:.4f}")

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Write to tensorboard 
        tensorboard_writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
        tensorboard_writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

    checkpoint_path = os.path.join(checkpoint_dir, f'{MODEL_TYPE}-checkpoint.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'running_loss': running_loss,
        'running_corrects': running_corrects
    }, checkpoint_path)

    # Export to ONNX format
    dummy_input = torch.randn(1, 3, img_height, img_width, device=device)
    onnx_path = os.path.join(checkpoint_dir, f'Birds-Classifier-{MODEL_TYPE}.onnx')
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=13)

    scheduler.step()
    epoch_end = time.time()  # End time for the epoch
    time_elapsed = epoch_end - epoch_start

    # Update epoch progress description for elapsed time
    epoch_progress.set_description(f"Epoch {epoch} Completed in {time_elapsed:.2f}s")

print('Training complete')
tensorboard_writer.close()
