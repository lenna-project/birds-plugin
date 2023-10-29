import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

MODEL_TYPE = 'MobileNetV2'
#MODEL_TYPE = 'EfficientNetB2'

# Define parameters
data_dir = './100-bird-species/'
img_height, img_width = 224, 224
num_labels = 525
checkpoint_dir = './checkpoints/'
checkpoint_file = f'{MODEL_TYPE}-checkpoint.pth'

# Load the validation dataset
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_height),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.47853944, 0.4732864, 0.47434163])
])

validation_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), data_transform)
validation_loader = DataLoader(validation_dataset, batch_size=5, shuffle=True)

# Define the model and load checkpoint
model = None

if MODEL_TYPE == 'MobileNetV2':
    model = models.mobilenet_v2()
elif MODEL_TYPE == 'EfficientNetB2':
    model = models.efficientnet_b2()

num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, num_labels)

checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    raise ValueError("Checkpoint file not found")

model.eval()

# Select 5 random images and predict
images, labels = next(iter(validation_loader))
with torch.no_grad():
    outputs = model(images)
    print(outputs)
    _, preds = torch.max(outputs, 1)

# Function to convert image for plotting
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.47853944, 0.4732864, 0.47434163])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1.001)

# Plot the images with predictions and save to a file
fig = plt.figure(figsize=(15, 10))

for i in range(5):
    ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
    imshow(images[i])
    ax.set_title(f"True: {validation_dataset.classes[labels[i]]}\nPred: {validation_dataset.classes[preds[i]]}", fontsize=10)

# Save the figure
plt.tight_layout()
plt.savefig('evaluation.jpg')
