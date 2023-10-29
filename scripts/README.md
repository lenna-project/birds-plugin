# Train the model

## Create virtualenv

```sh
virtualenv -p python3 .venv
source .venv/bin/activate
pip install tqdm tensorboard onnx
```

## optional prepare amd gpu

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

## Download Data

```sh
# kaggle datasets download -d gpiosenka/100-bird-species
sh download_train_data.sh
```

## Train

```sh
python train.py
```

# Export Model

```sh
cp checkpoints/Birds-Classifier-EfficientNetB2.onnx ../assets/
cp checkpoints/Birds-Classifier-MobileNetV2.onnx ../assets/
cp birds_labels.txt ../assets/
```
