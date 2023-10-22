# Train the model

## Create virtualenv

```sh
virtualenv -p python3 .venv
source .venv/bin/activate
pip install tensorflow
pip install kaggle pandas pillow numpy tf2onnx
```

## optional prepare amd gpu

```sh
sudo apt install rocm-opencl
pip install tensorflow-rocm
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
python export_onnx_model.py
cp birds_mobilenetv2.onnx ../assets/
cp birds_labels.txt ../assets/
```

## Download EfficientNetB2

```sh
python download_efficientnet.py
```

# torch train version

```sh
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
pip install tqdm
python train_torch.py
```
