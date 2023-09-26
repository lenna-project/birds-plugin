import logging
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

from keras.layers import Layer, Conv2D, MaxPool2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras import layers
from keras import Model

print("GPUs: ", tf.config.list_physical_devices('GPU'))

img_height = 224
img_width = 224
batch_size = 64
num_labels = 525

data_dir = './100-bird-species/'
data_dir_train = os.path.join(data_dir, 'train')
data_dir_valid = os.path.join(data_dir, 'valid')
data_dir_test = os.path.join(data_dir, 'test')

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    label_mode='categorical',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_valid,
    label_mode='categorical',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_test,
    label_mode='categorical',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


def normalize(img, label):
    return img / 255.0, label


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

train_dataset = (train_ds
                 .map(normalize)
                 .map(lambda x, y: (data_augmentation(x), y))
                 .prefetch(tf.data.AUTOTUNE))

valid_dataset = valid_ds.map(normalize)
test_dataset = test_ds.map(normalize)

def write_labels(labels, filename):
    with open(filename, 'w') as f:
        for class_name in labels:
            f.write(f'{class_name}\n')

    print(f'Class names written to {filename}')


def get_birds_mobilenet():
    pre_trained_model = MobileNetV2(
        include_top=False,
        input_shape=(img_height, img_width, 3),
        classifier_activation='softmax'
    )

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.output
    last_layer.trainable = True

    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(1024, activation='relu')(x)
    x = layers.Dense(num_labels, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)
    return model

write_labels(train_ds.class_names, 'birds_labels.txt')

model = get_birds_mobilenet()
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

checkpoint_path = "./checkpoints/birds_mobilenet/"

if os.path.exists(checkpoint_path) and os.path.isfile(os.path.join(checkpoint_path, 'saved_model.pb')):
    model = tf.keras.models.load_model(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
else:
    print("No checkpoints found, training from scratch.")


model_history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=200,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, verbose=0, save_freq="epoch")
    ])
