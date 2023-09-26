import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras import Model
import tf2onnx

img_height = 224
img_width = 224
num_labels = 525

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

model = get_birds_mobilenet()

checkpoint_path = "./checkpoints/birds_mobilenet/"
model.load_weights(checkpoint_path)

spec = (tf.TensorSpec((None, img_height, img_width, 3), tf.float32, name="input"),)
output_path = "birds_mobilenetv2.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
