from PIL import Image
from numpy import asarray
import lenna_mobilenet_plugin
print(lenna_mobilenet_plugin.description())

image = Image.open('assets/lenna.png')
data = asarray(image)
print(data.shape)

config = lenna_mobilenet_plugin.default_config()
print(config)
processed = lenna_mobilenet_plugin.process(config, data)
print(processed.shape)
Image.fromarray(processed).save('lenna_test_out.png')
