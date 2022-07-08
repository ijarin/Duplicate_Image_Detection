#import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
#import time
import yaml
from util import warmup, BuildLSHTable
from load_data import image_data
from model import load_model
from plot import visualize_lsh
print("check tensorflow version",tf. __version__)
with open("config.yaml", "r") as fp:
    args = yaml.safe_load(fp)
#import tensorflow_datasets as tfds
IMAGE_SIZE =args["imsize"]
NUM_IMAGES=args["num_image"]

#model and data
images,labels,train_ds,validation_ds=image_data(IMAGE_SIZE,NUM_IMAGES)
embedding_model=load_model(IMAGE_SIZE)
#tfds.disable_progress_bar()

#warmup the GPU
warmup(IMAGE_SIZE,embedding_model)

training_files = zip(images, labels)
lsh_builder = BuildLSHTable(embedding_model)
lsh_builder.train(training_files)

# First serialize the embedding model as a SavedModel.
embedding_model.save("embedding_model")

# Initialize the conversion parameters.
#params = tf.experimental.tensorrt.ConversionParams(precision_mode="FP16", maximum_cached_engines=16)

# Run the conversion.
#converter = tf.experimental.tensorrt.Converter( input_saved_model_dir="embedding_model", conversion_params=params)
#converter.convert()
#converter.save("tensorrt_embedding_model")
#Visualize set of validation image
validation_images = []
validation_labels = []

for image, label in validation_ds.take(100):
    image = tf.image.resize(image, (224, 224))
    validation_images.append(image.numpy())
    validation_labels.append(label.numpy())

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)
print("validation shape",validation_images.shape, validation_labels.shape)

#plot
for _ in range(5):
      visualize_lsh(lsh_builder,validation_images,validation_labels)
visualize_lsh(lsh_builder,validation_images,validation_labels)
