import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
tfds.disable_progress_bar()

def image_data(IMAGE_SIZE,NUM_IMAGES):

    train_ds, validation_ds = tfds.load(
        "tf_flowers", split=["train[:85%]", "train[85%:]"], as_supervised=True
    )

    #IMAGE_SIZE = 224
    #NUM_IMAGES = 1000

    images = []
    labels = []

    for (image, label) in train_ds.take(NUM_IMAGES):
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(image.numpy())
        labels.append(label.numpy())

    images = np.array(images)
    labels = np.array(labels)

    return images,labels, train_ds, validation_ds