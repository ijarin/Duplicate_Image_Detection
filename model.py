import tensorflow as tf

def load_model(IMAGE_SIZE):
    bit_model = tf.keras.models.load_model("flower_model_bit_0.96875")
    bit_model.count_params()

    #create an embedding model to generate feature vectors
    embedding_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3)),
            tf.keras.layers.Rescaling(scale=1.0 / 255),
            bit_model.layers[1],
            tf.keras.layers.Normalization(mean=0, variance=1),
        ],
        name="embedding_model",
    )

    print(embedding_model.summary())
    return embedding_model


