import numpy as np
import tensorflow as tf
from keras.preprocessing import image

MODEL = tf.keras.models.load_model("models/3")

CLASS_NAMES = ['Anthracnose', 'Bacterial_Spot', 'Bacterial_blight', 'Gall_Midge', 'Healthy',
               'Sooty_Mould']


def prediction(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return predicted_class, confidence
