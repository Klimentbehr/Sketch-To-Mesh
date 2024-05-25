import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model_save_path = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\models\geometric_shapes_model.keras"

model = tf.keras.models.load_model(model_save_path)

# mapping of class indices to class names (update this dictionary based on your training data)
# idk about thsi tbh
class_indices = {
    0: 'Cone',
    1: 'Cube',
    2: 'Cylinder',
    3: 'Sphere'
}

def predict_image(image_path):
    
    # load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))  # Resize to the input size of the model
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array /= 255.0  # Normalize the image
    
    # predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])  # Get the index of the highest probability
    
    # get the class name from the class indices
    predicted_class = class_indices[predicted_class_index]
    
    print(f"Predicted class: {predicted_class}")

test_image_path = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\Cube_3_angle_36.00_30.00_POINT_1000.png"
predict_image(test_image_path)
