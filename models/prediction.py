from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model_path = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\models\stm-v.0.7.keras"
model = load_model(model_path)

def predict_shape(img_path):
    img_width, img_height = 150, 150
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    classes = ['Cilinder', 'Cone', 'Cube', 'Pyramid', 'Sphere']
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class

# Example usage
img_path = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\oldD\test\Cube\Cube_3_4_aa655ae0-8f3b-4e22-afd9-3d845e1bab6a_outline_150.38_-55.73.png"

predicted_shape = predict_shape(img_path)
print(f'The predicted shape is: {predicted_shape}')
