from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import bpy
import json
import os

def predict_image(image_path):
    
    # load model, specify which version 
    #model_save_dir = os.path.abspath("\\Sketch-To-Mes\\ModelInfo\\stm-v.1.0.keras")
    model_save_dir = r"C:\Users\judah\Desktop\Code Stuffs\Filtered2\Filtered\Sketch-To-Mesh\ModelInfo"
    version = 1.0 # edit this to the current version you have developed
    model_save_path = os.path.join(model_save_dir, f'stm-v.{version}.keras')
    model = load_model(model_save_path)

    # load json file with indices because i fucked up and didnt save the class indices in the model training
    class_indices_path = os.path.join(model_save_dir, 'class_indices.json')
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    class_indices = {v: k for k, v in class_indices.items()}  # reverse the class indices dictionary
    
    img_width, img_height = 224, 224  # this is needed because the model was trained to analyze images of this dimension
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    predicted_class = class_indices[predicted_class_index]
    
    print(f"Predicted class: {predicted_class}")
    
    return(predicted_class)

# edit a cone to look like a pyramid
# there is no pyramid presets in blender
def add_pyramid(size=1, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=size, depth=size, location=location)

def create_temp(predicted_object_name):
    
    shapes_info = [
        {'func': bpy.ops.mesh.primitive_cube_add, 'param': 'size', 'name': 'Cube'},
        {'func': bpy.ops.mesh.primitive_uv_sphere_add, 'param': 'radius', 'name': 'Sphere'},
        {'func': bpy.ops.mesh.primitive_cone_add, 'param': 'radius1', 'name': 'Cone'},
        {'func': bpy.ops.mesh.primitive_cylinder_add, 'param': 'radius', 'name': 'Cilinder'},
        {'func': add_pyramid, 'param': 'size', 'name': 'Pyramid'}
    ]
    
    for shape_info in shapes_info:
        if(shape_info['name'] == predicted_object_name):
            kwargs = {shape_info['param']: 2, 'location': (0, 0, 0)}
            shape_info['func'](**kwargs)
            break
    
    return
    