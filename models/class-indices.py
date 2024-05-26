import json

# this is needed because i did not save the class indices while training the best model i currently have.
# this generates a json file with the indices basically
class_indices = {
    'Cilinder': 0,
    'Cone': 1,
    'Cube': 2,
    'Pyramid': 3,
    'Sphere': 4
}

class_indices_path = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\models\class_indices.json"
with open(class_indices_path, 'w') as f:
    json.dump(class_indices, f)
