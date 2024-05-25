import os

def count_files_in_directory(directory):
    counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            counts[class_name] = len(os.listdir(class_dir))
    return counts

# Assuming your dataset directory structure is like this:
# dataset/
# ├── test/
# │   ├── Cylinder/
# │   ├── Cone/
# │   ├── Cube/
# │   └── Sphere/
# ├── train/
# │   ├── Cylinder/
# │   ├── Cone/
# │   ├── Cube/
# │   └── Sphere/
# └── validation/
#     ├── Cylinder/
#     ├── Cone/
#     ├── Cube/
#     └── Sphere/

dataset_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets"

test_dir = os.path.join(dataset_dir, "test")
train_dir = os.path.join(dataset_dir, "train")
validation_dir = os.path.join(dataset_dir, "validation")

test_counts = count_files_in_directory(test_dir)
train_counts = count_files_in_directory(train_dir)
validation_counts = count_files_in_directory(validation_dir)

print("Test set counts:", test_counts)
print("Train set counts:", train_counts)
print("Validation set counts:", validation_counts)