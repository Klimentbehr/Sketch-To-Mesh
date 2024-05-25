import os
import shutil
import random

# Paths to the original dataset and the new train, validation, and test directories
original_dataset_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets"
train_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\train"
validation_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\validation"
test_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\test"

# Ensure the directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split ratios
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# Function to split the data
def split_data_by_size(original_dir, train_dir, validation_dir, test_dir, train_ratio, validation_ratio):
    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if os.path.isdir(class_dir):
            # Create dictionaries to hold images for each size
            size_images = {1: [], 2: [], 3: []}
            
            # Categorize images by size
            for image in os.listdir(class_dir):
                if '1_' in image:
                    size_images[1].append(image)
                elif '2_' in image:
                    size_images[2].append(image)
                elif '3_' in image:
                    size_images[3].append(image)
            
            # Shuffle images in each size category
            for size in size_images:
                random.shuffle(size_images[size])
            
            # Split images into train, validation, and test sets
            for size, images in size_images.items():
                train_count = int(train_ratio * len(images))
                validation_count = int(validation_ratio * len(images))
                
                train_images = images[:train_count]
                validation_images = images[train_count:train_count + validation_count]
                test_images = images[train_count + validation_count:]
                
                # Create directories for each size
                size_train_dir = os.path.join(train_dir, class_name, f'size_{size}')
                size_validation_dir = os.path.join(validation_dir, class_name, f'size_{size}')
                size_test_dir = os.path.join(test_dir, class_name, f'size_{size}')
                
                os.makedirs(size_train_dir, exist_ok=True)
                os.makedirs(size_validation_dir, exist_ok=True)
                os.makedirs(size_test_dir, exist_ok=True)
                
                # Copy images to train directory
                for image in train_images:
                    src = os.path.join(class_dir, image)
                    dst = os.path.join(size_train_dir, image)
                    try:
                        shutil.copy(src, dst)
                    except PermissionError as e:
                        print(f"PermissionError: {e} - src: {src} - dst: {dst}")
                
                # Copy images to validation directory
                for image in validation_images:
                    src = os.path.join(class_dir, image)
                    dst = os.path.join(size_validation_dir, image)
                    try:
                        shutil.copy(src, dst)
                    except PermissionError as e:
                        print(f"PermissionError: {e} - src: {src} - dst: {dst}")
                
                # Copy images to test directory
                for image in test_images:
                    src = os.path.join(class_dir, image)
                    dst = os.path.join(size_test_dir, image)
                    try:
                        shutil.copy(src, dst)
                    except PermissionError as e:
                        print(f"PermissionError: {e} - src: {src} - dst: {dst}")

# Split the dataset
split_data_by_size(original_dataset_dir, train_dir, validation_dir, test_dir, train_ratio, validation_ratio)

print("Dataset split into training, validation, and test sets by size.")
