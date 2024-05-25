import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# dataset organization
train_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\train"
validation_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\validation"
test_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\test"

# where to save the model so i can load it after
model_save_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\models"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, 'geometric_shapes_model.keras')

# Verify directory structure
def verify_directory_structure(base_dir):
    for root, dirs, files in os.walk(base_dir):
        print(f"Checking directory: {root}")
        if root != base_dir and len(files) == 0:
            print(f"Warning: No files found in directory {root}")

print("Training Directory Contents:")
verify_directory_structure(train_dir)

print("Validation Directory Contents:")
verify_directory_structure(validation_dir)

print("Test Directory Contents:")
verify_directory_structure(test_dir)

# Enhanced image data generator with augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

# Validation and test data generators (without augmentation)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Data loaders
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
)

# Enhanced model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
history = model.fit(train_generator, epochs=50, validation_data=validation_generator)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

model.save(model_save_path)
print(f'Model saved at: {model_save_path}')

# prediction function
# TODO: move this to another file
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    predicted_class = class_indices[predicted_class_index]
    
    print(f"Predicted class: {predicted_class}")

test_image_path = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\Cube_3_angle_36.00_30.00_POINT_1000.png"
predict_image(test_image_path)
