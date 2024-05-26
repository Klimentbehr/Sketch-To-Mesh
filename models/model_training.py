# import tensorflow as tf
# import os
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.applications import VGG16

# # Dataset organization
# train_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\train"
# validation_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\validation"
# test_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\test"

# # Where to save the model
# model_save_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\models"
# os.makedirs(model_save_dir, exist_ok=True)
# version = 0.7
# model_save_path = os.path.join(model_save_dir, f'stm-v.{version}.keras')

# # Image data generators with augmentation for training set
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255.0,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # validation and test data generators (without augmentation)
# # i tried aug earlier didnt get good results
# validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
# test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# # data loaders
# train_generator = train_datagen.flow_from_directory(
#     train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
# )

# # load pre-trained VGG16 model + higher level layers
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# # freeze the convolutional base
# # tbh no fucking clue what is this
# for layer in base_model.layers:
#     layer.trainable = False

# # Create the model
# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(5, activation='softmax')  # 5 classes: Cilinder, Cone, Cube, Pyramid, Sphere
# ])

# # compile the model
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # early stopping callback
# # idk why this is necessary
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # fit train
# history = model.fit(
#     train_generator,
#     epochs=50,
#     validation_data=validation_generator,
#     callbacks=[early_stopping]
# )

# # evaluate the model
# loss, accuracy = model.evaluate(test_generator)
# print(f'Test Accuracy: {accuracy * 100:.2f}%')

# model.save(model_save_path)
# print(f'Model saved at: {model_save_path}')

# ------------------------------------------------ next one was .7333 but still had 70% acc

# import tensorflow as tf
# import os
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.applications import VGG16

# 
# train_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\train"
# validation_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\validation"
# test_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\test"

#
# model_save_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\models"
# os.makedirs(model_save_dir, exist_ok=True)
# version = 0.9
# model_save_path = os.path.join(model_save_dir, f'stm-v.{version}.keras')

# 
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255.0,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# 
# validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
# test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# 
# train_generator = train_datagen.flow_from_directory(
#     train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
# )

# 
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# 
# for layer in base_model.layers:
#     layer.trainable = False

# 
# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(5, activation='softmax')  # 5 classes: Cilinder, Cone, Cube, Pyramid, Sphere
# ])

#
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 
# history = model.fit(
#     train_generator,
#     epochs=100,  # Increase the number of epochs
#     validation_data=validation_generator,
#     callbacks=[early_stopping]
# )
 
# 
# loss, accuracy = model.evaluate(test_generator)
# print(f'Test Accuracy: {accuracy * 100:.2f}%')

# model.save(model_save_path)
# print(f'Model saved at: {model_save_path}')

import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16

# setting up enviroment to speed up epoch learning
# took me ages without this, changed so i would use 100% of my cpu. need to be careful with this tho, not all cpus have 20 threads and 10 cores
os.environ["OMP_NUM_THREADS"] = "20"  # number of threads should be equal to the number of logical processors
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"  # typically set this to 2 or the number of physical cores

# configure TensorFlow to use multiple cores
tf.config.threading.set_intra_op_parallelism_threads(20)
tf.config.threading.set_inter_op_parallelism_threads(2)

train_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\train"
validation_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\validation"
test_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets\test"

model_save_dir = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\models"
os.makedirs(model_save_dir, exist_ok=True)
version = 1.0
model_save_path = os.path.join(model_save_dir, f'stm-v.{version}.keras')

# image data generators with augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# validation and test data generators (without augmentation)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# data loaders
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# load pre-trained VGG16 model + higher level layers
# TODO: have no clue, understand this
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# freeze the base model initially
# TODO: have no clue, try to understand this?
for layer in base_model.layers:
    layer.trainable = False

# ddd custom layers on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 classes: Cilinder, Cone, Cube, Pyramid, Sphere
])

# Compiling the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate reduction callbacks
# TODO: have no clue, try to understand this?
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model with frozen base layers
# TODO: have no clue, try to understand this?
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Unfreeze more layers in the base model for fine-tuning
# TODO: have no clue, try to understand this?
for layer in base_model.layers[-8:]:  # Unfreezing the last 8 layers
    layer.trainable = True

# Re-compile the model with a lower learning rate for fine-tuning
# TODO: have no clue, try to understand this?
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
# TODO: have no clue, try to understand this?
history_fine = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

model.save(model_save_path)
print(f'Model saved at: {model_save_path}')


