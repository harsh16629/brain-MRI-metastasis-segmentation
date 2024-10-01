import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, Model

# Load dataset paths
def load_data(data_dir):
    images = []
    masks = []
    for file_name in os.listdir(data_dir):
        if 'mask' in file_name:
            mask_path = os.path.join(data_dir, file_name)
            image_path = mask_path.replace('_mask', '')  # Get the corresponding image path

            mask = cv2.imread(mask_path, 0)
            image = cv2.imread(image_path, 0)

            if image is not None and mask is not None:
                images.append(image)
                masks.append(mask)

    return np.array(images), np.array(masks)

# Apply CLAHE preprocessing
def preprocess_image(image):
    """Apply CLAHE to improve contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Preprocess dataset
def preprocess_dataset(images):
    """Preprocess all images in the dataset."""
    preprocessed_images = [preprocess_image(img) for img in images]
    return np.array(preprocessed_images)

# Load and preprocess data
data_dir = 'Dataset'  # Update with your actual dataset directory
images, masks = load_data(data_dir)
images = preprocess_dataset(images)

# Normalize images and masks
images = images.astype('float32') / 255.0
masks = masks.astype('float32') / 255.0

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

def conv_block(inputs, num_filters):
    """A block that includes two convolutional layers followed by batch normalization and ReLU activation."""
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    return x

def unet_plus_plus(input_shape=(256, 256, 1), num_classes=1):
    """Defines the U-Net++ architecture."""
    inputs = layers.Input(input_shape)

    # Encoder Path
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    c5 = conv_block(p4, 1024)  # Bottleneck

    # Decoder Path with Nested Skip Connections (U-Net++)
    u4_0 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
    u4_0 = layers.Concatenate()([u4_0, c4])
    c4_0 = conv_block(u4_0, 512)
    
    u3_0 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c4_0)
    u3_0 = layers.Concatenate()([u3_0, c3])
    c3_0 = conv_block(u3_0, 256)
    
    u2_0 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c3_0)
    u2_0 = layers.Concatenate()([u2_0, c2])
    c2_0 = conv_block(u2_0, 128)
    
    u1_0 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c2_0)
    u1_0 = layers.Concatenate()([u1_0, c1])
    c1_0 = conv_block(u1_0, 64)

    # Nested skip connections
    u3_1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c4_0)
    u3_1 = layers.Concatenate()([u3_1, u3_0, c3])
    c3_1 = conv_block(u3_1, 256)
    
    u2_1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c3_1)
    u2_1 = layers.Concatenate()([u2_1, u2_0, c2])
    c2_1 = conv_block(u2_1, 128)
    
    u1_1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c2_1)
    u1_1 = layers.Concatenate()([u1_1, u1_0, c1])
    c1_1 = conv_block(u1_1, 64)
    
    # Output Layer
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(c1_1)
    
    # Define the model
    model = Model(inputs, outputs)
    
    return model

# Model configuration
input_shape = (256, 256, 1)  # Example input shape for MRI images
num_classes = 1  # Binary segmentation (1 class)
model = unet_plus_plus(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a ModelCheckpoint callback to save the best model during training
checkpoint = ModelCheckpoint('models/current_best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, callbacks=[checkpoint])

# Model summary
model.summary()

#saving the model
model.save('models/current_best_model.h5')
