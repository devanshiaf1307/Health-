import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Load the model
model_path = r"C:\Devansh College\New folder\Blood group detection\fingerprint-based-blood-group-detection-main\test\model_blood_group_detection_resnet.h5"
model = load_model(model_path)

# Create test dataset generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Define test dataset path
test_dir = r"C:\Devansh College\New folder\Blood group detection\fingerprint-based-blood-group-detection-main\dataset"

# Load images from test directory
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode="categorical"  # ✅ Fix: Use categorical instead of binary
)

# Evaluate the model on test dataset
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
