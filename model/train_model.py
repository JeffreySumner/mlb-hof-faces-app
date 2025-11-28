"""
Train and save the Hall of Fame face prediction model.
Run this script once to create the model file.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tensorflow import keras
from tensorflow.keras import layers

# Set random seeds
random.seed(42)
np.random.seed(42)

def load_training_data():
    """Load training data from the blog post directory."""
    # Path to training images
    data_dir = Path(__file__).parent.parent.parent / "posts" / "MLBHallOfFameFaces" / "data" / "PlayerImages"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data not found at {data_dir}")
    
    # Get all images
    all_files = list(data_dir.glob("*.jpg"))
    
    # Split into HOF and non-HOF
    hof_files = [f for f in all_files if f.name.startswith("hof_")]
    nothof_files = [f for f in all_files if f.name.startswith("nothof_")]
    
    print(f"Found {len(hof_files)} HOF images and {len(nothof_files)} non-HOF images")
    
    # Create train/test splits
    n_hof_train = min(50, int(len(hof_files) * 0.7))
    n_nothof_train = min(50, int(len(nothof_files) * 0.7))
    
    random.shuffle(hof_files)
    random.shuffle(nothof_files)
    
    hof_train = hof_files[:n_hof_train]
    hof_test = hof_files[n_hof_train:]
    
    nothof_train = nothof_files[:n_nothof_train]
    nothof_test = nothof_files[n_nothof_train:]
    
    # Balance test set
    n_test = min(len(hof_test), len(nothof_test))
    hof_test = hof_test[:n_test]
    nothof_test = nothof_test[:n_test]
    
    # Combine
    train_files = hof_train + nothof_train
    test_files = hof_test + nothof_test
    
    print(f"Training on {len(train_files)} images, testing on {len(test_files)} images")
    
    # Load and preprocess images
    def load_image(filepath):
        img = Image.open(filepath).convert('L')
        img = img.resize((32, 32))
        return np.array(img) / 255.0
    
    # Load training data
    train_x = np.array([load_image(f).flatten() for f in train_files])
    train_y = np.array([1] * len(hof_train) + [0] * len(nothof_train))
    
    # Load test data
    test_x = np.array([load_image(f).flatten() for f in test_files])
    test_y = np.array([1] * len(hof_test) + [0] * len(nothof_test))
    
    # One-hot encode labels
    train_labels = keras.utils.to_categorical(train_y, num_classes=2)
    test_labels = keras.utils.to_categorical(test_y, num_classes=2)
    
    return train_x, train_labels, test_x, test_labels


def build_model():
    """Build the neural network model."""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(1024,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    
    return model


def main():
    """Train and save the model."""
    print("Loading training data...")
    train_x, train_labels, test_x, test_labels = load_training_data()
    
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        train_x, train_labels,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_x, test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model_path = Path(__file__).parent / "hof_model.keras"
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    
    print("\n✅ Model training complete!")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
