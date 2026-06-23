import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

def load_dataset(base_dir, img_height=224, img_width=224, batch_size=32):
    """
    Loading dataset: train, valid and test. Instead of using an online dataset, i'm using an offline one
    """
    print("Attempting to load skin lesions dataset...")
    try:
        train_dir = os.path.join(base_dir, 'train')
        valid_dir = os.path.join(base_dir, 'valid')
        test_dir = os.path.join(base_dir, 'test')

        #Using keras to load from directory
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical', # Useful for multi class
            shuffle=True
        )
        
        valid_ds = tf.keras.utils.image_dataset_from_directory(
            valid_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False
        )

        # Extracting class name from train folder
        class_names = train_ds.class_names

        print("Dataset loaded")
        return train_ds, valid_ds, test_ds, class_names

    except Exception as e:
        print(f"An error occurred during load: {e}")
        return None, None, None, None

if __name__ == "__main__":
    DATASET_PATH = "./img_folder"
    
    # --- CRITICAL VARIABLE: EXPERIMENT NAME ---
    EXP_NAME = "exp03_dropout"

    # --- DEDICATED FOLDER ONLY FOR THIS EXPERIMENT ---
    # Now it will automatically create a folder named "risultati_exp03_dropout"
    RESULTS_DIR = f"risultati_{EXP_NAME}" 
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds, valid_ds, test_ds, class_names = load_dataset(DATASET_PATH)

  

    # We keep Data Augmentation from previous experiment
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    # --- NEW ARCHITECTURE MODEL WITH DROPOUT ---
    model = Sequential([
        layers.Input(shape=(224, 224, 3)),
        
        # 1. Data Augmentation
        data_augmentation,
        
        # 2. Normalization
        layers.Rescaling(1./255),
        
        # 3. Convolutional Network
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        
        # --- NEW THING: DROPOUT LAYER (Regularization) ---
        # 50% of neurons casually turned off to prevent overfitting
        layers.Dropout(0.5), 
        
        layers.Dense(7, activation='softmax') 
    ])

    # Compiling
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # --- TRAINING ---
    epochs = 20
    
    print(f"\nStarting training with Dropout (Experiment: {EXP_NAME})...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        verbose=1 
    )

    # --- VALUATION AND SAVING ---
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    
    y_true_flat = np.concatenate([np.argmax(y, axis=1) for x, y in test_ds], axis=0)
    y_pred = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Model Confusion Matrix - {EXP_NAME}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{EXP_NAME}_confusion_matrix.png"))
    plt.close()

    # Classification Report
    report = classification_report(y_true_flat, y_pred_classes, target_names=class_names)
    report_path = os.path.join(RESULTS_DIR, f"{EXP_NAME}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model Classification Report - {EXP_NAME}\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Test Loss: {loss:.4f}\n\