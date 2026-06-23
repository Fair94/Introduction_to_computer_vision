import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix

def load_dataset(base_dir, img_height=224, img_width=224, batch_size=32):
    print("Attempting to load dataset...")
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(base_dir, 'train'), image_size=(img_height, img_width),
            batch_size=batch_size, label_mode='categorical', shuffle=True
        )
        valid_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(base_dir, 'valid'), image_size=(img_height, img_width),
            batch_size=batch_size, label_mode='categorical', shuffle=False
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(base_dir, 'test'), image_size=(img_height, img_width),
            batch_size=batch_size, label_mode='categorical', shuffle=False
        )
        return train_ds, valid_ds, test_ds, train_ds.class_names
    except Exception as e:
        print(f"Loading error: {e}")
        return None, None, None, None

# We use "soft" weights for help the model on rare classes without make it explode
def get_soft_class_weights(train_ds):
    y_train = np.concatenate([y for x, y in train_ds], axis=0)
    y_train_indices = np.argmax(y_train, axis=1)
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    return {i: min(np.sqrt(weight), 4.0) for i, weight in enumerate(cw)}

if __name__ == "__main__":
    DATASET_PATH = "./img_folder"
    
    # --- EXPERIMENT 9 VARIABLE ---
    EXP_NAME = "exp09_transfer_learning"
    RESULTS_DIR = f"risultati_{EXP_NAME}" 
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds, valid_ds, test_ds, class_names = load_dataset(DATASET_PATH)
    soft_weights = get_soft_class_weights(train_ds)

    data_augmentation = Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    # --- TRANSFER LEARNING: THE GOOGLE'S BRAIN ---
    print("\nDownloading MobileNetV2 weights...")
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, # We cut the original head (that was used for 1000 classes)
        weights='imagenet' # We use the ImageNet experience
    )
    
    # We FREEZE the weights: we don't want to destroy the previous Google knowledge
    base_model.trainable = False 

    # Assembling the final architecture
    model = Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        
        # Specific preprocessing required by MobileNetV2 (pixels from -1 to 1)
        layers.Rescaling(1./127.5, offset=-1),
        
        base_model, # We insert the giant MobileNet block
        
        # New customized "Head" for our medical dataset
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax') 
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    epochs = 20
    
    print(f"\nStarting Transfer Learning training (Experiment: {EXP_NAME})...")
    # Note: since base_model is frozen, we are training ONLY the last layer. 
    # It will be surprisingly fast!
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        class_weight=soft_weights,
        verbose=1 
    )

    # --- VALUATION AND SAVINGS ---
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    
    y_true_flat = np.concatenate([np.argmax(y, axis=1) for x, y in test_ds], axis=0)
    y_pred = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true_flat, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Model Confusion Matrix - {EXP_NAME}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{EXP_NAME}_confusion_matrix.png"))
    plt.close()

    report = classification_report(y_true_flat, y_pred_classes, target_names=class_names)
    report_path = os.path.join(RESULTS_DIR, f"{EXP_NAME}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model Classification Report - {EXP_NAME} (Transfer Learning MobileNetV2)\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{EXP_NAME}: Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{EXP_NAME}: Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{EXP_NAME}_training_history.png"))
    plt.close()

    MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, f"{EXP_NAME}_model.h5")
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel {EXP_NAME} saved in {MODEL_SAVE_PATH}")