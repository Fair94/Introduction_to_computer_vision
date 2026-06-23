import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
    
    # --- EXPERIMENT 8.1 VARIABLE ---
    EXP_NAME = "exp08.1_custom_tweaked"
    RESULTS_DIR = f"risultati_{EXP_NAME}" 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, f"{EXP_NAME}_model.h5")

    train_ds, valid_ds, test_ds, class_names = load_dataset(DATASET_PATH)
    soft_weights = get_soft_class_weights(train_ds)

    # TWEAK 1: Added RandomContrast for highlight the lesions borders
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1) 
    ], name="data_augmentation")

    model = Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4), 
        layers.Dense(7, activation='softmax') 
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_score')]
    )

    # --- TWEAK 2 & 3: THE NEW GUARDIANS (CALLBACKS) ---
    
    # 1. Early Stopping: It stop if val_f1_score don't improve for 8 epochs
    early_stopping = EarlyStopping(
        monitor='val_f1_score',
        mode='max',
        patience=8,
        restore_best_weights=True, # Automatic reload the best version!
        verbose=1
    )

    # 2. Reduce LR on Plateau: Divide by half the step if it get stuck for 3 epochs
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_f1_score',
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print(f"\nStarting Tweaked training (Experiment: {EXP_NAME})...")
    # We raised epochs to 40, anyway it will stop by himself when finished to learn
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=40, 
        class_weight=soft_weights,
        callbacks=[early_stopping, lr_scheduler], # We insert the management intelligence
        verbose=1 
    )

    # --- VALUATION (The best weights are already restored from EarlyStopping) ---
    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved in {MODEL_SAVE_PATH}")

    loss, accuracy, f1 = model.evaluate(test_ds, verbose=0)
    
    y_true_flat = np.concatenate([y for x, y in test_ds], axis=0)
    y_true_classes = np.argmax(y_true_flat, axis=1)
    
    y_pred = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Standard savings...
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Model Confusion Matrix - {EXP_NAME}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{EXP_NAME}_confusion_matrix.png"))
    plt.close()

    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    report_path = os.path.join(RESULTS_DIR, f"{EXP_NAME}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model Classification Report - {EXP_NAME}\n")
        f.write(f"Tweak: EarlyStopping + LR Scheduler + Contrast Augmentation\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

    # Updated History Graph to show also Learning Rate (optional but cool to see)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_score'], label='Train F1')
    plt.plot(history.history['val_f1_score'], label='Val F1')
    plt.title('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{EXP_NAME}_training_history.png"))
    plt.close()
    print("Savings completed.")