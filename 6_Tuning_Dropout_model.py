import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

def load_dataset(base_dir, img_height=224, img_width=224, batch_size=32):
    print("Tentativo di caricamento del dataset di lesioni cutanee...")
    try:
        train_dir = os.path.join(base_dir, 'train')
        valid_dir = os.path.join(base_dir, 'valid')
        test_dir = os.path.join(base_dir, 'test')

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=(img_height, img_width),
            batch_size=batch_size, label_mode='categorical', shuffle=True
        )
        valid_ds = tf.keras.utils.image_dataset_from_directory(
            valid_dir, image_size=(img_height, img_width),
            batch_size=batch_size, label_mode='categorical', shuffle=False
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir, image_size=(img_height, img_width),
            batch_size=batch_size, label_mode='categorical', shuffle=False
        )

        class_names = train_ds.class_names
        print("Dataset loaded")
        return train_ds, valid_ds, test_ds, class_names

    except Exception as e:
        print(f"An error occurred during load: {e}")
        return None, None, None, None

# --- HYPERTUNING: FUNZIONE PER I PESI "DOLCI" ---
def get_soft_class_weights(train_ds):
    y_train = np.concatenate([y for x, y in train_ds], axis=0)
    y_train_indices = np.argmax(y_train, axis=1)
    
    # Calcolo pesi puri (quelli del disastro exp05)
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    
    # Tuning: Radice quadrata e tetto massimo a 4.0
    # In questo modo diamo importanza alle minoranze ma evitiamo l'esplosione dei gradienti
    soft_cw = {i: min(np.sqrt(weight), 4.0) for i, weight in enumerate(cw)}
    return soft_cw

if __name__ == "__main__":
    DATASET_PATH = "./img_folder"
    
    # --- VARIABILE ESPERIMENTO 6 ---
    EXP_NAME = "exp06_hypertuning"
    RESULTS_DIR = f"risultati_{EXP_NAME}" 
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds, valid_ds, test_ds, class_names = load_dataset(DATASET_PATH)

    print("\nCalcolo dei Soft Class Weights per l'Hypertuning...")
    soft_weights = get_soft_class_weights(train_ds)
    print(f"Pesi addolciti applicati: {soft_weights}")

    data_augmentation = Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    # --- ARCHITETTURA (Basata sul nostro modello migliore: Exp 3) ---
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
        
        # HYPERTUNING: Dropout leggermente abbassato (da 0.5 a 0.4)
        layers.Dropout(0.4), 
        
        layers.Dense(7, activation='softmax') 
    ])

    # --- HYPERTUNING: LEARNING RATE ---
    # Invece del default (0.001), usiamo 0.0001. Passi lenti e sicuri.
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=custom_optimizer,
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # Avendo abbassato il Learning Rate, potremmo aver bisogno di più epoche,
    # ma teniamo 20 per fare un confronto alla pari con gli altri esperimenti.
    epochs = 20
    
    print(f"\nInizio addestramento Hypertuning (Esperimento: {EXP_NAME})...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        class_weight=soft_weights, # Applichiamo i pesi calibrati
        verbose=1 
    )

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    
    y_true_flat = np.concatenate([np.argmax(y, axis=1) for x, y in test_ds], axis=0)
    y_pred = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Salvataggio Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Model Confusion Matrix - {EXP_NAME}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{EXP_NAME}_confusion_matrix.png"))
    plt.close()

    # Salvataggio Classification Report
    report = classification_report(y_true_flat, y_pred_classes, target_names=class_names)
    report_path = os.path.join(RESULTS_DIR, f"{EXP_NAME}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model Classification Report - {EXP_NAME}\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Test Loss: {loss:.4f}\n\n")
        f.write(report)

    # Salvataggio Grafici History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{EXP_NAME}: Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{EXP_NAME}: Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{EXP_NAME}_training_history.png"))
    plt.close()

    # Salvataggio Modello
    MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, f"{EXP_NAME}_model.h5")
    model.save(MODEL_SAVE_PATH)
    print(f"\nModello {EXP_NAME} salvato in {MODEL_SAVE_PATH}")