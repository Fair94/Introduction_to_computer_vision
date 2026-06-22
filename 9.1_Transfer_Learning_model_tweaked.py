import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

def load_dataset(base_dir, img_height=224, img_width=224, batch_size=32):
    print("Tentativo di caricamento del dataset...")
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
        print(f"Errore di caricamento: {e}")
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
    
    # --- VARIABILI ESPERIMENTI ---
    EXP_OLD = "exp09_transfer_learning"
    OLD_MODEL_PATH = f"risultati_{EXP_OLD}/{EXP_OLD}_model.h5"
    
    EXP_NAME = "exp09.1_transfer_finetuned"
    RESULTS_DIR = f"risultati_{EXP_NAME}" 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, f"{EXP_NAME}_model.h5")

    train_ds, valid_ds, test_ds, class_names = load_dataset(DATASET_PATH)
    soft_weights = get_soft_class_weights(train_ds)

    # 1. Ricreiamo l'esatta architettura dell'Exp 9
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1) # Aggiunto il contrasto clinico
    ], name="data_augmentation")

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False # Per ora lo teniamo congelato

    model = Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        layers.Rescaling(1./127.5, offset=-1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax') 
    ])

    # 2. Carichiamo i pesi già addestrati (La nostra testa non è più casuale!)
    print(f"\nCaricamento pesi dell'Esperimento 9 da: {OLD_MODEL_PATH}")
    model.load_weights(OLD_MODEL_PATH)
    print("Pesi caricati con successo.")

    # --- IL CUORE DEL FINE-TUNING ---
    print("\nInizio procedura di Unfreezing...")
    # Sblocchiamo il base_model
    base_model.trainable = True

    # MobileNetV2 ha 154 layers. Ne sblocchiamo solo gli ultimi 30.
    # Così manteniamo la capacità di riconoscere linee/curve (primi layer)
    # ma adattiamo la capacità di riconoscere texture complesse (ultimi layer).
    fine_tune_at = len(base_model.layers) - 30

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # 3. Ri-compiliamo il modello con un Learning Rate MICROSCOPICO
    # Se usassimo il LR standard, distruggeremmo i pesi di Google in un istante.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # <--- 0.00001
        loss='categorical_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_score')]
    )

    # 4. Impostiamo i Guardiani (Callbacks)
    early_stopping = EarlyStopping(
        monitor='val_f1_score', mode='max', patience=5, restore_best_weights=True, verbose=1
    )
    
    # Salviamo ad ogni miglioramento della F1
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH, monitor='val_f1_score', mode='max', save_best_only=True, verbose=0
    )

    print(f"\nInizio Fine-Tuning Avanzato (Esperimento: {EXP_NAME})...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=15, # Non servono troppe epoche per il fine-tuning
        class_weight=soft_weights,
        callbacks=[early_stopping, checkpoint],
        verbose=1 
    )

    # --- VALUTAZIONE FINALE ---
    print("\nValutazione sul Test Set...")
    model.load_weights(MODEL_SAVE_PATH) # Ricarica il migliore
    loss, accuracy, f1 = model.evaluate(test_ds, verbose=0)
    
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
        f.write(f"Model Classification Report - {EXP_NAME}\n")
        f.write(f"Tweak: MobileNetV2 Fine-Tuning (Top 30 layers unfreezed) + LR 1e-5\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

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
    print("Salvataggi completati.")