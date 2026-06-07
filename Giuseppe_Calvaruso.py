import tensorflow as tf
import os

# --- 1. PROJECT SETUP AND DATA LOADING ---

def load_dataset(base_dir, img_height=224, img_width=224, batch_size=32):
    """
    Carica il dataset dei nei dalle cartelle locali e restituisce i set di train, valid e test.
    """
    print("Tentativo di caricamento del dataset di lesioni cutanee...")
    try:
        train_dir = os.path.join(base_dir, 'train')
        valid_dir = os.path.join(base_dir, 'valid')
        test_dir = os.path.join(base_dir, 'test')

        # Caricamento tramite Keras dalle directory
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical', # Necessario per classificazione multi-classe
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

        # Estraiamo i nomi delle classi direttamente dalla cartella di train
        class_names = train_ds.class_names

        print("Dataset caricato con successo.")
        return train_ds, valid_ds, test_ds, class_names

    except Exception as e:
        print(f"Si è verificato un errore durante il caricamento del dataset: {e}")
        return None, None, None, None

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Project Kickoff: End-to-End Skin Lesion Classifier ---")
    print("Selected Dataset: Skin Cancer 7-Class")
    
    # Inserisci qui il percorso della cartella principale dove tieni 'train', 'valid' e 'test'
    DATASET_PATH = "./percorso/del/tuo/dataset" 

    # Caricamento del dataset
    train_ds, valid_ds, test_ds, class_names = load_dataset(DATASET_PATH)

    # Verifica che i dati siano stati caricati correttamente
    if train_ds is not None:
        print("\n--- Dataset Verification ---")
        print(f"Classi rilevate automaticamente: {class_names}")
        
        # Estraiamo un batch per vederne la forma
        for image_batch, labels_batch in train_ds.take(1):
            print(f"Forma del batch di immagini: {image_batch.shape}")
            print(f"Forma del batch di etichette: {labels_batch.shape}")