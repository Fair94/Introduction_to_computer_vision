import tensorflow as tf
import os

# --- 1. PROJECT SETUP, DATA LOADING AND EDA: EXPLORING DATA ANALYSIS ---

def load_dataset(base_dir, img_height=224, img_width=224, batch_size=32):
    """
    

    Loading dataset: train, valid and test
    """
    print("Tentativo di caricamento del dataset di lesioni cutanee...")
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

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Project Kickoff: End-to-End Skin Cancer Classifier ---")
    print("Selected Dataset: Skin Cancer 7-Class")
    
    # path where 'train', 'valid' and 'test' are located 
    DATASET_PATH = "./img_folder"

    # loading datasert
    train_ds, valid_ds, test_ds, class_names = load_dataset(DATASET_PATH)

    # Veryfing loading 
    if train_ds is not None:
        print("\n--- Dataset Verification ---")
        print(f"Classes: {class_names}")
        
        # Batch extracting
        for image_batch, labels_batch in train_ds.take(1):
            print(f"Shape: {image_batch.shape}")
            print(f"Label: {labels_batch.shape}")