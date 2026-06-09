import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# --- 1. PROJECT SETUP, DATA LOADING AND EDA: EXPLORING DATA ANALYSIS ---

def load_dataset(base_dir, img_height=224, img_width=224, batch_size=32):
    """
    

    Loading dataset: train, valid and test. Instead of using an online dataset, i'm using an offline one
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



# --- 3. BASELINE CONSTRUCTION ---

#
model = Sequential([
    #Pixel normalization between 0 and 1
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
    #Convolutional layer
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Flattening and classification
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax') 
])

# Model compiling 
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Baseline training 

epochs = 10

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs
)