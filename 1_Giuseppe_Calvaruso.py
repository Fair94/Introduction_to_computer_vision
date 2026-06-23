import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
# --- 1. PROJECT SETUP, DATA LOADING AND EDA: EXPLORING DATA ANALYSIS ---

def load_dataset(base_dir, img_height=224, img_width=224, batch_size=32):
    """
    

    Loading dataset: train, valid and test. Instead of using an online dataset, i'm using an offline one
    """
    print("Loading dataset")
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
    # 7 is the numbers of classes of tumors
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
    epochs=epochs,
    verbose =1 
)

#  First confusion matrix for a 10 epochs model
print("\nGenerating Confusion Matrix at 10 Epochs...")

# need to flatten the true label for comparing with prediction
y_true_flat = np.concatenate([np.argmax(y, axis=1) for x, y in test_ds], axis=0)
y_pred_10 = model.predict(test_ds)
y_pred_classes_10 = np.argmax(y_pred_10, axis=1)

cm_10 = confusion_matrix(y_true_flat, y_pred_classes_10)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_10, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Baseline Model - Confusion Matrix (10 Epochs)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

RESULTS_DIR = "baseline_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_10_epochs.png"))
plt.close()

print("\nGenerating and saving Classification Report...")

# Generating report for a 10 epoch model 
# Calculate loss and accuracy on test set before to write them in the report
loss, accuracy = model.evaluate(test_ds, verbose=0)
report = classification_report(y_true_flat, y_pred_classes_10, target_names=class_names)

report_path = os.path.join(RESULTS_DIR, "baseline_classification_report_10_epoch.txt")
with open(report_path, "w") as f:
    f.write("Baseline Model Classification Report\n")
    
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Test Loss: {loss:.4f}\n\n")
    f.write(report)

print(report)
print(f"Saved to {report_path}")

# After comparing the two models, I've choosed to save the 10 epoch model as baseline ---
MODEL_SAVE_PATH = "baseline_model.h5" 
print(f"\nSaving optimal baseline model (10 epochs) to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")


# Training with 20 epoch 
# i want to see if more epochs do better or cause overfitting
print("\n--- Continuing Training: Epochs 11-20 ---")
history_20 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    verbose=1
)

# Saving 20 epochs matrix
print("\nGenerating Confusion Matrix at 20 Epochs...")
y_pred_20 = model.predict(test_ds)
y_pred_classes_20 = np.argmax(y_pred_20, axis=1)

cm_20 = confusion_matrix(y_true_flat, y_pred_classes_20)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_20, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Baseline Model - Confusion Matrix (20 Epochs)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_20_epochs.png"))
plt.close()

print(f"\nBoth matrix are saved in the folder '{RESULTS_DIR}'!")



# Classification and report
print("\nGenerating and saving Classification Report...")

# 
report = classification_report(y_true_flat, y_pred_classes_20, target_names=class_names)

report_path = os.path.join(RESULTS_DIR, "baseline_classification_report_20_epochs.txt")
with open(report_path, "w") as f:
    f.write("Baseline Model Classification Report\n")
    
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Test Loss: {loss:.4f}\n\n")
    f.write(report)

print(report)
print(f"Saved to {report_path}")

#Saving the model 


#Plotting 

plt.figure(figsize=(12, 5))

#  Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Baseline: Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Baseline: Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
history_plot_path = os.path.join(RESULTS_DIR, "baseline_training_history.png")
plt.savefig(history_plot_path)
print(f"Training history plot saved to {history_plot_path}")

print("Baseline established. Ready for improvement experiments.")