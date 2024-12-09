import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class MotorImageryClassifier:
    def __init__(self, data_path):
        """
        Initialize the classifier with data preprocessing and model creation
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file with preprocessed features
        """
        # Load preprocessed data
        self.data = pd.read_csv(data_path)
        
        # Prepare features and labels
        self.X = self.data.drop(['task', 'subject'], axis=1)
        self.y = self.data['task']
        
        # Perform preprocessing
        self.preprocess_data()
        
    def preprocess_data(self):
        """
        Preprocess the data for deep learning:
        - Encode labels
        - Scale features
        - Split into train and test sets
        """
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, 
            self.y_encoded, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.y_encoded
        )
        
    def build_model(self):
        """
        Build a deep neural network for motor imagery classification
        
        Returns:
        --------
        tf.keras.Model
            Compiled deep learning model
        """
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=(self.X_train.shape[1],)),
            
            # Hidden layers with dropout for regularization
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(16, activation='relu'),
            
            # Output layer with softmax for multi-class classification
            keras.layers.Dense(len(np.unique(self.y_encoded)), activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs=100, batch_size=32):
        """
        Train the deep learning model
        
        Parameters:
        -----------
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        
        Returns:
        --------
        History object from model training
        """
        # Build the model
        self.model = self.build_model()
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self):
        """
        Evaluate the trained model and generate comprehensive metrics
        """
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(
            self.y_test, 
            y_pred_classes, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        # plt.show()
        
    def plot_training_history(self, history):
        """
        Plot training and validation metrics
        
        Parameters:
        -----------
        history : History object from model training
        """
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        # plt.show()

def main():
    # Path to the preprocessed CSV file
    data_path = '/home/maverick/Documents/PROJECTS/Final_year_2/comprehensive_motor_imagery_features_for_Code4.csv'
    
    # Initialize the classifier
    classifier = MotorImageryClassifier(data_path)
    
    # Train the model
    training_history = classifier.train_model(epochs=100)
    
    # Plot training history
    classifier.plot_training_history(training_history)
    
    # Evaluate the model
    classifier.evaluate_model()

if __name__ == "__main__":
    main()