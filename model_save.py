import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class MotorImageryModelExporter:
    def __init__(self, data_path, output_dir='motor_imagery_model'):
        """
        Initialize the model exporter with data preprocessing and model creation
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file with preprocessed features
        output_dir : str, optional
            Directory to save model and preprocessing artifacts
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Load preprocessed data
        self.data = pd.read_csv(data_path)
        
        # Prepare features and labels
        self.X = self.data.drop(['task', 'subject'], axis=1)
        self.y = self.data['task']
        
        # Perform preprocessing
        self.preprocess_data()
        
    def preprocess_data(self):
        """
        Preprocess the data for deep learning and save preprocessing artifacts
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
            keras.layers.Input(shape=(self.X_train.shape[1],)),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(16, activation='relu'),
            
            keras.layers.Dense(len(np.unique(self.y_encoded)), activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_export_model(self, epochs=100, batch_size=32):
        """
        Train the model and export all necessary artifacts
        
        Parameters:
        -----------
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        
        Returns:
        --------
        Trained model
        """
        # Build and train the model
        model = self.build_model()
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            self.X_train, 
            self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Export model
        model_path = os.path.join(self.output_dir, 'motor_imagery_model.h5')
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Export label encoder
        label_encoder_path = os.path.join(self.output_dir, 'label_encoder.joblib')
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"Label Encoder saved to {label_encoder_path}")
        
        # Export scaler
        scaler_path = os.path.join(self.output_dir, 'feature_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"Feature Scaler saved to {scaler_path}")
        
        # Evaluate and print model performance
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        return model

def main():
    # Path to the preprocessed CSV file
    data_path = '/home/maverick/Documents/PROJECTS/Final_year_2/comprehensive_motor_imagery_features_for_Code4.csv'
    
    # Initialize and train the model exporter
    model_exporter = MotorImageryModelExporter(data_path)
    model_exporter.train_and_export_model()

if __name__ == "__main__":
    main()