import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TransformerEEGClassifier:
    def __init__(self, data_path, output_dir='transformer_eeg_model'):
        """
        Initialize Transformer-based EEG Signal Classifier
        
        Parameters:
        -----------
        data_path : str
            Path to the preprocessed EEG feature CSV
        output_dir : str, optional
            Directory to save model artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        self.data = pd.read_csv(data_path)
        
        self.X = self.data.drop(['task', 'subject'], axis=1)
        self.y = self.data['task']
        
        self.preprocess_data()
    
    def preprocess_data(self):
        """
        Preprocess data for Transformer architecture
        """
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        self.X_transformed = self.X_scaled.reshape(
            self.X_scaled.shape[0], 1, self.X_scaled.shape[1]
        )
        
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_transformed, 
            self.y_encoded, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.y_encoded
        )
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        Transformer Encoder Block
        
        Parameters:
        -----------
        inputs : tensor
            Input tensor
        head_size : int
            Size of attention heads
        num_heads : int
            Number of attention heads
        ff_dim : int
            Dimension of feed-forward network
        dropout : float, optional
            Dropout rate
        
        Returns:
        --------
        tensor
            Transformed input
        """
        x = layers.MultiHeadAttention(
            key_dim=head_size, 
            num_heads=num_heads, 
            dropout=dropout
        )(inputs, inputs)
        
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
        
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    
    def build_transformer_model(self):
        """
        Build Transformer-based EEG Classification Model
        
        Returns:
        --------
        tf.keras.Model
            Compiled Transformer model
        """
        inputs = keras.Input(shape=(1, self.X_train.shape[2]))
        
        head_size = 256
        num_heads = 4
        ff_dim = 4
        num_transformer_blocks = 4
        mlp_units = [128, 64]
        mlp_dropout = 0.4
        dropout = 0.25
        
        x = inputs
        
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        
        outputs = layers.Dense(
            len(np.unique(self.y_encoded)), 
            activation="softmax"
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_export_model(self, epochs=200, batch_size=32):
        """
        Train Transformer model and export artifacts
        
        Parameters:
        -----------
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Training batch size
        
        Returns:
        --------
        Trained Transformer model
        """
        # Build model
        model = self.build_transformer_model()
        
        # Print model summary
        model.summary()
        
        # Early stopping and model checkpointing
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            restore_best_weights=True
        )
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(self.output_dir, 'best_transformer_model.h5'),
            save_best_only=True,
            monitor='val_accuracy'
        )
        
        history = model.fit(
            self.X_train, 
            self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        self._export_artifacts(model)
        
        return model
    
    def _export_artifacts(self, model):
        """
        Export model, label encoder, and scaler
        
        Parameters:
        -----------
        model : tf.keras.Model
            Trained Transformer model
        """
        model_path = os.path.join(self.output_dir, 'transformer_eeg_model.h5')
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        label_encoder_path = os.path.join(self.output_dir, 'label_encoder.joblib')
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"Label Encoder saved to {label_encoder_path}")
        
        scaler_path = os.path.join(self.output_dir, 'feature_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"Feature Scaler saved to {scaler_path}")

def main():
    data_path = '/home/maverick/Documents/PROJECTS/Final_year_2/comprehensive_motor_imagery_features_for_Code4.csv'
    
    transformer_classifier = TransformerEEGClassifier(data_path)
    transformer_classifier.train_and_export_model()

if __name__ == "__main__":
    main()