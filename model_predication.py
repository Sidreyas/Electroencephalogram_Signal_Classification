import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

class MotorImageryPredictor:
    def __init__(self, model_dir='motor_imagery_model'):
        """
        Initialize predictor with saved model and preprocessing artifacts
        
        Parameters:
        -----------
        model_dir : str, optional
            Directory containing saved model and preprocessing files
        """
        # Load model
        model_path = os.path.join(model_dir, 'motor_imagery_model.h5')
        self.model = tf.keras.models.load_model(model_path)
        
        # Load label encoder
        label_encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
        self.label_encoder = joblib.load(label_encoder_path)
        
        # Load feature scaler
        scaler_path = os.path.join(model_dir, 'feature_scaler.joblib')
        self.scaler = joblib.load(scaler_path)
    
    def predict(self, features):
        """
        Predict motor imagery task from input features
        
        Parameters:
        -----------
        features : pd.DataFrame or np.ndarray
            Input features for prediction
        
        Returns:
        --------
        dict
            Prediction results with class and probabilities
        """
        # Ensure features are in numpy array format
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Make predictions
        probabilities = self.model.predict(scaled_features)
        
        # Get predicted classes
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        # Prepare results
        results = []
        for label, prob, raw_prob in zip(predicted_labels, predicted_classes, probabilities):
            results.append({
                'predicted_task': label,
                'predicted_class': int(prob),
                'probabilities': dict(zip(
                    self.label_encoder.classes_, 
                    raw_prob
                ))
            })
        
        return results if len(results) > 1 else results[0]

def main():
    # Example usage
    predictor = MotorImageryPredictor()
    
    # Load some example data (replace with your actual data)
    example_data_path = 'comprehensive_motor_imagery_features_for_Code4.csv'
    example_data = pd.read_csv(example_data_path)
    
    # Take a few rows for prediction (excluding known labels)
    sample_features = example_data.drop(['task', 'subject'], axis=1).iloc[28137:28147]
    
    # Make predictions
    predictions = predictor.predict(sample_features)
    
    # Print predictions
    print("Predictions:")
    for pred in predictions:
        print("\nPredicted Task:", pred['predicted_task'])
        print("Probabilities:")
        for task, prob in pred['probabilities'].items():
            print(f"{task}: {prob:.4f}")

if __name__ == "__main__":
    main()