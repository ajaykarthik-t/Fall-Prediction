import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_prepare_data(file_path, test_size=0.2):
    """Load the data and split into train and test sets"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Remove timestamp for model training
    X = df.drop(['Timestamp', 'Label'], axis=1)
    y = df['Label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=42, stratify=y)
    
    print(f"Data loaded from {file_path}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, X.columns

def train_random_forest(X_train, X_test, y_train, y_test, output_model_path):
    """Train a Random Forest model and evaluate performance"""
    print("\n==== Training Random Forest Model ====")
    
    # Create pipeline with scaling and model
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train model
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Walking Steadily', 'About to Fall', 'User Fell']))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Walking Steadily', 'About to Fall', 'User Fell'],
                yticklabels=['Walking Steadily', 'About to Fall', 'User Fell'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest')
    
    # Create directory if it doesn't exist
    if not os.path.exists('model_evaluation'):
        os.makedirs('model_evaluation')
    
    plt.savefig('model_evaluation/rf_confusion_matrix.png')
    plt.close()
    
    # Feature importance
    if hasattr(rf_pipeline['rf'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': rf_pipeline['rf'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance - Random Forest')
        plt.tight_layout()
        plt.savefig('model_evaluation/rf_feature_importance.png')
        plt.close()
        
        print("\nTop 5 most important features:")
        print(feature_importance.head())
    
    # Save model
    joblib.dump(rf_pipeline, output_model_path)
    print(f"Random Forest model saved to {output_model_path}")
    
    return rf_pipeline, accuracy

def prepare_sequence_data(X, y, sequence_length=10):
    """Prepare sequential data for LSTM model"""
    X_seq = []
    y_seq = []
    
    # Convert to numpy arrays
    X_array = X.values
    y_array = y.values
    
    # Create sequences
    for i in range(len(X_array) - sequence_length):
        X_seq.append(X_array[i:i+sequence_length])
        # Use the label of the last element in the sequence
        y_seq.append(y_array[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

def train_lstm(X_train, X_test, y_train, y_test, feature_names, output_model_path, sequence_length=10):
    """Train an LSTM model for sequential prediction"""
    print("\n==== Training LSTM Model ====")
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to sequence data
    X_train_seq, y_train_seq = prepare_sequence_data(
        pd.DataFrame(X_train_scaled, columns=feature_names), 
        y_train, 
        sequence_length
    )
    
    X_test_seq, y_test_seq = prepare_sequence_data(
        pd.DataFrame(X_test_scaled, columns=feature_names), 
        y_test, 
        sequence_length
    )
    
    print(f"LSTM sequence data shape: {X_train_seq.shape}")
    
    # Convert labels to categorical
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train_seq, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_seq, num_classes=num_classes)
    
    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, X_train.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Summary
    model.summary()
    
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train_seq, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_seq, y_test_cat),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test_seq, y_test_cat)
    print(f"LSTM Test Accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_evaluation/lstm_training_history.png')
    plt.close()
    
    # Predictions
    y_pred_prob = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Classification report
    print("\nClassification Report (LSTM):")
    print(classification_report(y_test_seq, y_pred, 
                              target_names=['Walking Steadily', 'About to Fall', 'User Fell']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_seq, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Walking Steadily', 'About to Fall', 'User Fell'],
                yticklabels=['Walking Steadily', 'About to Fall', 'User Fell'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - LSTM')
    plt.savefig('model_evaluation/lstm_confusion_matrix.png')
    plt.close()
    
    # Save model components for later use
    # 1. Save the Keras model
    model.save(output_model_path)
    # 2. Save the scaler
    joblib.dump(scaler, 'lstm_scaler.pkl')
    # 3. Save sequence length as a simple text file
    with open('lstm_sequence_length.txt', 'w') as f:
        f.write(str(sequence_length))
    
    print(f"LSTM model saved to {output_model_path}")
    print(f"Scaler saved to lstm_scaler.pkl")
    print(f"Sequence length saved to lstm_sequence_length.txt")
    
    return model, accuracy, scaler, sequence_length

def compare_models(rf_accuracy, lstm_accuracy):
    """Compare the performance of both models"""
    models = ['Random Forest', 'LSTM']
    accuracies = [rf_accuracy, lstm_accuracy]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.savefig('model_evaluation/model_comparison.png')
    plt.close()
    
    print("\n==== Model Comparison ====")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"LSTM Accuracy: {lstm_accuracy:.4f}")
    
    if rf_accuracy > lstm_accuracy:
        print("Random Forest performed better on this dataset.")
        return 'Random Forest'
    else:
        print("LSTM performed better on this dataset.")
        return 'LSTM'

if __name__ == "__main__":
    # Make sure the output directories exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data('fall_risk_data.csv')
    
    # Train Random Forest model
    rf_model, rf_accuracy = train_random_forest(X_train, X_test, y_train, y_test, 'models/rf_fall_risk_model.pkl')
    
    # Train LSTM model
    lstm_model, lstm_accuracy, lstm_scaler, sequence_length = train_lstm(
        X_train, X_test, y_train, y_test, feature_names, 'models/lstm_fall_risk_model'
    )
    
    # Compare models
    better_model = compare_models(rf_accuracy, lstm_accuracy)
    
    # Save the best model as the default model
    if better_model == 'Random Forest':
        joblib.dump(rf_model, 'fall_risk_model.pkl')
        with open('model_type.txt', 'w') as f:
            f.write('rf')
        print("Random Forest model saved as the default model: fall_risk_model.pkl")
    else:
        # For LSTM, we need to create a helper text file to indicate that we're using LSTM
        # This will be needed by the Streamlit app to know which model to load
        with open('model_type.txt', 'w') as f:
            f.write('lstm')
        print("LSTM model saved as the default model (models/lstm_fall_risk_model)")
    
    print("\nTraining complete! The best model has been saved and is ready for use in the Streamlit app.")