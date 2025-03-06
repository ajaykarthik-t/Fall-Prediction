import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import time
import plotly.graph_objects as go
import plotly.express as px
from collections import deque

# Set page configuration
st.set_page_config(
    page_title="Fall Risk Prediction System",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .normal-walk {
        color: green;
        font-size: 24px;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid green;
    }
    .about-to-fall {
        color: orange;
        font-size: 24px;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 165, 0, 0.1);
        border: 1px solid orange;
    }
    .user-fell {
        color: red;
        font-size: 24px;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid red;
    }
    .main-header {
        color: #1E88E5;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #0277BD;
        font-size: 24px;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to check if model files exist
def check_model_files():
    # Check if the model type file exists
    if not os.path.exists('model_type.txt'):
        st.error("Model type file not found. Please run train.py first.")
        return False
    
    # Read model type
    with open('model_type.txt', 'r') as f:
        model_type = f.read().strip()
    
    if model_type == 'rf':
        # Check if Random Forest model exists
        if not os.path.exists('fall_risk_model.pkl'):
            st.error("Random Forest model file not found. Please run train.py first.")
            return False
    elif model_type == 'lstm':
        # Check if LSTM model and related files exist
        if not os.path.exists('models/lstm_fall_risk_model'):
            st.error("LSTM model directory not found. Please run train.py first.")
            return False
        if not os.path.exists('lstm_scaler.pkl'):
            st.error("LSTM scaler file not found. Please run train.py first.")
            return False
        if not os.path.exists('lstm_sequence_length.txt'):
            st.error("LSTM sequence length file not found. Please run train.py first.")
            return False
    else:
        st.error(f"Unknown model type: {model_type}")
        return False
    
    return True

# Function to load the model
def load_model():
    # Read model type
    with open('model_type.txt', 'r') as f:
        model_type = f.read().strip()
    
    if model_type == 'rf':
        # Load Random Forest model
        model = joblib.load('fall_risk_model.pkl')
        return model, model_type, None, None
    else:
        # Load LSTM model and related components
        model = tf.keras.models.load_model('models/lstm_fall_risk_model')
        scaler = joblib.load('lstm_scaler.pkl')
        with open('lstm_sequence_length.txt', 'r') as f:
            sequence_length = int(f.read().strip())
        return model, model_type, scaler, sequence_length

# Function to prepare data for prediction
def prepare_data_for_prediction(df, model_type, scaler=None, sequence_length=None):
    # Remove timestamp and label if they exist
    features = df.copy()
    if 'Timestamp' in features.columns:
        features = features.drop('Timestamp', axis=1)
    if 'Label' in features.columns:
        features = features.drop('Label', axis=1)
    
    if model_type == 'rf':
        # For Random Forest, just return the features
        return features
    else:
        # For LSTM, prepare sequences
        # Scale the data
        features_scaled = scaler.transform(features)
        
        # Create sequences
        X_seq = []
        for i in range(len(features_scaled) - sequence_length + 1):
            X_seq.append(features_scaled[i:i+sequence_length])
        
        return np.array(X_seq)

# Function to make predictions
def make_prediction(model, data, model_type):
    if model_type == 'rf':
        return model.predict(data)
    else:
        # For LSTM, predict on sequences
        y_pred_prob = model.predict(data)
        return np.argmax(y_pred_prob, axis=1)

# Function to create sensor data visualizations
def plot_sensor_data(df, predictions=None):
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot pressure sensor readings
        st.subheader("Pressure Sensor Readings")
        fig = px.line(df, x='Timestamp', y=['Pressure_1', 'Pressure_2', 'Pressure_10', 'Pressure_12'],
                     title="Pressure Sensor Readings Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot accelerometer readings
        st.subheader("Accelerometer Readings")
        fig = px.line(df, x='Timestamp', y=['Accel_X', 'Accel_Y', 'Accel_Z'],
                     title="Accelerometer Readings Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Plot gyroscope readings
        st.subheader("Gyroscope Readings")
        fig = px.line(df, x='Timestamp', y=['Gyro_X', 'Gyro_Y', 'Gyro_Z'],
                     title="Gyroscope Readings Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # If predictions are provided, plot them
        if predictions is not None:
            st.subheader("Fall Risk Predictions")
            
            # Create a DataFrame for predictions
            pred_df = pd.DataFrame({
                'Timestamp': df['Timestamp'].iloc[len(df) - len(predictions):].values,
                'Prediction': predictions
            })
            
            # Map numerical predictions to labels
            pred_df['PredictionLabel'] = pred_df['Prediction'].map({
                0: 'Walking Steadily',
                1: 'About to Fall',
                2: 'User Fell'
            })
            
            # Create a color map
            colors = {
                'Walking Steadily': 'green',
                'About to Fall': 'orange',
                'User Fell': 'red'
            }
            
            fig = px.line(pred_df, x='Timestamp', y='Prediction',
                         title="Fall Risk Predictions Over Time",
                         color='PredictionLabel', color_discrete_map=colors)
            
            st.plotly_chart(fig, use_container_width=True)

# Function to display statistics
def display_statistics(df, predictions):
    st.markdown('<div class="sub-header">Data Statistics</div>', unsafe_allow_html=True)
    
    # Create 3 columns
    col1, col2, col3 = st.columns(3)
    
    # Count occurrences of each prediction
    prediction_counts = pd.Series(predictions).value_counts().to_dict()
    
    with col1:
        st.metric(
            label="Total Data Points", 
            value=len(df)
        )
    
    with col2:
        walking_steadily = prediction_counts.get(0, 0)
        about_to_fall = prediction_counts.get(1, 0)
        user_fell = prediction_counts.get(2, 0)
        
        if len(predictions) > 0:
            walking_percent = (walking_steadily / len(predictions)) * 100
            st.metric(
                label="Walking Steadily", 
                value=f"{walking_steadily} ({walking_percent:.1f}%)"
            )
    
    with col3:
        if len(predictions) > 0:
            risk_percent = ((about_to_fall + user_fell) / len(predictions)) * 100
            st.metric(
                label="At Risk (About to Fall + Fell)", 
                value=f"{about_to_fall + user_fell} ({risk_percent:.1f}%)"
            )

# Function to display the overall risk status
def display_risk_status(predictions):
    if len(predictions) == 0:
        return
    
    # Get the most recent 10 predictions or all if less than 10
    recent_predictions = predictions[-min(10, len(predictions)):]
    
    # Count occurrences of each class
    walking_steadily = sum(1 for p in recent_predictions if p == 0)
    about_to_fall = sum(1 for p in recent_predictions if p == 1)
    user_fell = sum(1 for p in recent_predictions if p == 2)
    
    # Determine overall status
    if user_fell > 0:
        status_class = "user-fell"
        status_text = "⚠️ EMERGENCY: USER HAS FALLEN ⚠️"
        status_desc = "The system has detected a fall. Immediate assistance may be required."
    elif about_to_fall > len(recent_predictions) * 0.3:  # 30% threshold
        status_class = "about-to-fall"
        status_text = "⚠️ WARNING: USER AT RISK OF FALLING ⚠️"
        status_desc = "The system has detected unstable walking patterns that indicate a risk of falling."
    else:
        status_class = "normal-walk"
        status_text = "✅ USER IS WALKING NORMALLY"
        status_desc = "The system detects normal walking patterns with no immediate fall risk."
    
    st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
    st.markdown(f"<p>{status_desc}</p>", unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.markdown('<div class="main-header">🚶 Fall Risk Prediction System</div>', unsafe_allow_html=True)
    
    # st.markdown("""
    # # <div class="info-box">
    # # This application detects fall risk based on sensor data from pressure sensors, accelerometers, and gyroscopes.
    # # Upload a CSV file with sensor data to predict if a person is:
    # # <ul>
    # #     <li>✅ Walking Normally</li>
    # #     <li>⚠️ About to Fall (Warning)</li>
    # #     <li>❌ Has Fallen (Emergency)</li>
    # # </ul>
    # # </div>
    # """, unsafe_allow_html=True)
    
    # Check if the model files exist
    if not check_model_files():
        st.stop()
    
    # Load the model
    model, model_type, scaler, sequence_length = load_model()
    
    st.sidebar.header("Upload Data")
    st.sidebar.markdown("Upload a CSV file with sensor data to analyze fall risk.")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    # If example data option is chosen
    use_example = st.sidebar.checkbox("Use example data")
    
    # Add some sample test files in a dropdown
    example_files = []
    if os.path.exists('test_data'):
        example_files = [f for f in os.listdir('test_data') if f.endswith('.csv')]
    
    selected_example = None
    if use_example and example_files:
        selected_example = st.sidebar.selectbox("Select an example file", example_files)
    
    if uploaded_file is not None or (use_example and selected_example is not None):
        # Load the data
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        else:
            example_path = os.path.join('test_data', selected_example)
            df = pd.read_csv(example_path)
            st.sidebar.success(f"Loaded example file: {selected_example}")
        
        # Display the raw data
        with st.expander("Raw Data (First 10 rows)"):
            st.dataframe(df.head(10))
        
        # Check if the data has the required columns
        required_columns = [
            'Timestamp', 'Pressure_1', 'Pressure_2', 'Pressure_10', 'Pressure_12',
            'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
            st.stop()
        
        # Prepare data for prediction
        X = prepare_data_for_prediction(df, model_type, scaler, sequence_length)
        
        # Make predictions
        with st.spinner('Analyzing data and generating predictions...'):
            predictions = make_prediction(model, X, model_type)
            
            # For LSTM, the predictions will be shorter than the original data
            # because we need a sequence of observations to make a prediction
            if model_type == 'lstm':
                st.info(f"Note: The first {sequence_length-1} data points don't have predictions due to sequence requirements.")
        
        # Display the risk status
        display_risk_status(predictions)
        
        # Display statistics
        display_statistics(df, predictions)
        
        # Plot the data
        plot_sensor_data(df, predictions)
        
        # Show detailed analysis
        st.markdown('<div class="sub-header">Detailed Sensor Analysis</div>', unsafe_allow_html=True)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Motion Patterns", "Correlations", "Risk Assessment"])
        
        with tab1:
            # Motion pattern analysis
            st.subheader("3D Motion Visualization")
            
            # Create a 3D scatter plot of accelerometer data
            fig = px.scatter_3d(df, x='Accel_X', y='Accel_Y', z='Accel_Z', 
                               color='Timestamp', color_continuous_scale='viridis',
                               title="3D Accelerometer Motion Path")
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a heatmap of sensor activity over time
            st.subheader("Sensor Activity Heatmap")
            
            # Prepare data for heatmap
            sensors = ['Pressure_1', 'Pressure_2', 'Pressure_10', 'Pressure_12', 
                      'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
            
            # Sample some timestamps for the heatmap
            n_samples = min(50, len(df))
            sample_indices = np.linspace(0, len(df)-1, n_samples, dtype=int)
            heatmap_data = df.iloc[sample_indices][sensors]
            
            # Create the heatmap
            fig = px.imshow(heatmap_data.T, 
                           labels=dict(x="Time Progression", y="Sensor", color="Value"),
                           title="Sensor Activity Over Time",
                           color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            # Correlation analysis
            st.subheader("Sensor Correlation Matrix")
            
            # Calculate correlations
            corr_matrix = df[sensors].corr()
            
            # Plot correlation matrix
            fig = px.imshow(corr_matrix, 
                           text_auto=True,
                           labels=dict(x="Sensor", y="Sensor", color="Correlation"),
                           color_continuous_scale="RdBu_r",
                           title="Sensor Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strongest correlations
            st.subheader("Strongest Correlations")
            
            # Get upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find top 5 correlations
            strongest_corr = upper.stack().sort_values(ascending=False)[:5]
            
            for (sensor1, sensor2), corr_value in strongest_corr.items():
                st.write(f"**{sensor1}** and **{sensor2}**: {corr_value:.3f}")
                # Plot the relationship
                fig = px.scatter(df, x=sensor1, y=sensor2, 
                                title=f"Relationship between {sensor1} and {sensor2} (Correlation: {corr_value:.3f})",
                                trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
                
        with tab3:
            # Risk assessment tab
            st.subheader("Fall Risk Assessment")
            
            # Show prediction distribution
            prediction_counts = pd.Series(predictions).value_counts().sort_index()
            labels = {0: "Walking Steadily", 1: "About to Fall", 2: "User Fell"}
            colors = ["green", "orange", "red"]
            
            # Create a pie chart of prediction distribution
            fig = px.pie(
                values=prediction_counts.values,
                names=[labels.get(i, f"Class {i}") for i in prediction_counts.index],
                title="Distribution of Predictions",
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show risk over time
            st.subheader("Risk Level Over Time")
            
            # Calculate a rolling risk score (for visualization purposes)
            window_size = min(20, len(predictions))
            if len(predictions) > 0:
                # Create a time series of predictions
                pred_df = pd.DataFrame({
                    'Timestamp': df['Timestamp'].iloc[len(df) - len(predictions):].values,
                    'Prediction': predictions
                })
                
                # Calculate a risk score (0-100)
                # Walking normally = 0, About to fall = 50, User fell = 100
                pred_df['RiskScore'] = pred_df['Prediction'].map({0: 0, 1: 50, 2: 100})
                
                # Calculate rolling average risk score
                if len(pred_df) >= window_size:
                    pred_df['RollingRiskScore'] = pred_df['RiskScore'].rolling(window=window_size).mean()
                    
                    # Plot the rolling risk score
                    fig = px.line(pred_df.dropna(), x='Timestamp', y='RollingRiskScore',
                                 title=f"Rolling Average Risk Score (Window Size: {window_size})")
                    
                    # Add risk zones
                    fig.add_shape(type="rect", x0=pred_df['Timestamp'].min(), x1=pred_df['Timestamp'].max(), 
                                 y0=0, y1=25, line=dict(width=0), fillcolor="green", opacity=0.2)
                    fig.add_shape(type="rect", x0=pred_df['Timestamp'].min(), x1=pred_df['Timestamp'].max(), 
                                 y0=25, y1=75, line=dict(width=0), fillcolor="orange", opacity=0.2)
                    fig.add_shape(type="rect", x0=pred_df['Timestamp'].min(), x1=pred_df['Timestamp'].max(), 
                                 y0=75, y1=100, line=dict(width=0), fillcolor="red", opacity=0.2)
                    
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Identify high-risk time periods
                    st.subheader("High Risk Time Periods")
                    
                    # Find consecutive periods of high risk
                    high_risk_threshold = 50
                    pred_df['HighRisk'] = pred_df['RollingRiskScore'] > high_risk_threshold
                    
                    # Group consecutive high risk periods
                    pred_df['RiskGroup'] = (pred_df['HighRisk'] != pred_df['HighRisk'].shift()).cumsum()
                    high_risk_periods = pred_df[pred_df['HighRisk']].groupby('RiskGroup')
                    
                    if len(high_risk_periods) > 0:
                        # Display high risk periods
                        for i, (_, group) in enumerate(high_risk_periods):
                            if len(group) > 0:
                                start_time = group['Timestamp'].iloc[0]
                                end_time = group['Timestamp'].iloc[-1]
                                avg_risk = group['RollingRiskScore'].mean()
                                st.warning(f"High Risk Period {i+1}: From {start_time} to {end_time} (Average Risk: {avg_risk:.1f}%)")
                    else:
                        st.success("No high-risk periods detected in the data.")
            
            # Show recommendations based on risk assessment
            st.subheader("Recommendations")
            
            # Count risk levels in recent predictions
            recent_count = min(30, len(predictions))
            if recent_count > 0:
                recent_predictions = predictions[-recent_count:]
                walking_steadily = sum(1 for p in recent_predictions if p == 0)
                about_to_fall = sum(1 for p in recent_predictions if p == 1)
                user_fell = sum(1 for p in recent_predictions if p == 2)
                
                if user_fell > 0:
                    st.error("""
                    **Emergency Recommendations:**
                    1. Check on the user immediately
                    2. Assess for injuries
                    3. Provide assistance to help the user get up safely
                    4. Consider medical evaluation
                    5. Review environmental hazards that may have contributed to the fall
                    """)
                elif about_to_fall > recent_count * 0.2:  # More than 20% at risk
                    st.warning("""
                    **Preventive Recommendations:**
                    1. Encourage the user to slow down and be cautious
                    2. Ensure the walking path is clear of obstacles
                    3. Check footwear for proper support
                    4. Consider using a walking aid if available
                    5. Monitor for signs of fatigue or dizziness
                    """)
                else:
                    st.success("""
                    **Maintenance Recommendations:**
                    1. Continue monitoring
                    2. Ensure the user maintains good walking form
                    3. Regular exercise to improve balance and strength
                    4. Stay hydrated
                    5. Maintain good lighting in walking areas
                    """)

    # Add information about the system in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About the System")
    st.sidebar.info("""
    This Fall Risk Prediction System uses machine learning to analyze sensor data and predict fall risk in real-time.
    
    Features:
    - Real-time fall risk detection
    - Detailed sensor analysis
    - Risk visualization and assessment
    - Preventive recommendations
    
    For more information, contact support.
    """)
    
    # Add version information
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** 1.0.0")
    st.sidebar.markdown("**Last Updated:** March 2025")

# Run the application
if __name__ == "__main__":
    main()