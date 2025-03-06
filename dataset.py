import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_timestamp_sequence(num_samples, start_time=None, frequency_ms=100):
    """Generate a sequence of timestamps with a specified frequency in milliseconds."""
    if start_time is None:
        start_time = datetime.now()
    
    timestamps = [start_time + timedelta(milliseconds=i * frequency_ms) for i in range(num_samples)]
    return timestamps

def generate_walking_steadily_data(num_samples, noise_level=0.1):
    """Generate sensor data for walking steadily (label 0)."""
    # Base walking pattern
    # Time in seconds (converted to radians for sinusoidal patterns)
    time = np.linspace(0, num_samples/10, num_samples)
    
    # Pressure sensors - normal walking has balanced pressure distribution
    pressure_1 = 2.5 + 0.3 * np.sin(time * 2) + noise_level * np.random.randn(num_samples)
    pressure_2 = 2.3 + 0.3 * np.sin(time * 2 + np.pi/2) + noise_level * np.random.randn(num_samples)
    pressure_10 = 2.2 + 0.4 * np.sin(time * 2 + np.pi) + noise_level * np.random.randn(num_samples)
    pressure_12 = 2.4 + 0.4 * np.sin(time * 2 + 3*np.pi/2) + noise_level * np.random.randn(num_samples)
    
    # Accelerometer data - steady walking shows regular patterns
    accel_x = 0.1 * np.sin(time * 4) + noise_level * np.random.randn(num_samples)
    accel_y = 0.8 + 0.3 * np.sin(time * 4) + noise_level * np.random.randn(num_samples)
    accel_z = 9.8 + 0.5 * np.sin(time * 8) + noise_level * np.random.randn(num_samples)  # Gravity + vertical motion
    
    # Gyroscope data - regular rotational patterns during walking
    gyro_x = 5 * np.sin(time * 3) + noise_level * 2 * np.random.randn(num_samples)
    gyro_y = 8 * np.sin(time * 3 + np.pi/4) + noise_level * 2 * np.random.randn(num_samples)
    gyro_z = 3 * np.sin(time * 3 + np.pi/2) + noise_level * 2 * np.random.randn(num_samples)
    
    # Vital signs - normal walking has stable heart rate and oxygen levels
    heart_rate = 75 + 5 * np.sin(time * 0.5) + noise_level * 3 * np.random.randn(num_samples)
    spo2 = 97 + noise_level * 1.5 * np.random.randn(num_samples)
    
    # Clip SpO2 to realistic values
    spo2 = np.clip(spo2, 94, 100)
    
    # Create labels (all 0 for steady walking)
    labels = np.zeros(num_samples)
    
    return {
        'Pressure_1': pressure_1,
        'Pressure_2': pressure_2,
        'Pressure_10': pressure_10,
        'Pressure_12': pressure_12,
        'Accel_X': accel_x,
        'Accel_Y': accel_y,
        'Accel_Z': accel_z,
        'Gyro_X': gyro_x,
        'Gyro_Y': gyro_y,
        'Gyro_Z': gyro_z,
        'Heart_Rate': heart_rate,
        'SpO2': spo2,
        'Label': labels
    }

def generate_about_to_fall_data(num_samples, noise_level=0.2):
    """Generate sensor data for about to fall (label 1)."""
    # Time in seconds (converted to radians for sinusoidal patterns)
    time = np.linspace(0, num_samples/10, num_samples)
    
    # Pressure sensors - uneven pressure distribution indicating instability
    pressure_1 = 3.0 + 0.8 * np.sin(time * 3) + noise_level * np.random.randn(num_samples)
    pressure_2 = 1.5 + 1.0 * np.sin(time * 3 + np.pi/4) + noise_level * np.random.randn(num_samples)
    pressure_10 = 1.0 + 1.2 * np.sin(time * 3 + np.pi/2) + noise_level * np.random.randn(num_samples)
    pressure_12 = 3.5 + 0.7 * np.sin(time * 3 + 3*np.pi/4) + noise_level * np.random.randn(num_samples)
    
    # Accelerometer data - irregular patterns indicating loss of balance
    accel_x = 1.0 * np.sin(time * 5) + 0.5 * np.cos(time * 7) + noise_level * 1.5 * np.random.randn(num_samples)
    accel_y = 2.0 + 1.0 * np.sin(time * 6) - 0.4 * np.cos(time * 4) + noise_level * 1.5 * np.random.randn(num_samples)
    accel_z = 9.8 + 1.5 * np.sin(time * 9) + noise_level * 1.5 * np.random.randn(num_samples)
    
    # Gyroscope data - faster, more variable rotational movements
    gyro_x = 15 * np.sin(time * 4) + 10 * np.sin(time * 6) + noise_level * 5 * np.random.randn(num_samples)
    gyro_y = 20 * np.sin(time * 4 + np.pi/3) + noise_level * 5 * np.random.randn(num_samples)
    gyro_z = 12 * np.sin(time * 5 + np.pi/2) + 8 * np.cos(time * 3) + noise_level * 5 * np.random.randn(num_samples)
    
    # Vital signs - slightly elevated due to instability/stress
    heart_rate = 90 + 10 * np.sin(time * 0.5) + noise_level * 5 * np.random.randn(num_samples)
    spo2 = 96 + noise_level * 2 * np.random.randn(num_samples)
    
    # Clip SpO2 to realistic values
    spo2 = np.clip(spo2, 93, 100)
    
    # Create labels (all 1 for about to fall)
    labels = np.ones(num_samples)
    
    return {
        'Pressure_1': pressure_1,
        'Pressure_2': pressure_2,
        'Pressure_10': pressure_10,
        'Pressure_12': pressure_12,
        'Accel_X': accel_x,
        'Accel_Y': accel_y,
        'Accel_Z': accel_z,
        'Gyro_X': gyro_x,
        'Gyro_Y': gyro_y,
        'Gyro_Z': gyro_z,
        'Heart_Rate': heart_rate,
        'SpO2': spo2,
        'Label': labels
    }

def generate_user_fell_data(num_samples, noise_level=0.3):
    """Generate sensor data for user fell (label 2)."""
    # Time in seconds (converted to radians for sinusoidal patterns)
    time = np.linspace(0, num_samples/10, num_samples)
    
    # Simulate the fall happening in the first half of the sequence
    fall_point = num_samples // 3
    
    # Initialize data arrays
    pressure_1 = np.zeros(num_samples)
    pressure_2 = np.zeros(num_samples)
    pressure_10 = np.zeros(num_samples)
    pressure_12 = np.zeros(num_samples)
    accel_x = np.zeros(num_samples)
    accel_y = np.zeros(num_samples)
    accel_z = np.zeros(num_samples)
    gyro_x = np.zeros(num_samples)
    gyro_y = np.zeros(num_samples)
    gyro_z = np.zeros(num_samples)
    heart_rate = np.zeros(num_samples)
    spo2 = np.zeros(num_samples)
    
    # Before fall - similar to "about to fall" but more extreme
    for i in range(fall_point):
        t = time[i]
        
        # Pressure becomes very uneven just before fall
        pressure_1[i] = 4.0 + 1.5 * np.sin(t * 5) + noise_level * np.random.randn()
        pressure_2[i] = 0.8 + 0.5 * np.sin(t * 5 + np.pi/4) + noise_level * np.random.randn()
        pressure_10[i] = 0.5 + 0.3 * np.sin(t * 5 + np.pi/2) + noise_level * np.random.randn()
        pressure_12[i] = 3.5 + 1.0 * np.sin(t * 5 + 3*np.pi/4) + noise_level * np.random.randn()
        
        # Acceleration shows increasing instability
        accel_x[i] = 2.0 * np.sin(t * 6) + 1.0 * np.cos(t * 8) + noise_level * 2 * np.random.randn()
        accel_y[i] = 3.0 + 2.0 * np.sin(t * 7) - 0.8 * np.cos(t * 5) + noise_level * 2 * np.random.randn()
        accel_z[i] = 9.8 + 2.5 * np.sin(t * 10) + noise_level * 2 * np.random.randn()
        
        # Gyroscope shows rapid rotation
        gyro_x[i] = 25 * np.sin(t * 5) + 15 * np.sin(t * 7) + noise_level * 8 * np.random.randn()
        gyro_y[i] = 35 * np.sin(t * 5 + np.pi/3) + noise_level * 8 * np.random.randn()
        gyro_z[i] = 20 * np.sin(t * 6 + np.pi/2) + 15 * np.cos(t * 4) + noise_level * 8 * np.random.randn()
        
        # Vital signs start to elevate
        heart_rate[i] = 95 + 15 * np.sin(t * 0.6) + noise_level * 5 * np.random.randn()
        spo2[i] = 95 + noise_level * 2 * np.random.randn()
    
    # The actual fall - sudden dramatic changes in accelerometer and gyroscope
    fall_duration = min(20, num_samples - fall_point)
    for i in range(fall_point, fall_point + fall_duration):
        progress = (i - fall_point) / fall_duration
        
        # Pressure sensors show rapid changes during fall
        pressure_1[i] = 4.0 * (1 - progress) + noise_level * 2 * np.random.randn()
        pressure_2[i] = 0.8 * (1 - progress) + noise_level * 2 * np.random.randn()
        pressure_10[i] = 0.5 * (1 - progress) + noise_level * 2 * np.random.randn()
        pressure_12[i] = 3.5 * (1 - progress) + noise_level * 2 * np.random.randn()
        
        # Acceleration shows the fall - large spike followed by impact
        if progress < 0.5:  # During fall
            accel_x[i] = 5.0 + progress * 10 + noise_level * 3 * np.random.randn()
            accel_y[i] = 3.0 - progress * 8 + noise_level * 3 * np.random.randn()
            accel_z[i] = 9.8 - progress * 15 + noise_level * 3 * np.random.randn()  # Gravity decreases during free fall
        else:  # Impact
            accel_x[i] = -8.0 + noise_level * 4 * np.random.randn()
            accel_y[i] = -5.0 + noise_level * 4 * np.random.randn()
            accel_z[i] = -12.0 + noise_level * 4 * np.random.randn()  # Impact force
        
        # Gyroscope shows rapid rotation then sudden stop
        if progress < 0.7:
            gyro_x[i] = 80 * np.sin(progress * 10) + noise_level * 10 * np.random.randn()
            gyro_y[i] = 100 * np.cos(progress * 10) + noise_level * 10 * np.random.randn()
            gyro_z[i] = 60 * np.sin(progress * 15) + noise_level * 10 * np.random.randn()
        else:
            gyro_x[i] = 20 * np.exp(-5 * (progress - 0.7)) + noise_level * 5 * np.random.randn()
            gyro_y[i] = 25 * np.exp(-5 * (progress - 0.7)) + noise_level * 5 * np.random.randn()
            gyro_z[i] = 15 * np.exp(-5 * (progress - 0.7)) + noise_level * 5 * np.random.randn()
        
        # Vital signs spike during fall
        heart_rate[i] = 110 + 20 * progress + noise_level * 8 * np.random.randn()
        spo2[i] = 94 - progress * 2 + noise_level * 3 * np.random.randn()
    
    # After fall - minimal foot pressure, low movement, elevated HR
    for i in range(fall_point + fall_duration, num_samples):
        # Almost no pressure on sensors (person is on ground)
        pressure_1[i] = 0.2 + noise_level * 0.2 * np.random.randn()
        pressure_2[i] = 0.1 + noise_level * 0.2 * np.random.randn()
        pressure_10[i] = 0.1 + noise_level * 0.2 * np.random.randn()
        pressure_12[i] = 0.2 + noise_level * 0.2 * np.random.randn()
        
        # Minimal acceleration (mostly just noise)
        accel_x[i] = 0.1 + noise_level * 0.5 * np.random.randn()
        accel_y[i] = 0.1 + noise_level * 0.5 * np.random.randn()
        accel_z[i] = 9.8 + noise_level * 0.5 * np.random.randn()  # Gravity is still present
        
        # Minimal rotation
        gyro_x[i] = noise_level * 2 * np.random.randn()
        gyro_y[i] = noise_level * 2 * np.random.randn()
        gyro_z[i] = noise_level * 2 * np.random.randn()
        
        # Heart rate elevated and then slowly decreasing, oxygen may be slightly lowered
        time_since_fall = (i - (fall_point + fall_duration)) / (num_samples - (fall_point + fall_duration))
        heart_rate[i] = 130 - time_since_fall * 20 + noise_level * 5 * np.random.randn()
        spo2[i] = 92 + time_since_fall * 3 + noise_level * 2 * np.random.randn()
    
    # Clip SpO2 to realistic values
    spo2 = np.clip(spo2, 90, 100)
    
    # Create labels (all 2 for user fell)
    labels = np.ones(num_samples) * 2
    
    return {
        'Pressure_1': pressure_1,
        'Pressure_2': pressure_2,
        'Pressure_10': pressure_10,
        'Pressure_12': pressure_12,
        'Accel_X': accel_x,
        'Accel_Y': accel_y,
        'Accel_Z': accel_z,
        'Gyro_X': gyro_x,
        'Gyro_Y': gyro_y,
        'Gyro_Z': gyro_z,
        'Heart_Rate': heart_rate,
        'SpO2': spo2,
        'Label': labels
    }

def generate_mixed_sequence(total_samples, sequence_length=100):
    """Generate a mixed sequence with transitions between different states."""
    # Create a list to store individual sequences
    all_sequences = []
    
    # Track the number of samples generated
    samples_generated = 0
    
    while samples_generated < total_samples:
        # Randomly choose a state (with more weight to steady walking)
        state = np.random.choice([0, 1, 2], p=[0.6, 0.25, 0.15])
        
        # Determine sequence length (vary it slightly for realism)
        curr_seq_length = min(sequence_length + np.random.randint(-20, 20), total_samples - samples_generated)
        
        # Generate data for the chosen state
        if state == 0:
            data = generate_walking_steadily_data(curr_seq_length)
        elif state == 1:
            data = generate_about_to_fall_data(curr_seq_length)
        else:
            data = generate_user_fell_data(curr_seq_length)
        
        # Add to our sequences list
        all_sequences.append(data)
        
        # Update the counter
        samples_generated += curr_seq_length
    
    # Combine all the sequences
    combined_data = {}
    for key in all_sequences[0].keys():
        combined_data[key] = np.concatenate([seq[key] for seq in all_sequences])
    
    # Trim to exact required length
    for key in combined_data:
        combined_data[key] = combined_data[key][:total_samples]
    
    return combined_data

def create_dataset(total_samples=10000, output_file='fall_risk_data.csv'):
    """Create the main dataset with mixed sequences."""
    # Generate mixed data
    data_dict = generate_mixed_sequence(total_samples)
    
    # Generate timestamps
    timestamps = generate_timestamp_sequence(total_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Pressure_1': data_dict['Pressure_1'],
        'Pressure_2': data_dict['Pressure_2'],
        'Pressure_10': data_dict['Pressure_10'],
        'Pressure_12': data_dict['Pressure_12'],
        'Accel_X': data_dict['Accel_X'],
        'Accel_Y': data_dict['Accel_Y'],
        'Accel_Z': data_dict['Accel_Z'],
        'Gyro_X': data_dict['Gyro_X'],
        'Gyro_Y': data_dict['Gyro_Y'],
        'Gyro_Z': data_dict['Gyro_Z'],
        'Heart_Rate': data_dict['Heart_Rate'],
        'SpO2': data_dict['SpO2'],
        'Label': data_dict['Label'].astype(int)
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Main dataset created successfully with {total_samples} samples.")
    print(f"Class distribution:")
    print(df['Label'].value_counts())
    
    return df

def create_test_files(num_files=20, samples_per_file=200, output_dir='test_data'):
    """Create test files with various walking patterns."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_files = []
    
    # Create test files
    for i in range(num_files):
        # Determine the primary class for this file (with some randomness)
        # More files with normal walking, but ensure we have some of each class
        if i < 10:
            primary_class = 0  # Walking steadily
        elif i < 15:
            primary_class = 1  # About to fall
        else:
            primary_class = 2  # User fell
        
        # Generate the data
        if primary_class == 0:
            data = generate_walking_steadily_data(samples_per_file, noise_level=0.15)
        elif primary_class == 1:
            data = generate_about_to_fall_data(samples_per_file, noise_level=0.25)
        else:
            data = generate_user_fell_data(samples_per_file, noise_level=0.35)
        
        # Generate timestamps
        timestamps = generate_timestamp_sequence(samples_per_file)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'Pressure_1': data['Pressure_1'],
            'Pressure_2': data['Pressure_2'],
            'Pressure_10': data['Pressure_10'],
            'Pressure_12': data['Pressure_12'],
            'Accel_X': data['Accel_X'],
            'Accel_Y': data['Accel_Y'],
            'Accel_Z': data['Accel_Z'],
            'Gyro_X': data['Gyro_X'],
            'Gyro_Y': data['Gyro_Y'],
            'Gyro_Z': data['Gyro_Z'],
            'Heart_Rate': data['Heart_Rate'],
            'SpO2': data['SpO2'],
            'Label': data['Label'].astype(int)
        })
        
        # Create filename based on primary class
        if primary_class == 0:
            filename = f"{output_dir}/test_walking_steadily_{i+1}.csv"
        elif primary_class == 1:
            filename = f"{output_dir}/test_about_to_fall_{i+1}.csv"
        else:
            filename = f"{output_dir}/test_user_fell_{i+1}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        test_files.append(filename)
        
        print(f"Created test file {i+1}/{num_files}: {filename}")
    
    return test_files

def plot_sample_data(df, output_file='data_visualization.png'):
    """Plot sample data for visualization."""
    # Create a figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    
    # Get a subset of the data for plotting
    sample_data = df.iloc[:500].copy()
    
    # Plot pressure sensors
    axs[0].plot(sample_data['Pressure_1'], label='Pressure 1')
    axs[0].plot(sample_data['Pressure_2'], label='Pressure 2')
    axs[0].plot(sample_data['Pressure_10'], label='Pressure 10')
    axs[0].plot(sample_data['Pressure_12'], label='Pressure 12')
    axs[0].set_title('Pressure Sensors')
    axs[0].set_ylabel('Pressure (N/cm²)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot accelerometer data
    axs[1].plot(sample_data['Accel_X'], label='X')
    axs[1].plot(sample_data['Accel_Y'], label='Y')
    axs[1].plot(sample_data['Accel_Z'], label='Z')
    axs[1].set_title('Accelerometer Data')
    axs[1].set_ylabel('Acceleration (g)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot gyroscope data
    axs[2].plot(sample_data['Gyro_X'], label='X')
    axs[2].plot(sample_data['Gyro_Y'], label='Y')
    axs[2].plot(sample_data['Gyro_Z'], label='Z')
    axs[2].set_title('Gyroscope Data')
    axs[2].set_ylabel('Angular Velocity (dps)')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot vital signs and labels
    ax3_1 = axs[3]
    ax3_2 = ax3_1.twinx()
    
    ax3_1.plot(sample_data['Heart_Rate'], 'r-', label='Heart Rate')
    ax3_1.set_ylabel('Heart Rate (BPM)', color='r')
    ax3_1.tick_params(axis='y', labelcolor='r')
    
    ax3_2.plot(sample_data['SpO2'], 'b-', label='SpO2')
    ax3_2.set_ylabel('SpO2 (%)', color='b')
    ax3_2.tick_params(axis='y', labelcolor='b')
    
    # Add class labels as background color
    for i in range(len(sample_data) - 1):
        if sample_data['Label'].iloc[i] == 0:
            color = 'green'
            alpha = 0.1
        elif sample_data['Label'].iloc[i] == 1:
            color = 'yellow'
            alpha = 0.2
        else:
            color = 'red'
            alpha = 0.2
        
        axs[3].axvspan(i, i+1, color=color, alpha=alpha)
    
    axs[3].set_title('Vital Signs and Class Labels')
    axs[3].set_xlabel('Sample Index')
    
    # Add a legend for labels
    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color='green', alpha=0.3, label='Walking Steadily (0)')
    yellow_patch = mpatches.Patch(color='yellow', alpha=0.3, label='About to Fall (1)')
    red_patch = mpatches.Patch(color='red', alpha=0.3, label='User Fell (2)')
    axs[3].legend(handles=[green_patch, yellow_patch, red_patch], loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Sample data visualization saved to {output_file}")

if __name__ == "__main__":
    # Create main dataset
    df = create_dataset(total_samples=10000, output_file='fall_risk_data.csv')
    
    # Create test files
    test_files = create_test_files(num_files=20, samples_per_file=200, output_dir='test_data')
    
    # Visualize some of the data
    plot_sample_data(df, output_file='data_visualization.png')
    
    print("Dataset generation complete!")