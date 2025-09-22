import h5py
import os
import numpy as np
import pandas as pd
from scipy.stats import sem 
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# Dataset file paths
test1_file_path = './Dataset/Test1.mat'
test2_file_path = './Dataset/Test2.mat'
test3_file_path = './Dataset/Test3.mat'

def inspect_mat_file(file_path: str):
    """
    Inspects a MATLAB v7.3 .mat file (HDF5 format) and prints its structure.

    Args:
        file_path (str): The path to the .mat file.
    """
    
    def print_hdf5_structure(group, indent=""):
        """A recursive function to print the structure."""
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                print(f"{indent}Group: {key}")
                print_hdf5_structure(item, indent + "  ")
            elif isinstance(item, h5py.Dataset):
                print(f"{indent}Dataset: '{key}'")
                print(f"{indent}  - Shape: {item.shape}")
                print(f"{indent}  - Dtype: {item.dtype}")
            else:
                print(f"{indent}Unknown item: {key}")

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"--- Structure of: {file_path} ---")
            print_hdf5_structure(f)
            print("---------------------------------")

    except FileNotFoundError:
        print(f"Error: The file was not found at the path: {file_path}")
    except OSError:
        print(f"Error: Could not open the file. It might not be a valid v7.3 .mat file (HDF5).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_eeg_data_interactively(mat_file_path: str) -> np.ndarray:
    """
    Inspects a .mat file, identifies potential EEG datasets, and prompts the
    user to select the correct one if ambiguous. Also handles transposition.

    Args:
        mat_file_path (str): The path to the .mat file.

    Returns:
        np.ndarray: A NumPy array with the EEG data in (channels, samples) format, or None.
    """
    file_name = os.path.basename(mat_file_path)
    print(f"\n--- Inspecting file: {file_name} ---")
    
    try:
        with h5py.File(mat_file_path, 'r') as f:
            candidate_keys = []
            # Find all 2D numerical datasets that could be EEG data
            for key, item in f.items():
                if isinstance(item, h5py.Dataset) and item.ndim == 2 and np.issubdtype(item.dtype, np.number):
                    candidate_keys.append(key)

            if not candidate_keys:
                print("Error: No suitable 2D numerical datasets found in the file.")
                return None

            eeg_key = None
            if len(candidate_keys) == 1:
                eeg_key = candidate_keys[0]
                print(f"Automatically selected the only available dataset: '{eeg_key}'")
            else:
                print("Multiple potential datasets found. Please choose one:")
                for i, key in enumerate(candidate_keys):
                    print(f"  {i+1}: '{key}' (Shape: {f[key].shape})")
                
                while eeg_key is None:
                    try:
                        choice = int(input("Enter the number of the correct dataset: ")) - 1
                        if 0 <= choice < len(candidate_keys):
                            eeg_key = candidate_keys[choice]
                        else:
                            print("Invalid number. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")

            # --- Load and process the selected dataset ---
            print(f"Loading data from dataset '{eeg_key}'...")
            eeg_data = np.array(f[eeg_key])
            
            # Intelligent transpose: assume channels is the smaller dimension
            if eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T
                print(f"Data has been transposed to shape: {eeg_data.shape}")
            else:
                print(f"Data shape is {eeg_data.shape}. No transpose needed.")

            return eeg_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_eeg_duration(file_path: str, fs: int):
    """
    Calculates and prints the total duration of an EEG signal from a .mat file.

    Args:
        file_path (str): The path to the .mat file.
        fs (int): The sampling frequency of the device.
    """
    file_name = os.path.basename(file_path)
    try:
        with h5py.File(file_path, 'r') as f:
            # Assumes the data is stored under the key 'y'
            if 'y' not in f:
                print(f"Error: Dataset 'y' not found in {file_name}")
                return

            # Shape is typically (samples, channels)
            num_samples = f['y'].shape[0]
            num_channels = f['y'].shape[1]

            # Calculate duration
            total_seconds = num_samples / fs
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60

            # Print the results for the current file
            print(f"--- Analysis for: {file_name} ---")
            print(f"  - Total Samples: {num_samples}")
            print(f"  - Total Duration: {minutes} minutes and {seconds:.2f} seconds")
            print("-" * (20 + len(file_name)))


    except FileNotFoundError:
        print(f"Error: The file was not found at the path: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_name}: {e}")


def plot_raw_eeg_channels(eeg_data: np.ndarray, fs: int, file_name: str):
    """
    Plots all channels of the raw EEG data in separate subplots.
    """
    if eeg_data is None:
        print("Skipping plot due to loading error.")
        return
        
    num_channels, num_samples = eeg_data.shape
    time_vector = np.arange(num_samples) / fs

    fig, axes = plt.subplots(num_channels-1, 1, figsize=(18, 12), sharex=True, squeeze=False)

    fig.suptitle(f"Raw EEG Signals for {file_name}", fontsize=16)

    for i in range(1, num_channels):

        ax = axes[i-1, 0]
        ax.plot(time_vector, eeg_data[i, :])
        ax.set_ylabel(f"Channel {i}\n(μV)")

        ax.set_xticks(np.arange(0, time_vector[-1] + 1, 60))

        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    fig.canvas.manager.set_window_title(f"Raw EEG Signals - {file_name}")

    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def apply_bandpass_filter(data: np.ndarray, fs: int, low_cut: float, high_cut: float, order: int = 4) -> np.ndarray:
    """
    Applies a band-pass filter to the EEG data.
    """
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data


def plot_all_channels_comparison(raw_data: np.ndarray, filtered_data: np.ndarray, fs: int, file_name: str, time_offset_sec: int = 0):
    """
    Plots a comparison of raw vs. filtered signals for all provided channels.
    """
    if raw_data is None or filtered_data is None:
        return
        
    num_channels, num_samples = raw_data.shape
    # Create a time vector that accounts for the initial cutoff
    time_vector = np.arange(num_samples) / fs + time_offset_sec

    fig, axes = plt.subplots(num_channels, 1, figsize=(18, num_channels * 2.5), sharex=True, squeeze=False)
    
    fig.suptitle(f"Raw vs. Filtered EEG Signals for {file_name}", fontsize=16)

    # Use the final specified plotting structure
    for i in range(num_channels):
        ax = axes[i, 0]

        # Plot raw signal with some transparency
        ax.plot(time_vector, raw_data[i, :], color='gray', alpha=0.7, label='Raw')
        # Plot filtered signal on top
        ax.plot(time_vector, filtered_data[i, :], color='red', alpha=0.9, label='Filtered')

        # The channel label is now simply 'i+1' (e.g., Channel 1, 2, ...)
        # We add 1 to the index because we already removed the original Channel 0
        ax.set_ylabel(f"Channel {i+1}\n(μV)")
        ax.legend(loc='upper right')

        # Set the major x-axis ticks to 60-second intervals
        # Adjust the start of the range to match the time offset
        tick_start = np.ceil(time_vector[0] / 60) * 60
        ax.set_xticks(np.arange(tick_start, time_vector[-1] + 1, 60))
        
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Set the window title
    fig.canvas.manager.set_window_title(f"Filter Comparison for {file_name}")

    # Set the x-label on the last subplot
    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_filtered_eeg(filtered_data: np.ndarray, fs: int, file_name: str, time_offset_sec: int = 0):
    """
    Plots only the filtered EEG signals for all provided channels.
    """
    if filtered_data is None:
        return
        
    num_channels, num_samples = filtered_data.shape
    # Create a time vector that accounts for the initial cutoff
    time_vector = np.arange(num_samples) / fs + time_offset_sec

    fig, axes = plt.subplots(num_channels, 1, figsize=(18, num_channels * 2.5), sharex=True, squeeze=False)
    
    fig.suptitle(f"Filtered EEG Signals for {file_name}", fontsize=16)

    # Use the final specified plotting structure
    for i in range(num_channels):
        ax = axes[i, 0]

        # Plot only the filtered signal
        ax.plot(time_vector, filtered_data[i, :], color='red')

        # The channel label starts from 1 for the plotted data
        # (e.g., index 0 of the data is the 1st channel plotted)
        ax.set_ylabel(f"Channel {i+1}\n(μV)")

        # Set the major x-axis ticks to 60-second intervals
        tick_start = np.ceil(time_vector[0] / 60) * 60
        ax.set_xticks(np.arange(tick_start, time_vector[-1] + 1, 60))
        
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Set the window title
    fig.canvas.manager.set_window_title(f"Filtered EEG for {file_name}")

    # Set the x-label on the last subplot
    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def interpolate_outliers_by_threshold(data: np.ndarray, threshold_uv: float) -> np.ndarray:
    """
    Finds outliers by an amplitude threshold and replaces them with linearly interpolated values.
    """
    print(f"Finding and interpolating outliers with a threshold of +/- {threshold_uv} uV...")
    cleaned_data = data.copy()
    num_channels = data.shape[0]

    outlier_indices = np.abs(cleaned_data) > threshold_uv
    num_outliers = np.sum(outlier_indices)
    
    cleaned_data[outlier_indices] = np.nan

    for i in range(num_channels):
        channel_series = pd.Series(cleaned_data[i, :])
        interpolated_channel = channel_series.interpolate(method='linear', limit_direction='both')
        cleaned_data[i, :] = interpolated_channel.to_numpy()

    print(f"Found and interpolated {num_outliers} outlier data points.")
    return cleaned_data


def plot_preprocessing_steps(raw_data, filtered_data, final_data, fs, channel_to_plot, time_offset_sec, threshold):
    """
    Plots the signal at three stages: Raw, Filtered, and Final (Filtered + Interpolated).
    Adds vertical gridlines every 60 seconds.
    """
    num_samples = raw_data.shape[1]
    time_vector = np.arange(num_samples) / fs + time_offset_sec
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle(f"EEG Preprocessing Steps (Channel {channel_to_plot+1})", fontsize=16)

    xticks = np.arange(60, time_vector[-1] + 1, 60)

    # 1. Raw Data
    axes[0].plot(time_vector, raw_data[channel_to_plot, :], label='Raw Signal')
    axes[0].set_title('Step 1: Raw Signal')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].set_xticks(xticks)
    axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[0].legend()

    # 2. After Band-pass Filtering
    axes[1].plot(time_vector, filtered_data[channel_to_plot, :], label='Filtered Signal', color='orange')
    axes[1].set_title(f'Step 2: After {LOW_CUT}-{HIGH_CUT} Hz Band-pass Filter')
    axes[1].set_ylabel('Amplitude (μV)')
    axes[1].set_xticks(xticks)
    axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[1].legend()

    # 3. After Outlier Interpolation
    axes[2].plot(time_vector, final_data[channel_to_plot, :], label='Final Cleaned Signal', color='red')
    axes[2].set_title(f'Step 3: After Outlier Interpolation (Threshold = {threshold} μV)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude (μV)')
    axes[2].set_xticks(xticks)
    axes[2].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[2].legend()

    fig.canvas.manager.set_window_title(f"Preprocessing Steps - Channel {channel_to_plot+1}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def epoch_data(eeg_data: np.ndarray, fs: int, conditions: dict, time_offset_sec: int) -> dict:
    """
    Segments the preprocessed data into epochs based on the conditions dictionary.
    """
    epoched_data = {}
    print("Epoching data into conditions...")
    for condition_name, (start_time, end_time) in conditions.items():
        start_sample_relative = int((start_time - time_offset_sec) * fs)
        end_sample_relative = int((end_time - time_offset_sec) * fs)
        
        start_sample_relative = max(0, start_sample_relative)
        end_sample_relative = min(eeg_data.shape[1], end_sample_relative)

        epoched_data[condition_name] = eeg_data[:, start_sample_relative:end_sample_relative]
        
    return epoched_data


def calculate_weighted_band_powers(eeg_segment: np.ndarray, fs: int, band_weights: dict, channel_map: dict) -> dict:
    """
    Calculates the weighted average power for all major frequency bands.
    """
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)}
    
    # Calculate PSD for all channels in the segment
    freqs, psd = welch(eeg_segment, fs=fs, nperseg=fs*2)
    
    # Calculate the average power within each band for each channel
    power_per_channel = {}
    for band_name, (low_freq, high_freq) in bands.items():
        freq_indices = np.where((freqs >= low_freq) & (freqs < high_freq))[0]
        # Shape: (num_channels,)
        power_per_channel[band_name] = np.mean(psd[:, freq_indices], axis=1)

    # Calculate the final weighted average power for each band
    weighted_powers = {}
    for band_name, weights in band_weights.items():
        numerator = 0
        denominator = 0
        for channel_name, weight in weights.items():
            channel_index = channel_map[channel_name]
            numerator += power_per_channel[band_name][channel_index] * weight
            denominator += weight
        
        if denominator > 0:
            weighted_powers[band_name] = numerator / denominator
        else:
            weighted_powers[band_name] = 0

    return weighted_powers


def plot_absolute_power_bar_chart(power_data: dict, file_name_info: str):
    """
    Creates a grouped bar chart of the mean absolute power for all conditions.
    """
    conditions = list(power_data.keys())
    band_names = list(power_data[conditions[0]].keys())
    
    x = np.arange(len(band_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e'] # Blue, Green, Orange

    for i, condition in enumerate(conditions):
        powers = list(power_data[condition].values())
        # The offset places the bars for each condition next to each other
        offset = width * (i - 1)
        rects = ax.bar(x + offset, powers, width, label=condition, color=colors[i])
        ax.bar_label(rects, padding=3, fmt='%.3f')

    ax.set_ylabel('Average Weighted Power (μV²/Hz)')
    ax.set_title(f' Band Power Comparison for {file_name_info}')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.legend()
    fig.tight_layout()
    fig.canvas.manager.set_window_title(f"Band Power Comparison - {file_name_info}")
    plt.show()


def segment_into_chunks(eeg_data: np.ndarray, fs: int, chunk_duration_sec: int = 10) -> list:
    """
    Segments the EEG data into consecutive, non-overlapping chunks of a fixed duration.

    Args:
        eeg_data (np.ndarray): The preprocessed EEG data in (channels, samples) format.
        fs (int): The sampling frequency.
        chunk_duration_sec (int): The duration of each chunk in seconds.

    Returns:
        list: A list of NumPy arrays, where each array is a single chunk of EEG data.
    """
    num_channels, num_samples = eeg_data.shape
    chunk_size_samples = int(chunk_duration_sec * fs)
    
    num_chunks = num_samples // chunk_size_samples
    
    print(f"Segmenting data into {num_chunks} chunks of {chunk_duration_sec} seconds each...")
    
    # Use a list comprehension for a concise way to create the chunks
    segmented_data = [eeg_data[:, i * chunk_size_samples : (i + 1) * chunk_size_samples] for i in range(num_chunks)]
    
    print("Segmentation complete.")
    return segmented_data


# --- Main execution block ---
if __name__ == "__main__":

    # test1_file_path, test2_file_path, test3_file_path
    dataset_paths = [test1_file_path, test2_file_path, test3_file_path]
    
    # --- Configuration ---
    FS = 256
    LOW_CUT = 1.0
    HIGH_CUT = 40.0
    START_TIME_SEC = 60
    END_TIME_SEC = 240
    OUTLIER_THRESHOLD = 50.0
    CHUNK_DURATION = 10 # Each new data point will be a 10-second chunk

    CHANNEL_MAP = {'T7': 0, 'T8': 1, 'F3': 2, 'F4': 3, 'Fz': 4, 'Cz': 5}
    
    BAND_WEIGHTS = {
        'Delta': {'F3': 3, 'F4': 3, 'Fz': 3, 'Cz': 3, 'T7': 2, 'T8': 2},
        'Theta': {'Fz': 3, 'Cz': 3, 'F3': 2, 'F4': 2, 'T7': 1, 'T8': 1},
        'Alpha': {'T7': 3, 'T8': 3, 'F3': 2, 'F4': 2, 'Fz': 1, 'Cz': 1},
        'Beta':  {'F3': 3, 'F4': 3, 'Fz': 3, 'Cz': 2, 'T7': 1, 'T8': 1},
        'Gamma': {'Fz': 3, 'Cz': 3, 'F3': 2, 'F4': 2, 'T7': 1, 'T8': 1}
    }
    
    # This dictionary will store all the final results
    all_chunk_results = {}

    # --- Loop through each file ---
    for file_path in dataset_paths:
        file_name = os.path.basename(file_path)
        full_raw_data = load_eeg_data_interactively(file_path)
        
        if full_raw_data is not None:
            # 1. Preprocessing Pipeline
            start_sample = int(START_TIME_SEC * FS); end_sample = int(END_TIME_SEC * FS)
            if end_sample > full_raw_data.shape[1]: end_sample = full_raw_data.shape[1]
            raw_data_segment = full_raw_data[1:, start_sample:end_sample]
            
            filtered_data = apply_bandpass_filter(raw_data_segment, FS, LOW_CUT, HIGH_CUT)
            preprocessed_data = interpolate_outliers_by_threshold(filtered_data, threshold_uv=OUTLIER_THRESHOLD)
            
            # 2. Segment the clean data into 10-second chunks
            data_chunks = segment_into_chunks(preprocessed_data, FS, chunk_duration_sec=CHUNK_DURATION)
            
            # This list will hold the power analysis for each chunk of the current file
            file_chunk_powers = []
            
            # 3. Loop through each chunk and calculate its band powers
            for i, chunk in enumerate(data_chunks):
                chunk_start_time = START_TIME_SEC + (i * CHUNK_DURATION)
                print(f"  Analyzing chunk {i+1}/{len(data_chunks)} (Time: {chunk_start_time}s - {chunk_start_time + CHUNK_DURATION}s)...")
                
                band_powers = calculate_weighted_band_powers(chunk, FS, BAND_WEIGHTS, CHANNEL_MAP)
                
                # Add time information to the results for context
                band_powers['start_time_s'] = chunk_start_time
                file_chunk_powers.append(band_powers)

            # Store the results for the current file
            all_chunk_results[file_name] = file_chunk_powers

    # --- 4. Print the final results in a structured way ---
    if all_chunk_results:
        print("\n\n=============================================")
        print("--- Final Chunk-Based Power Analysis Results ---")
        print("=============================================")
        for file_name, chunk_list in all_chunk_results.items():
            print(f"\n--- Results for {file_name} ---")
            for chunk_result in chunk_list:
                # Format the output for better readability
                time_info = f"Time: {chunk_result['start_time_s']}s"
                delta = f"Delta: {chunk_result['Delta']:.3f}"
                theta = f"Theta: {chunk_result['Theta']:.3f}"
                alpha = f"Alpha: {chunk_result['Alpha']:.3f}"
                beta = f"Beta: {chunk_result['Beta']:.3f}"
                gamma = f"Gamma: {chunk_result['Gamma']:.3f}"
                print(f"  - {time_info:<15} | {delta:<18} | {theta:<18} | {alpha:<18} | {beta:<17} | {gamma}")
        print("---------------------------------------------")

    # '     Full analysis pipeline for all dataset files  '
    # dataset_paths = [test1_file_path, test2_file_path, test3_file_path]
    
    # # --- Configuration ---
    # FS = 256
    # LOW_CUT = 1.0
    # HIGH_CUT = 40.0
    # START_TIME_SEC = 60
    # END_TIME_SEC = 240
    # OUTLIER_THRESHOLD = 50.0

    # EXPERIMENTAL_CONDITIONS = {
    #     'Calm Music': (60, 120),
    #     'Silence': (120, 180),
    #     'Energetic Music': (180, 240)
    # }
    
    # CHANNEL_MAP = {'T7': 0, 'T8': 1, 'F3': 2, 'F4': 3, 'Fz': 4, 'Cz': 5}
    
    # BAND_WEIGHTS = {
    #     'Delta': {'F3': 3, 'F4': 3, 'Fz': 3, 'Cz': 3, 'T7': 2, 'T8': 2},
    #     'Theta': {'Fz': 3, 'Cz': 3, 'F3': 2, 'F4': 2, 'T7': 1, 'T8': 1},
    #     'Alpha': {'T7': 3, 'T8': 3, 'F3': 2, 'F4': 2, 'Fz': 1, 'Cz': 1},
    #     'Beta':  {'F3': 3, 'F4': 3, 'Fz': 3, 'Cz': 2, 'T7': 1, 'T8': 1},
    #     'Gamma': {'Fz': 3, 'Cz': 3, 'F3': 2, 'F4': 2, 'T7': 1, 'T8': 1}
    # }

    # # --- Loop through each file, process it, and plot the results ---
    # for file_path in dataset_paths:
    #     file_name = os.path.basename(file_path)
    #     full_raw_data = load_eeg_data_interactively(file_path)
        
    #     if full_raw_data is not None:
    #         # 1. Preprocessing Pipeline
    #         start_sample = int(START_TIME_SEC * FS); end_sample = int(END_TIME_SEC * FS)
    #         if end_sample > full_raw_data.shape[1]: end_sample = full_raw_data.shape[1]
    #         raw_data_segment = full_raw_data[1:, start_sample:end_sample]
            
    #         filtered_data = apply_bandpass_filter(raw_data_segment, FS, LOW_CUT, HIGH_CUT)
    #         preprocessed_data = interpolate_outliers_by_threshold(filtered_data, threshold_uv=OUTLIER_THRESHOLD)
            
    #         # 2. Epoching
    #         epochs = epoch_data(preprocessed_data, FS, EXPERIMENTAL_CONDITIONS, time_offset_sec=START_TIME_SEC)
            
    #         # 3. Calculate weighted powers for each condition
    #         condition_powers = {name: calculate_weighted_band_powers(data_epoch, FS, BAND_WEIGHTS, CHANNEL_MAP) for name, data_epoch in epochs.items()}
            
    #         print(f"\n--- Calculated Powers for {file_name} ---")
    #         import json
    #         print(json.dumps(condition_powers, indent=2))
    #         print("-" * (35 + len(file_name)))
            
    #         # 4. Plot the comparison chart for the current subject
    #         plot_absolute_power_bar_chart(condition_powers, file_name)



    '     Averaging rule across three dataset files  '
    # dataset_paths = [test1_file_path, test2_file_path, test3_file_path]
    
    # # This dictionary will store the results for all subjects
    # all_subject_results = {}

    # # --- Step 1: Process each subject individually ---
    # for file_path in dataset_paths:
    #     file_name = os.path.basename(file_path)
    #     full_raw_data = load_eeg_data_interactively(file_path)
        
    #     if full_raw_data is not None:
    #         # Preprocessing pipeline
    #         start_sample = int(START_TIME_SEC * FS); end_sample = int(END_TIME_SEC * FS)
    #         if end_sample > full_raw_data.shape[1]: end_sample = full_raw_data.shape[1]
    #         raw_data_segment = full_raw_data[1:, start_sample:end_sample]
            
    #         filtered_data = apply_bandpass_filter(raw_data_segment, FS, LOW_CUT, HIGH_CUT)
    #         preprocessed_data = interpolate_outliers_by_threshold(filtered_data, threshold_uv=OUTLIER_THRESHOLD)
            
    #         # Calculate and store powers for the entire preprocessed segment
    #         print(f"Calculating overall band powers for {file_name}...")
    #         overall_powers = calculate_weighted_band_powers(preprocessed_data, FS, BAND_WEIGHTS, CHANNEL_MAP)
    #         all_subject_results[file_name] = overall_powers

    #         # Print the results for the current subject
    #         print(f"\n--- Calculated Overall Powers for {file_name} ---")
    #         for band, power_val in overall_powers.items():
    #             print(f"    - {band}: {power_val:.4f}")
    #         print("-" * (40 + len(file_name)))

    # # --- Step 2: Calculate and print the average across all subjects ---
    # if len(all_subject_results) > 0:
    #     # Structure to hold all power values for averaging
    #     aggregated_powers = {band: [] for band in BAND_WEIGHTS.keys()}

    #     # Collect data from each subject
    #     for subject_data in all_subject_results.values():
    #         for band, power_val in subject_data.items():
    #             aggregated_powers[band].append(power_val)
        
    #     # Calculate the mean
    #     mean_powers = {band: 0 for band in BAND_WEIGHTS.keys()}
    #     for band, values in aggregated_powers.items():
    #         mean_powers[band] = np.mean(values)

    #     # Print the final averaged results
    #     print("\n\n=============================================")
    #     print("--- Overall Average Powers (All Subjects) ---")
    #     print("=============================================")
    #     for band, power_val in mean_powers.items():
    #         print(f"    - {band}: {power_val:.4f}")
    #     print("---------------------------------------------")



    '     Epoching pipeline for all dataset files  '

    # dataset_paths = [
    #     test1_file_path,
    #     test2_file_path,
    #     test3_file_path
    # ]
    
    # # --- Configure parameters ---
    # FS = 256
    # LOW_CUT = 1.0
    # HIGH_CUT = 40.0
    # START_TIME_SEC = 60
    # END_TIME_SEC = 240
    # OUTLIER_THRESHOLD = 50.0

    # # Define the experimental conditions with their absolute start and end times in seconds
    # EXPERIMENTAL_CONDITIONS = {
    #     'Calm Music': (60, 120),
    #     'Silence': (120, 180),
    #     'Energetic Music': (180, 240)
    # }

    # # This dictionary will store the epoched data for all files
    # all_epoched_data = {}

    # for file_path in dataset_paths:
    #     full_raw_data = load_eeg_data_interactively(file_path)
        
    #     if full_raw_data is not None:
    #         # 1. Slice data: Remove channel 0 and select the overall time window
    #         start_sample = int(START_TIME_SEC * FS)
    #         end_sample = int(END_TIME_SEC * FS)
    #         if end_sample > full_raw_data.shape[1]: end_sample = full_raw_data.shape[1]
            
    #         raw_data_segment = full_raw_data[1:, start_sample:end_sample]
            
    #         # 2. Apply band-pass filter
    #         print(f"Applying a {LOW_CUT}-{HIGH_CUT} Hz band-pass filter...")
    #         filtered_data = apply_bandpass_filter(raw_data_segment, FS, LOW_CUT, HIGH_CUT)
            
    #         # 3. Interpolate outliers
    #         preprocessed_data = interpolate_outliers_by_threshold(filtered_data, threshold_uv=OUTLIER_THRESHOLD)
            
    #         # 4. Epoch the preprocessed data into conditions
    #         epochs = epoch_data(preprocessed_data, FS, EXPERIMENTAL_CONDITIONS, time_offset_sec=START_TIME_SEC)
            
    #         # Store the result and print a confirmation
    #         file_name = os.path.basename(file_path)
    #         all_epoched_data[file_name] = epochs
            
    #         print(f"\n--- Epoching complete for {file_name} ---")
    #         for condition, data in epochs.items():
    #             print(f"  - Condition '{condition}' has shape: {data.shape}")
    #         print("-" * (30 + len(file_name)))

    # # Now, 'all_epoched_data' contains the segmented data for all your files.
    # # For example, to access the 'Calm Music' epoch from the first test file:
    # # calm_music_test1 = all_epoched_data['Test1.mat']['Calm Music']
    # # print("\nExample access:")
    # # print(f"Shape of Calm Music data for Test1.mat: {calm_music_test1.shape}")


    '     Applies band-pass filtering and outlier interpolation, then plots all three steps for a single dataset file  '
    # # Assuming 'test1_file_path' is defined elsewhere in your script
    # FILE_PATH = test1_file_path
    # FS = 256  # Note: You had 260 in your snippet, I'm using 256 as before.
    # LOW_CUT = 1.0
    # HIGH_CUT = 40.0
    
    # # --- UPDATED: Define the time window for analysis ---
    # START_TIME_SEC = 60      # Start at minute 1
    # END_TIME_SEC = 240       # End at minute 4
    # OUTLIER_THRESHOLD = 50.0  # Using the last agreed-upon threshold
    
    # # 1. Load the raw data
    # full_raw_data = load_eeg_data_interactively(FILE_PATH)
    
    # if full_raw_data is not None:
    #     # 2. Exclude the first channel AND select the time segment
    #     start_sample = int(START_TIME_SEC * FS)
    #     end_sample = int(END_TIME_SEC * FS)

    #     # Ensure the end sample is not out of bounds
    #     if end_sample > full_raw_data.shape[1]:
    #         end_sample = full_raw_data.shape[1]

    #     raw_data_for_analysis = full_raw_data[1:, start_sample:end_sample]
    #     print(f"Selected segment from {START_TIME_SEC}s to {end_sample/FS:.2f}s.")
        
    #     # 3. Apply the band-pass filter FIRST
    #     filtered_data = apply_bandpass_filter(raw_data_for_analysis, FS, LOW_CUT, HIGH_CUT)
        
    #     # 4. Find and interpolate outliers on the filtered data LAST
    #     final_cleaned_data = interpolate_outliers_by_threshold(filtered_data, threshold_uv=OUTLIER_THRESHOLD)
        
    #     # 5. Plot the results to see the effect of each step
    #     channel_to_plot_in_analysis = 0 
    #     original_channel_number = 2

    #     plot_preprocessing_steps(
    #         raw_data_for_analysis,
    #         filtered_data,
    #         final_cleaned_data,
    #         fs=FS,
    #         channel_to_plot=channel_to_plot_in_analysis,
    #         time_offset_sec=START_TIME_SEC,
    #         threshold=OUTLIER_THRESHOLD
    #     )


    '     Applies band-pass filtering and plots only the filtered data for each dataset file, excluding first channel and first 60 seconds  '
    # dataset_paths = [
    #     test1_file_path,
    #     test2_file_path,
    #     test3_file_path
    # ]
    
    # # --- Configure parameters here ---
    # FS = 256
    # LOW_CUT = 1.0
    # HIGH_CUT = 40.0
    
    # # --- Define the time window for analysis ---
    # START_TIME_SEC = 60  # Start at minute 1
    # END_TIME_SEC = 240   # End at minute 4

    # for file_path in dataset_paths:
        
    #     # Load the full, original data
    #     full_raw_data = load_eeg_data_interactively(file_path)
        
    #     if full_raw_data is not None:
    #         # --- Exclude the first channel AND select the time segment ---
    #         start_sample = int(START_TIME_SEC * FS)
    #         end_sample = int(END_TIME_SEC * FS)
            
    #         # Ensure the end sample is not out of bounds
    #         if end_sample > full_raw_data.shape[1]:
    #             print(f"Warning: The specified end time is beyond the data length for {os.path.basename(file_path)}. Adjusting to max length.")
    #             end_sample = full_raw_data.shape[1]

    #         # Slice the data: remove channel 0 and select the time window
    #         raw_data_for_analysis = full_raw_data[1:, start_sample:end_sample]
    #         print(f"Selected segment from {START_TIME_SEC}s to {end_sample/FS:.2f}s.")
    #         print(f"Shape for analysis: {raw_data_for_analysis.shape}")
            
    #         # Apply the band-pass filter
    #         print(f"Applying a {LOW_CUT}-{HIGH_CUT} Hz band-pass filter...")
    #         filtered_data_for_analysis = apply_bandpass_filter(raw_data_for_analysis, FS, LOW_CUT, HIGH_CUT)
    #         print("Filtering complete.")

    #         file_name = os.path.basename(file_path)

    #         # Plot only the filtered data
    #         plot_filtered_eeg(
    #             filtered_data_for_analysis,
    #             FS,
    #             file_name,
    #             time_offset_sec=START_TIME_SEC
    #         )


    '     Applies band-pass filtering and plots only the filtered data for each dataset file, excluding first channel and first 60 seconds  '
    # dataset_paths = [
    #     test1_file_path,
    #     test2_file_path,
    #     test3_file_path
    # ]
    
    # # --- Configure other parameters here ---
    # FS = 256
    # LOW_CUT = 1.0
    # HIGH_CUT = 40.0
    # CUTOFF_SECONDS = 60

    # for file_path in dataset_paths:
        
    #     # Load the full, original data
    #     full_raw_data = load_eeg_data_interactively(file_path)
        
    #     if full_raw_data is not None:
    #         # --- Exclude the first channel (index 0) AND the first 60 seconds ---
    #         start_sample = int(CUTOFF_SECONDS * FS)
    #         raw_data_for_analysis = full_raw_data[1:, start_sample:]
    #         print(f"Removed first channel and first {CUTOFF_SECONDS} seconds.")
    #         print(f"Shape for analysis: {raw_data_for_analysis.shape}")
            
    #         # Apply the band-pass filter
    #         print(f"Applying a {LOW_CUT}-{HIGH_CUT} Hz band-pass filter...")
    #         filtered_data_for_analysis = apply_bandpass_filter(raw_data_for_analysis, FS, LOW_CUT, HIGH_CUT)
    #         print("Filtering complete.")

    #         file_name = os.path.basename(file_path)

    #         # Plot only the filtered data
    #         plot_filtered_eeg(
    #             filtered_data_for_analysis,
    #             FS,
    #             file_name,
    #             time_offset_sec=CUTOFF_SECONDS
    #         )


    '     Applies band-pass filtering and plots comparison for each dataset file, excluding first channel and first 60 seconds  '
    # dataset_paths = [
    #     test1_file_path,
    #     test2_file_path,
    #     test3_file_path
    # ]
    
    # # --- Configure other parameters here ---
    # FS = 256
    # LOW_CUT = 1.0
    # HIGH_CUT = 40.0
    # CUTOFF_SECONDS = 60 # UPDATED: Set cutoff to 60 seconds

    # for file_path in dataset_paths:
        
    #     # Load the full, original data
    #     full_raw_data = load_eeg_data_interactively(file_path)
        
    #     if full_raw_data is not None:
    #         # --- Exclude the first channel (index 0) AND the first 60 seconds ---
    #         start_sample = int(CUTOFF_SECONDS * FS)
    #         raw_data_for_analysis = full_raw_data[1:, start_sample:]
    #         print(f"Removed first channel and first {CUTOFF_SECONDS} seconds.")
    #         print(f"Shape for analysis: {raw_data_for_analysis.shape}")
            
    #         # Apply the band-pass filter
    #         print(f"Applying a {LOW_CUT}-{HIGH_CUT} Hz band-pass filter...")
    #         filtered_data_for_analysis = apply_bandpass_filter(raw_data_for_analysis, FS, LOW_CUT, HIGH_CUT)
    #         print("Filtering complete.")

    #         file_name = os.path.basename(file_path)

    #         # Plot the comparison for all remaining channels
    #         plot_all_channels_comparison(
    #             raw_data_for_analysis,
    #             filtered_data_for_analysis,
    #             FS,
    #             file_name,
    #             time_offset_sec=CUTOFF_SECONDS
    #         )


    '     Calculates and prints the duration of each dataset file  '
    # # Create a list of all paths to process
    # dataset_paths = [
    #     test1_file_path,
    #     test2_file_path,
    #     test3_file_path
    # ]
    
    # # --- Set your sampling frequency ---
    # FS = 256

    # # --- Loop through each file and get its duration ---
    # for path in dataset_paths:
    #     get_eeg_duration(path, FS)


    '     Applies band-pass filtering and plots comparison for each dataset file  '
    # dataset_paths = [
    #     test1_file_path,
    #     test2_file_path,
    #     test3_file_path
    # ]
    
    # # --- Configure other parameters here ---
    # FS = 256
    # LOW_CUT = 1.0
    # HIGH_CUT = 40.0

    # # Loop through each file path in the list
    # for file_path in dataset_paths:
        
    #     # Load the data
    #     full_raw_data = load_eeg_data_interactively(file_path)
        
    #     if full_raw_data is not None:
    #         # --- Exclude the first channel (index 0) ---
    #         raw_data_for_analysis = full_raw_data[1:, :]
    #         print(f"Removed first channel. Shape for analysis: {raw_data_for_analysis.shape}")
            
    #         # Apply the band-pass filter to the remaining channels
    #         print(f"Applying a {LOW_CUT}-{HIGH_CUT} Hz band-pass filter...")
    #         filtered_data_for_analysis = apply_bandpass_filter(raw_data_for_analysis, FS, LOW_CUT, HIGH_CUT)
    #         print("Filtering complete.")

    #         # Get the simple file name for the plot title
    #         file_name = os.path.basename(file_path)

    #         # Plot the comparison for all channels
    #         plot_all_channels_comparison(
    #             raw_data_for_analysis,
    #             filtered_data_for_analysis,
    #             FS,
    #             file_name
    #         )


    '     Inspects the .mat file structure'
    # # --- Configure your parameters here ---
    # FILE_PATH = test1_file_path  # Enter the path to your .mat file

    # inspect_mat_file(FILE_PATH)


    '     Plots all channels of the raw EEG data for each dataset file  '
    # DATASET_PATHS = [
    #     test1_file_path,
    #     test2_file_path,
    #     test3_file_path
    # ]
    # FS = 256
    # for file_path in DATASET_PATHS:
    #     raw_data = load_eeg_data_interactively(file_path)
    #     if raw_data is not None:
    #         file_name_for_title = os.path.basename(file_path)
    #         plot_raw_eeg_channels(raw_data, FS, file_name_for_title)
