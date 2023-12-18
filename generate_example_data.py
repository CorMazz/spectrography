"""A script to generate dummy data for the spectrogram project."""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from datetime import datetime, timedelta

########################################################################################################################
########################################################################################################################
# Functions
########################################################################################################################
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Generate Signal
# ----------------------------------------------------------------------------------------------------------------------

def generate_signal(properties, sampling_frequency=100):
    """
    Generated with ChatGPT
    Generate a signal by combining multiple periodic signals with specified properties.

    Parameters:
    - properties (list of tuples): Each tuple contains the properties of a periodic signal,
                                   where each tuple is (num_cycles, amplitude, frequency, noise_percentage).
    - sampling_frequency (float): Sampling frequency for generating the time array.

    Returns:
    - combined_time (numpy.ndarray): Combined time array for the entire signal.
    - combined_signal (numpy.ndarray): Combined signal generated by combining individual periodic signals.
    """
    combined_time = np.array([])
    combined_signal = np.array([])

    for prop in properties:
        num_cycles, amplitude, frequency, noise_percentage = prop
        time = np.arange(0, num_cycles * 1/frequency, 1/sampling_frequency)
        signal = amplitude * np.sin(2 * np.pi * frequency * time)

        # Add Gaussian noise if specified
        if noise_percentage > 0:
            rng = np.random.default_rng(seed=11)
            noise = rng.normal(0, amplitude * noise_percentage, len(time))
            signal += noise

        # Adjust the time for each section to be unique
        time += combined_time[-1] if combined_time.size > 0 else 0

        combined_time = np.concatenate((combined_time, time))
        combined_signal = np.concatenate((combined_signal, signal))

    return combined_time, combined_signal

# ----------------------------------------------------------------------------------------------------------------------
# Convert to Datetime
# ----------------------------------------------------------------------------------------------------------------------

def convert_to_datetime(time_array, start_date=datetime(2023, 1, 1)):
    """
    Generated with ChatGPT
    Convert a time array to datetime objects with a specified start date.

    Parameters:
    - time_array (numpy.ndarray): Time array to be converted.
    - start_date (datetime): Start date for the conversion.

    Returns:
    - datetime_array (list): List of datetime objects.
    """
    datetime_array = [start_date + timedelta(seconds=float(seconds)) for seconds in time_array]
    return datetime_array


########################################################################################################################
########################################################################################################################
# Main
########################################################################################################################
########################################################################################################################

if __name__ == "__main__":  # Added this in case I want to import those functions externally

    # ------------------------------------------------------------------------------------------------------------------
    # Define Properties
    # ------------------------------------------------------------------------------------------------------------------

    # Define properties for each section: 
    section_properties = [
        
        # (num_cycles, amplitude, frequency, noise_percentage)
    
        (5, 0.01, 1, 10),  # Noisy filler data
        (3, 50, 0.05, 0.01),  # Num cycles, high amplitude, low frequency, 1% Gaussian noise
        (3, 0.01, 1, 10),  # Noisy filler data
        (10000, 0.5, 30, 0.02),   # Num cycles, low amplitude, high frequency, 2% Gaussian noise
        (3, 0.01, 1, 10),  # Noisy filler data
        (5, 50, 0.05, 0.01),  # Num cycles, high amplitude, low frequency, 1% Gaussian noise
        (20000, 0.5, 30, 0.02),   # Num cycles, low amplitude, high frequency, 2% Gaussian noise
        (7, 50, 0.05, 0.01),  # Num cycles, high amplitude, low frequency, 1% Gaussian noise
        (5, 0.01, 1, 10),  # Noisy filler data
    ]
    
    # Set the sampling frequency
    sampling_frequency = 149
    
    # ------------------------------------------------------------------------------------------------------------------
    # Generate Signal
    # ------------------------------------------------------------------------------------------------------------------
    
    # Generate combined signal using the specified properties and sampling frequency
    time, signal = generate_signal(section_properties, sampling_frequency)
    
    # Convert time axis to datetime objects
    datetime_array = convert_to_datetime(time)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Plot and Save Original Signal
    # ------------------------------------------------------------------------------------------------------------------
    
    # Plotting the combined signal using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datetime_array, y=signal, mode='lines', name='Signal'))
    fig.update_layout(title='Signal', xaxis_title='Time', yaxis_title='Amplitude')
    fig.show(renderer='browser')
    fig.write_html("./plots/initial_data.html")
    
    # Turn the signal into a Polars dataframe and write it to a CSV
    df = pl.DataFrame({'Time': convert_to_datetime(time), 'Signal': signal})
    df.write_csv("./example_data.csv")
    

     
    

