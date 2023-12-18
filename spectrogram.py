import os
import glob
import pickle
import datetime
import numpy as np
import polars as pl
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from scipy.signal import spectrogram
from sklearn.pipeline import make_pipeline
from plotly_resampler import FigureResampler
from sklearn.preprocessing import StandardScaler


"""
Notes:
    It may be possible to optimize this by not parsing the entire datetime column to datetime and just grabbing the
    first two values to figure out the sampling frequency, saving runtime.
    
    
    I have determined that we want < 3 second temporal resolution on the spectrogram because the data switches between
    high frequency and low frequency within 3 seconds. You can't get both perfect time resolution and perfect frequency
    resolution. This is the uncertainty principle (just like quantum mechanics). You have to trade frequency resolution 
    for time resolution. In our case, since we have data sampled so frequently, we can get pretty good resolution in
    both domains. We'll sacrifice some frequency resolution to have the temporal resolution that we want. 
    
    See this link https://www.youtube.com/watch?v=MBnnXbOM5S4 for more details.
    
    Down sampling does not seem to improve .csv loading time. All of the data is loaded, then it is down sampled. This 
    simply expedites the cluster labeling time. Use that as you will. 
"""


########################################################################################################################
# Inputs
########################################################################################################################

plot_original_data = False  # Plot the original data using Plotly resampler
debug_plots = True  # Plot optional plots to see what the script is doing behind the scenes
plot_all_clusters = True  # Plot the final results

data_folder = r"."
files = glob.glob(os.path.join(data_folder, "example_data*.csv")) 

time_col = "Time"  # The column name for the time
y_col = "Signal"  # The column name for the dependent data to be clustered


frequency_ranges = {  # The different frequency ranges to be identified
    # "name": (min, max) in Hz
    "Low Freq": (0, 5),
    "High Freq": (25, 45)
}

"""
This is how many seconds we want between times we determine the frequency. The smaller this is, the longer the run time.
See the note above. 
"""
temporal_resolution = 1.5  # seconds


"""
Eps is the greediness of the clustering. Larger eps means larger clusters. Too large and all the data will be one 
cluster. Too small and all the data will be considered noise. Min samples is the minimum number of points in the 
spectrogram space (see the dominant frequencies plot in debug mode) for a cluster to be considered a cluster.
"""
clustering_parameters = {"eps": 0.2, "min_samples": 50}


"""
We have data sampled way faster than we really need it. We can down sample our data slightly to improve runtime.
"""
read_every_nth_row = 1


########################################################################################################################
########################################################################################################################
# Main
########################################################################################################################
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
#%% Lazily Read the CSVs
# ----------------------------------------------------------------------------------------------------------------------

# Wrap the code with tqdm to add a progress bar
with tqdm(total=1, desc="Loading Data") as pbar:
    # Originally this was done lazily since the CSV files had superfluous columns, and the data
    # was split between multiple CSV files, each with 1e6 data points. Now it isn't necessary...
    
    df = pl.scan_csv(files, try_parse_dates=True).gather_every(read_every_nth_row).collect()
    pbar.update(1)  # Update the progress bar

# Need to convert to Numpy because Polars still isn't widely supported in the Python ecosystem
with tqdm(total=1, desc="Converting Data to Numpy") as pbar:
    y = np.squeeze(df.select(df.columns[1]).to_numpy())
    pbar.update(1)  # Update the progress bar

# Optionally plot the raw data
if plot_original_data:
    with tqdm(total=1, desc="Plotting Original Data. This may take a while...") as pbar:

        x = np.squeeze(df.select(df.columns[0]).to_numpy())  # Get the time column in numpy

        fig = FigureResampler(go.Figure())
        fig.add_trace(
            go.Scattergl(
                name='Vertical Displacement',
                showlegend=True
            ),
            hf_x=x,
            hf_y=y,
        )
        pbar.update(1)
        fig.show_dash()

# ----------------------------------------------------------------------------------------------------------------------
#%% Calculate the Spectrogram
# ----------------------------------------------------------------------------------------------------------------------

with tqdm(total=1, desc="Calculating Spectrogram") as pbar:
    # First determine the sampling frequency (fs)
    # Convert to float by dividing a time delta by another TD
    fs = datetime.timedelta(seconds=1) / (df[time_col].gather([1])[0] - df[time_col].gather([0])[0])

    # Determine the samples per window and window overlap to get the desired temporal resolution
    # https://dsp.stackexchange.com/questions/42428/understanding-overlapping-in-stft
    # I did the math to determine the formula for nperseg and noverlap Assume we want 12.5% overlap (default from Scipy)
    overlap_fraction = 0.125
    nperseg = int(temporal_resolution * fs / (1-overlap_fraction))
    noverlap = int(overlap_fraction*nperseg)

    f, t, Sxx = spectrogram(y, fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")

    # Zoom the data to the range we care about (below max detection frequency)
    max_detection_frequency = np.max(np.array(list(frequency_ranges.values())))
    idx = f < max_detection_frequency*1.1
    f, Sxx = f[idx], Sxx[idx, :]
    pbar.update(1)

# Create a Plotly heatmap
if debug_plots:
    # Plot this on a log scale because the magnitudes are very, very, very different...
    fig = go.Figure(data=go.Heatmap(z=10*np.log10(Sxx), x=t, y=f, colorscale='Viridis'))

    # Set axis labels
    fig.update_layout(
        xaxis_title='Time [sec]',
        yaxis_title='Frequency [Hz]',
        title='Spectrogram'
    )
    fig.show(renderer="browser")
    fig.write_html("./plots/spectrogram.html")

# ----------------------------------------------------------------------------------------------------------------------
#%% Identify Dominant Frequency in the Spectrogram
# ----------------------------------------------------------------------------------------------------------------------

# This is fast, don't even time this
dominant_frequencies = f[np.argmax(Sxx, axis=0)]

if debug_plots:
    # Plot this on a log scale because the magnitudes are very, very, very different...
    fig = go.Figure(data=go.Scatter(x=t, y=dominant_frequencies))

    # Set axis labels
    fig.update_layout(
        xaxis_title='Time [sec]',
        yaxis_title='Dominant Frequency [Hz]',
        title='Dominant Frequency Plot'
    )
    fig.show(renderer="browser")
    fig.write_html("./plots/dominant_frequency_plot.html")

# ----------------------------------------------------------------------------------------------------------------------
#%% Temporally Cluster the Dominant Frequencies
# ----------------------------------------------------------------------------------------------------------------------


with tqdm(total=1, desc="Temporally Clustering Dominant Frequencies") as pbar:
    pipeline = make_pipeline(StandardScaler(), DBSCAN(**clustering_parameters))
    X = np.column_stack((t, dominant_frequencies))  # Cluster with time as a factor
    cluster_labels = pipeline.fit_predict(X)

    # Identify the cluster frequencies
    cluster_frequencies = {}
    for cluster_label in sorted(np.unique(cluster_labels)):
        cluster_frequency = np.mean(dominant_frequencies[cluster_labels == cluster_label])
        cluster_frequencies[cluster_label] = cluster_frequency

    # Label the frequency as the correct type
    cluster_types = {}

    for cluster_label, cluster_frequency in cluster_frequencies.items():
        if cluster_label == -1:  # -1 is the noise cluster
            cluster_types[cluster_label] = "Noise"
        else:
            # Search for the first frequency range that the cluster frequency falls in and set that as the label
            for i, (cluster_type, (f_min, f_max)) in enumerate(frequency_ranges.items()):
                if f_min <= cluster_frequency <= f_max:
                    cluster_types[cluster_label] = cluster_type
                    break
                if i == len(frequency_ranges):
                    raise ValueError("A frequency was detected that does not fall within the desired frequency ranges.")

    pbar.update(1)

if debug_plots:
    # Create a scatter plot with different colors for each cluster
    fig = go.Figure()

    for cluster_label in np.unique(cluster_labels):
        cluster_points = X[cluster_labels == cluster_label]
        fig.add_trace(
            go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=(
                    f'Cluster {cluster_label}: Freq = {cluster_frequencies[cluster_label]:.2f}'
                    if cluster_label != -1 else "Noise"
                )
            )
        )

    # Set axis labels
    fig.update_layout(
        xaxis_title='Time [sec]',
        yaxis_title='Dominant Frequency [Hz]',
        title='Dominant Frequency Clusters'
    )

    fig.show(renderer="browser")
    fig.write_html("./plots/clustered_dominant_frequencies.html")

# ----------------------------------------------------------------------------------------------------------------------
#%% Identify the Start and End of the Clusters
# ----------------------------------------------------------------------------------------------------------------------

with tqdm(total=1, desc="Labeling Clusters") as pbar:
    cluster_times = {}
    for cluster_label in np.unique(cluster_labels):
        cluster_time = (t[cluster_labels == cluster_label])
        cluster_times[cluster_label] = (np.min(cluster_time), np.max(cluster_time))

# ----------------------------------------------------------------------------------------------------------------------
#%% Label the Original Data by Cluster
# ----------------------------------------------------------------------------------------------------------------------

    test_start_time = df[time_col].gather([1])[0]  # Get the start time of the test
    df = df.lazy()  # Return to lazy execution mode
    df = df.with_columns(pl.lit(-1).alias("Cluster"))  # Add a cluster label column with the default value of -1
    df = df.with_columns(pl.lit("Noise").alias("Cluster Type"))

    # Loop over each cluster
    for cluster_label, (start_time, end_time) in cluster_times.items():
        # Skip the noise cluster
        if cluster_label == -1:
            continue
        
        # Start and end time are in seconds, referenced to 0. We want them to be relative to the start of the dataframe
        # and as datetime objects so that we can add them to the start time
        cluster_start_time = test_start_time + datetime.timedelta(seconds=start_time)
        cluster_end_time = test_start_time + datetime.timedelta(seconds=end_time)

        # Label data points within the cluster time range with the cluster label
        mask = (pl.col(time_col) >= cluster_start_time) & (pl.col(time_col) <= cluster_end_time)
        df = df.with_columns([
            pl.when(mask).then(pl.lit(cluster_label)).otherwise(pl.col("Cluster")).alias("Cluster"),
            pl.when(mask)
                .then(
                    pl.lit(cluster_types[cluster_label]))
                .otherwise(
                    pl.col("Cluster Type")
                )
                .alias("Cluster Type")
        ])

    df = df.collect()
    pbar.update(1)

# ----------------------------------------------------------------------------------------------------------------------
#%% Plot the Final Results
# ----------------------------------------------------------------------------------------------------------------------

if plot_all_clusters:
    with tqdm(total=len(cluster_times), desc="Plotting Final Results. This may take a while...") as pbar:

        fig = FigureResampler(go.Figure())

        for cluster_label in cluster_times:

            fig.add_trace(
                go.Scattergl(
                    name=(
                        f'Cluster {cluster_label}: {cluster_types[cluster_label]} Freq = '
                        f'{cluster_frequencies[cluster_label]:.2f}'
                        if cluster_label != -1 else "Noise"
                    ),
                    showlegend=True,
                    mode="markers",
                ),
                hf_x=np.squeeze(df.filter(pl.col("Cluster") == cluster_label).select(time_col).to_numpy()),
                hf_y=np.squeeze(df.filter(pl.col("Cluster") == cluster_label).select(y_col).to_numpy()),
            )
            pbar.update(1)

        fig.update_layout(
            xaxis_title=time_col,
            yaxis_title=y_col,
            title='Final Identified Clusters'
        )

        fig.show_dash()



