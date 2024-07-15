import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pywt
import soundfile as sf

def save_spectrogram_image(S, sr, output_path, title):
    # Save the spectrogram as a matplot image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_spectrograms(segment_path, genre, track_name, segment_name, output_base_path):
    # Load the audio segment
    y, sr = sf.read(segment_path)
    
    # Create output directories for each transformation
    stft_dir = os.path.join(output_base_path, 'STFT', genre, track_name)
    wavelet_dir = os.path.join(output_base_path, 'Wavelet', genre, track_name)
    mfcc_dir = os.path.join(output_base_path, 'MFCC', genre, track_name)
    os.makedirs(stft_dir, exist_ok=True)
    os.makedirs(wavelet_dir, exist_ok=True)
    os.makedirs(mfcc_dir, exist_ok=True)
    
    # Short-Time Fourier Transform (STFT)
    D = np.abs(librosa.stft(y))
    DB = librosa.amplitude_to_db(D, ref=np.max) # From amplitude to dB
    stft_output_path = os.path.join(stft_dir, f"{segment_name}_stft.png")
    save_spectrogram_image(DB, sr, stft_output_path, f"STFT Spectrogram of {segment_name}")
    
    # Wavelet Transform
    coeffs, freqs = pywt.cwt(y, scales=np.arange(1, 128), wavelet='morl', sampling_period=1/sr)
    wavelet_output_path = os.path.join(wavelet_dir, f"{segment_name}_wavelet.png")
    plt.figure(figsize=(10, 4))
    plt.imshow(np.abs(coeffs), extent=[0, len(y)/sr, 1, 128], cmap='PRGn', aspect='auto')
    # [0, len(y)/sr]: Maps the x-axis from 0 to the duration of the audio signal in seconds
    # [1, 128]: Maps the y-axis from 1 to 128, which corresponds to the scales used in the Wavelet Transform.
    plt.title(f"Wavelet Spectrogram of {segment_name}")
    plt.tight_layout()
    plt.savefig(wavelet_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Mel-Frequency Cepstral Coefficients (MFCC)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_output_path = os.path.join(mfcc_dir, f"{segment_name}_mfcc.png")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCC of {segment_name}")
    plt.tight_layout()
    plt.savefig(mfcc_output_path)
    plt.close()

def process_segments(input_base_path, output_base_path):
    for genre in os.listdir(input_base_path):
        genre_path = os.path.join(input_base_path, genre)
        if os.path.isdir(genre_path):
            for track_name in os.listdir(genre_path):
                track_path = os.path.join(genre_path, track_name)
                if os.path.isdir(track_path):
                    for segment_file in os.listdir(track_path):
                        segment_path = os.path.join(track_path, segment_file)
                        segment_name, _ = os.path.splitext(segment_file)
                        generate_spectrograms(segment_path, genre, track_name, segment_name, output_base_path)

input_base_path = 'Segmented_genres'
output_base_path = 'Transformed_spectrograms'

# Process segments to generate and save spectrograms
process_segments(input_base_path, output_base_path)
