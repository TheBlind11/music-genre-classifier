import os
import librosa
import numpy as np
import soundfile as sf

def get_bpm(filename):
    try:
        y, sr = librosa.load(filename)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return tempo
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def process_dataset(dataset_path):
    genres = os.listdir(dataset_path)
    bpm_data = []
    
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if os.path.isdir(genre_path):
            tracks = os.listdir(genre_path)
            for track in tracks:
                track_path = os.path.join(genre_path, track)
                if track_path.endswith('.wav'):
                    bpm = get_bpm(track_path)
                    if bpm is not None:
                        bpm_data.append((genre, track, bpm))
                        print(f"Processed {track}, in {genre}: BPM = {bpm}")
                    else:
                        continue
    return bpm_data

def get_segments(filename, bpm, segment_lenght=5):
    try:
        y, sr = librosa.load(filename)
        beat_duration = 60 / bpm # BPS duration
        segment_duration = beat_duration * segment_lenght # Duration of each segment in seconds
        segment_samples = int(segment_duration * sr)
        
        segments = []
        for start_sample in range(0, len(y), segment_samples): # From 0 to track lenght with segment lenght steps
            end_sample = start_sample + segment_samples
            segment = y[start_sample:end_sample]
            if len(segment) == segment_samples:
                segments.append(segment)
            
        return segments, sr
    except Exception as e:
        print(f"Error processing {filename} for segmentation: {e}")
        return None, None

def process_save_segment(dataset_path, bpm_data, output_path):
    for genre, track, bpm in bpm_data:
        if bpm is None:
            continue
        
        track_path = os.path.join(dataset_path, genre, track)
        segments, sr = get_segments(track_path, bpm)
        
        if segments is not None:
            track_name = os.path.splitext(track)[0] # Remove .wav extension
            track_output_path = os.path.join(output_path, genre, track_name)
            os.makedirs(track_output_path, exist_ok=True)
            
            for i, segment in enumerate(segments):
                segment_filename = f"{track_name}_segment_{i+1}.wav"
                segment_path = os.path.join(track_output_path, segment_filename)
                sf.write(segment_path, segment, sr)
                print(f"Saved segment {i+1} for {track} in {genre}")


dataset_path = "Data/genres_original"
output_path = "Segmented_genres"
bpm_data = process_dataset(dataset_path)

process_save_segment(dataset_path, bpm_data, output_path)
