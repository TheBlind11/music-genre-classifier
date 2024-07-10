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

def get_segment(filename, bpm, segment_lenght=5):
    try:
        y, sr = librosa.load(filename)
        beat_duration = 60 / bpm # BPS duration
        segment_duration = beat_duration * segment_lenght # Duration of each segment in seconds
        segment_samples = int(segment_duration * sr)
        
        segment = y[:segment_samples] # Take the first segment
            
        return segment, sr
    except Exception as e:
        print(f"Error processing {filename} for segmentation: {e}")
        return None, None

def process_save_segment(dataset_path, bpm_data, output_path):
    for genre, track, bpm in bpm_data:
        if bpm is None:
            continue
        
        track_path = os.path.join(dataset_path, genre, track)
        segment, sr = get_segment(track_path, bpm)
        
        if segment is not None:
            genre_output_path = os.path.join(output_path, genre)
            os.makedirs(genre_output_path, exist_ok=True)
            
            segment_name = f"{os.path.splitext(track)[0]}_segment.wav"
            segment_path = os.path.join(genre_output_path, segment_name)
            sf.write(segment_path, segment, sr)
            print(f"Saved segment for {track} in {genre}")


dataset_path = "Data/genres_original"
output_path = "Segmented_genres"
bpm_data = process_dataset(dataset_path)

process_save_segment(dataset_path, bpm_data, output_path)
