import os
import shutil
import torch
import librosa
import numpy as np
from panns_inference import SoundEventDetection, labels
from demucs import separate as demucs_separate

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extract audio from a video file using ffmpeg.
    """
    os.system(f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{output_audio_path}\" -y")

def load_audio_file(audio_path):
    """Load an audio file using librosa."""
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]
    return torch.tensor(audio, dtype=torch.float32)

def detect_music_events(audio_path, threshold=0.5, time_between_music_events=3, min_music_length=3):
    """Detect music events in an audio file."""
    waveform = load_audio_file(audio_path)
    sed = SoundEventDetection(checkpoint_path=None, device='cpu')
    framewise_output = sed.inference(waveform)
    timestamps = librosa.frames_to_time(np.arange(framewise_output.shape[1]), sr=32000, hop_length=640)
    timestamps /= 2 # Adjust for segmentation
    music_event_timestamps = []
    start_time = None
    music_label_indices = [27, 28, 29, 30, 32, 33, 34, 35, 36, 37]
    music_label_indices.extend(range(137, 284))
    
    for i in range(framewise_output.shape[1]):
        music_probabilities = framewise_output[0, i, music_label_indices]
        if np.any(music_probabilities > threshold):
            if start_time is None:
                start_time = timestamps[i]
        elif start_time is not None:
            end_time = timestamps[i]
            if end_time - start_time >= min_music_length:  # Check if the music event is at least 3 seconds long
                music_event_timestamps.append((round(start_time, 2), round(end_time, 2)))
            start_time = None
    if start_time is not None:
        end_time = timestamps[-1]
        if end_time - start_time >= min_music_length:  # Check if the music event is at least 3 seconds long
            music_event_timestamps.append((round(start_time, 2), round(end_time, 2)))
    
    # Combine music events that are close together
    combined_music_event_timestamps = []
    if music_event_timestamps:
        current_start_time = music_event_timestamps[0][0]
        current_end_time = music_event_timestamps[0][1]
        for i, timestamp in enumerate(music_event_timestamps[1:], start=1):
            if timestamp[0] - current_end_time < time_between_music_events:
                current_end_time = timestamp[1]
            else:
                combined_music_event_timestamps.append((current_start_time, current_end_time))
                current_start_time = timestamp[0]
                current_end_time = timestamp[1]
        combined_music_event_timestamps.append((current_start_time, current_end_time))
    
    return combined_music_event_timestamps

def remove_music_with_demucs(audio_path, output_path):
    # Separate audio into sources using demucs
    demucs_separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", "-o", output_path, audio_path])

def process_video_for_music_detection(video_path, output_folder="output"):
    """
    Process a video to detect music intervals, save music video clips, and remove music from those clips.
    """
    os.makedirs(output_folder, exist_ok=True)
    audio_path = os.path.join(output_folder, "temp_audio.wav")
    extract_audio_from_video(video_path, audio_path)

    print("Detecting music intervals...")
    music_intervals = detect_music_events(audio_path, threshold=0.1, time_between_music_events=3, min_music_length=3)

    music_detected_clips_dir = os.path.join(output_folder, "music_detected_clips")
    full_audio_clips_dir = os.path.join(music_detected_clips_dir, "full_audio")
    vocals_only_clips_dir = os.path.join(music_detected_clips_dir, "vocals_only")
    music_only_clips_dir = os.path.join(music_detected_clips_dir, "music_only")

    os.makedirs(music_detected_clips_dir, exist_ok=True)
    os.makedirs(full_audio_clips_dir, exist_ok=True)
    os.makedirs(vocals_only_clips_dir, exist_ok=True)
    os.makedirs(music_only_clips_dir, exist_ok=True)

    clip_paths = []
    for i, (start, end) in enumerate(music_intervals):
        try:
            # Step 1: Save video clip with music
            music_clip_path = os.path.join(full_audio_clips_dir, f"clip_{i + 1}_{start:.2f}-{end:.2f}.mp4")
            os.system(f"ffmpeg -i \"{video_path}\" -ss {start} -to {end} -c:v h264_videotoolbox -crf 18 -c:a aac -b:a 128k \"{music_clip_path}\" -y")
            clip_paths.append(music_clip_path)

            # Step 2: Extract audio from the music clip
            audio_clip_path = os.path.join(output_folder, f"temp_audio_clip_{i + 1}_{start:.2f}-{end:.2f}.wav")
            extract_audio_from_video(music_clip_path, audio_clip_path)

            # Step 3: Remove music using Demucs
            remove_music_with_demucs(audio_clip_path, output_folder)

            # Step 4: Combine silent video with vocals-only and no vocals audio
            vocals_only_clip_file = os.path.join(vocals_only_clips_dir, f"clip_{i + 1}_{start:.2f}-{end:.2f}.mp4")
            vocals_only_audio_file = os.path.join(output_folder, "mdx_extra", f"temp_audio_clip_{i + 1}_{start:.2f}-{end:.2f}", "vocals.mp3")
            os.system(f"ffmpeg -i \"{music_clip_path}\" -i \"{vocals_only_audio_file}\" -c:v copy -c:a aac -map 0:v -map 1:a:0? \"{vocals_only_clip_file}\" -y")
            
            music_only_clip_file = os.path.join(music_only_clips_dir, f"clip_{i + 1}_{start:.2f}-{end:.2f}.mp4")
            music_only_audio_file = os.path.join(output_folder, "mdx_extra", f"temp_audio_clip_{i + 1}_{start:.2f}-{end:.2f}", "no_vocals.mp3")
            os.system(f"ffmpeg -i \"{music_clip_path}\" -i \"{music_only_audio_file}\" -c:v copy -c:a aac -map 0:v -map 1:a:0? \"{music_only_clip_file}\" -y")

            # Clean up temporary files
            os.remove(audio_clip_path)            
            
        except Exception as e:
            print(f"Error processing clip {i + 1}: {str(e)}")
            continue
        
    # Clean up temporary files
    os.remove(audio_path)
    shutil.rmtree(os.path.join(output_folder, "mdx_extra"))
        
    # Step 5: Save timestamps
    # timestamps_file = os.path.join(output_folder, "music_timestamps.txt")
    # with open(timestamps_file, "w") as f:
    #     for start, end in music_intervals:
    #         f.write(f"{start:.2f},{end:.2f}\n")
    # print(f"Timestamps saved to: {timestamps_file}")

# Example Usage
video_path = 'path/to/your/video.mp4'
output_folder = 'path/to/your/output/folder/'
process_video_for_music_detection(video_path, output_folder)
