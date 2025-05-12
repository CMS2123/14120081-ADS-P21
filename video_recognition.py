# video_recognition.py

import os
import cv2
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import platform
import sys
import shutil

# OCR and speech recognition availability checks
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("pytesseract not installed. OCR functionality will be limited.")

try:
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("speech_recognition or pydub not installed. Speech recognition will be disabled.")

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
    def extract_frames(self, skip_frames=30, output_dir='extracted_frames'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        frames = []
        frame_paths = []
        count = 0
        success = True
        while success:
            success, frame = self.video.read()
            if not success:
                break
            if count % skip_frames == 0:
                frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame)
                frame_paths.append(frame_path)
            count += 1
        return frames, frame_paths
    
    def extract_audio(self, output_path='extracted_audio.wav'):
        try:
            ffmpeg_locations = [
                'ffmpeg',
                '/usr/bin/ffmpeg',
                '/usr/local/bin/ffmpeg',
                '/opt/homebrew/bin/ffmpeg',
                '/opt/local/bin/ffmpeg',
                'C:\\ffmpeg\\bin\\ffmpeg.exe',
                os.path.expanduser('~/ffmpeg/bin/ffmpeg')
            ]
            ffmpeg_cmd = None
            for location in ffmpeg_locations:
                try:
                    subprocess.run([location, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    ffmpeg_cmd = location
                    print(f"Found ffmpeg at: {location}")
                    break
                except (FileNotFoundError, PermissionError):
                    continue
            if ffmpeg_cmd is None:
                print("ffmpeg not found. Audio extraction skipped.")
                return None
            command = [
                ffmpeg_cmd, "-i", self.video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2", output_path, "-y"
            ]
            subprocess.run(command, check=True)
            return output_path if os.path.exists(output_path) else None
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def extract_text_from_frames(self, frames=None, frame_paths=None):
        if not OCR_AVAILABLE:
            print("OCR not available.")
            return []
        if frames is None and frame_paths is None:
            frames, frame_paths = self.extract_frames()
        all_text = []
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            print(f"Tesseract OCR error: {e}")
            return []
        for i, item in enumerate(frame_paths if frames is None else frames):
            try:
                image = cv2.imread(item) if frames is None else item
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                kernel = np.ones((1, 1), np.uint8)
                opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
                text = pytesseract.image_to_string(opening)
                if text.strip():
                    frame_idx = i if frames is None else i * 30
                    timestamp = frame_idx / self.fps if self.fps > 0 else 0
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    time_str = f"{minutes:02d}:{seconds:02d}"
                    all_text.append({
                        "frame": frame_idx,
                        "timestamp": time_str,
                        "text": text.strip()
                    })
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
        return all_text

    def transcribe_audio(self, audio_path=None):
        if not SPEECH_RECOGNITION_AVAILABLE:
            print("Speech recognition not available.")
            return []
        audio_path = audio_path or self.extract_audio()
        if not audio_path or not os.path.exists(audio_path):
            print("No audio to transcribe.")
            return []
        try:
            sound = AudioSegment.from_wav(audio_path)
            chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=sound.dBFS-14, keep_silence=500)
            os.makedirs("audio_chunks", exist_ok=True)
            transcriptions = []
            recognizer = sr.Recognizer()
            for i, chunk in enumerate(chunks):
                chunk_path = f"audio_chunks/chunk{i}.wav"
                chunk.export(chunk_path, format="wav")
                chunk_start_time = sum(len(c) for c in chunks[:i]) / 1000.0
                time_str = f"{int(chunk_start_time // 60):02d}:{int(chunk_start_time % 60):02d}"
                with sr.AudioFile(chunk_path) as source:
                    audio = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio)
                        transcriptions.append({"segment": i, "timestamp": time_str, "text": text})
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        print(f"Speech API error: {e}")
            return transcriptions
        except Exception as e:
            print(f"Transcription error: {e}")
            return []

    def analyze_motion(self, skip_frames=5):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = self.video.read()
        if not ret:
            return []
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_data = []
        frame_idx = 0
        while True:
            for _ in range(skip_frames):
                ret = self.video.grab()
                if not ret:
                    break
            if not ret:
                break
            ret, frame = self.video.retrieve()
            if not ret:
                break
            frame_idx += skip_frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            avg_motion = np.mean(diff)
            motion_data.append((frame_idx / self.fps, avg_motion))
            prev_gray = gray
        return motion_data

    def generate_summary(self):
        frames, frame_paths = self.extract_frames()
        audio_path = self.extract_audio()
        texts = self.extract_text_from_frames(frames, frame_paths)
        transcriptions = self.transcribe_audio(audio_path) if audio_path else []
        motion_data = self.analyze_motion()
        return {
            "video_path": self.video_path,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "extracted_frames": frame_paths,
            "audio_path": audio_path,
            "texts_from_frames": texts,
            "audio_transcriptions": transcriptions,
            "motion_data": motion_data
        }

    def visualize_results(self, summary=None):
        summary = summary or self.generate_summary()
        fig = plt.figure(figsize=(15, 14))
        max_frames = min(5, len(summary["extracted_frames"]))
        for i in range(max_frames):
            ax = fig.add_subplot(3, max_frames, i+1)
            img = Image.open(summary["extracted_frames"][i])
            ax.imshow(np.array(img))
            ax.set_title(f"Frame {i}")
            ax.axis('off')
        if summary["motion_data"]:
            ax = fig.add_subplot(3, 1, 2)
            times, motions = zip(*summary["motion_data"])
            ax.plot(times, motions)
            ax.set_title("Motion Analysis")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Motion Magnitude")
        ax = fig.add_subplot(3, 1, 3)
        ax.axis('off')
        text_summary = f"Video: {os.path.basename(summary['video_path'])}\nDuration: {summary['duration']:.2f} sec\nFPS: {summary['fps']:.2f}\n"
        text_summary += "\nSample OCR:\n"
        for item in summary["texts_from_frames"][:3]:
            text_summary += f"- {item['timestamp']}: {item['text'][:50]}...\n"
        text_summary += "\nSample Transcription:\n"
        for item in summary["audio_transcriptions"][:3]:
            text_summary += f"- {item['timestamp']}: {item['text'][:50]}...\n"
        ax.text(0, 1, text_summary, fontsize=10, va='top')
        plt.tight_layout()
        output_file = f"{os.path.splitext(os.path.basename(summary['video_path']))[0]}_analysis.png"
        plt.savefig(output_file)
        plt.show()
        self.create_text_report(summary)

    def create_text_report(self, summary):
        output_file = f"{os.path.splitext(os.path.basename(summary['video_path']))[0]}_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"VIDEO REPORT\n============\n\n")
            f.write(f"Path: {summary['video_path']}\nDuration: {summary['duration']:.2f}s\nFrame Count: {summary['frame_count']}\nFPS: {summary['fps']:.2f}\n\n")
            f.write(f"TEXT FROM FRAMES:\n-----------------\n")
            for item in summary["texts_from_frames"]:
                f.write(f"[{item['timestamp']}] {item['text']}\n")
            f.write(f"\nAUDIO TRANSCRIPTION:\n---------------------\n")
            for item in summary["audio_transcriptions"]:
                f.write(f"[{item['timestamp']}] {item['text']}\n")
            f.write(f"\nMOTION ANALYSIS:\n----------------\n")
            if summary["motion_data"]:
                motions = [m for _, m in summary["motion_data"]]
                avg_motion = sum(motions) / len(motions)
                f.write(f"Average Motion: {avg_motion:.2f}\n")
            else:
                f.write("No motion data.\n")
        print(f"Report saved to {output_file}")

    def close(self):
        self.video.release()

def process_videos_in_folder(folder_path=None):
    folder_path = folder_path or os.getcwd()
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
    if not video_files:
        print(f"No MP4 videos found in {folder_path}")
        return
    results = []
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"\nProcessing: {video_file}")
        processor = VideoProcessor(video_path)
        try:
            summary = processor.generate_summary()
            processor.visualize_results(summary)
            results.append(summary)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            processor.close()
    print(f"\n✅ Processed {len(results)} videos")
    return results

if __name__ == "__main__":
    process_videos_in_folder()
