import feedparser
import whisper
import os
import json
from urllib.request import Request, urlopen
import subprocess
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv
import threading
import time
import sys

load_dotenv()

print(os.getenv("GROQ_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def find_dict_by_value(list_of_dicts, key, value):
    for dict_item in list_of_dicts:
        if key in dict_item and dict_item[key] == value:
            return dict_item
    return None

FFMPEG_PATH = os.getenv("FFMPEG_PATH")
os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ["PATH"]

def check_ffmpeg():
    try:
        subprocess.run([FFMPEG_PATH, '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not check_ffmpeg():
    print(f"Error: FFmpeg not found at {FFMPEG_PATH}")
    exit(1)

print("FFmpeg found successfully!")

# Timer function
def display_timer():
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        sys.stdout.write(f"\rElapsed Time: {elapsed_time:.2f} seconds")
        sys.stdout.flush()
        time.sleep(1)

def start_timer():
    timer_thread = threading.Thread(target=display_timer, daemon=True)
    timer_thread.start()

# Start the timer
start_timer()

# Parse the RSS feed
d = feedparser.parse(os.getenv("PODCAST_URL"))
episode = d['entries'][0]
print(episode["links"])

audio = find_dict_by_value(episode["links"], 'type', 'audio/mpeg')
print(audio)

print("Downloading podcast episode...")

# Create a Request object with a User-Agent
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
req = Request(audio['href'], headers=headers)

# Download with progress bar
response = urlopen(req)
total_size = int(response.headers.get('content-length', 0))
block_size = 1024  # 1 Kibibyte

# Create folders for chunks, transcriptions, and VTT files
chunks_folder = "chunks"
audio_folder = os.path.join(chunks_folder, "audio")
transcriptions_folder = os.path.join(chunks_folder, "transcriptions")
vtt_folder = os.path.join(chunks_folder, "vtt")

os.makedirs(audio_folder, exist_ok=True)
os.makedirs(transcriptions_folder, exist_ok=True)
os.makedirs(vtt_folder, exist_ok=True)

audio_file = os.path.join(audio_folder, 'audio.mp3')
progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc='Downloading')
with open(audio_file, 'wb') as file:
    while True:
        buffer = response.read(block_size)
        if not buffer:
            break
        file.write(buffer)
        progress_bar.update(len(buffer))
progress_bar.close()

print("\nDownload complete!")

# Split the audio file into 5 MB chunks
chunk_prefix = os.path.join(audio_folder, 'audio_chunk')
output_pattern = f'{chunk_prefix}_%03d.mp3'

print("\nSplitting audio into 5 MB chunks...")
subprocess.run([FFMPEG_PATH, '-i', audio_file, '-f', 'segment', '-segment_time', '600', output_pattern], check=True)

print("Audio split complete!\n")

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")

# Transcribe each chunk and save the output
total_duration = 0.0
chunk_index = 0
vtt_files = []

while True:
    chunk_filename = f"{chunk_prefix}_{chunk_index:03d}.mp3"
    if not os.path.exists(chunk_filename):
        break
    
    print(f"\nTranscribing {chunk_filename}...")
    result = model.transcribe(chunk_filename, fp16=False)
    
    # Save transcription to a JSON file
    output_json = os.path.join(transcriptions_folder, f"transcription_{chunk_index:03d}.json")
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)
    
    print(f"Saved transcription to {output_json}")
    
    # Convert JSON to VTT
    vtt_filename = os.path.join(vtt_folder, f"transcription_{chunk_index:03d}.vtt")
    with open(vtt_filename, 'w', encoding='utf-8') as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        for segment in result["segments"]:
            start_time = segment["start"] + total_duration
            end_time = segment["end"] + total_duration
            text = segment["text"]
            
            # Convert time format to VTT format (HH:MM:SS.mmm)
            start_vtt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:.3f}"
            end_vtt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{end_time % 60:.3f}"
            
            vtt_file.write(f"{start_vtt.replace('.', ',')} --> {end_vtt.replace('.', ',')}\n{text}\n\n")
    
    print(f"Saved VTT to {vtt_filename}")
    vtt_files.append(vtt_filename)
    
    total_duration += result["segments"][-1]["end"]
    chunk_index += 1

print("\nAll chunks transcribed and saved!")

# Merge all VTT files
merged_vtt_file = os.path.join(vtt_folder, "merged_transcription.vtt")

print("\nMerging VTT files...")
with open(merged_vtt_file, 'w', encoding='utf-8') as outfile:
    outfile.write("WEBVTT\n\n")
    for vtt_file in vtt_files:
        with open(vtt_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()[2:]  # Skip the WEBVTT header
            outfile.writelines(lines)

    
with open(merged_vtt_file, 'r', encoding='utf-8') as infile:
    vtt_data = infile.read()

print(f"Merged VTT saved to {merged_vtt_file}")

# Generate Newsletter
print("\nGenerating newsletter...")
print(json.dumps(vtt_data))
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are going to turn a VTT file into a newsletter. Using markdown, have a minimum of 5 paragraphs with at least 6 sentences. The VTT file might not be totally correct. In the newsletter, you should correct it based on this data. Here is the podcast episode data: " + json.dumps(episode) + ". Here is the VTT file: " + json.dumps(vtt_data),
        },
        {
            "role": "user",
            "content": "Give me the newsletter.",
        }
    ],

    model="llama3-8b-8192",
    temperature=0.5,
    top_p=1,
    stop=None,
    stream=False,
)

print(chat_completion.choices[0].message.content)

with open("newsletter.md", "w") as f:
    f.write(chat_completion.choices[0].message.content)
