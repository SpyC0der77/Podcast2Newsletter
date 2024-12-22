import feedparser
import whisper
import os
import json
from urllib.request import Request, urlopen
import subprocess
from tqdm import tqdm
from dotenv import load_dotenv
import threading
import time
import sys
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from pybars import Compiler

load_dotenv()

print(os.getenv("GEMINI_API_KEY"))
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def find_dict_by_value(list_of_dicts, key, value):
    for dict_item in list_of_dicts:
        if key in dict_item and dict_item[key] == value:
            return dict_item
    return None

FFMPEG_PATH = r"C:\Users\Carter\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
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
with open("episode.json", "w") as f:
    f.write(json.dumps(episode, indent=4))

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

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_schema": content.Schema(
    type = content.Type.OBJECT,
    enum = [],
    required = ["title", "summary", "sections"],
    properties = {
      "title": content.Schema(
        type = content.Type.STRING,
      ),
      "summary": content.Schema(
        type = content.Type.STRING,
      ),
      "sections": content.Schema(
        type = content.Type.ARRAY,
        items = content.Schema(
          type = content.Type.OBJECT,
          enum = [],
          required = ["timestamp", "header", "content"],
          properties = {
            "timestamp": content.Schema(
              type = content.Type.NUMBER,
            ),
            "header": content.Schema(
              type = content.Type.STRING,
            ),
            "content": content.Schema(
              type = content.Type.STRING,
            ),
          },
        ),
      ),
    },
  ),
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-8b",
  generation_config=generation_config,
  system_instruction="You are going to turn a VTT file into a newsletter using markdown. The VTT file might not be totally correct. In the newsletter, you should correct it based on the episode data given. For each thing they talk about, put it in a seperate section. Don't just respond with the transcript, respond with a summary of each part. For the timestamp, extract the starting timestamp from the vtt data you are given, in seconds.",
)

chat_session = model.start_chat(
  history=[
  ]
)

with open("episode.json", "r") as f:
    episode = json.load(f)

with open("chunks/vtt/merged_transcription.vtt", "r") as f:
    vtt_data = json.load(f)
response = chat_session.send_message("Podcast episode data: " + json.dumps(episode) + ". Here is the VTT file: " + json.dumps(vtt_data))

print(response.text)

with open("response.json", "w") as f:
    f.write(response.text)

json_input = json.loads(response.text)
# Base URL for timestamps
base_url = audio['href']

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# Preprocess JSON to include formatted timestamps
for section in json_input["sections"]:
    section["formatted_timestamp"] = format_timestamp(section["timestamp"])

# Handlebars template
markdown_template = """
# {{title}}

{{summary}}

{{#each sections}}
## {{header}}

{{content}}

[{{formatted_timestamp}}]({{../base_url}}#t={{timestamp}})

{{/each}}
"""

# Compile the template
compiler = Compiler()
template = compiler.compile(markdown_template)

# Add the base_url to the JSON input
json_input["base_url"] = base_url

# Generate Markdown
output = template(json_input)

# Write to a file
with open("newsletter.md", "w") as file:
    file.write(output)

print("Markdown newsletter with formatted timestamps generated successfully.")
