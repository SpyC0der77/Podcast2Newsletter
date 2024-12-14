# Podcast to Newsletter Automation

This project automates the process of downloading podcast episodes, transcribing them using OpenAI's Whisper model, and generating a newsletter using Groq's LLM. It also provides a timer to track the runtime of the script.

## Features

- **RSS Feed Parsing**: Downloads the latest podcast episode from a given RSS feed.
- **Audio Download**: Downloads podcast episodes with a progress bar.
- **Audio Splitting**: Splits large audio files into smaller chunks using FFmpeg.
- **Transcription**: Transcribes audio chunks using OpenAI's Whisper model.
- **VTT Generation**: Converts transcriptions to VTT (Web Video Text Tracks) format.
- **Newsletter Generation**: Uses Groq's LLM to generate a newsletter from the VTT file.
- **Timer**: Displays a real-time timer showing the script's execution time.

## Prerequisites

### Software Requirements

- **Python 3.8 or higher**
- **FFmpeg** (Ensure it is installed and available in the specified path)

### Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**

   Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory with the following variables:

   ```plaintext
   GROQ_API_KEY=your_groq_api_key
   PODCAST_URL=your_podcast_rss_feed_url
   FFMPEG_PATH=path_to_your_ffmpeg_executable
   ```

   Replace `your_groq_api_key`, `your_podcast_rss_feed_url`, and `path_to_your_ffmpeg_executable` with your actual API key, podcast RSS feed URL, and the path to your FFmpeg executable.

## Usage

Run the script with:

```bash
python script.py
```

The script will:

1. Download the latest podcast episode.
2. Split the audio into chunks.
3. Transcribe each chunk and generate VTT files.
4. Merge the VTT files.
5. Generate a newsletter in markdown format.

The generated newsletter will be saved as `newsletter.md`.

## Timer

A timer will display at the bottom of the terminal, showing the elapsed time throughout the execution of the script.

## Project Structure

```
.
├── chunks
│   ├── audio
│   ├── transcriptions
│   └── vtt
├── script.py
├── requirements.txt
└── .env
```

## Dependencies

- **feedparser**: For parsing the RSS feed.
- **whisper**: OpenAI's Whisper model for audio transcription.
- **tqdm**: For displaying progress bars.
- **groq**: For interacting with the Groq API.
- **python-dotenv**: For managing environment variables.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please submit an issue or pull request for any improvements or suggestions.
