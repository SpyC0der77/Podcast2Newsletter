# Podcast2Newsletter

Podcast2Newsletter is a Python-based tool that automates the process of converting podcast episodes into newsletters. It downloads podcast audio, transcribes it, summarizes key points, and formats them into a Markdown newsletter.

---

## Features

- **Podcast Download:** Fetches podcast episodes from an RSS feed.
- **Audio Processing:** Splits audio files into manageable chunks using FFmpeg.
- **Transcription:** Converts audio to text using the Whisper ASR model.
- **Summary and Newsletter Generation:** Summarizes transcriptions and formats them into a newsletter in Markdown.
- **Handlebars Template:** Provides an easy way to customize newsletter formatting.

---

## Requirements

- Python 3.7+
- FFmpeg
- OpenAI Whisper
- Google Generative AI API (Gemini)
- Required Python Libraries:
  - `feedparser`
  - `whisper`
  - `dotenv`
  - `pybars`
  - `tqdm`
  - `google-generativeai`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/podcast2newsletter.git
   cd podcast2newsletter
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure FFmpeg is installed and its path is set in the `.env` file.

---

## Configuration

Create a `.env` file in the root directory and add the following environment variables:

```env
PODCAST_URL=<Your Podcast RSS Feed URL>
FFMPEG_PATH=<Path to FFmpeg executable>
GEMINI_API_KEY=<Your Google Gemini API Key>
```

---

## Usage

1. Run the script:
   ```bash
   python podcast2newsletter.py
   ```

2. The script will:
   - Download the latest podcast episode from the RSS feed.
   - Split the audio into chunks.
   - Transcribe each chunk.
   - Generate a Markdown newsletter summarizing the episode.

3. Output files:
   - Transcriptions: Stored in `chunks/transcriptions/`.
   - VTT Files: Stored in `chunks/vtt/`.
   - Final Newsletter: `newsletter.md`

---

## How It Works

1. **Podcast Download:**
   - Parses the RSS feed and downloads the latest episode audio.

2. **Audio Chunking:**
   - Splits the audio into smaller chunks for easier processing.

3. **Transcription:**
   - Transcribes each chunk using Whisper ASR.

4. **VTT Merging:**
   - Merges transcriptions into a single VTT file.

5. **Summary Generation:**
   - Generates a summarized Markdown newsletter using Google Gemini API and Handlebars templates.

6. **Markdown Output:**
   - The final newsletter is saved as `newsletter.md` with clickable timestamps.

---

## Example Output

```markdown
# Episode Title

Episode Summary

## Section Header

Section content.

[00:05:30](<Podcast Episode URL>#t=330)

## Another Section Header

More content.

[00:10:15](<Podcast Episode URL>#t=615)
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributions

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---
