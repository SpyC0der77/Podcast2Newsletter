import os
import json
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from pybars import Compiler

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type=content.Type.OBJECT,
        enum=[],
        required=["title", "summary", "sections"],
        properties={
            "title": content.Schema(
                type=content.Type.STRING,
            ),
            "summary": content.Schema(
                type=content.Type.STRING,
            ),
            "sections": content.Schema(
                type=content.Type.ARRAY,
                items=content.Schema(
                    type=content.Type.OBJECT,
                    enum=[],
                    required=["timestamp", "header", "content"],
                    properties={
                        "timestamp": content.Schema(
                            type=content.Type.NUMBER,
                        ),
                        "header": content.Schema(
                            type=content.Type.STRING,
                        ),
                        "content": content.Schema(
                            type=content.Type.STRING,
                        ),
                    },
                ),
            ),
        },
    ),
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

system_instruction = (
    "You are tasked with creating a detailed newsletter from podcast data. "
    "The newsletter should include a title, summary, and sections summarizing "
    "key parts of the content. Each section should have at least one detailed paragraph, "
    "explaining the topic discussed in depth. Timestamps should be formatted in HH:MM:SS, and "
    "each section should be professional and exclude advertisements. "
    "The content is sourced from a JSON object and an uploaded audio file."
)

print("Please upload the podcast audio file.")

def process_uploaded_file(audio_file_path):
    # Read episode metadata
    with open("episode.json", "r") as f:
        episode_data = json.load(f)

    # Generate Newsletter using Gemini
    print("Generating newsletter with Gemini...")

    chat_session = model.start_chat(history=[])

    response = chat_session.send_message(
        f"System: {system_instruction}\nPodcast episode: {json.dumps(episode_data)}\nAudio file uploaded: {audio_file_path}"
    )

    json_input = json.loads(response.text)

    # Base URL for timestamps
    base_url = audio_file_path

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

    # Save the markdown
    newsletter_file = "newsletter.md"
    with open(newsletter_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Markdown newsletter with detailed segments saved to {newsletter_file}")

# Example call to process the uploaded file (to be replaced by actual file handling logic in a multimodal environment)
# Assuming the uploaded file is saved as "uploaded_audio.mp3"
process_uploaded_file("uploaded_audio.mp3")
