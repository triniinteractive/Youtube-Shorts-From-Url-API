# YouTube URL to Shorts API

This API is designed for YouTube content creators to process videos into engaging short clips. It uses advanced AI analysis to identify viral-worthy sections of videos and supports transcript generation, video trimming, and face tracking.
Features

    Video Downloading
        Download videos from YouTube using yt-dlp.
        Supports downloading videos in the best available quality.

    Transcript Extraction
        Retrieve video transcripts using the YouTubeTranscriptApi.
        Save transcripts in JSON format for further processing.

    AI Analysis with OpenAI
        Analyze video transcripts to identify the most engaging, viral-worthy sections using GPT-based models.
        Suggests timestamps and durations for short-form video creation.

    Video Trimming and Editing
        Automatically crop videos to a 9:16 aspect ratio suitable for platforms like YouTube Shorts or TikTok.
        Add captions dynamically based on the transcript.

    Face Tracking
        Advanced face detection using OpenCV for tracking single or multiple faces.
        Smooth transitions with intelligent cropping and multi-face handling.

    RESTful API
        Easy-to-use endpoints for processing videos:
            /process_video: Main endpoint for processing a YouTube video into short clips.

## Installation
### Prerequisites

Ensure you have the following installed on your system:

    Python 3.8 or higher
    ffmpeg (required by moviepy for video processing)
    Fonts for captions (e.g., DejaVuSans.ttf)

### Python Dependencies

Install required Python libraries using pip:

pip install Flask yt-dlp youtube-transcript-api moviepy openai pillow numpy opencv-python

API Key Configuration

Set your OpenAI API key in the code:

client = OpenAI(
    api_key='your-openai-key-here'
)

Alternatively, use environment variables for security.
Directory Setup

Ensure the following directories exist for storing processed files:

    videos/downloads – For downloaded videos.
    videos/shorts – For generated short videos.
    videos/transcripts – For extracted transcripts.

Run this setup script:

from pathlib import Path

for folder in ['videos/downloads', 'videos/shorts', 'videos/transcripts']:
    Path(folder).mkdir(parents=True, exist_ok=True)

Usage
Running the API

Start the Flask server:

python app.py

The API will be accessible at http://127.0.0.1:5000.
API Endpoint: /process_video
Method: GET

Processes a YouTube video and returns JSON data about the generated short videos.

Parameters:

    url (string): The YouTube video URL.
    title (string): Title for the video (optional).

Example Request:

GET /process_video?url=https://www.youtube.com/watch?v=exampleID&title=My+Video

Response:

    output_paths (list): Paths to generated short videos.
    section_details (list): Details about each section, including duration and path.

Sample Workflow

    Download and Process Video
    Input a YouTube URL to process the video.
    Example output: ['videos/shorts/videoID_short_0.mp4', ...]

    Review Generated Clips
    Access generated short clips in the videos/shorts directory.

    Customize Captions
    Modify the captions or add branding as needed.

Contributing

    Fork the repository.
    Create a feature branch (git checkout -b feature-name).
    Commit your changes (git commit -m "Feature description").
    Push to the branch (git push origin feature-name).
    Open a Pull Request.

License

This project is licensed under the MIT License.