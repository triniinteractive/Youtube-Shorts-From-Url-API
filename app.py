from flask import Flask, request, jsonify, send_from_directory
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip, TextClip, ColorClip
import os
import re
import numpy as np
import json
from openai import OpenAI
import cv2
from PIL import Image, ImageDraw, ImageFont
from moviepy.config import change_settings



client = OpenAI(
    api_key = 'your-openai-key-here',  # This is the default and can be omitted
)


app = Flask(__name__)

# Get the script directory and setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, 'videos')
DOWNLOADS_DIR = os.path.join(VIDEO_DIR, 'downloads')
SHORTS_DIR = os.path.join(VIDEO_DIR, 'shorts')
TRANSCRIPTS_DIR = os.path.join(VIDEO_DIR, 'transcripts')

def ensure_directories():
    """Create all necessary directories"""
    for directory in [DOWNLOADS_DIR, SHORTS_DIR, TRANSCRIPTS_DIR]:
        os.makedirs(directory, exist_ok=True)

def get_video_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def download_video(url):
    video_id = get_video_id(url)
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(DOWNLOADS_DIR, '%(id)s.%(ext)s')
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return os.path.join(DOWNLOADS_DIR, f"{info['id']}.{info['ext']}")
        
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        return None


def split_transcript(transcript, max_chars=20000):
    """Split transcript into smaller chunks while maintaining context"""
    chunks = []
    current_chunk = []
    current_length = 0
    
    for entry in transcript:
        entry_text = f"[{entry['start']:.1f}s] {entry['text']}\n"
        if current_length + len(entry_text) > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append(entry)
        current_length += len(entry_text)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def analyze_transcript_with_gpt(transcript):
    # Split transcript into manageable chunks
    transcript_chunks = split_transcript(transcript)
    all_sections = []
    
    for chunk in transcript_chunks:
        # Combine chunk entries into text
        full_text = ""
        for entry in chunk:
            full_text += f"[{entry['start']:.1f}s] {entry['text']}\n"
        
        prompt = f"""
        Analyze this segment of a video transcript and identify the sections that would make great viral short-form videos (30-180 seconds each).
        For each section, provide:
        1. Start timestamp (in seconds)
        2. Duration (in seconds)

        The transcript with timestamps is below:
        {full_text}

        Respond in this exact JSON format:
        {{
            "sections": [
                {{
                    "start_time": X,
                    "duration": Y,
                }},
                ...
            ]
        }}

        Ensure each section is:
        - Between 30-180 seconds
        - Contains complete thoughts/ideas
        - Has high viral potential
        - Doesn't overlap with other sections
        - Analyze properly to ensure start and duration are unique and doesn't overlap into another segment.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"You are a viral content analyzer specializing in identifying the most engaging parts of videos. The current video segment we are analyzing is {full_text}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content.strip())
            
            # Add sections from this chunk
            for section in analysis['sections']:
                all_sections.append({
                    'start': float(section['start_time']),
                    'duration': float(section['duration'])
                })
                
        except Exception as e:
            print(f"Error analyzing transcript chunk with GPT: {e}")
            continue
    
    # Sort all sections by virality score
    all_sections.sort(key=lambda x: x['virality_score'], reverse=True)
    return all_sections

def find_viral_sections(transcript, min_duration=30, max_duration=180):
    # Use GPT to analyze transcript and find viral sections
    viral_sections = analyze_transcript_with_gpt(transcript)
    
    # Validate and adjust durations if necessary
    for section in viral_sections:
        if section['duration'] > max_duration:
            section['duration'] = max_duration
        elif section['duration'] < min_duration:
            section['duration'] = min_duration
        
        # Ensure start time is valid
        if section['start'] < 0:
            section['start'] = 0
    
    return viral_sections


def detect_faces(frame):
    """Enhanced face detection using multiple cascades"""
    # Load both frontal and profile face cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Detect faces using both cascades
    frontal_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    profile_faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Combine and remove overlapping detections
    all_faces = list(frontal_faces) + list(profile_faces)
    if len(all_faces) == 0:
        return np.array([])
    
    return np.array(all_faces)

def get_face_center(frame_width, frame_height, faces, prev_center=None):
    """Get smoothed center point of detected faces"""
    if len(faces) == 0:
        return prev_center if prev_center is not None else frame_width // 2
    
    # Calculate weighted center based on face sizes
    centers = []
    weights = []
    for (x, y, w, h) in faces:
        center_x = x + w//2
        centers.append(center_x)
        weights.append(w * h)  # Larger faces get more weight
    
    current_center = int(np.average(centers, weights=weights))
    
    # Smooth movement using interpolation
    if prev_center is not None:
        smoothing_factor = 0.3
        return int(prev_center * (1 - smoothing_factor) + current_center * smoothing_factor)
    
    return current_center

def process_frame(get_frame, t, frame_size, prev_center=None):
    """Enhanced frame processing with smoother tracking"""
    frame = get_frame(t)
    faces = detect_faces(frame)
    
    frame_height, frame_width = frame_size
    target_width = int(frame_height * (9/16))
    
    # Get smoothed center position
    center_x = get_face_center(frame_width, frame_height, faces, prev_center)
    
    if len(faces) <= 1:
        # Single face or no face tracking
        left = max(0, min(center_x - target_width//2, frame_width - target_width))
        return frame[:, left:left+target_width], center_x
    else:
        # Enhanced multi-face handling
        splits = []
        face_regions = []
        
        # Sort faces from left to right
        sorted_faces = sorted(faces, key=lambda x: x[0])
        
        for (x, y, w, h) in sorted_faces:
            # Calculate region of interest
            center_x = x + w//2
            context_width = int(w * 1.5)  # Add some context around the face
            left = max(0, min(center_x - context_width//2, frame_width - target_width))
            
            # Avoid overlapping regions
            if not face_regions or left > face_regions[-1][1]:
                face_regions.append((left, left + target_width))
                splits.append(frame[:, left:left+target_width])
        
        if splits:
            # Resize splits to maintain aspect ratio
            split_height = frame_height // len(splits)
            resized_splits = [cv2.resize(split, (target_width, split_height)) for split in splits]
            return np.vstack(resized_splits), center_x
        
        # Fallback to center crop if no valid splits
        left = max(0, min(center_x - target_width//2, frame_width - target_width))
        return frame[:, left:left+target_width], center_x

def get_captions_for_segment(transcript, start_time, end_time):
    """Extract and format captions for a specific time segment without added delays."""
    relevant_captions = []
    
    for entry in transcript:
        if start_time <= entry['start'] < end_time:
            # Adjust timing relative to clip start
            relative_start = entry['start'] - start_time
            
            # Ensure caption doesn't extend beyond clip duration
            duration = min(
                entry['duration'],
                end_time - entry['start'],
                3.0  # Maximum caption duration of 4 seconds
            )
            
            relevant_captions.append({
                'text': entry['text'],
                'start': relative_start,
                'duration': duration
            })
    
    return relevant_captions


def create_caption_frame(text, size, font_size=30):
    """Create a caption frame with background"""
    width, height = size
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    padding = 15
    bg_height = font_size + padding * 2
    bg_y = height - bg_height - 50
    
    draw.rectangle(
        [(0, bg_y), (width, bg_y + bg_height)],
        fill=(0, 0, 0, 160)
    )
    
    # Update font path for Ubuntu
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Center text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    text_y = bg_y + padding
    
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
    return np.array(img)


def create_short(video_path, start_time, duration, output_path, transcript):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Creating short: video_path={video_path}, start_time={start_time}, duration={duration}")

    try:
        with VideoFileClip(video_path) as video:
            # Cut the clip
            clip = video.subclip(start_time, start_time + duration)

            # Calculate dimensions for 9:16 aspect ratio
            target_aspect_ratio = 9 / 16
            if clip.w / clip.h > target_aspect_ratio:
                new_width = int(clip.h * target_aspect_ratio)
                x_center = (clip.w - new_width) / 2
                cropped_clip = clip.crop(x1=x_center, width=new_width)
            else:
                new_height = int(clip.w / target_aspect_ratio)
                y_center = (clip.h - new_height) / 2
                cropped_clip = clip.crop(y1=y_center, height=new_height)

            # Get captions
            captions = get_captions_for_segment(transcript, start_time, start_time + duration)

            # Create caption overlays
            caption_clips = []
            for caption in captions:
                txt_clip = (TextClip(
                    caption['text'],
                    fontsize=20,
                    color='white',
                    bg_color='black',
                    size=(cropped_clip.w * 0.9, None),
                    method='caption')
                    .set_duration(caption['duration'])
                    .set_start(caption['start'])
                    .set_position(('center', 0.8), relative=True))
                caption_clips.append(txt_clip)

            # Compose the final video
            final_clip = CompositeVideoClip([cropped_clip] + caption_clips, size=cropped_clip.size)

            # Write the video file
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate='2000k',
                audio_bitrate='128k',
                fps=24,
                preset='medium',
                threads=2,
                ffmpeg_params=["-strict", "-2"]
            )

        if os.path.exists(output_path):
            print(f"Short created successfully at {output_path}")
        else:
            print(f"Short not found at {output_path}")
    except Exception as e:
        print(f"Error while creating short: {e}")
        raise


@app.route('/process_video', methods=['GET'])
def process_video():
    url = request.args.get('url')
    # title = request.args.get('title')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
        
    # if not title:
    #     return jsonify({'error': 'No Title provided'}), 400
    
    video_id = get_video_id(url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400
    
    try:
        # Ensure all directories exist
        ensure_directories()
        
        # Download video
        video_path = download_video(url)
        
        # Get and save transcript
        transcript = get_transcript(video_id)
        if not transcript:
            return jsonify({'error': 'Could not get transcript'}), 400
        
        # Save transcript
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f'{video_id}.json')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        
        # Find viral sections
        viral_sections = find_viral_sections(transcript)
        
        # Process each viral section
        output_paths = []
        section_details = []
        
        for i, section in enumerate(viral_sections):
            try:
                # Define output path for the current section
                output_path = os.path.join(SHORTS_DIR, f'{video_id}_short_{i}.mp4')
                
                # Create the short video for the section
                create_short(
                    video_path,
                    section['start'],
                    section['duration'],
                    output_path,
                    transcript  # Pass the transcript to create_short
                )
        
                # Convert to relative path for response
                rel_path = os.path.relpath(output_path, SCRIPT_DIR)
        
                # Verify the file was created
                if os.path.exists(output_path):
                    output_paths.append(rel_path)
                    section_details.append({
                        'path': rel_path,
                        'duration': section['duration']
                    })
                else:
                    print(f"Warning: Output file not created for section {i}")
        
            except Exception as e:
                # Log the error and continue with the next section
                print(f"Error processing section {i}: {e}")
    
        return jsonify({
            'message': 'Success',
            'video_path': os.path.relpath(video_path, SCRIPT_DIR),
            'transcript_path': os.path.relpath(transcript_path, SCRIPT_DIR),
            'shorts_created': section_details
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/videos/<path:filename>', methods=['GET'])
def serve_video(filename):
    """Serve video files from the videos directory."""
    return send_from_directory(VIDEO_DIR, filename)


if __name__ == '__main__':
    # Ensure directories exist when starting the app
    ensure_directories()
    app.run(debug=True)