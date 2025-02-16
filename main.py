import json
import os
import cv2
import moviepy.editor as mp #pip install moviepy==1.0.3
from moviepy.editor import *
import numpy as np
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
from pymediainfo import MediaInfo
from PIL import Image
from datetime import datetime, date
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import math
from pydub import AudioSegment
from pathlib import Path

target_width = 1280
target_height = 720
target_width = 2560
target_height = 1920

target_params = {'target_width': target_width, 'target_height': target_height, 'target_fps': 25, 'image_duration_sec': 2.5}

sounds = ["FIVE OF A KIND - Density & Time.mp3",
"Helium - TrackTribe.mp3",
"In Eternity We'll Meet - Aakash Gandhi.mp3",
"Just Dance - Patrick Patrikios.mp3",
"LITE BRITE - Density & Time.mp3",
"Mer-Ka-Ba - Jesse Gallagher.mp3",
"Rubix Cube - Audionautix.mp3",
"SPRING OF DECEPTION - Density & Time.mp3",
"Spring Thaw - Asher Fulero.mp3",
"The Sea Beneath Our Feet - Puddle of Infinity.mp3",
"TORSION - Density & Time.mp3"]

"""
# list music 
directory = Path('c:/Music/')
for file in directory.glob('*.mp3'):  # Replace '*.txt' with your pattern
    print(f'"{file.name}",')
"""

def get_file_type(filename):
    _, ext = os.path.splitext(filename.lower())
    if ext in ['.jpg', '.jpeg', '.png', '.mp4']:
        return ext[1:]
    return 'other'


def get_system_time(path):
    creation_date = os.path.getctime(path)
    formatted_date = datetime.fromtimestamp(creation_date).strftime("%Y-%m-%d %H:%M:%S")
    return formatted_date

def get_image_dimensions(image_path):
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Get image dimensions
            width, height = img.size

            # Get image creation date
            creation_date = img._getexif().get(0x9003) if hasattr(img, '_getexif') and hasattr(img._getexif(), 'get') \
                else get_system_time(image_path)

            # Return all collected information
            return width, height, creation_date

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None, None

def get_video_info(video_path):
    try:

        video = mp.VideoFileClip(video_path)

        width = int(video.w)
        height = int(video.h)
        duration = video.duration
        fps = video.fps
        rotation = video.rotation

        # Estimate number of frames (this is an approximation)
        num_frames = int(duration * fps)

        # Get video creation date
        try:
            media_info = MediaInfo.parse(video_path)
            for track in media_info.tracks:
                if hasattr(track, 'encoded_date'):
                    if 'UTC' in track.encoded_date:
                        creation_date = datetime.strptime(track.encoded_date, "%Y-%m-%d %H:%M:%S %Z")
                    else:
                        creation_date = datetime.strptime(track.encoded_date, "%Y-%m-%d %H:%M:%S")
                    break
        except Exception as e:
            creation_date = os.path.getctime(video_path)

        # Estimate bitrate (this is a rough estimate and may vary)
        bitrate = int(width * height * fps * 8 * 1024 / (1024 * 1024))  # in Mbps
        video.close()

        # Open the video file

        return {
            'width': width,
            'height': height,
            'duration': duration,
            'fps': fps,
            'num_frames': num_frames,
            'bitrate': bitrate,
            'is_image': False,
            'creation_date': creation_date,
            'rotation': rotation
        }
    except Exception as e:
        print(e)


def get_file_attrs(item, item_path):
    file_info = {
        'name': item,
        'path': item_path,
        'size': os.path.getsize(item_path),
        'type': get_file_type(item)
    }

    if file_info['type'] in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']:
        width, height, dt = get_image_dimensions(item_path)
        if width is not None:
            file_info['width'] = width
            file_info['height'] = height
            file_info['creation_date'] = dt
            file_info['is_image'] = True
            file_info['rotation'] = 0
        else:
            file_info['width'] = file_info['height'] = None
    elif file_info['type'] in ['mp4', 'mpeg4', 'avi', 'mov', 'wmv', 'flv', 'ogv', 'mkv', 'ts', 'm4v', 'divx', 'vob', 'asf', 'f4v', 'vp9', 'vp8', 'hevc', 'avchd']:
        video_info = get_video_info(item_path)
        file_info.update(video_info)

    return file_info


def scan_directory(root_path):
    file_tree = {}

    def walk_and_collect(current_path, file_tree):
        for item in os.listdir(current_path):
            if item == "bad":
                continue
            item_path = os.path.join(current_path, item)
            #print(item_path)

            if os.path.isdir(item_path):
                if "dirs" not in file_tree:
                    file_tree['dirs'] = {}
                file_tree['dirs'][item] = {}
                walk_and_collect(item_path, file_tree['dirs'][item])
            else:
                if "files" not in file_tree:
                    file_tree['files'] = {}
                fi = file_tree['files'][item] = get_file_attrs(item, item_path)
                if 'width' in fi and 'height' in fi:
                    print(f"{item_path}: {fi['width']}x{fi['height']}")
                else:
                    print(f"{item_path}: ERROR")

    walk_and_collect(root_path, file_tree)
    return file_tree



def parse_date(dt):
    """
    Parse input into a datetime object.

    Args:
        dt (date, datetime, str): Input date/time value

    Returns:
        datetime: Parsed datetime object if successful
        None: If parsing fails
    """
    if isinstance(dt, (datetime, date)):
        # If already a datetime or date object, return as-is
        return dt

    try:
        # Try to parse as ISO format string
        return datetime.fromisoformat(dt)
    except:
        pass

    for format in ["%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"]:
        try:
            # Try to parse as standard Python date string format
            return datetime.strptime(dt, format)
        except:
            pass

    try:
        # Try to parse as YYYYMMDD format
        return datetime.strptime(dt, "%Y%m%d")
    except:
        pass

    # If all parsing attempts fail, return None
    return None


def make_into_sequence(file_tree, target_params):
    sequence = []
    for name in file_tree['files']:
        fi = file_tree['files'][name]
        if 'is_image' not in fi:
            continue
        p = {'fi': fi}
        p['name'] = name
        p['path'] = fi['path']
        p['target_width'] = target_params['target_width']
        p['target_height'] = target_params['target_height']
        p['width'] = fi['width']
        p['height'] = fi['height']
        p['is_image'] = fi['is_image']
        p['rotation'] = fi['rotation']
        p['repeat'] = 1 if not fi['is_image'] else int(target_params['image_duration_sec'] * target_params['target_fps'] + 0.5)
        if 'creation_date' in fi:
            p['creation_date'] = parse_date(fi['creation_date']) 
        elif 'date' in fi:
            p['creation_date'] = parse_date(p['date'])
        del p['fi']
        if 'creation_date' not in p or p['creation_date'] is None:
            p['creation_date'] = parse_date(get_system_time(p['path']))
        sequence.append(p)
    sequence = sorted(sequence, key=lambda x: x['creation_date'])
    print(sequence)
    return sequence



def resize_to_fit(w, h, target_width, target_height):
    # Calculate initial aspect ratio
    initial_aspect_ratio = w / h
    # First pass: Fit by width
    if w > target_width:
        new_width = target_width
        new_height = int(new_width / initial_aspect_ratio)
    elif w < target_width:
        new_width = target_width
        new_height = int(new_width / initial_aspect_ratio)
    else:
        new_width = w
        new_height = h
    if target_height < new_height:
        new_height2 = target_height
        new_width2 = int((new_width * target_height) / new_height)
        new_height = new_height2
        new_width = new_width2
    return new_height, new_width



def apply_sound_to_silence(base_video_clip, silences, sound_track, output_file_name, fps):
    """
    Applies an audio track to specific silence intervals in a video file.

    Args:
        video_file (str): Path to the video file
        silences (list): List of (start, end) tuples representing silence intervals in seconds
        sound_track (str): Path to the audio file to apply

    Returns:
        str: Path to the processed video file
    """
    # Load audio
    sound = AudioFileClip(sound_track)

    # Convert video duration to milliseconds for pydub compatibility
    #video_duration_ms = video.duration * 1000

    # Create silent base audio matching video duration
    #base_audio = AudioSegment.silent(duration=video_duration_ms)

    audios = []
    sound_breaks = sorted(set([0.0] + [start for start, end in silences] + [end for start, end in silences] + [base_video_clip.duration]))
    print_sounds = True
    for i in range(len(sound_breaks) - 1):
        start = sound_breaks[i]
        end = sound_breaks[i + 1]
        if (start, end) in silences:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            # Calculate how many times we need to repeat the sound
            interval_length = end - start
            # Calculate how many times we need to repeat the sound
            repeats_needed = int(np.ceil(interval_length / sound.duration))

            # Create a list of clips to concatenate
            clips_to_concatenate = [sound] * repeats_needed

            # Concatenate the clips
            repeated_sound = concatenate_audioclips(clips_to_concatenate)

            # Trim to exact interval length
            final_sound = repeated_sound.subclip(0, interval_length)

            audios.append(final_sound)
            if print_sounds:
                print(f'sound track {start:.2f} {end:.2f}')
        else:
            # Extract audio within specified interval
            audio_clip = base_video_clip.subclip(start, end).audio
            audios.append(audio_clip)
            if print_sounds:
                print(f'sound track {start:.2f} {end:.2f}')

    #final_audio = CompositeAudioClip(audios)

    final_audio = concatenate_audioclips(audios)

    #mp3 = os.path.join(os.path.splitext(output_file_name)[0], ".mp3")

    #final_audio.write_audiofile(mp3, codec="libmp3lame", bitrate="128k", fps=fps)

    # Close the clip to free resources
    #final_audio.close()

    # Combine video with new audio
    final_video = base_video_clip.set_audio(final_audio)
    return final_video


def create_video(output_file_name, target_params, sequence, sound_track):
    # Extract target parameters
    target_width = target_params['target_width']
    target_height = target_params['target_height']
    target_fps = target_params['target_fps']
    image_duration_sec = target_params['image_duration_sec']

    clips = []
    # Create a list to hold the resized images
    #resized_images = []

    # Process each image in the sequence
    for item in sequence:
        image_path = item['path']
        if item['is_image']:
            add_image_to_video(clips, image_path, item, target_fps, target_height, target_width)
        else:
            add_video_to_video(clips, image_path, item, target_fps, target_height, target_width)

    pos = 0
    last_silent = True
    start_silence = 0
    silences = []
    for clip in clips:
        if isinstance(clip, VideoFileClip):
            if last_silent and pos > start_silence:
                silences.append((start_silence, pos))
                last_silent = False
            pos += clip.duration
            start_silence = pos
        else:
            pos += clip.duration
            last_silent = True

    if last_silent and pos > start_silence:
        silences.append((start_silence, pos))

    # Concatenate all clips into one video clip
    final_clip = mp.concatenate_videoclips(clips, method="compose")

    # Write the video
    final_clip = apply_sound_to_silence(final_clip, silences, sound_track, output_file_name, target_fps)
    final_clip.write_videofile(output_file_name, fps=target_fps, codec='libx264')

    print(f"Video '{output_file_name}' has been created successfully.")

def resize_rgb_image(img, new_width, new_height, rotation=0):
    #img = rotate(img, (rotation + 180) % 360)
    img = cv2.resize(img, (new_width, new_height))
    #img = rotate(img, rotation)
    return img

def rotate(img, rotation):
    if rotation == 90:
        img = np.rot90(img, 1)
    elif rotation == 180:
        img = np.rot90(img, 2)
    elif rotation == 270:
        img = np.rot90(img, 3)
    return img


def process_video_image0(image):
    global target_width
    global target_height
    h = image.shape[0]
    w = image.shape[1]
    new_height, new_width = resize_to_fit(w, h, target_width, target_height)
    resized_img = resize_rgb_image(image, new_width, new_height, 0)
    image2 = expand_image(resized_img, target_height, target_width)
    return image2

def process_video_image90(image):
    global target_width
    global target_height
    image = rotate(image, 270)
    h = image.shape[0]
    w = image.shape[1]
    new_height, new_width = resize_to_fit(w, h, target_width, target_height)
    resized_img = resize_rgb_image(image, new_height, new_width, 0)
    image2 = expand_image(resized_img, target_width, target_height)
    image2 = rotate(image2, 90)
    return image2

def process_video_image270(image):
    global target_width
    global target_height
    #return image
    image = rotate(image, 90)
    h = image.shape[0]
    w = image.shape[1]
    new_height, new_width = resize_to_fit(w, h, target_width, target_height)
    resized_img = resize_rgb_image(image, new_height, new_width, 0)
    image2 = expand_image(resized_img, target_width, target_height)
    image2 = rotate(image2, 270)
    return image2

def add_video_to_video(clips, video_path, item, target_fps, target_height, target_width):

    w = item['width']
    h = item['height']
    new_height, new_width = resize_to_fit(w, h, target_width, target_height)

    # Load the existing video file
    resized_clip = VideoFileClip(video_path)#.fx(vfx.resize, height=new_height)

    """    
    text_clip = TextClip(
        video_path,
        fontsize=30,  # Adjust size as needed
        color='white',
        bg_color='black'  # Optional background for better visibility
    ).set_position(('right', 'bottom')).set_duration(resized_clip.duration)

    # Combine video and text
    resized_clip = CompositeVideoClip([resized_clip, text_clip])
    """

    # Check if video is longer than l seconds and trim if necessary
    l = 20000000000
    if resized_clip.duration > l:
        resized_clip = resized_clip.subclip(0, l)

    # Apply custom function to each frame
    if item['rotation'] == 0:
        resized_clip = resized_clip.fl_image(process_video_image0)
    elif item['rotation'] == 90:
        resized_clip = resized_clip.fl_image(process_video_image90)
    elif item['rotation'] == 180:
        resized_clip = resized_clip.fl_image(process_video_image0) # 180 same as 0
    elif item['rotation'] == 270:
        resized_clip = resized_clip.fl_image(process_video_image90) # 270 same as 0

    print(f"({w}, {h}) -- ({new_width}, {new_height}) -- ({target_width}, {target_height}) -- {item['rotation']}")

    # Set the new frame rate
    resized_clip = resized_clip.set_fps(target_fps)

    # Adjust duration if needed
    duration = int(resized_clip.duration)
    resized_clip = resized_clip.subclip(0, duration)

    clips.append(resized_clip)

    print(video_path)

def add_image_to_video(clips, image_path, item, target_fps, target_height, target_width):
    # Open the image
    with Image.open(image_path) as img:
        # Convert image to RGB mode
        img = img.convert('RGB')

        # Apply custom function to each frame
        if item['rotation'] in (0, 180):
            w = img.width
            h = img.height
            new_height, new_width = resize_to_fit(w, h, target_width, target_height)
            resized_img = np.array(img.resize((new_width, new_height)).copy())
            expanded_img = expand_image(resized_img, target_height, target_width)
            print(f"resized ({w}, {h}) --> ({new_width}, {new_height}) -- {item['rotation']}")
        else:
            h = img.width
            w = img.height
            new_height, new_width = resize_to_fit(w, h, target_width, target_height)
            resized_img = np.array(img.resize((new_height, new_width)).copy())
            print(f"resized ({h}, {w}) --> ({new_height}, {new_width}) -- {item['rotation']}")
            expanded_img = expand_image(resized_img, target_width, target_height)

        clips.append(mp.ImageSequenceClip([expanded_img for _ in range(item['repeat'])], fps=target_fps))

def expand_image(img, target_height, target_width):
    new_height = img.shape[0]
    new_width = img.shape[1]
    expanded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    dh = (target_height - new_height) // 2
    dw = (target_width - new_width) // 2
    expanded_img[dh:new_height + dh, dw:new_width + dw] = img[:new_height, :new_width]
    return expanded_img

def apply_sound(output_file_name):
    """
    Detects silent intervals longer than 2 seconds in a video file.

    Args:
        output_file_name (str): Path to the video file

    Returns:
        list: List of tuples containing (start_time, end_time) for each silent interval
    """

    # audio = AudioFileClip(data[i]["file"])

    try:
        # Extract audio from video file
        sound = AudioSegment.from_file(output_file_name)

        # Split audio on silence
        chunks = split_on_silence(
            sound,
            min_silence_len=2000,  # Minimum silence duration in milliseconds (2 seconds)
            silence_thresh=-40,  # Silence threshold in dBFS (adjust based on your needs)
            keep_silence=True  # Keep the silent portions
        )

        # Convert chunks to time intervals
        intervals = []
        current_pos = 0

        for chunk in chunks:
            start_time = current_pos  # Convert milliseconds to seconds
            end_time = (current_pos + len(chunk))

            # Only include intervals longer than 2 seconds
            #if end_time - start_time >= 2:
            intervals.append((math.ceil(start_time), math.floor(end_time), chunk.dBFS))

            current_pos = end_time

        return intervals

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

sounds = ['c:\\Music\\' + s for s in sounds]
# Specify the folder to scan
folder_to_scan = 'c:\\Photo\\Турция 2023\\'
video_source_path = 'c:\\Photo\\Турция 2023\\'
output_path = 'c:\\VideoMontage\\Турция2023\\'

video_name = 'Олюдениз По Морю'
video_name = 'Ксантос'
video_name = 'Бабадаг Подьем и Спуск'
video_name = 'test'
folder_to_scan = os.path.join(video_source_path, video_name)
output_file_name = os.path.join(output_path, video_name + '.mp4')

result = scan_directory(folder_to_scan)
sequence = make_into_sequence(result, target_params)

#create_video(output_file_name, target_params, sequence, sounds[0])
#vn = output_file_name.replace("/", "\\")
#os.system(f'"C:\\Program Files\\VideoLAN\\VLC\\vlc.exe" "{vn}"')

result = scan_directory(video_source_path)
for i, dir in enumerate(result['dirs']):
    print(dir)
    video_name = dir
    folder_to_scan = os.path.join(video_source_path, video_name)
    output_file_name = os.path.join(output_path, video_name + '.mp4')
    result = scan_directory(folder_to_scan)
    #target_params = {'target_width': 1280, 'target_height': 720, 'target_fps': 25, 'image_duration_sec': 3.5}
    sequence = make_into_sequence(result, target_params)
    sound_index = i % len(sounds)
    try:
        create_video(output_file_name, target_params, sequence, sounds[sound_index])
    except Exception as e:
        print(e)
        continue

#print(json.dumps(result, indent=2))
#print(json.dumps(sequence, indent=2))

