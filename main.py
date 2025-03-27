from faster_whisper import WhisperModel
import math
import gradio as gr
from moviepy import VideoFileClip
import requests



def extract_audio(input_video_name):
    # Define the input video file and output audio file
    mp3_file = "audio.mp3"
    # Load the video clip
    video_clip = VideoFileClip(input_video_name)

    # Extract the audio from the video clip
    audio_clip = video_clip.audio
    duration = audio_clip.duration
    print(f"Audio duration: {duration}")
    # Write the audio to a separate file
    audio_clip.write_audiofile(mp3_file)

    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()

    print("Audio extraction successful!")
    return mp3_file, duration

def download_video(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    video_file = "video.mp4"
    with open(video_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    print("Video downloaded successfully!")
    return video_file

def word_level_transcribe(audio, max_segment_duration=2.0):  # Set your desired max duration here
    model = WhisperModel("tiny", device="cpu")
    segments, info = model.transcribe(audio, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1500), word_timestamps=True, log_progress=True)
    segments = list(segments)  # The transcription will actually run here.
    wordlevel_info = []
    for segment in segments:
        for word in segment.words:
          print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
          wordlevel_info.append({'word':word.word,'start':word.start,'end':word.end})
    return wordlevel_info

def create_subtitles(wordlevel_info):
    punctuation_marks = {'.', '!', '?', ',', ';', ':', '—', '-', '。', '！', '？'}  # Add/remove punctuation as needed
    subtitles = []
    line = []

    for word_data in wordlevel_info:
        line.append(word_data)
        current_word = word_data['word']

        # Check if current word ends with punctuation or line reached 5 words
        ends_with_punct = current_word and (current_word[-1] in punctuation_marks)

        if ends_with_punct or len(line) == 5:
            # Create a new subtitle segment
            subtitle = {
                "word": " ".join(item["word"] for item in line),
                "start": line[0]["start"],
                "end": line[-1]["end"],
                "textcontents": line.copy()
            }
            subtitles.append(subtitle)
            line = []

    # Add remaining words if any
    if line:
        subtitle = {
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line.copy()
        }
        subtitles.append(subtitle)

    # Remove gaps between segments by extending the previous segment's end time
    for i in range(1, len(subtitles)):
        prev_subtitle = subtitles[i - 1]
        current_subtitle = subtitles[i]

        # Extend the previous segment's end time to the start of the current segment
        prev_subtitle["end"] = current_subtitle["start"]

    return subtitles

def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"
    return formatted_time

def generate_subtitle_file(language, segments, input_video_name):
    subtitle_file = f"sub-{input_video_name}.{language}.srt"
    text = ""
    for index, segment in enumerate(segments):
        segment_start = format_time(segment['start'])
        segment_end = format_time(segment['end'])
        text += f"{str(index+1)} \n"
        text += f"{segment_start} --> {segment_end} \n"
        text += f"{segment['word']} \n"
        text += "\n"
    f = open(subtitle_file, "w", encoding='utf8')
    f.write(text)
    f.close()
    return subtitle_file

def transcribe(url):
    
    video = download_video(url)
    mp3_file, duration = extract_audio(video)
    print("transcribe")
    wordlevel_info=word_level_transcribe(mp3_file)
    subtitles = create_subtitles(wordlevel_info)
    subtitle_file = generate_subtitle_file('fa', subtitles, 'video_subtitled')
    return subtitle_file, video, mp3_file

with gr.Blocks() as demo:
    gr.Markdown("Start typing below and then click **Run** to see the progress and final output.")
    with gr.Column():
        #audio_in = gr.Audio(type="filepath")
        url = gr.Text()
        srt_file = gr.File()
        btn = gr.Button("Create")
        video_file_output = gr.Video(label="Result Video")
        mp3_file = gr.Audio(type="filepath")
        btn.click(
            fn=transcribe,
            inputs=url,
            outputs=[srt_file, video_file_output, mp3_file],
            concurrency_limit=4
        )

demo.launch(debug=True)
