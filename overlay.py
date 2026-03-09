import os
import re
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Set paths 
BASE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = os.path.join(BASE, "FFMPEG", "bin", "ffmpeg.exe")
OUTPUT_FOLDER = os.path.join(BASE, "overlays")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(FFMPEG):
    raise FileNotFoundError(f"ffmpeg.exe not found: {FFMPEG}")

# Overlay function 
def overlay(name, matrix3d, video_clip_path, audio_path, Cx_name,
            plot_fps=5, overlay_position="topright", cleanup=True):
    
    
    print(f"Matrix shape: {matrix3d.shape}")
    phi, theta, t_total = matrix3d.shape

    # Temporary folder for plots
    plot_dir = os.path.join(OUTPUT_FOLDER, f"{name}_plotframes")
    os.makedirs(plot_dir, exist_ok=True)

    # Unified color scale per session
    vmin = matrix3d.min()
    vmax = matrix3d.max()

    # Save plot frames
    for t in range(t_total):
        frame = matrix3d[:, :, t].T
        plt.figure(figsize=(4,4))
        plt.imshow(frame, cmap='jet', origin='lower')
        plt.gca().invert_xaxis()  # reverse x-axis
        plt.axis('off')
        plt.tight_layout(pad=0)
        #plt.savefig(os.path.join(plot_dir, f"plot_{t:04d}.png"), dpi=100)
        plt.savefig( os.path.join(plot_dir, f"plot_{t:04d}.png"),
            dpi=100,
            bbox_inches='tight',      # crops extra whitespace
            pad_inches=0              # removes padding around the figure
        )
        #plt.show()
        plt.close()
        

    # Convert frames to video
    plot_video = os.path.join(OUTPUT_FOLDER, f"{name}_plots.mp4")
    subprocess.run([
        FFMPEG, "-y",
        "-framerate", str(plot_fps),
        "-i", os.path.join(plot_dir, "plot_%04d.png"),
        "-pix_fmt", "yuv420p",
        plot_video
    ])

    # Overlay plots on video
    overlaid_video = os.path.join(OUTPUT_FOLDER, f"{Cx_name}_overlaid.mp4")

    subprocess.run([
        FFMPEG, "-y",
        "-i", video_clip_path,  # base video
        "-i", plot_video,       # overlay video
        "-filter_complex",
        "[1:v][0:v]scale2ref[ovr][base];"
        "[ovr]format=rgba,colorchannelmixer=aa=0.5[ovr_alpha];"
        "[base][ovr_alpha]overlay=0:0:shortest=1[vid]",
        "-map", "[vid]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-shortest",
        overlaid_video
    ])
    

    print("The overlaid video is", overlaid_video, '\n')
    print("The audio path is", audio_path,'n')
    # Add audio
    final_output = os.path.join(OUTPUT_FOLDER, f"{Cx_name}_final.mp4")
    subprocess.run([
        FFMPEG, "-y",
        "-i", overlaid_video,
        "-i", audio_path,
        "-c:v", "copy",
        "-filter:a", "pan=stereo|FL=c0|FR=c1", 
        "-c:a", "aac",
        "-shortest",
        final_output
    ])
   
    # Cleanup temporary files
    if cleanup:
        shutil.rmtree(plot_dir)
        os.remove(plot_video)
        os.remove(overlaid_video)

    print(f"Video saved to: {final_output}")
    return final_output

# Batch function 
def data_for_overlay(Cx_name, matrix2d, audio_name, phi, theta, num, video_folder, audio_folder,
               plot_fps=5, overlay_position="topright"):
    """
    matrix2d: shape (d, t), d = phi*theta, averaged over f
    num: part number to extract from video
    """
    Cx_audio = os.path.basename(Cx_name[0]) if isinstance(Cx_name, (list, tuple)) else os.path.basename(Cx_name)
    audio_folder = "_".join(Cx_audio.split("_")[-4:])   # the last part of the name
    audio_folder = os.path.splitext(audio_folder)[0] #it removes the extension eg ".wav"

    video_folder = "_".join(Cx_audio.split("_")[-4:-1])   # the last part of the name including the extension
    #video_folder = os.path.splitext(video_folder)[0] #it removes the extension eg ".wav"
    print("The video folder is", video_folder, '\n')

    print("the audio key is", audio_folder,'\n')
    # Reshape 2D matrix to 3D
    d, t_total = matrix2d.shape
    if d != phi * theta:
        raise ValueError(f"Mismatch: d={d} != phi*theta={phi*theta}")
    matrix3d = matrix2d.reshape(phi, theta, t_total)

    # Construct filenames
    #audio_name = f"data_for_overlay{num}"  # example
    #video_base = re.sub(r"_part\d+$", "", audio_name)
    #print("the video path is", video_base, '\n')

    #video_file = os.path.join(video_folder, ".mp4")
    #audio_file = os.path.join(audio_folder, ".wav")
    video_path = video_folder + ".mp4"
    audio_path = audio_folder + ".wav"
    print("Video file is", video_path,'\n')
    print("Audio file is",audio_path,'\n')

    BASE = os.path.dirname(os.path.abspath(__file__))
    video_file  = os.path.join(BASE, "..", "dataset", "Cx_videos", video_path)
    audio_file  = os.path.join(BASE, "..", "dataset", "Cx_data", audio_path)

    # Validate files
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video not found: {video_file}")
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio not found: {audio_file}")

    Cx_name = os.path.splitext(Cx_name)[0] #it removes the extension eg ".wav"
    # Extract video clip for this part
    part_duration = 5
    start_time = (num - 1) * part_duration
    clip_path = os.path.join(OUTPUT_FOLDER, f"{video_folder}_split{num}_clip.mp4")
    print("The clipped path is", clip_path, '\n')
    #print("Clip size:", os.path.getsize(clip_path))
    subprocess.run([
        FFMPEG, "-y",
        "-ss", str(start_time),
        "-t", str(part_duration),
        "-i", video_file,
        "-c:v", "copy",
        "-c:a", "copy",
        clip_path
    ])
    print("Clip size2:", os.path.getsize(clip_path))

    # Call overlay
    final_video = overlay(
        name=Cx_name,
        matrix3d=matrix3d,
        video_clip_path=clip_path,
        audio_path=audio_file,
        Cx_name = Cx_name,
        plot_fps=plot_fps,
        overlay_position=overlay_position
    )
    print("The video path for the final video is", clip_path,'\n')
    print("Extracting clip from:", video_file)

    # Cleanup clip
    if os.path.exists(clip_path):
        os.remove(clip_path)

    return final_video

# Example usage 
if __name__ == "__main__":
    # Example data
    phi, theta = 8, 8
    t_total = 10
    matrix2d = np.random.rand(phi*theta, t_total)  # averaged over f
    num = 1
    video_folder = "../dataset/Cx_videos"
    audio_folder = "../dataset/Cx_audios"
    Cx_name = "Cx4_ref_fold4_room10_mix001_split3"
    audio_name = "fold4_room10_mix001_split3.wav"

    final_video_path = data_for_overlay(Cx_name, matrix2d, audio_name, phi, theta, num, video_folder, audio_folder,
               plot_fps=5, overlay_position="topright")
    print("Final overlayed video:", final_video_path)

