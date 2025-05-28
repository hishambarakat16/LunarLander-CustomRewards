# create_gif.py (updated with higher default FPS)
import argparse
import glob
import os
import datetime
import imageio
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing the MP4 video")
    parser.add_argument("--output-file", type=str, default="agent_performance.gif", help="Output GIF file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second in the output GIF")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to include")
    return parser.parse_args()

def create_gif(args):
    # Find MP4 file in the directory
    mp4_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    
    if not mp4_files:
        print(f"No MP4 files found in {args.video_dir}")
        return
    
    # Use the first MP4 file found
    video_path = mp4_files[0]
    print(f"Found video: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video has {total_frames} frames at {original_fps} fps")
    
    # Determine which frames to capture
    frames_to_capture = []
    if args.max_frames is not None and total_frames > args.max_frames:
        # Take frames at regular intervals
        step = total_frames // args.max_frames
        frames_to_capture = list(range(0, total_frames, step))[:args.max_frames]
        print(f"Will capture {len(frames_to_capture)} frames")
    else:
        # Skip frames to make the GIF faster but maintain motion
        # Capture every 2nd frame by default to make it faster
        skip_frames = 2
        frames_to_capture = list(range(0, total_frames, skip_frames))
        print(f"Will capture {len(frames_to_capture)} frames (every {skip_frames}th frame)")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("./gifs", os.path.basename(args.video_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with datetime
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(args.output_file)[0]
    output_path = os.path.join(output_dir, f"{base_filename}_{now}.gif")
    
    # Extract frames and create GIF
    frames = []
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in frames_to_capture:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            
        current_frame += 1
    
    cap.release()
    
    print(f"Extracted {len(frames)} frames, creating GIF...")
    
    # Convert fps to duration in milliseconds
    duration = 1000.0 / args.fps
    
    # Save as GIF
    imageio.mimsave(output_path, frames, duration=duration)
    
    print(f"GIF created: {output_path}")

if __name__ == "__main__":
    args = parse_args()
    create_gif(args)
