import os
import subprocess
from pathlib import Path
from datetime import datetime


def get_most_recent_dvr(input_dir):
    """Get the most recent .DVR.mp4 file from the directory"""
    dvr_files = list(Path(input_dir).glob("*.DVR.mp4"))
    if not dvr_files:
        raise FileNotFoundError("No .DVR files found in the directory")

    # Sort by modification time, most recent first
    dvr_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dvr_files[0]


def normalize_video(input_path, output_path):
    """
    Normalize video to 1080p, 5fps, and crop to notification area.
    Uses NVIDIA hardware acceleration for RTX 3070.
    """

    # Build ffmpeg command with hardware acceleration
    cmd = [
        'ffmpeg',
        '-hwaccel', 'cuda',  # Use NVIDIA GPU for decoding
        '-hwaccel_output_format', 'cuda',
        '-i', str(input_path),

        # Video filters - do scaling and cropping on GPU
        '-vf', (
            'scale_cuda=1920:1080:'  # Scale to 1080p on GPU
            'force_original_aspect_ratio=decrease,'
            'hwdownload,'  # Download from GPU memory for crop
            'format=nv12,'
            'crop=310:110:790:160'  # Crop: width:height:x:y
        ),

        # Output settings
        '-r', '5',  # 5 fps
        '-c:v', 'h264_nvenc',  # Use NVIDIA encoder
        '-preset', 'p4',  # Balanced preset for RTX 3070
        '-cq', '23',  # Constant quality mode (good quality/size ratio)
        '-an',  # Remove audio
        '-y',  # Overwrite output
        str(output_path)
    ]

    # Run ffmpeg
    print(f"Processing: {input_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error processing video: {result.stderr}")
        return False

    return True


def main():
    # Configuration
    input_dir = r"D:\game\Rocket League"
    output_dir = Path("./normalized_clips")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Get most recent DVR file
    try:
        most_recent = get_most_recent_dvr(input_dir)
        print(f"Found most recent file: {most_recent.name}")
        print(f"Modified: {datetime.fromtimestamp(most_recent.stat().st_mtime)}")

    except FileNotFoundError as e:
        print(e)
        return

    # Create output path with same filename but .mp4 extension
    output_path = output_dir / f"{most_recent.stem}_normalized.mp4"

    # Create mapping file to track original filenames
    mapping_file = output_dir / "filename_mapping.txt"

    # Process the video
    start_time = datetime.now()
    success = normalize_video(most_recent, output_path)

    if success:
        processing_time = (datetime.now() - start_time).total_seconds()

        # Save filename mapping
        with open(mapping_file, 'a') as f:
            f.write(f"{output_path.name} -> {most_recent.name}\n")

        # Get file sizes for comparison
        original_size = most_recent.stat().st_size / (1024 ** 2)  # MB
        output_size = output_path.stat().st_size / (1024 ** 2)  # MB

        print(f"\n✓ Processing complete!")
        print(f"  Time: {processing_time:.1f} seconds")
        print(f"  Original size: {original_size:.1f} MB")
        print(f"  Output size: {output_size:.1f} MB")
        print(f"  Reduction: {(1 - output_size / original_size) * 100:.1f}%")
        print(f"  Output: {output_path}")
    else:
        print("✗ Processing failed")


if __name__ == "__main__":
    main()