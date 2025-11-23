import cv2
import pytesseract
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Fix for Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Setup paths
SOURCE_DIR = Path(r"D:\game\Rocket League")
OUTPUT_DIR = Path(r"D:\RL_Dataset")
NORMALIZED_DIR = OUTPUT_DIR / "normalized"
OUTPUT_DIR.mkdir(exist_ok=True)
NORMALIZED_DIR.mkdir(exist_ok=True)

# Notification area
ROI = (830, 190, 270, 90)

# Find clips
clips = list(SOURCE_DIR.glob("*.mp4"))
clips.sort(key=lambda x: x.stat().st_mtime, reverse=True)
print(f"Found {len(clips)} clips")


def normalize_with_gpu(input_path, output_path):
    """Try different GPU acceleration methods"""

    # Method 1: GPU encoding only (most compatible)
    cmd = f'ffmpeg -i "{input_path}" -vf scale=1920:1080 -r 30 -c:v h264_nvenc -preset p1 -tune hq "{output_path}" -y -hide_banner -loglevel error'

    if os.system(cmd) == 0:
        return True

    # Method 2: Try with different GPU preset
    print("    Trying alternate GPU method...")
    cmd = f'ffmpeg -i "{input_path}" -vf scale=1920:1080 -r 30 -c:v h264_nvenc -preset fast "{output_path}" -y -hide_banner -loglevel error'

    if os.system(cmd) == 0:
        return True

    # Method 3: Fast CPU with all optimizations
    print("    Using optimized CPU...")
    cmd = f'ffmpeg -i "{input_path}" -vf scale=1920:1080 -r 30 -c:v libx264 -preset ultrafast -crf 23 -threads 0 "{output_path}" -y -hide_banner -loglevel error'

    return os.system(cmd) == 0


def show_screenshot(frame, text, time):
    """Show frame and get keyboard input"""
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Found: '{text}' at {time:.1f}s\n\nPress Y for YES, N for NO", fontsize=16)
    ax.axis('off')

    from matplotlib.patches import Rectangle
    x, y, w, h = ROI
    ax.add_patch(Rectangle((x, y), w, h, fill=False, color='red', linewidth=3))

    plt.show(block=False)
    plt.pause(0.1)

    print(f"    Is '{text}' correct? Y/N: ", end='', flush=True)

    while True:
        response = input().strip().lower()
        if response in ['y', 'yes']:
            plt.close()
            return True
        elif response in ['n', 'no']:
            plt.close()
            return False
        else:
            print("    Please enter Y or N: ", end='', flush=True)


# Process clips
for i, clip_path in enumerate(clips[:5]):
    print(f"\n[{i + 1}/5] {clip_path.name}")

    # Normalize
    normalized_path = NORMALIZED_DIR / f"{clip_path.stem}_norm.mp4"

    if not normalized_path.exists():
        print("  Normalizing...")
        if not normalize_with_gpu(clip_path, normalized_path):
            print("  Failed to normalize, skipping...")
            continue
        print("  Normalized successfully")
    else:
        print("  Already normalized")

    # Open video
    cap = cv2.VideoCapture(str(normalized_path))
    fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"  Scanning {duration:.1f}s...")

    frame_num = 0
    skip_until_frame = 0
    events_found = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Progress
        if frame_num % (fps * 5) == 0 and frame_num > 0:
            print(f"    {frame_num / fps:.0f}s / {duration:.0f}s")

        # Skip if in cooldown
        if frame_num < skip_until_frame:
            frame_num += 1
            continue

        # Check every 15 frames
        if frame_num % 15 == 0:
            # OCR
            x, y, w, h = ROI
            crop = frame[y:y + h, x:x + w]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(thresh).strip()

            # Check for events
            if any(kw in text.upper() for kw in ['GOAL', 'SAVE', 'ASSIST', '+100', '+50', '+75', '+25']):
                time = frame_num / fps
                print(f"\n    Event: '{text}' at {time:.1f}s")

                # Show and ask
                if show_screenshot(frame, text, time):
                    print(f"    ✓ Confirmed")
                    events_found += 1

                    # Skip ahead 3 seconds
                    skip_until_frame = frame_num + (fps * 3)

                    # Determine type
                    text_upper = text.upper()
                    if 'GOAL' in text_upper or '+100' in text:
                        event_type = 'goal'
                    elif 'SAVE' in text_upper or '+50' in text or '+75' in text:
                        event_type = 'save'
                    elif 'ASSIST' in text_upper or '+25' in text:
                        event_type = 'assist'
                    else:
                        event_type = 'other'

                    # Extract segment
                    event_dir = OUTPUT_DIR / event_type
                    event_dir.mkdir(exist_ok=True)

                    start_time = max(0, time - 7)
                    output_file = event_dir / f"{clip_path.stem}_{int(time)}s.mp4"

                    cmd = f'ffmpeg -i "{normalized_path}" -ss {start_time} -t 7 -c copy "{output_file}" -y -hide_banner -loglevel error'
                    if os.system(cmd) == 0:
                        print(f"    → Saved to {event_type}/")
                else:
                    print(f"    ✗ Rejected")
                    skip_until_frame = frame_num + fps

        frame_num += 1

    cap.release()
    print(f"  Done. Found {events_found} events")

print(f"\nComplete! Check {OUTPUT_DIR}")