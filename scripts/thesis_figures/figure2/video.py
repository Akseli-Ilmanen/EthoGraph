#!/usr/bin/env python3
"""Create diagonal shift visualization from MP4 frames."""

from pathlib import Path
from typing import Optional, Tuple, List, Union
import cv2
import numpy as np
import base64
from io import BytesIO


def parse_frame_spec(spec: str, total_frames: int, fps: float) -> List[int]:
    """
    Parse frame specification string into frame indices.
    
    Formats:
        "0,10,20,30" - Exact frame indices
        "0-100:10" - Range with step (start-end:step)
        "0s,1s,2s" - Time-based (seconds)
        "10%" - Percentage intervals
        "start,middle,end" - Named positions
    
    Args:
        spec: Frame specification string
        total_frames: Total frames in video
        fps: Frames per second
        
    Returns:
        List of frame indices
    """
    spec = spec.strip()
    
    # Named positions
    if spec in ["start", "middle", "end"]:
        if spec == "start":
            return list(range(min(10, total_frames)))
        elif spec == "middle":
            mid = total_frames // 2
            return list(range(max(0, mid - 5), min(total_frames, mid + 5)))
        else:  # end
            return list(range(max(0, total_frames - 10), total_frames))
    
    # Percentage intervals
    if spec.endswith("%"):
        percent = float(spec[:-1]) / 100
        step = int(total_frames * percent)
        return list(range(0, total_frames, max(1, step)))
    
    # Time-based (seconds)
    if "s" in spec:
        times = []
        for part in spec.split(","):
            if part.strip().endswith("s"):
                time_sec = float(part.strip()[:-1])
                times.append(int(time_sec * fps))
        return [t for t in times if 0 <= t < total_frames]
    
    # Range with step
    if "-" in spec and ":" in spec:
        range_part, step_part = spec.split(":")
        start, end = map(int, range_part.split("-"))
        step = int(step_part)
        return list(range(start, min(end + 1, total_frames), step))
    
    # Range without step
    if "-" in spec:
        start, end = map(int, spec.split("-"))
        return list(range(start, min(end + 1, total_frames)))
    
    # Comma-separated indices
    if "," in spec:
        return [int(x) for x in spec.split(",") if 0 <= int(x) < total_frames]
    
    # Single number (interpret as number of frames)
    try:
        n = int(spec)
        return list(np.linspace(0, total_frames - 1, n, dtype=int))
    except ValueError:
        raise ValueError(f"Invalid frame specification: {spec}")


def get_frames_by_motion(
    video_path: str,
    n_frames: int = 10,
    motion_threshold: float = 30.0
) -> List[int]:
    """
    Select frames with highest motion/scene changes.
    
    Args:
        video_path: Path to video file
        n_frames: Number of frames to select
        motion_threshold: Minimum motion score to consider
        
    Returns:
        Frame indices with highest motion
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames for motion detection
        sample_interval = max(1, total_frames // 100)
        motion_scores = []
        prev_gray = None
        
        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 120))  # Reduce size for speed
            
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                score = np.mean(diff)
                if score > motion_threshold:
                    motion_scores.append((i, score))
            
            prev_gray = gray
        
        # Sort by motion score and select top frames
        motion_scores.sort(key=lambda x: x[1], reverse=True)
        selected = sorted([idx for idx, _ in motion_scores[:n_frames]])
        
        return selected if selected else list(np.linspace(0, total_frames - 1, n_frames, dtype=int))
        
    finally:
        cap.release()


def create_diagonal_movie_viz_svg(
    video_path: str,
    output_path: str = "movie_visualization.svg",
    frames: Union[str, List[int], int] = 10,
    frame_size: Tuple[int, int] = (200, 150),
    shift_offset: Tuple[int, int] = (50, 50),
    background_color: str = "#ffffff",
    use_motion_detection: bool = False,
) -> None:
    """
    Create diagonal shift SVG visualization from video frames.
    
    Args:
        video_path: Path to input MP4 file
        output_path: Path for output SVG
        frames: Frame specification (string format, list of indices, or count)
        frame_size: Size of each frame (width, height)
        shift_offset: Diagonal shift between frames (x, y)
        background_color: Background color (hex)
        use_motion_detection: Select frames with highest motion
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine frame indices
        if use_motion_detection:
            n_frames = int(frames) if isinstance(frames, (int, str)) else len(frames)
            frame_indices = get_frames_by_motion(video_path, n_frames)
        elif isinstance(frames, str):
            frame_indices = parse_frame_spec(frames, total_frames, fps)
        elif isinstance(frames, list):
            frame_indices = frames
        else:
            frame_indices = list(np.linspace(0, total_frames - 1, frames, dtype=int))
        
        n_frames = len(frame_indices)
        print(f"Using frames: {frame_indices}")
        
        # Calculate SVG dimensions
        border_size = 2
        crop_size = min(224, frame_size[0], frame_size[1])
        bordered_width = crop_size + 2 * border_size
        bordered_height = crop_size + 2 * border_size
        svg_width = bordered_width + shift_offset[0] * (n_frames - 1)
        svg_height = bordered_height + shift_offset[1] * (n_frames - 1)
        
        # Start SVG
        svg_parts = [
            f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{svg_width}" height="{svg_height}" fill="{background_color}"/>',
        ]
        
        # Process frames in reverse order for correct layering
        for i, frame_idx in enumerate(reversed(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # Process frame
            resized = cv2.resize(frame, frame_size)
            
            # Center crop
            h, w = resized.shape[:2]
            crop_size_actual = min(224, h, w)
            center_x, center_y = w // 2, h // 2
            crop_half = crop_size_actual // 2
            
            x1 = max(0, center_x - crop_half)
            x2 = min(w, center_x + crop_half)
            y1 = max(0, center_y - crop_half)
            y2 = min(h, center_y + crop_half)
            
            cropped = resized[y1:y2, x1:x2]
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', cropped)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Calculate position
            reverse_i = n_frames - 1 - i
            x = reverse_i * shift_offset[0]
            y = (n_frames - 1 - reverse_i) * shift_offset[1]
            
            # Add to SVG with border
            svg_parts.append(f'<g transform="translate({x},{y})">')
            # Border rectangle
            svg_parts.append(
                f'<rect x="0" y="0" width="{crop_size_actual + 2*border_size}" '
                f'height="{crop_size_actual + 2*border_size}" fill="#333333"/>'
            )
            # Image
            svg_parts.append(
                f'<image x="{border_size}" y="{border_size}" '
                f'width="{crop_size_actual}" height="{crop_size_actual}" '
                f'href="data:image/png;base64,{img_base64}"/>'
            )
            svg_parts.append('</g>')
        
        svg_parts.append('</svg>')
        
        # Write SVG
        with open(output_path, 'w') as f:
            f.write('\n'.join(svg_parts))
        
        print(f"SVG visualization saved to: {output_path}")
        
    finally:
        cap.release()


def create_diagonal_movie_viz(
    video_path: str,
    output_path: str = "movie_visualization.png",
    frames: Union[str, List[int], int] = 10,
    frame_size: Tuple[int, int] = (200, 150),
    shift_offset: Tuple[int, int] = (50, 50),
    background_color: Tuple[int, int, int] = (255, 255, 255),
    use_motion_detection: bool = False,
    output_format: str = "png",
) -> None:
    """
    Create diagonal shift visualization from video frames.
    
    Args:
        video_path: Path to input MP4 file
        output_path: Path for output image
        frames: Frame specification (string format, list of indices, or count)
        frame_size: Size of each frame (width, height)
        shift_offset: Diagonal shift between frames (x, y)
        background_color: Canvas background BGR color or hex for SVG
        use_motion_detection: Select frames with highest motion
        output_format: Output format ("png" or "svg")
    """
    if output_format == "svg" or output_path.lower().endswith('.svg'):
        # Convert BGR to hex if needed
        if isinstance(background_color, tuple):
            bg_hex = f"#{background_color[2]:02x}{background_color[1]:02x}{background_color[0]:02x}"
        else:
            bg_hex = background_color
        return create_diagonal_movie_viz_svg(
            video_path, output_path, frames, frame_size, 
            shift_offset, bg_hex, use_motion_detection
        )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine frame indices
        if use_motion_detection:
            n_frames = int(frames) if isinstance(frames, (int, str)) else len(frames)
            frame_indices = get_frames_by_motion(video_path, n_frames)
        elif isinstance(frames, str):
            frame_indices = parse_frame_spec(frames, total_frames, fps)
        elif isinstance(frames, list):
            frame_indices = frames
        else:  # int
            frame_indices = list(np.linspace(0, total_frames - 1, frames, dtype=int))
        
        n_frames = len(frame_indices)
        print(f"Using frames: {frame_indices}")
        
        # Calculate canvas size (account for border and crop)
        border_size = 2
        crop_size = min(224, frame_size[0], frame_size[1])
        bordered_width = crop_size + 2 * border_size
        bordered_height = crop_size + 2 * border_size
        canvas_width = bordered_width + shift_offset[0] * (n_frames - 1)
        canvas_height = bordered_height + shift_offset[1] * (n_frames - 1)
        canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)
        
        for i, frame_idx in enumerate(reversed(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # First resize to frame_size
            resized = cv2.resize(frame, frame_size)
            
            # Then center crop to 224x224
            h, w = resized.shape[:2]
            crop_size = min(224, h, w)  # Don't exceed resized dimensions
            
            # Calculate center crop coordinates
            center_x, center_y = w // 2, h // 2
            crop_half = crop_size // 2
            
            x1 = max(0, center_x - crop_half)
            x2 = min(w, center_x + crop_half)
            y1 = max(0, center_y - crop_half)
            y2 = min(h, center_y + crop_half)
            
            # Crop the resized frame
            cropped = resized[y1:y2, x1:x2]
            
            # Calculate position (bottom-left to top-right diagonal)
            # Reversed: last frame at bottom-left (front), first frame at top-right (back)
            reverse_i = n_frames - 1 - i
            x = reverse_i * shift_offset[0]
            y = (n_frames - 1 - reverse_i) * shift_offset[1]
            
            # Add subtle border (darker for white background)
            border_size = 2
            bordered = cv2.copyMakeBorder(
                cropped, border_size, border_size, border_size, border_size, 
                cv2.BORDER_CONSTANT, 
                value=(50, 50, 50)
            )
            
            # Place frame on canvas
            h, w = bordered.shape[:2]
            # Ensure we don't exceed canvas boundaries
            y_end = min(y + h, canvas.shape[0])
            x_end = min(x + w, canvas.shape[1])
            canvas[y:y_end, x:x_end] = bordered[:y_end-y, :x_end-x]
        
        cv2.imwrite(output_path, canvas)
        print(f"Visualization saved to: {output_path}")
        
    finally:
        cap.release()


def create_grid_viz(
    video_path: str,
    output_path: str = "movie_grid.png",
    frames: Union[str, List[int], int] = "10%",
    grid_cols: Optional[int] = None,
    frame_size: Tuple[int, int] = (200, 150),
    padding: int = 10,
) -> None:
    """
    Create grid layout visualization of frames.
    
    Args:
        video_path: Path to input MP4 file
        output_path: Path for output image
        frames: Frame specification
        grid_cols: Number of columns (auto if None)
        frame_size: Size of each frame
        padding: Padding between frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Parse frames
        if isinstance(frames, str):
            frame_indices = parse_frame_spec(frames, total_frames, fps)
        elif isinstance(frames, list):
            frame_indices = frames
        else:
            frame_indices = list(np.linspace(0, total_frames - 1, frames, dtype=int))
        
        n_frames = len(frame_indices)
        
        # Calculate grid dimensions
        if grid_cols is None:
            grid_cols = int(np.ceil(np.sqrt(n_frames)))
        grid_rows = int(np.ceil(n_frames / grid_cols))
        
        # Create canvas
        canvas_width = grid_cols * frame_size[0] + (grid_cols + 1) * padding
        canvas_height = grid_rows * frame_size[1] + (grid_rows + 1) * padding
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            resized = cv2.resize(frame, frame_size)
            
            # Calculate grid position
            row = i // grid_cols
            col = i % grid_cols
            x = col * (frame_size[0] + padding) + padding
            y = row * (frame_size[1] + padding) + padding
            
            # Add timestamp overlay
            time_sec = frame_idx / fps
            text = f"{time_sec:.1f}s"
            cv2.putText(resized, text, (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            canvas[y:y + frame_size[1], x:x + frame_size[0]] = resized
        
        cv2.imwrite(output_path, canvas)
        print(f"Grid visualization saved to: {output_path}")
        
    finally:
        cap.release()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create movie frame visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Frame specification examples:
  10                  - 10 evenly spaced frames
  0,50,100,150       - Specific frame indices
  0-100:10           - Every 10th frame from 0 to 100
  0s,1s,5s,10s       - Frames at specific seconds
  10%%                - Every 10%% of the video
  start,middle,end   - Named positions
        """
    )
    parser.add_argument("video", help="Path to MP4 file")
    parser.add_argument("-o", "--output", default="movie_viz.png", help="Output path (.png or .svg)")
    parser.add_argument("-f", "--frames", default="10", help="Frame specification")
    parser.add_argument("--motion", action="store_true", help="Select frames with highest motion")
    parser.add_argument("--grid", action="store_true", help="Use grid layout instead of diagonal")
    parser.add_argument("--svg", action="store_true", help="Export as SVG")
    parser.add_argument("--size", type=int, nargs=2, default=[200, 150], 
                       metavar=("W", "H"), help="Frame size")
    
    args = parser.parse_args()
    
    # Auto-detect format from extension or use --svg flag
    output_format = "svg" if (args.svg or args.output.lower().endswith('.svg')) else "png"
    output_path = args.output
    if args.svg and not output_path.endswith('.svg'):
        output_path = output_path.rsplit('.', 1)[0] + '.svg'
    
    if args.grid:
        create_grid_viz(
            args.video,
            output_path,
            frames=args.frames,
            frame_size=tuple(args.size)
        )
    else:
        create_diagonal_movie_viz(
            args.video,
            output_path,
            frames=args.frames,
            frame_size=tuple(args.size),
            use_motion_detection=args.motion,
            output_format=output_format
        )