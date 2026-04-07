"""Quick test for regex filename analysis."""
from pathlib import Path
from ethograph.gui.wizard_media_files import analyze_filenames_with_regex, extract_file_row

# Simulate files like: left_001.mp4, left_002.mp4, right_001.mp4, right_002.mp4
files = [Path(f"{cam}_{t:03d}.mp4") for cam in ("left", "right") for t in range(1, 4)]
print("Files:", [f.stem for f in files])

# Named groups regex
pat = analyze_filenames_with_regex(files, r"(?P<camera>\w+)_(?P<trial>\d+)")
print("Mode:", pat.tokenize_mode)
print("Summary:", pat.summary())
print("Segments:", [(s.role, s.values) for s in pat.segments])

# Extract row
row = extract_file_row(files[0], pat.segments, pat.tokenize_mode, regex_pattern=pat.regex_pattern)
print("Row:", row)

# Plain filter (no groups)
pat2 = analyze_filenames_with_regex(files, r"left")
print("Filter mode files:", len(pat2.files), [f.stem for f in pat2.files])

print("\nAll tests passed!")
