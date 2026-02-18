# Troubleshooting

## Error Messages

1. If you run into a problem, I recommend `Ctrl + S` to save labels `.nc` file, and then restart the GUI. Generally recommended to save semi-regularly!
2. If you are starting a new project with different type of data, it may be helpful to click on the `Reset gui_settings.yaml` to avoid `KeyError`/`AttributeError` from previously selected `..._sel` config.


## FAQ

### Opening .tsv label files in Excel

Excel on Windows may not correctly parse `.tsv` files when double-clicked due to regional delimiter settings. To open correctly:

1. Open Excel
2. **File → Open → Browse**
3. Change file filter to **"All Files (\*.\*)"**
4. Select the `.tsv` file
5. Excel will launch the Text Import Wizard — select **"Tab"** as delimiter


### Video seek warnings with `.avi` / `.mov` files

If you see warnings like `Seek problem with frame 206! pts: 208; target: 206`, your video container format has unreliable keyframe indexing. Frame display may be off by 1-2 frames when scrubbing or seeking.

**Fix:** Transcode to MP4 with H.264 for frame-accurate seeking:

Batch — all `.avi` files in a folder:

```bash
# Linux / macOS / Git Bash
for f in *.avi; do ffmpeg -y -i "$f" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "${f%.avi}.mp4"; done

# Windows CMD
for %f in (*.avi) do ffmpeg -y -i "%f" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "%~nf.mp4"

# Windows PowerShell
Get-ChildItem *.avi | ForEach-Object { ffmpeg -y -i $_.FullName -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "$($_.DirectoryName)\$($_.BaseName).mp4" }
```


## Issues

Feel free to raise any GitHub issues [here](https://github.com/Akseli-Ilmanen/EthoGraph/issues).