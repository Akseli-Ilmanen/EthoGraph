# Troubleshooting

Report bugs on [GitHub Issues](https://github.com/Akseli-Ilmanen/EthoGraph/issues).

---

## Quick fixes

| Problem | Solution |
|---------|----------|
| Unexpected error in the GUI | Save labels (`Ctrl + S`), then restart the GUI. Save semi-regularly! |
| Error during data loading | Click **Reset gui_settings.yaml** in the I/O widget to reset the state of the GUI. |

---

## FAQ

??? info "My dataset format is not supported"

    I/O support for new data formats is actively being expanded. If your format is not yet represented, please send a sample dataset to [akseli.ilmanen@gmail.com](mailto:akseli.ilmanen@gmail.com) and I will work on adding loading support for it.

??? note "Video seek warnings with `.avi` / `.mov` files"

    If you see warnings like `Seek problem with frame 206! pts: 208; target: 206`, your video container format has unreliable keyframe indexing. Frame display may be off by 1-2 frames when scrubbing or seeking.

    **Quick fix:** Ignore inacurate frame-seeking, and suppress warning in `Navigation controls` using `Filter warnings` checkbox.

    **Proper fix:** Transcode to MP4 with H.264 for frame-accurate seeking:

    ```bash
    # Linux / macOS / Git Bash
    for f in *.avi; do ffmpeg -y -i "$f" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "${f%.avi}.mp4"; done

    # Windows CMD
    for %f in (*.avi) do ffmpeg -y -i "%f" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "%~nf.mp4"

    # Windows PowerShell
    Get-ChildItem *.avi | ForEach-Object { ffmpeg -y -i $_.FullName -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "$($_.DirectoryName)\$($_.BaseName).mp4" }
    ```

??? note "Opening `.tsv` label files in Excel"

    Excel on Windows may not correctly parse `.tsv` files when double-clicked due to regional delimiter settings. To open correctly:

    1. Open Excel → **File → Open → Browse**
    2. Change file filter to **"All Files (\*.\*)"**
    3. Select the `.tsv` file
    4. In the Text Import Wizard, select **Tab** as delimiter
