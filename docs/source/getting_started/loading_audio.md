(target-loading-audio)=
# From an audio file

Use this path for acoustic or vocal data, with or without video.

Supported formats: `.wav`, `.mp3`, `.mp4`, `.flac`. If your `.mp4` video contains audio, point both fields at the same file.

The **Create dialog** handles single recordings. For multiple separate microphone files or multiple trials, see {doc}`multi_trial`.

---

## Steps

```{tip}
{doc}`Install EthoGraph <../getting_started/installation>` if you haven't already, then launch via shortcut or:
`conda activate ethograph && ethograph launch`

In the **I/O widget**, click **Create with own data** — the wizard opens.
```

1. Under **Single trial**, select: **3) Generate from audio file**
2. Click **Next** — the dialog opens
3. Set **Audio file** (`.wav`, `.mp3`, `.mp4`, `.flac`)
4. Optionally set **Video file** — frame rate is auto-detected; audio sample rate is read-only (auto-detected)
5. Set **Output path** for the generated `session.nc`
6. Click **Generate .nc file**
7. The I/O widget auto-populates -> click **Load**

---

## Multichannel audio

If all microphones are stored in a **single multichannel `.wav`** file, load it directly — EthoGraph separates channels automatically.

For **multiple separate `.wav` files** (one per mic), use the multi-trial wizard: see {ref}`Multi-trial setup — session-wide audio <target-loading-script>`.

---

## Dialog fields

| Field | Notes |
|-------|-------|
| **Audio file** | `.wav`, `.mp3`, `.mp4`, `.flac` |
| **Video file** | Optional |
| **Output path** | |
