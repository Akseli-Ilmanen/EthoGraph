(target-loading-audio)=
# From an audio file

Use this path for acoustic or vocal data, with or without video.

Supported formats: `.wav`, `.mp3`, `.mp4`, `.flac`. If your `.mp4` video contains audio, drop the same file — it is used as both.

Single recordings load by **drag & drop**. For multiple separate microphone files or multiple trials, see {doc}`multi_trial`.

---

## Load it — drag & drop

```{tip}
{doc}`Install EthoGraph <../getting_started/installation>` if you haven't already, then launch via shortcut or:
`conda activate ethograph && ethograph launch`
```

1. On the start page, drag your **audio file** (and optionally a **video**) onto the **Drag & drop** zone.
2. Click **Load**.

No questions are asked — the sample rate and video frame rate are read from the files automatically.

---

## Multichannel audio

If all microphones are stored in a **single multichannel `.wav`** file, drop it directly — EthoGraph separates channels automatically.

For **multiple separate `.wav` files** (one per mic), use the multi-trial wizard: see {ref}`Multi-trial setup — session-wide audio <target-loading-script>`.
