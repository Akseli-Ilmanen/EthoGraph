(target-labelling-gui)=
# Labelling in the GUI

## Creating labels

To create a new behavioural label:

1. Press one of the number/letters keys (see image below) to activate a specific behavioural label.
2. Click twice on the line plot to define the start and end boundaries of the label. For point events, click only once.
3. The label will be created and displayed with a color-coded overlay.


![keyboard](../../_static/media/keyboard.png)

---

## Who the labels are about

The **Individual** section at the top of the right sidebar is shown for every
panel except the video, whatever backend the data came from. It holds two
dropdowns:

- **Individual** — the animal performing the behaviour. Switching it switches
  the labels you see and create, exactly as before.
- **Recipient** — for dyadic interactions (one bird mounting another, one
  animal grooming another). It is `None` by default, meaning a solo behaviour.

Together they are the *subject* of a label, and each (individual, recipient)
pair is an independent track: with a recipient chosen, only that pair's labels
are drawn and clickable, and a new label is stored against that pair. Switching
the recipient therefore gives you a fresh canvas for the next interaction
without touching what you already labelled.

The pair is stored per label in the TSV's `individual` and `individual_rec`
columns — see {ref}`the column reference <target-exporting-labels>`.

---

## Playing back labels

Once you've created a label, click on it and press `v` to play the segment.

You can also use `Left` / `Right` to navigate individual frames, or right-click
to jump to specific timepoints. Over time you may find this faster than
playing the video.

---

## Editing and deleting labels

Use the labels widget interface:

- **Edit**: Select a label (Left-click), press `Ctrl + E`, then click twice to define new start/stop boundary.
- **Delete**: Select a label (Left-click), press `Ctrl + D` to delete.

---

## Changepoint correction

See {doc}`../changepoints/correction` for how label boundaries are snapped to
detected changepoints.

---

## Importing / exporting labels

Labels are imported and exported from the top-bar **File** menu: **Import
labels…**, **Import predictions…**, and **Export labels…** each open their
own panel. See {doc}`importing` and {doc}`exporting` for details.
