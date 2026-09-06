# Which individual a panel shows

Every panel is in exactly one of two modes, and its title says which:

```
speed — bird_1 (sidebar)              follows the Individual combo
speed — bird_2 (pinned)               ignores it
cam-1 (front.mp4) — bird_1 (sidebar)
```

Change the combo and every `(sidebar)` panel moves while every `(pinned)`
panel stays. With one individual in the dataset none of this appears: no
suffix, no pin button.

## The controls

```
Individual:  [ bird_1 ▾ ]  📌
```

- **The combo** is the sidebar's individual: what every following panel
  shows, and the default for a new panel.
- **📌** opens one radio choice for the panel you last clicked (a line
  plot, heatmap or camera view):

  ```
  Panel: speed — bird_2 (pinned)
  ○ Follow sidebar (bird_1)
  ● bird_2
  ○ bird_3
  ────────────────
    Unpin all panels (follow sidebar)
  ```

  Exactly one entry is checked. Choosing a name pins the panel; choosing
  "Follow sidebar" unpins it. "Unpin all panels" puts every panel back on
  the combo, which is the way to show one individual everywhere.
- A feature panel's ⠿ title-bar menu has the same radio choice under
  **Individual**. Camera views live in the shell's docks and use the sidebar
  button only.

## What a pin changes

| panel | pinned to `bird_2` |
|---|---|
| line plot / heatmap | the individual dim of its feature is pinned to `bird_2`, whatever the combo says |
| camera view | the pose overlay shows `bird_2`'s keypoints only (the frame is unchanged) |
| any panel | the label overlay draws `bird_2`'s labels, whoever they are directed at |

## Whose label a click places

The **labelling subject** is the individual of the panel you last clicked,
by the same rule. The bottom bar says `labelling: bird_2` whenever the
dataset has more than one animal. On a paired layout, clicking bird_2's
pinned plot and placing a label lands it on bird_2 while the combo still
says bird_1; clicking a following panel puts the subject back on the
combo's individual. The receiver combo names whom the next label is directed at; it never filters.

## A two-bird layout

1. Combo says bird_1. Add a line plot and a camera view: both `(sidebar)`.
2. Duplicate each, click the copy, 📌 → bird_2: the copies read `(pinned)`.
3. Switch the combo to bird_3: the originals follow, the copies stay.
4. To see one bird everywhere: 📌 → Unpin all panels, then pick it in the
   combo.

## Where it lives

- A feature panel's pin is `panel_state["individual"]`, saved with the
  panel layout (`local_settings.yaml`); absent = follow.
- A camera view's pin is `app_state.camera_individuals`, keyed by the
  view's layout key (`primary`, or an extra view's dock key), restored when
  the extra cameras are rebuilt.
- One rule, `app_state.panel_individual(panel)`, resolves all of it;
  `selected_individual()` is that rule applied to the last clicked panel;
  `panel_mode_suffix(panel)` is the one place the title suffix is spelled.

Covered by `tests/test_unit/test_panel_individual.py` + `tests/test_integration/test_panel_pin.py`.

## Not done here

- The add-panel popup has no individual choice: pin after adding
  (duplicate a panel, pin the copy). The popup lists sources, not a form.
- Cropping a camera view to a pinned individual's mask (per-individual
  video masks): crop is display-only and keyed by camera name today; it
  would become keyed by (camera, individual) with a time-varying box.
- Named layouts (a saved "pairs" and "single" layout to switch between).
