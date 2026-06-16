"""Global keyboard shortcut bindings for the ethograph GUI."""

import logging

from napari.layers import Image, Labels, Points, Shapes, Surface, Tracks
from qtpy.QtWidgets import QMenu

logger = logging.getLogger(__name__)


def override_napari_shortcuts(viewer):
    number_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    qwerty_row = ["q", "w", "e", "r", "t", "z", "u", "i", "o", "p"]
    home_row = ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"]
    control_row = ["e", "d", "f", "i", "k", "c", "m", "t", "n", "p"]
    other = ["y", "space", "Up", "Down", "v", "x"]

    combos = [
        "Ctrl-a",
        "Ctrl-s",
        "Ctrl-x",
        "Ctrl-v",
        "Ctrl-l",
        "Ctrl-enter",
        "Ctrl-d",
        "Ctrl-e",
        "Ctrl-f",
        "Ctrl-i",
        "Ctrl-k",
        "Ctrl-c",
        "Ctrl-m",
        "Ctrl-t",
        "Ctrl-Left",
        "Ctrl-Right",
        "Shift-A",
        "Shift-S",
        "Shift-E",
        "Shift-F",
        "Shift-C",
        "Shift-K",
    ]

    all_keys = number_keys + qwerty_row + home_row + control_row + other + combos
    layer_types = [Image, Points, Shapes, Labels, Tracks, Surface]

    for layer_type in layer_types:
        for key in all_keys:
            try:
                if hasattr(layer_type, "bind_key"):
                    layer_type.bind_key(key, None)
            except (KeyError, ValueError, AttributeError) as e:
                logger.warning("Could not unbind %s from %s: %s", key, layer_type.__name__, e)

    for key in all_keys:
        if hasattr(viewer, "keymap") and key in viewer.keymap:
            del viewer.keymap[key]
        if hasattr(viewer, "_keymap") and key in viewer._keymap:
            del viewer._keymap[key]

    if viewer.layers.selection.active:
        active_layer = viewer.layers.selection.active
        for key in all_keys:
            if hasattr(active_layer, "keymap") and key in active_layer.keymap:
                del active_layer.keymap[key]

    if hasattr(viewer, "window") and viewer.window:
        window = viewer.window
        if hasattr(window, "file_menu"):
            for action in window.file_menu.actions():
                if "save" in action.text().lower():
                    action.setShortcut("")

        if hasattr(window, "_qt_window"):
            menubar = window._qt_window.menuBar()
            for menu in menubar.findChildren(QMenu):
                for action in menu.actions():
                    if action.shortcut().toString() in ["Ctrl+S", "Ctrl+A", "Ctrl+M"]:
                        action.setShortcut("")


def bind_global_shortcuts(meta_widget):
    viewer = meta_widget.viewer
    app_state = meta_widget.app_state
    labels_widget = meta_widget.labels_widget
    data_widget = meta_widget.data_widget
    navigation_widget = meta_widget.navigation_widget
    plot_settings_widget = meta_widget.plot_settings_widget
    changepoints_widget = meta_widget.changepoints_widget
    io_widget = meta_widget.io_widget
    plot_container = meta_widget.plot_container

    override_napari_shortcuts(viewer)

    @viewer.bind_key("ctrl+s", overwrite=True)
    def save_labels(v):
        app_state.save_labels()

    @viewer.bind_key("space", overwrite=True)
    def toggle_pause_resume(v):
        data_widget.toggle_pause_resume()
        navigation_widget._sync_play_icon()

    @viewer.bind_key("v", overwrite=True)
    def play_segment(v):
        labels_widget._play_segment()

    @viewer.bind_key("shift+Left", overwrite=True)
    def step_window_backward(v):
        navigation_widget.step_window_backward()

    @viewer.bind_key("shift+Right", overwrite=True)
    def step_window_forward(v):
        navigation_widget.step_window_forward()

    @viewer.bind_key("Left", overwrite=True)
    def step_frame_backward(v):
        navigation_widget.step_frame_backward()

    @viewer.bind_key("Right", overwrite=True)
    def step_frame_forward(v):
        navigation_widget.step_frame_forward()

    @viewer.bind_key("Down", overwrite=True)
    def next_trial(v):
        navigation_widget.next_trial()

    @viewer.bind_key("Up", overwrite=True)
    def prev_trial(v):
        navigation_widget.prev_trial()

    @viewer.bind_key("ctrl+Down", overwrite=True)
    def next_channel(v):
        meta_widget._cycle_channel(+1)

    @viewer.bind_key("ctrl+Up", overwrite=True)
    def prev_channel(v):
        meta_widget._cycle_channel(-1)

    @viewer.bind_key("ctrl+p", overwrite=True)
    def toggle_sync(v):
        btn = getattr(navigation_widget, "sync_toggle_btn", None)
        if btn is None:
            return
        next_index = (btn.currentIndex() + 1) % btn.count()
        btn.setCurrentIndex(next_index)

    @viewer.bind_key("ctrl+y", overwrite=True)
    def toggle_label_pred(v):
        if data_widget is not None:
            data_widget.toggle_predictions_slot()

    if "Ctrl-A" in viewer.keymap:
        del viewer.keymap["Ctrl-A"]

    @viewer.bind_key("ctrl+a", overwrite=True)
    def toggle_autoscale(v):
        autoscale_status = plot_settings_widget.autoscale_checkbox.isChecked()
        plot_settings_widget.autoscale_checkbox.setChecked(not autoscale_status)

    @viewer.bind_key("ctrl+l", overwrite=True)
    def toggle_lock(v):
        lock_status = plot_settings_widget.lock_axes_checkbox.isChecked()
        plot_settings_widget.lock_axes_checkbox.setChecked(not lock_status)

    @viewer.bind_key("ctrl+enter", overwrite=True)
    def apply_plot_settings(v):
        plot_settings_widget.apply_button.click()

    @viewer.bind_key("ctrl+v", overwrite=True)
    def human_verified(v):
        io_widget._human_verification_true(mode="single_trial")

    @viewer.bind_key("ctrl+b", overwrite=True)
    def toggle_changepoint_correction(v):
        checkbox = changepoints_widget.changepoint_correction_checkbox
        checkbox.setChecked(not checkbox.isChecked())

    @viewer.bind_key("ctrl+r", overwrite=True)
    def refresh_lineplot(v):
        data_widget.refresh_lineplot()

    @viewer.bind_key("ctrl+=", overwrite=True)
    def increase_spacing(v):
        pc = plot_container
        if pc and pc.is_ephystrace():
            buf = pc.ephys_trace_plot.buffer
            buf.channel_spacing = min(buf.channel_spacing + 0.5, 20.0)
            xmin, xmax = pc.get_current_xlim()
            pc.ephys_trace_plot.update_plot_content(xmin, xmax)

    @viewer.bind_key("ctrl+-", overwrite=True)
    def decrease_spacing(v):
        pc = plot_container
        if pc and pc.is_ephystrace():
            buf = pc.ephys_trace_plot.buffer
            buf.channel_spacing = max(buf.channel_spacing - 0.5, 0.5)
            xmin, xmax = pc.get_current_xlim()
            pc.ephys_trace_plot.update_plot_content(xmin, xmax)

    def _jump_spike(delta: int):
        if not plot_container or not plot_container.is_ephystrace():
            return
        plot_container.ephys_trace_plot.jump_to_spike(delta)

    @viewer.bind_key("alt+Right", overwrite=True)
    def next_spike(v):
        _jump_spike(+1)

    @viewer.bind_key("alt+Left", overwrite=True)
    def prev_spike(v):
        _jump_spike(-1)

    @viewer.bind_key("ctrl+space", overwrite=True)
    def stop_recording(v):
        record_btn = getattr(navigation_widget, "record_button", None)
        if record_btn is not None:
            record_btn._stop_recording()

    # Label activation grid layout
    number_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    qwerty_row = ["q", "w", "e", "r", "t", "z", "u", "i", "o", "p"]
    home_row = ["a", "s", "d", "f", "g", "h", "j", "k", "l"]

    for i, key in enumerate(number_keys):
        labels = i + 1 if key != "0" else 10
        viewer.bind_key(key, lambda v, mk=labels: labels_widget.activate_label(mk), overwrite=True)

    for i, key in enumerate(qwerty_row):
        viewer.bind_key(key, lambda v, mk=i + 11: labels_widget.activate_label(mk), overwrite=True)

    for i, key in enumerate(home_row):
        viewer.bind_key(key, lambda v, mk=i + 21: labels_widget.activate_label(mk), overwrite=True)

    @viewer.bind_key("ctrl+e", overwrite=True)
    def edit_label(v):
        labels_widget._edit_label()

    @viewer.bind_key("ctrl+d", overwrite=True)
    def delete_label(v):
        labels_widget._delete_label()

    @viewer.bind_key("shift+b", overwrite=True)
    def switch_branch(v):
        labels_widget.toggle_branch()

    @viewer.bind_key("ctrl+f", overwrite=True)
    def toggle_features(v):
        app_state.toggle_key_sel("features", data_widget)

    @viewer.bind_key("ctrl+i", overwrite=True)
    def toggle_individuals(v):
        app_state.toggle_key_sel("individual", data_widget)

    @viewer.bind_key("ctrl+k", overwrite=True)
    def toggle_keypoints(v):
        app_state.toggle_key_sel("keypoint", data_widget)

    @viewer.bind_key("ctrl+c", overwrite=True)
    def cycle_cameras(v):
        combo = getattr(data_widget, "primary_camera_combo", None)
        if combo is not None and combo.count() > 1:
            next_index = (combo.currentIndex() + 1) % combo.count()
            combo.setCurrentIndex(next_index)

    @viewer.bind_key("ctrl+m", overwrite=True)
    def toggle_mics(v):
        app_state.toggle_key_sel("mics", data_widget)

    @viewer.bind_key("ctrl+h", overwrite=True)
    def cycle_neural_view(v):
        data_widget.cycle_neural_view()

    @viewer.bind_key("ctrl+g", overwrite=True)
    def cycle_view_mode(v):
        data_widget.cycle_view_mode()

    @viewer.bind_key("shift+a", overwrite=True)
    def toggle_audiotrace(v):
        cb = data_widget.audiotrace_checkbox
        if cb.isEnabled():
            cb.setChecked(not cb.isChecked())

    @viewer.bind_key("shift+s", overwrite=True)
    def toggle_spectrogram(v):
        cb = data_widget.spectrogram_checkbox
        if cb.isEnabled():
            cb.setChecked(not cb.isChecked())

    @viewer.bind_key("shift+e", overwrite=True)
    def toggle_ephys(v):
        cb = getattr(data_widget, "phy_viewer_checkbox", None)
        if cb and cb.isEnabled():
            cb.setChecked(not cb.isChecked())

    @viewer.bind_key("shift+f", overwrite=True)
    def toggle_featureplot(v):
        cb = data_widget.featureplot_checkbox
        if cb.isEnabled():
            cb.setChecked(not cb.isChecked())

    @viewer.bind_key("shift+c", overwrite=True)
    def toggle_video_viewer(v):
        cb = data_widget.video_viewer_checkbox
        cb.setChecked(not cb.isChecked())

    @viewer.bind_key("ctrl+Right", overwrite=True)
    def next_changepoint(v):
        changepoints_widget.jump_changepoint(+1)

    @viewer.bind_key("ctrl+Left", overwrite=True)
    def prev_changepoint(v):
        changepoints_widget.jump_changepoint(-1)

    @viewer.bind_key("shift+k", overwrite=True)
    def toggle_space_keypoint(v):
        sp = getattr(data_widget, "space_plot", None)
        if sp is None or not sp.isVisible():
            return
        sp.toggle_keypoint()
