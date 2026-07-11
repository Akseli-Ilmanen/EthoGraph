"""Global keyboard shortcut bindings for the ethograph GUI.

Shortcuts are QShortcuts on the main window (napari keymaps are gone).
Plain-letter shortcuts are guarded so they don't fire while the user is
typing in a text field or spin box.
"""

import logging

from qtpy.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QPlainTextEdit, QTextEdit

logger = logging.getLogger(__name__)

_TEXT_WIDGETS = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)


def _typing_in_text_field() -> bool:
    widget = QApplication.focusWidget()
    if widget is None:
        return False
    if isinstance(widget, _TEXT_WIDGETS):
        return True
    if isinstance(widget, QComboBox) and widget.isEditable():
        return True
    return False


def bind_global_shortcuts(meta_widget):
    shell = meta_widget.shell
    app_state = meta_widget.app_state
    labels_widget = meta_widget.labels_widget
    data_widget = meta_widget.data_widget
    navigation_widget = meta_widget.navigation_widget
    plot_settings_widget = meta_widget.plot_settings_widget
    changepoints_widget = meta_widget.changepoints_widget
    io_widget = meta_widget.io_widget
    plot_container = meta_widget.plot_container

    shell.clear_shortcuts()

    def bind(key, callback, guarded=False):
        """Bind *key*; guarded shortcuts are ignored while typing in a field."""

        def run():
            if guarded and _typing_in_text_field():
                return
            callback()

        shell.bind_shortcut(key, run)

    # --- Playback / navigation ---
    bind("Ctrl+S", app_state.save_labels)

    def toggle_pause_resume():
        data_widget.toggle_pause_resume()
        navigation_widget._sync_play_icon()

    bind("Space", toggle_pause_resume, guarded=True)
    bind("V", labels_widget._play_segment, guarded=True)
    bind("Shift+Left", navigation_widget.step_window_backward)
    bind("Shift+Right", navigation_widget.step_window_forward)
    bind("Left", navigation_widget.step_frame_backward, guarded=True)
    bind("Right", navigation_widget.step_frame_forward, guarded=True)
    bind("Down", navigation_widget.next_trial, guarded=True)
    bind("Up", navigation_widget.prev_trial, guarded=True)
    bind("Ctrl+Down", lambda: meta_widget._cycle_channel(+1))
    bind("Ctrl+Up", lambda: meta_widget._cycle_channel(-1))

    def toggle_sync():
        btn = getattr(navigation_widget, "sync_toggle_btn", None)
        if btn is None:
            return
        next_index = (btn.currentIndex() + 1) % btn.count()
        btn.setCurrentIndex(next_index)

    bind("Ctrl+P", toggle_sync)
    bind("Ctrl+Y", data_widget.toggle_predictions_slot)

    def toggle_autoscale():
        autoscale_status = plot_settings_widget.autoscale_checkbox.isChecked()
        plot_settings_widget.autoscale_checkbox.setChecked(not autoscale_status)

    bind("Ctrl+A", toggle_autoscale)

    def toggle_lock():
        lock_status = plot_settings_widget.lock_axes_checkbox.isChecked()
        plot_settings_widget.lock_axes_checkbox.setChecked(not lock_status)

    bind("Ctrl+L", toggle_lock)
    bind("Ctrl+Return", plot_settings_widget.apply_button.click)
    bind("Ctrl+V", lambda: io_widget._human_verification_true(mode="single_trial"))

    def toggle_changepoint_correction():
        checkbox = changepoints_widget.changepoint_correction_checkbox
        checkbox.setChecked(not checkbox.isChecked())

    bind("Ctrl+B", toggle_changepoint_correction)
    bind("Ctrl+R", data_widget.refresh_lineplot)

    def _change_spacing(delta: float):
        pc = plot_container
        if pc and pc.is_ephystrace():
            buf = pc.ephys_trace_plot.buffer
            buf.channel_spacing = min(max(buf.channel_spacing + delta, 0.5), 20.0)
            xmin, xmax = pc.get_current_xlim()
            pc.ephys_trace_plot.update_plot_content(xmin, xmax)

    bind("Ctrl+=", lambda: _change_spacing(+0.5))
    bind("Ctrl+-", lambda: _change_spacing(-0.5))

    def _jump_spike(delta: int):
        if not plot_container or not plot_container.is_ephystrace():
            return
        plot_container.ephys_trace_plot.jump_to_spike(delta)

    bind("Alt+Right", lambda: _jump_spike(+1))
    bind("Alt+Left", lambda: _jump_spike(-1))

    def stop_recording():
        record_btn = getattr(navigation_widget, "record_button", None)
        if record_btn is not None:
            record_btn._stop_recording()

    bind("Ctrl+Space", stop_recording)

    # --- Label activation grid layout ---
    number_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    qwerty_row = ["Q", "W", "E", "R", "T", "Z", "U", "I", "O", "P"]
    home_row = ["A", "S", "D", "F", "G", "H", "J", "K", "L"]

    for i, key in enumerate(number_keys):
        labels = i + 1 if key != "0" else 10
        bind(key, lambda mk=labels: labels_widget.activate_label(mk), guarded=True)

    for i, key in enumerate(qwerty_row):
        bind(key, lambda mk=i + 11: labels_widget.activate_label(mk), guarded=True)

    for i, key in enumerate(home_row):
        bind(key, lambda mk=i + 21: labels_widget.activate_label(mk), guarded=True)

    bind("Ctrl+E", labels_widget._edit_label)
    bind("Ctrl+D", labels_widget._delete_label)
    bind("Shift+B", labels_widget.toggle_branch, guarded=True)
    bind("Ctrl+F", lambda: app_state.toggle_key_sel("features", data_widget))
    bind("Ctrl+I", lambda: app_state.toggle_key_sel("individual", data_widget))
    bind("Ctrl+K", lambda: app_state.toggle_key_sel("keypoint", data_widget))

    def cycle_cameras():
        combo = getattr(data_widget, "primary_camera_combo", None)
        if combo is not None and combo.count() > 1:
            next_index = (combo.currentIndex() + 1) % combo.count()
            combo.setCurrentIndex(next_index)

    bind("Ctrl+C", cycle_cameras)
    bind("Ctrl+M", lambda: app_state.toggle_key_sel("mics", data_widget))
    bind("Ctrl+H", data_widget.cycle_neural_view)
    bind("Ctrl+G", data_widget.cycle_view_mode)

    bind("Ctrl+Right", lambda: changepoints_widget.jump_changepoint(+1))
    bind("Ctrl+Left", lambda: changepoints_widget.jump_changepoint(-1))

    def toggle_space_keypoint():
        sp = getattr(data_widget, "space_plot", None)
        if sp is None or not sp.isVisible():
            return
        sp.toggle_keypoint()

    bind("Shift+K", toggle_space_keypoint, guarded=True)
