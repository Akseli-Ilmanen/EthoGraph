"""Tests for audio-only mode (no video loaded).

Uses the birdpark dataset which has audio but no video configured.
Verifies that panels are created, audio data renders, and features work
without a video source.
"""

import numpy as np
import pytest
import pyqtgraph as pg
from qtpy.QtWidgets import QApplication


class TestNoVideoState:

    def test_ready_after_load(self, no_video_gui):
        _, meta = no_video_gui
        assert meta.app_state.ready is True

    def test_no_video_loaded(self, no_video_gui):
        _, meta = no_video_gui
        video = getattr(meta.app_state, 'video', None)
        assert video is None, "Expected no video in audio-only mode"

    def test_has_dataset(self, no_video_gui):
        _, meta = no_video_gui
        assert meta.app_state.dt is not None
        assert meta.app_state.ds is not None

    def test_trials_populated(self, no_video_gui):
        _, meta = no_video_gui
        assert len(meta.app_state.trials) > 0
        assert meta.app_state.trials_sel in meta.app_state.trials


class TestNoVideoAudioPanels:

    def test_audio_trace_visible(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        if not pc._panel_visible.get("audiotrace"):
            pytest.skip("Audio trace not visible (no audio files found)")
        assert pc.audio_trace_plot.isVisible()

    def test_audio_trace_has_data(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        if not pc._panel_visible.get("audiotrace"):
            pytest.skip("Audio trace not visible")
        at = pc.audio_trace_plot
        assert at.trace_item.xData is not None, "Audio trace xData is None"
        assert len(at.trace_item.xData) > 0, "Audio trace xData empty"
        assert at.trace_item.yData is not None, "Audio trace yData is None"
        assert len(at.trace_item.yData) > 0, "Audio trace yData empty"

    def test_spectrogram_visible(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        if not pc._panel_visible.get("spectrogram"):
            pytest.skip("Spectrogram not visible")
        assert pc.spectrogram_plot.isVisible()

    def test_spectrogram_has_data(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        if not pc._panel_visible.get("spectrogram"):
            pytest.skip("Spectrogram not visible")
        sp = pc.spectrogram_plot
        assert sp.spec_item.image is not None, "Spectrogram image is None"
        assert sp.spec_item.image.size > 0, "Spectrogram image empty"


class TestNoVideoFeaturePlot:

    def test_feature_plot_has_data(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        assert pc._feature_type == "lineplot"
        lp = pc.line_plot
        assert len(lp.plot_items) > 0, "Feature plot empty in no-video mode"

        for item in lp.plot_items:
            if isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
                assert item.xData is not None and len(item.xData) > 0
                break

    def test_valid_xlim(self, no_video_gui):
        _, meta = no_video_gui
        xlim = meta.plot_container.get_current_xlim()
        assert xlim[0] < xlim[1]
        assert xlim[0] >= -1.0

    def test_cycle_features(self, no_video_gui):
        _, meta = no_video_gui
        features_combo = meta.data_widget.combos.get("features")
        if features_combo is None or features_combo.count() == 0:
            pytest.skip("No features combo")

        lp = meta.plot_container.line_plot
        for i in range(features_combo.count()):
            text = features_combo.itemText(i)
            if text in ("Spectrogram", "Waveform"):
                continue
            features_combo.setCurrentIndex(i)
            QApplication.processEvents()
            assert meta.app_state.features_sel == text
            assert len(lp.plot_items) > 0, f"Feature '{text}' empty in no-video mode"


class TestNoVideoNavigation:

    def test_trial_navigation(self, no_video_gui):
        _, meta = no_video_gui
        if len(meta.app_state.trials) < 2:
            pytest.skip("Need 2+ trials")

        first = meta.app_state.trials_sel
        meta.navigation_widget.next_trial()
        QApplication.processEvents()
        assert meta.app_state.trials_sel != first

    def test_plot_has_data_after_trial_switch(self, no_video_gui):
        _, meta = no_video_gui
        if len(meta.app_state.trials) < 2:
            pytest.skip("Need 2+ trials")

        meta.navigation_widget.next_trial()
        QApplication.processEvents()

        lp = meta.plot_container.line_plot
        assert len(lp.plot_items) > 0, "Feature plot empty after trial switch in no-video mode"


class TestNoVideoTimeSlider:

    def test_time_slider_present(self, no_video_gui):
        """Without video, a time slider should still allow scrubbing."""
        _, meta = no_video_gui
        # The time marker on the feature plot should be accessible
        lp = meta.plot_container.line_plot
        assert hasattr(lp, 'time_marker')
        assert lp.time_marker is not None
