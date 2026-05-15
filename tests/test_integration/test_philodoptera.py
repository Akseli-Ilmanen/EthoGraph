"""Philodoptera integration tests: audio + video + pose."""


class TestPhilodoptera:
    def test_state_after_load(self, philodoptera_gui):
        _, meta = philodoptera_gui
        s = meta.app_state
        assert s.ready is True
        assert s.dt is not None
        assert s.has_audio is True

    def test_audio_and_feature_panels(self, philodoptera_gui):
        _, meta = philodoptera_gui
        pc = meta.plot_container
        assert pc.audio_trace_plot is not None
        assert pc.spectrogram_plot is not None
        assert pc._feature_plot is not None

    def test_features_include_speed(self, philodoptera_gui):
        _, meta = philodoptera_gui
        combo = meta.data_widget.combos.get("features")
        assert combo is not None
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "speed" in items
