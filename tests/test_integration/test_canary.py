"""Canary integration tests: audio-only dataset."""


class TestCanary:
    def test_state_after_load(self, canary_gui):
        _, meta = canary_gui
        s = meta.app_state
        assert s.ready is True
        assert s.dt is not None
        assert s.ds is not None
        assert s.has_audio is True

    def test_no_video(self, canary_gui):
        _, meta = canary_gui
        assert meta.app_state.video is None

    def test_audio_panels_created(self, canary_gui):
        _, meta = canary_gui
        pc = meta.plot_container
        assert pc.audio_trace_plot is not None
        assert pc.spectrogram_plot is not None

    def test_valid_xlim(self, canary_gui):
        _, meta = canary_gui
        xlim = meta.plot_container.get_current_xlim()
        assert xlim[0] < xlim[1]
        assert xlim[0] >= -1.0
