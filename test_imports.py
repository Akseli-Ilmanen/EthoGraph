"""Verify full import chain after modality migration."""
from ethograph.gui.modality import FileSource, WindowedBuffer, XarraySource, SourceData
from ethograph.gui.plots_audiotrace import AudioTracePlot
from ethograph.gui.plots_spectrogram import SpectrogramPlot, SpectrogramBuffer
from ethograph.gui.data_sources import build_audio_source
from ethograph.gui.plots_ephystrace import EphysTracePlot, EphysTraceBuffer
from ethograph.gui.plots_lineplot import LinePlot
from ethograph.gui.plots_heatmap import HeatmapPlot
from ethograph.gui.plots_container import UnifiedPanelContainer
print("ALL IMPORTS OK")
