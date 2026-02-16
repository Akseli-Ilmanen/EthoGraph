"""Constants used across the GUI module, that should be rarely modified by the user."""

# =============================================================================
# UI DIMENSIONS
# =============================================================================

# Labels table (widgets_labels.py)
LABELS_TABLE_MAX_HEIGHT = 120
LABELS_TABLE_ROW_HEIGHT = 18
LABELS_TABLE_ID_COLUMN_WIDTH = 20
LABELS_TABLE_COLOR_COLUMN_WIDTH = 20

# Label overlay box on video (widgets_labels.py)
LABELS_OVERLAY_BOX_WIDTH = 180
LABELS_OVERLAY_BOX_HEIGHT = 50
LABELS_OVERLAY_BOX_MARGIN = 5
LABELS_OVERLAY_TEXT_SIZE = 18
LABELS_OVERLAY_FALLBACK_SIZE = (100, 100)

# Plot container (plot_container.py, widgets_meta.py)
PLOT_CONTAINER_MIN_HEIGHT = 250
PLOT_CONTAINER_SIZE_HINT_HEIGHT = 300
DOCK_WIDGET_BOTTOM_MARGIN = 50

# Layout spacing (widgets_data.py, widgets_labels.py)
DEFAULT_LAYOUT_SPACING = 2
DEFAULT_LAYOUT_MARGIN = 2

# =============================================================================
# PLOT SETTINGS
# =============================================================================

# Axis locking (plots_base.py)
LOCKED_RANGE_MIN_FACTOR = 0.8  # window_size * 0.8 when locked
LOCKED_RANGE_MAX_FACTOR = 1.5  # window_size * 1.5 when locked
AXIS_LIMIT_PADDING_RATIO = 0.05  # 5% of data range as padding for xMin/xMax

# Label drawing (plot_container.py)
PREDICTION_LABELS_HEIGHT_RATIO = 0.10  # Height as fraction of y-range
SPECTROGRAM_LABELS_HEIGHT_RATIO = 0.10  # Height as fraction of y-range
SPECTROGRAM_OVERLAY_OPACITY = 0.6
PREDICTION_FALLBACK_Y_TOP = 20000
PREDICTION_FALLBACK_Y_HEIGHT = 2000
SPECTROGRAM_FALLBACK_Y_HEIGHT = 1600

# Zoom thresholds for spectrogram overlay refresh (plot_container.py)
SPECTROGRAM_OVERLAY_ZOOM_OUT_THRESHOLD = 0.5  # Refresh when width < old * 0.5
SPECTROGRAM_OVERLAY_ZOOM_IN_THRESHOLD = 2.0   # Refresh when width > old * 2.0

# Changepoint line styles based on zoom level (plot_container.py)
CP_ZOOM_VERY_OUT_THRESHOLD = 10.0  # seconds visible
CP_ZOOM_MEDIUM_THRESHOLD = 2.0     # seconds visible
CP_LINE_WIDTH_THIN = 0.1
CP_LINE_WIDTH_MEDIUM = 1.0
CP_LINE_WIDTH_THICK = 2.0

# =============================================================================
# TIMING / DEBOUNCE
# =============================================================================

SPECTROGRAM_DEBOUNCE_MS = 50
SPECTROGRAM_OVERLAY_DEBOUNCE_MS = 100
ENVELOPE_OVERLAY_DEBOUNCE_MS = 100

# =============================================================================
# AUDIO / SPECTROGRAM
# =============================================================================

# Buffer settings (plots_spectrogram.py)
DEFAULT_BUFFER_MULTIPLIER = 5.0
BUFFER_COVERAGE_MARGIN = 0.1  # 10% margin for buffer coverage check

# Frequency limits (plots_spectrogram.py)
DEFAULT_FALLBACK_MAX_FREQUENCY = 25000  # Hz, fallback when audio not loaded

# =============================================================================
# DATA PROCESSING
# =============================================================================
MIN_LABEL_LEN = 2 ## For hand labelling


# Z-index values for layering
Z_INDEX_BACKGROUND = -20
Z_INDEX_LABELS = -10
Z_INDEX_PREDICTIONS = 10
Z_INDEX_CHANGEPOINTS = 50
Z_INDEX_TIME_MARKER = 1000
Z_INDEX_LABELS_OVERLAY = 1000

# =============================================================================
# COLORS (RGBA tuples)
# =============================================================================

# Changepoint colors (plot_container.py)
CP_COLOR_WAVEFORM = (0, 0, 0, 200)      # Black for waveform plot
CP_COLOR_SPECTROGRAM = (255, 255, 255, 200)  # White for spectrogram

# Dataset changepoint method colors
CP_METHOD_COLORS = {
    'troughs': (100, 100, 255, 200),        # Blue
    'turning_points': (100, 255, 100, 200), # Green
    'ruptures': (255, 165, 0, 200),         # Orange
    'default': (200, 200, 200, 200),        # Gray fallback
}

# Scatter plot settings
CP_SCATTER_SIZE = 8
CP_SCATTER_Y_POSITION_RATIO = 0.05  # 5% from bottom of y-range

# Envelope overlay (plot_container.py)
ENVELOPE_OVERLAY_COLOR = (255, 165, 0, 120)
ENVELOPE_OVERLAY_WIDTH = 2
