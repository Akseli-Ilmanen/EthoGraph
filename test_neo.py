import neo

reader = neo.io.IntanIO(r"C:\Users\aksel\Desktop\Freddy_all\ses-000_date-20250527_02_raw\ephys\Freddy_250527_160530\info.rhd")
block = reader.read_block()
segment = block.segments[0]
analog_signal = segment.analogsignals[0]
# Only plot first 10,000 samples
num_samples = 10000
times = analog_signal.times[:num_samples]
data = analog_signal[:num_samples, 0]  # This line is retained for backward compatibility
print(data.shape)
num_channels = analog_signal.shape[1]
print(f"Shape: {analog_signal.shape}, Channels: {num_channels}")
channels_to_plot = 4
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt import QtCore
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Intan .rhd Ephys Trace")

# Create plots for 4 channels, hide axes, add offset
plots = []
offset = 5000  # Adjust for your signal scale
for ch in range(channels_to_plot):
    plot = win.addPlot()
    plot.hideAxis('left')
    plot.hideAxis('bottom')
    plots.append(plot)
    win.nextRow()

# Add slider
slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
slider.setMinimum(0)
slider.setMaximum(analog_signal.shape[0] - num_samples)
slider.setValue(0)
slider.setSingleStep(num_samples // 10)
slider.setPageStep(num_samples)
slider.setTickInterval(num_samples // 10)
slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

# Embed slider in a window
slider_win = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()
layout.addWidget(win)
layout.addWidget(slider)
slider_win.setLayout(layout)
slider_win.setWindowTitle("Intan .rhd Ephys Trace with Slider")
slider_win.setStyleSheet("background-color: black;")
slider_win.show()

def update_plots(pos):
    start = pos
    end = start + num_samples
    times = analog_signal.times[start:end]
    for ch, plot in enumerate(plots):
        plot.clear()
        data_ch = analog_signal[start:end, ch].flatten()
        data_ch_offset = data_ch + ch * offset
        plot.plot(times, data_ch_offset, pen=pg.mkPen('w'))

slider.valueChanged.connect(update_plots)
update_plots(0)

QtWidgets.QApplication.instance().exec_()