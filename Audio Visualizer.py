import sys
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QComboBox, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QSizePolicy, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QPen, QColor
import pyqtgraph as pg
import threading 
import collections 
import subprocess

class BarMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.levels = [0.0, 0.0]  # Left and Right levels, expected range 0.0 to 1.0

    def setLevels(self, left, right):
        # Sets the level for the left and right channels.
        # Clamps the values between 0.0 and 1.0.
        self.levels[0] = max(0.0, min(1.0, left))
        self.levels[1] = max(0.0, min(1.0, right))
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        margin = 3
        label_height = 15 
        bar_area_height = height - label_height
        
        bar_width = (width - 3 * margin) // 2

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRect(margin, margin, bar_width, bar_area_height - 2 * margin)
        painter.drawRect(2 * margin + bar_width, margin, bar_width, bar_area_height - 2 * margin)

        painter.setBrush(QColor(0, 255, 255))
        for i, level in enumerate(self.levels):
            bar_height = int(level * (bar_area_height - 2 * margin))
            x = margin + i * (bar_width + margin)
            y = bar_area_height - margin - bar_height
            painter.drawRect(x, y, bar_width, bar_height)

        painter.setPen(QColor(0, 0, 0))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(margin, bar_area_height, bar_width, label_height, Qt.AlignCenter, "L")
        painter.drawText(2 * margin + bar_width, bar_area_height, bar_width, label_height, Qt.AlignCenter, "R")
        
class Oscilloscope(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer - @peeksxx on TG")
        # PLEASE don't remove my watermark..........I deserve credit...
        self.resize(1200, 950)

        # Audio State
        self.stream = None
        self.lock = threading.Lock()
        
        self.SCROLL_BUFFER_SECONDS = 4
        self.FALLOFF_RATE = 0.05
        self.FFT_AVERAGE_FACTOR = 0.1
        self.BUFFER_SIZES = [512, 1024, 2048, 4096, 8192]

        # Layouts
        main_layout = QHBoxLayout()
        meter_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        control_layout = QHBoxLayout()

        # Bar Meter
        self.meter = BarMeter()
        self.meter.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.meter.setMinimumWidth(100)
        meter_layout.addWidget(self.meter)
        main_layout.addLayout(meter_layout)

        # Controls & Plots
        # Audio Device Dropdown
        self.device_dropdown = QComboBox()
        control_layout.addWidget(QLabel("Audio Device:"))
        control_layout.addWidget(self.device_dropdown)
        
        # Buffer Size Dropdown
        self.buffer_size_dropdown = QComboBox()
        self.buffer_size_dropdown.addItems([str(s) for s in self.BUFFER_SIZES])
        self.buffer_size_dropdown.setCurrentIndex(1) # Default to 1024
        control_layout.addWidget(QLabel("Buffer Size:"))
        control_layout.addWidget(self.buffer_size_dropdown)
        control_layout.addStretch(1)
        right_layout.addLayout(control_layout)

        # Oscilloscope Plot
        self.plot_widget = pg.PlotWidget(title="Real-time Waveform")
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Samples')
        self.plot = self.plot_widget.plot(pen='c')
        right_layout.addWidget(self.plot_widget)
        
        # FFT Plot
        self.fft_plot_widget = pg.PlotWidget(title="Real-time FFT (Logarithmic Frequency)")
        self.fft_plot_widget.setLabel('left', 'Magnitude', units='dB')
        self.fft_plot_widget.setLabel('bottom', 'Frequency', units='kHz')
        self.fft_plot_widget.addLegend()
        self.fft_plot_widget.setLogMode(x=True, y=False)
        self.fft_plot_widget.setYRange(-120, 0)
        self.fft_plot_current = self.fft_plot_widget.plot(pen='y', name='Current FFT')
        self.fft_plot_average = self.fft_plot_widget.plot(pen='r', name='Average FFT')
        right_layout.addWidget(self.fft_plot_widget)

        # Scrolling Waveform Plot
        self.scroll_plot_widget = pg.PlotWidget(title="Scrolling Waveform History")
        self.scroll_plot_widget.setLabel('left', 'Amplitude')
        self.scroll_plot_widget.setLabel('bottom', 'Time', units='s')
        self.scroll_plot = self.scroll_plot_widget.plot(pen='w')
        right_layout.addWidget(self.scroll_plot_widget)

        main_layout.addLayout(right_layout, stretch=1)
        self.setLayout(main_layout)

        # Audio Device Setup
        self.populate_devices()

        # Connections
        self.device_dropdown.currentIndexChanged.connect(self.start_stream)
        self.buffer_size_dropdown.currentIndexChanged.connect(self.on_buffer_size_changed)

        # Metering & Data Buffers
        self.meter_levels = [0.0, 0.0]

        # Final Setup
        # Initialize all audio parameters and data buffers based on the default dropdown values
        self.on_buffer_size_changed()
        

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000 // 60) # ~60 FPS limit

    def populate_devices(self):
        # Finds and populates the audio device dropdown
        host_apis = sd.query_hostapis()
        self.input_devices = [
            d for d in sd.query_devices()
            if d['max_input_channels'] > 0 and host_apis[d['hostapi']]['name'] == 'Windows DirectSound'
        ]

        if not self.input_devices:
            QMessageBox.critical(self, "Error", "No Windows DirectSound input devices found. The application cannot continue.")
            QTimer.singleShot(0, self.close) 
            return

        stereo_mix_index = -1
        for i, device in enumerate(self.input_devices):
            self.device_dropdown.addItem(device['name'])
            if "Stereo Mix" in device['name']:
                stereo_mix_index = i

        if stereo_mix_index != -1:
            self.device_dropdown.setCurrentIndex(stereo_mix_index)
        else:
            self.prompt_enable_stereo_mix()
            print("Stereo Mix not found. Please enable it in sound settings if desired.")

    def prompt_enable_stereo_mix(self):
        # Displays a message box prompting the user to enable Stereo Mix
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Stereo Mix Not Found")
        msg_box.setInformativeText("To capture desktop audio, 'Stereo Mix' is often required.\n"
                                   "Please enable it in your Windows Sound settings.")
        msg_box.setWindowTitle("Information")
        open_settings_button = msg_box.addButton("Open Sound Settings", QMessageBox.ActionRole)
        msg_box.addButton(QMessageBox.Ok)
        
        msg_box.exec_()

        if msg_box.clickedButton() == open_settings_button:
            try:
                subprocess.Popen(['control', 'mmsys.cpl', ',1'])
                self.close() # Close the app after opening settings
            except Exception as e:
                print(f"Could not open sound settings: {e}")
                QMessageBox.warning(self, "Error", f"Could not open sound settings automatically.\nError: {e}")

    def on_buffer_size_changed(self):
        # Handles changes in buffer size, recalculating dependant parameters and restarting the stream
        self.buffer_size = int(self.buffer_size_dropdown.currentText())

        if self.buffer_size == 4096:
            self.samplerate = 22050
        elif self.buffer_size == 8192:
            self.samplerate = 11025
        else:  # For 512, 1024, 2048
            self.samplerate = 44100

        # (Re)initialize all data buffers based on the new parameters
        self.ydata = np.zeros((self.buffer_size, 2))
        self.xdata = np.arange(self.buffer_size)
        
        self.fft_xdata = np.fft.rfftfreq(self.buffer_size, 1/self.samplerate) / 1000
        self.fft_ydata_avg = np.zeros(len(self.fft_xdata))

        scroll_buffer_len = int(self.samplerate * self.SCROLL_BUFFER_SECONDS)
        self.scroll_waveform_deque = collections.deque(maxlen=scroll_buffer_len)
        self.scroll_xdata = np.arange(scroll_buffer_len) / self.samplerate

        self.start_stream()

    def _stop_stream(self):
        # Safely stops and closes the current audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def start_stream(self):
        # Starts a new audio stream with the current settings
        self._stop_stream()
        
        if not self.input_devices: # Don't try to start if no devices were found
             return
        
        device_info = self.input_devices[self.device_dropdown.currentIndex()]
        
        # Use the lesser of 2 or the device's max input channels, with a minimum of 1
        input_channels = max(1, min(2, device_info['max_input_channels']))

        try:
            self.stream = sd.InputStream(
                device=device_info['index'],
                channels=input_channels,
                samplerate=self.samplerate,
                callback=self.audio_callback,
                blocksize=self.buffer_size
            )
            self.stream.start()
            print(f"Stream started: {device_info['name']} ({input_channels}ch @ {self.samplerate}Hz)")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            QMessageBox.critical(self, "Audio Stream Error", f"Failed to start stream with '{device_info['name']}':\n{e}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        with self.lock:
            if indata.shape[1] == 1:  # Mono
                # Duplicate mono data into both channels for consistent processing
                self.ydata[:, 0] = self.ydata[:, 1] = indata[:, 0]
            else:  # Stereo
                self.ydata[:] = indata

            # Add left channel to the scrolling waveform buffer
            self.scroll_waveform_deque.extend(indata[:, 0])

    def update_plot(self):
        # Periodically called by the QTimer to update the GUI
        with self.lock:
            # Copy data to avoid issues if audio_callback modifies it during the plot update
            ydata_safe = self.ydata.copy()

        # Waveform plot (left channel)
        self.plot.setData(self.xdata, ydata_safe[:, 0])
        
        # VU Meter
        peak_left = np.max(np.abs(ydata_safe[:, 0]))
        peak_right = np.max(np.abs(ydata_safe[:, 1]))
        
        # Apply falloff and update with new peak
        self.meter_levels[0] = max(peak_left, self.meter_levels[0] - self.FALLOFF_RATE)
        self.meter_levels[1] = max(peak_right, self.meter_levels[1] - self.FALLOFF_RATE)
        self.meter.setLevels(self.meter_levels[0], self.meter_levels[1])

        # FFT Plot (left channel)
        fft_data = np.fft.rfft(ydata_safe[:, 0] * np.hanning(self.buffer_size))
        fft_magnitude = np.abs(fft_data) * 2 / self.buffer_size
        db_fft_magnitude = 20 * np.log10(fft_magnitude + 1e-10) # Add epsilon to avoid log(0)
        
        self.fft_plot_current.setData(self.fft_xdata, db_fft_magnitude)
        
        # Update running average for FFT
        self.fft_ydata_avg = (self.FFT_AVERAGE_FACTOR * db_fft_magnitude) + \
                             ((1 - self.FFT_AVERAGE_FACTOR) * self.fft_ydata_avg)
        self.fft_plot_average.setData(self.fft_xdata, self.fft_ydata_avg)

        # Scrolling Waveform Plot
        scroll_data_safe = np.array(self.scroll_waveform_deque)
        self.scroll_plot.setData(self.scroll_xdata[:len(scroll_data_safe)], scroll_data_safe)
            
    def closeEvent(self, event):
        # Ensures the audio stream is properly closed
        self._stop_stream()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    osc = Oscilloscope()
    osc.show()
    sys.exit(app.exec_())