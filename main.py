import sys
import numpy as np
import cv2
import sounddevice as sd
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QFontDatabase, QFont, QColor
import pyqtgraph as pg
from scipy import signal

# Global theme config that can be modified
GLOBAL_THEME = {
    "bg": QColor("#050505"),
    "bg2": QColor("#0a0a0a"),
    "text": QColor("#00ff00"),
    "btn_bg": QColor("#001a00"),
    "btn_hover": QColor("#004400"),
    "grid": QColor("#333333"),
    "sig_main": QColor("#00ff00"), 
    "sig_sec": QColor("#00aa00"),  
    "sig_res": QColor("#00ff00"),  
}
# All available signal types
SIGNAL_TYPES = [
    "Seno", "Coseno", "Cuadrada", "Triangular", "Diente sierra",
    "Sinc (Seno Cardinal)", "Exponencial", "Logarítmica",
    "Delta", "Impulso Cuadrado"
]

class SignalObject:
    def __init__(self, type='Seno', freq=5.0, amp=1.0, shift=0.0):
        self.type = type
        self.freq = freq
        self.amp = amp
        self.shift = shift
        self.active = True
        # Time limits
        self.limits_enabled = False
        self.t_min = 0.0
        self.t_max = 1.0

    def get_data(self, t):
        if not self.active: return np.zeros_like(t)
        
        # Apply time shift
        t_shifted = t - self.shift
        
        # Generate the raw signal
        data = self._generate_raw(t_shifted, t)
        
        # Apply time limits if enabled (zero out outside [t_min, t_max])
        if self.limits_enabled:
            mask = (t < self.t_min) | (t > self.t_max)
            data[mask] = 0.0
        
        return data

    def get_data_unlimited(self, t):
        """Return signal data WITHOUT applying time limits (for ghost curve)."""
        if not self.active: return np.zeros_like(t)
        t_shifted = t - self.shift
        return self._generate_raw(t_shifted, t)

    def _generate_raw(self, t_shifted, t):
        if self.type == 'Seno':
            return self.amp * np.sin(2 * np.pi * self.freq * t_shifted)
        if self.type == 'Coseno':
            return self.amp * np.cos(2 * np.pi * self.freq * t_shifted)
        if self.type == 'Cuadrada':
            return self.amp * signal.square(2 * np.pi * self.freq * t_shifted)
        if self.type == 'Triangular':
            return self.amp * signal.sawtooth(2 * np.pi * self.freq * t_shifted, 0.5)
        if self.type == 'Diente sierra':
            return self.amp * signal.sawtooth(2 * np.pi * self.freq * t_shifted)
        if self.type == 'Sinc (Seno Cardinal)':
            # np.sinc(x) = sin(pi*x)/(pi*x), so we scale by freq
            return self.amp * np.sinc(self.freq * t_shifted)
        if self.type == 'Exponencial':
            # Decaying exponential: amp * e^(-freq * |t|)
            return self.amp * np.exp(-self.freq * np.abs(t_shifted))
        if self.type == 'Logarítmica':
            # amp * sign(t) * log(1 + freq*|t|) for a symmetric log shape
            return self.amp * np.sign(t_shifted) * np.log1p(self.freq * np.abs(t_shifted))
        if self.type == 'Delta':
            data = np.zeros_like(t)
            idx = np.argmin(np.abs(t_shifted))
            data[idx] = self.amp
            return data
        if self.type == 'Impulso Cuadrado':
            data = np.zeros_like(t)
            width = 1.0 / (2.0 * max(self.freq, 0.001))
            mask = np.abs(t_shifted) < (width / 2.0)
            data[mask] = self.amp
            return data
            
        return np.zeros_like(t)

class SignalControlWidget(QtWidgets.QFrame):
    changed = QtCore.pyqtSignal()
    removed = QtCore.pyqtSignal(object)

    def __init__(self, signal_obj, disable_remove=False):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.sig = signal_obj
        layout = QtWidgets.QVBoxLayout(self)

        # Tipo
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(SIGNAL_TYPES)
        self.combo.setCurrentText(self.sig.type)
        self.combo.currentTextChanged.connect(self.update_params)
        layout.addWidget(QtWidgets.QLabel("Tipo de Onda:"))
        layout.addWidget(self.combo)

        # Helper to create slider + double spinbox
        def create_slider_spinbox(label_text, min_val, max_val, step, current_val, callback):
            lbl = QtWidgets.QLabel(label_text)
            layout.addWidget(lbl)
            
            row = QtWidgets.QHBoxLayout()
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            spin = QtWidgets.QDoubleSpinBox()
            
            # Use a factor to map float to int slider (100 steps per unit)
            factor = 100.0
            slider.setRange(int(min_val * factor), int(max_val * factor))
            slider.setValue(int(current_val * factor))
            
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(current_val)
            
            # Connect them
            def on_slider(val):
                spin.blockSignals(True)
                spin.setValue(val / factor)
                spin.blockSignals(False)
                callback()
                
            def on_spin(val):
                slider.blockSignals(True)
                slider.setValue(int(val * factor))
                slider.blockSignals(False)
                callback()
                
            slider.valueChanged.connect(on_slider)
            spin.valueChanged.connect(on_spin)
            
            row.addWidget(slider)
            row.addWidget(spin)
            layout.addLayout(row)
            return slider, spin

        self.f_slider, self.f_spin = create_slider_spinbox("Frecuencia (Hz):", 0.1, 100.0, 1.0, self.sig.freq, self.update_params)
        self.a_slider, self.a_spin = create_slider_spinbox("Amplitud:", 0.1, 10.0, 0.1, self.sig.amp, self.update_params)
        self.s_slider, self.s_spin = create_slider_spinbox("Desplazamiento (s):", -2.0, 2.0, 0.1, self.sig.shift, self.update_params)

        # --- Time Limits Section ---
        self.chk_limits = QtWidgets.QCheckBox("Habilitar Límites de Tiempo")
        self.chk_limits.setChecked(self.sig.limits_enabled)
        self.chk_limits.stateChanged.connect(self.update_params)
        layout.addWidget(self.chk_limits)
        
        self.limits_container = QtWidgets.QWidget()
        limits_layout = QtWidgets.QVBoxLayout(self.limits_container)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        
        # t_min
        row_min = QtWidgets.QHBoxLayout()
        row_min.addWidget(QtWidgets.QLabel("t mín:"))
        self.spin_tmin = QtWidgets.QDoubleSpinBox()
        self.spin_tmin.setRange(-10.0, 10.0)
        self.spin_tmin.setSingleStep(0.05)
        self.spin_tmin.setValue(self.sig.t_min)
        self.spin_tmin.valueChanged.connect(self.update_params)
        row_min.addWidget(self.spin_tmin)
        
        # t_max
        row_min.addWidget(QtWidgets.QLabel("t máx:"))
        self.spin_tmax = QtWidgets.QDoubleSpinBox()
        self.spin_tmax.setRange(-10.0, 10.0)
        self.spin_tmax.setSingleStep(0.05)
        self.spin_tmax.setValue(self.sig.t_max)
        self.spin_tmax.valueChanged.connect(self.update_params)
        row_min.addWidget(self.spin_tmax)
        
        limits_layout.addLayout(row_min)
        layout.addWidget(self.limits_container)
        self.limits_container.setVisible(self.sig.limits_enabled)

        if not disable_remove:
            btn_del = QtWidgets.QPushButton("Eliminar")
            btn_del.setStyleSheet("background-color: #330000; color: #ff3333; border: 1px solid #ff3333;")
            btn_del.clicked.connect(lambda: self.removed.emit(self))
            layout.addWidget(btn_del)

    def update_params(self):
        self.sig.type = self.combo.currentText()
        self.sig.freq = self.f_spin.value()
        self.sig.amp = self.a_spin.value()
        self.sig.shift = self.s_spin.value()
        self.sig.limits_enabled = self.chk_limits.isChecked()
        self.sig.t_min = self.spin_tmin.value()
        self.sig.t_max = self.spin_tmax.value()
        self.limits_container.setVisible(self.sig.limits_enabled)
        self.changed.emit()

class FFTTabWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.fs = 1000  
        self.t = np.linspace(0, 2, self.fs * 2, endpoint=False) # 2 seconds
        self.signals = [] 

        layout = QtWidgets.QHBoxLayout(self)

        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(340)
        self.side_layout = QtWidgets.QVBoxLayout(sidebar)
        btn_add = QtWidgets.QPushButton("+ Añadir Señal")
        btn_add.clicked.connect(self.add_signal)
        self.side_layout.addWidget(btn_add)
        
        btn_play = QtWidgets.QPushButton("🔊 Reproducir Audio")
        btn_play.clicked.connect(self.play_audio)
        self.side_layout.addWidget(btn_play)
        
        btn_stop = QtWidgets.QPushButton("⏹ Detener Audio")
        btn_stop.clicked.connect(self.stop_audio)
        self.side_layout.addWidget(btn_stop)

        # ── PAM Modulation Section ──
        pam_frame = QtWidgets.QFrame()
        pam_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        pam_layout = QtWidgets.QVBoxLayout(pam_frame)

        self.chk_pam = QtWidgets.QCheckBox("Habilitar Modulación PAM")
        self.chk_pam.stateChanged.connect(self.update_plots)
        pam_layout.addWidget(self.chk_pam)

        self.pam_container = QtWidgets.QWidget()
        pam_inner = QtWidgets.QVBoxLayout(self.pam_container)
        pam_inner.setContentsMargins(0, 0, 0, 0)

        pam_inner.addWidget(QtWidgets.QLabel("Tipo de PAM:"))
        self.combo_pam = QtWidgets.QComboBox()
        self.combo_pam.addItems(["PAM Natural", "PAM Instantánea"])
        self.combo_pam.currentTextChanged.connect(self.update_plots)
        pam_inner.addWidget(self.combo_pam)

        pam_inner.addWidget(QtWidgets.QLabel("Frecuencia pulsos (Hz):"))
        row_pf = QtWidgets.QHBoxLayout()
        self.pam_freq_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pam_freq_slider.setRange(1, 200)
        self.pam_freq_slider.setValue(20)
        self.pam_freq_spin = QtWidgets.QDoubleSpinBox()
        self.pam_freq_spin.setRange(1, 200)
        self.pam_freq_spin.setValue(20)
        self.pam_freq_spin.setSingleStep(1)
        self.pam_freq_slider.valueChanged.connect(lambda v: (self.pam_freq_spin.blockSignals(True), self.pam_freq_spin.setValue(v), self.pam_freq_spin.blockSignals(False), self.update_plots()))
        self.pam_freq_spin.valueChanged.connect(lambda v: (self.pam_freq_slider.blockSignals(True), self.pam_freq_slider.setValue(int(v)), self.pam_freq_slider.blockSignals(False), self.update_plots()))
        row_pf.addWidget(self.pam_freq_slider)
        row_pf.addWidget(self.pam_freq_spin)
        pam_inner.addLayout(row_pf)

        pam_inner.addWidget(QtWidgets.QLabel("Ciclo de trabajo (%):"))
        row_pd = QtWidgets.QHBoxLayout()
        self.pam_duty_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pam_duty_slider.setRange(1, 99)
        self.pam_duty_slider.setValue(30)
        self.pam_duty_spin = QtWidgets.QDoubleSpinBox()
        self.pam_duty_spin.setRange(1, 99)
        self.pam_duty_spin.setValue(30)
        self.pam_duty_spin.setSingleStep(1)
        self.pam_duty_spin.setSuffix(" %")
        self.pam_duty_slider.valueChanged.connect(lambda v: (self.pam_duty_spin.blockSignals(True), self.pam_duty_spin.setValue(v), self.pam_duty_spin.blockSignals(False), self.update_plots()))
        self.pam_duty_spin.valueChanged.connect(lambda v: (self.pam_duty_slider.blockSignals(True), self.pam_duty_slider.setValue(int(v)), self.pam_duty_slider.blockSignals(False), self.update_plots()))
        row_pd.addWidget(self.pam_duty_slider)
        row_pd.addWidget(self.pam_duty_spin)
        pam_inner.addLayout(row_pd)

        pam_layout.addWidget(self.pam_container)
        self.pam_container.setVisible(False)
        self.chk_pam.stateChanged.connect(lambda st: self.pam_container.setVisible(st != 0))
        self.side_layout.addWidget(pam_frame)
        # ── End PAM Section ──

        self.scroll = QtWidgets.QScrollArea()
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.scroll_content)
        self.side_layout.addWidget(self.scroll)

        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground(GLOBAL_THEME['bg'])

        self.p1 = self.win.addPlot(title="Señal Resultante (Tiempo)")
        self.win.nextRow()
        self.p2 = self.win.addPlot(title="FFT (Frecuencia)")
        self.p1.showGrid(x=True, y=True)
        self.p2.showGrid(x=True, y=True)
        self.p2.setLabel('bottom', 'Frecuencia', units='Hz')

        # Ghost curve (unlimited signal) — dashed, semi-transparent
        ghost_pen = pg.mkPen(color=(150, 150, 150, 100), width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.curve_t_ghost = self.p1.plot(pen=ghost_pen)

        self.curve_t = self.p1.plot(pen=pg.mkPen(GLOBAL_THEME['sig_main'], width=2))

        # PAM modulated curve — distinct color
        pam_pen = pg.mkPen(color='#00e5ff', width=2)
        self.curve_t_pam = self.p1.plot(pen=pam_pen)

        self.curve_f = self.p2.plot(pen=pg.mkPen(GLOBAL_THEME['sig_main'], width=2))

        layout.addWidget(sidebar)
        layout.addWidget(self.win)

    def apply_colors(self):
        self.win.setBackground(GLOBAL_THEME['bg'])
        self.p1.getAxis('bottom').setPen(pg.mkPen(GLOBAL_THEME['text']))
        self.p1.getAxis('left').setPen(pg.mkPen(GLOBAL_THEME['text']))
        self.p1.setTitle("Señal Resultante (Tiempo)", color=GLOBAL_THEME['text'].name())
        
        self.p2.getAxis('bottom').setPen(pg.mkPen(GLOBAL_THEME['text']))
        self.p2.getAxis('left').setPen(pg.mkPen(GLOBAL_THEME['text']))
        self.p2.setTitle("FFT (Frecuencia)", color=GLOBAL_THEME['text'].name())

        # update grids
        for plot in [self.p1, self.p2]:
            grid_pen = pg.mkPen(GLOBAL_THEME['grid'], width=1)
            plot.getAxis('bottom').setGrid(150) # enable
            plot.getAxis('left').setGrid(150)
            plot.getAxis('bottom').setPen(pg.mkPen(GLOBAL_THEME['text']))
            
        self.update_plots()

    def add_signal(self):
        new_sig = SignalObject()
        self.signals.append(new_sig)
        
        control = SignalControlWidget(new_sig)
        control.changed.connect(self.update_plots)
        control.removed.connect(self.remove_signal)
        
        self.scroll_layout.addWidget(control)
        self.update_plots()

    def remove_signal(self, widget):
        if widget.sig in self.signals:
            self.signals.remove(widget.sig)
        widget.deleteLater()
        QtCore.QTimer.singleShot(10, self.update_plots)

    def _generate_pam(self, total_y):
        """Apply PAM modulation to the total signal."""
        pam_freq = self.pam_freq_spin.value()
        duty = self.pam_duty_spin.value() / 100.0  # fraction
        pam_type = self.combo_pam.currentText()

        # Period of the pulse train
        T_pulse = 1.0 / pam_freq
        # Position within each pulse period
        t_mod = np.mod(self.t, T_pulse)
        # Pulse is ON when t_mod < duty * T_pulse
        pulse_on = t_mod < (duty * T_pulse)

        if pam_type == "PAM Natural":
            # Natural: signal passes through during pulse, zero otherwise
            pam_signal = np.where(pulse_on, total_y, 0.0)
        else:
            # Instantaneous (Sample & Hold): sample at start of each pulse, hold that value
            pam_signal = np.zeros_like(total_y)
            # Find sample indices (start of each pulse period)
            period_samples = int(T_pulse * self.fs)
            if period_samples < 1:
                period_samples = 1
            for start in range(0, len(self.t), period_samples):
                sample_val = total_y[start]
                end = min(start + period_samples, len(self.t))
                # Hold the sampled value during the ON portion of the pulse
                for j in range(start, end):
                    if pulse_on[j]:
                        pam_signal[j] = sample_val

        return pam_signal

    def update_plots(self):
        total_y = np.zeros_like(self.t)
        total_y_full = np.zeros_like(self.t)
        has_limits = False

        for s in self.signals:
            total_y += s.get_data(self.t)
            total_y_full += s.get_data_unlimited(self.t)
            if s.limits_enabled and s.active:
                has_limits = True

        # Ghost curve: show full (unlimited) signal when limits are active
        if has_limits:
            ghost_pen = pg.mkPen(color=(150, 150, 150, 100), width=1, style=QtCore.Qt.PenStyle.DashLine)
            self.curve_t_ghost.setData(self.t, total_y_full)
            self.curve_t_ghost.setPen(ghost_pen)
            self.curve_t_ghost.setVisible(True)
        else:
            self.curve_t_ghost.setVisible(False)

        # Determine which signal to use for FFT
        pam_active = self.chk_pam.isChecked()
        if pam_active:
            pam_signal = self._generate_pam(total_y)
            self.curve_t_pam.setData(self.t, pam_signal)
            self.curve_t_pam.setPen(pg.mkPen(color='#00e5ff', width=2))
            self.curve_t_pam.setVisible(True)
            fft_input = pam_signal
        else:
            self.curve_t_pam.setVisible(False)
            fft_input = total_y

        self.curve_t.setData(self.t, total_y)
        self.curve_t.setPen(pg.mkPen(GLOBAL_THEME['sig_main'], width=2))

        # FFT
        n = len(self.t)
        yf = np.fft.rfft(fft_input)
        xf = np.fft.rfftfreq(n, 1/self.fs)
        mag = np.abs(yf) * (2.0 / n)

        self.curve_f.setData(xf[:200], mag[:200])
        self.curve_f.setPen(pg.mkPen(GLOBAL_THEME['sig_main'], width=2))

    def play_audio(self):
        """Sonify the signal shape by mapping amplitude to audible pitch.
        
        Instead of playing the raw waveform (which sounds like noise at low
        frequencies), this maps the signal's value to a frequency range
        that the human ear can easily perceive:
          - High signal value → high pitch
          - Low signal value  → low pitch
        This way, a sine wave sounds like a smooth up-and-down sweep,
        a square wave sounds like alternating between two tones, etc.
        """
        total_y = np.zeros_like(self.t)
        for s in self.signals:
            total_y += s.get_data(self.t)
        
        if np.max(np.abs(total_y)) == 0:
            return
        
        # --- Sonification parameters ---
        audio_sr = 44100        # Standard audio sample rate
        duration = 3.0          # Seconds of audio output (slow enough to hear the shape)
        freq_low = 200.0        # Hz - pitch when signal is at minimum
        freq_high = 800.0       # Hz - pitch when signal is at maximum
        
        # Normalize signal to [0, 1] range for frequency mapping
        y_min, y_max = np.min(total_y), np.max(total_y)
        if y_max - y_min == 0:
            normalized = np.ones_like(total_y) * 0.5
        else:
            normalized = (total_y - y_min) / (y_max - y_min)
        
        # Create the audio time axis
        num_audio_samples = int(audio_sr * duration)
        t_audio = np.linspace(0, duration, num_audio_samples, endpoint=False)
        
        # Interpolate the normalized signal shape to match audio length
        t_norm = np.linspace(0, duration, len(normalized), endpoint=False)
        freq_envelope = np.interp(t_audio, t_norm, normalized)
        
        # Map [0, 1] to [freq_low, freq_high]
        instantaneous_freq = freq_low + (freq_high - freq_low) * freq_envelope
        
        # Generate audio by integrating the instantaneous frequency (phase accumulation)
        phase = np.cumsum(2 * np.pi * instantaneous_freq / audio_sr)
        audio = 0.7 * np.sin(phase).astype(np.float32)
        
        try:
            sd.stop()
            sd.play(audio, samplerate=audio_sr)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error de Audio", f"No se pudo reproducir: {e}")

    def stop_audio(self):
        """Stop any currently playing audio."""
        try:
            sd.stop()
        except Exception:
            pass


class ConvTabWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.fs = 250  
        self.t = np.linspace(-5, 5, self.fs * 10, endpoint=False)
        self.dt = self.t[1] - self.t[0]
        self.signals = [] 

        self.shift_index = 0
        self.conv_result = np.zeros_like(self.t)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.animate_step)

        self.setup_ui()
        self.add_signal(tipo='Seno')
        self.add_signal(tipo='Cuadrada')
        self.reset_animation()

    def apply_colors(self):
        self.win.setBackground(GLOBAL_THEME['bg'])
        self.p1.getAxis('bottom').setPen(pg.mkPen(GLOBAL_THEME['text']))
        self.p1.getAxis('left').setPen(pg.mkPen(GLOBAL_THEME['text']))
        self.p1.setTitle("Proceso: f(τ) [Fija] y g(t-τ) [Móvil]", color=GLOBAL_THEME['text'].name())

        self.p2.getAxis('bottom').setPen(pg.mkPen(GLOBAL_THEME['text']))
        self.p2.getAxis('left').setPen(pg.mkPen(GLOBAL_THEME['text']))
        self.p2.setTitle("Resultado de la Convolución (f * g)(t)", color=GLOBAL_THEME['text'].name())

        # Update specific signal curves
        self.curve_f.setPen(pg.mkPen(GLOBAL_THEME['sig_main'], width=2))
        self.curve_g.setPen(pg.mkPen(GLOBAL_THEME['sig_sec'], width=2))
        self.curve_prod.setPen(pg.mkPen(GLOBAL_THEME['text'], width=1, style=QtCore.Qt.PenStyle.DashLine))
        
        fill_color = QColor(GLOBAL_THEME['sig_res'])
        fill_color.setAlpha(80)
        self.fill.setBrush(fill_color)
        
        self.curve_conv.setPen(pg.mkPen(GLOBAL_THEME['sig_res'], width=3))

    def setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(300)
        self.side_layout = QtWidgets.QVBoxLayout(sidebar)
        
        self.btn_play = QtWidgets.QPushButton("▶ Reproducir / Pausa")
        self.btn_play.clicked.connect(self.toggle_animation)
        self.side_layout.addWidget(self.btn_play)
        
        self.btn_reset = QtWidgets.QPushButton("⏹ Reiniciar")
        self.btn_reset.clicked.connect(self.reset_animation)
        self.side_layout.addWidget(self.btn_reset)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.scroll_content)
        self.side_layout.addWidget(self.scroll)

        self.win = pg.GraphicsLayoutWidget()
        
        self.p1 = self.win.addPlot(title="Proceso: f(τ) [Fija] y g(t-τ) [Móvil]")
        self.p1.showGrid(x=True, y=True)
        self.p1.setYRange(-5, 5)
        
        self.curve_f = self.p1.plot(name="f(t)")
        self.curve_g = self.p1.plot(name="g(t-tau)")
        self.curve_prod = self.p1.plot()
        
        self.curve_zero = self.p1.plot(self.t, np.zeros_like(self.t), pen=None)
        
        # Init fill dummy brush
        self.fill = pg.FillBetweenItem(self.curve_zero, self.curve_prod, brush=(0,255,0,80))
        self.p1.addItem(self.fill)

        self.win.nextRow()
        
        self.p2 = self.win.addPlot(title="Resultado de la Convolución (f * g)(t)")
        self.p2.showGrid(x=True, y=True)
        self.curve_conv = self.p2.plot()

        self.apply_colors() # initial color apply
        layout.addWidget(sidebar)
        layout.addWidget(self.win)

    def add_signal(self, tipo='Seno'):
        if len(self.signals) >= 2:
            return 
        new_sig = SignalObject(type=tipo)
        self.signals.append(new_sig)
        
        control = SignalControlWidget(new_sig, disable_remove=True)
        control.changed.connect(self.reset_animation)
        
        self.scroll_layout.addWidget(control)

    def toggle_animation(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(20)

    def reset_animation(self):
        self.timer.stop()
        self.shift_index = 0
        self.conv_result = np.zeros_like(self.t)
        self.curve_conv.setData(self.t, self.conv_result)
        self.update_plots_static()

    def update_plots_static(self):
        if len(self.signals) < 2: return
        
        f_data = self.signals[0].get_data(self.t)
        g_data_raw = self.signals[1].get_data(self.t)
        g_data_inv = np.flip(g_data_raw) 
        
        self.curve_f.setData(self.t, f_data)
        
        # We start animation shift at left-most edge:
        g_shifted = np.roll(g_data_inv, -len(self.t))
        g_shifted[:-len(self.t)] = 0
        self.curve_g.setData(self.t, g_shifted)
        self.curve_prod.setData(self.t, np.zeros_like(self.t))
        
    def animate_step(self):
        if len(self.signals) < 2: return
        
        f_data = self.signals[0].get_data(self.t)
        g_data_raw = self.signals[1].get_data(self.t)
        g_data_inv = np.flip(g_data_raw)

        shift = self.shift_index - len(self.t)
        g_shifted = np.roll(g_data_inv, shift)
        
        if shift < 0:
            g_shifted[shift:] = 0
        else:
            g_shifted[:shift] = 0

        product = f_data * g_shifted
        
        self.curve_g.setData(self.t, g_shifted)
        self.curve_prod.setData(self.t, product)
        
        area = np.sum(product) * self.dt
        
        current_time_index = self.shift_index
        if current_time_index < len(self.t):
            self.conv_result[current_time_index] = area
            self.curve_conv.setData(self.t[:current_time_index], self.conv_result[:current_time_index])

        self.shift_index += max(1, int(len(self.t) / 200)) # Advance
        
        if self.shift_index >= len(self.t):
            self.timer.stop()


class EpicyclesTabWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.image_points = []     # Raw complex points from image
        self.fourier_coefs = []    # Computed and sorted DFT coefficients
        self.path_trace = []       # Storing the drawn path
        self.time = 0.0
        self.dt = 0.0
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.animate_step)
        
        # UI Setup
        layout = QtWidgets.QHBoxLayout(self)
        
        # Sidebar
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(300)
        side_layout = QtWidgets.QVBoxLayout(sidebar)
        
        btn_load = QtWidgets.QPushButton("Cargar Imagen B/N")
        btn_load.clicked.connect(self.load_image)
        side_layout.addWidget(btn_load)
        
        self.lbl_info = QtWidgets.QLabel("Puntos: 0")
        side_layout.addWidget(self.lbl_info)
        
        side_layout.addWidget(QtWidgets.QLabel("Círculos a dibujar:"))
        self.spin_circles = QtWidgets.QSpinBox()
        self.spin_circles.setRange(1, 10000)
        self.spin_circles.setValue(100)
        self.spin_circles.valueChanged.connect(self.reset_animation)
        side_layout.addWidget(self.spin_circles)
        
        side_layout.addWidget(QtWidgets.QLabel("Velocidad de dibujo (Salto dt):"))
        self.spin_speed = QtWidgets.QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 100.0)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setSingleStep(0.5)
        self.spin_speed.valueChanged.connect(self.update_speed)
        side_layout.addWidget(self.spin_speed)

        # Controles
        row_play = QtWidgets.QHBoxLayout()
        btn_play = QtWidgets.QPushButton("▶ Reproducir")
        btn_play.clicked.connect(self.toggle_animation)
        btn_reset = QtWidgets.QPushButton("⏹ Reset")
        btn_reset.clicked.connect(self.reset_animation)
        row_play.addWidget(btn_play)
        row_play.addWidget(btn_reset)
        side_layout.addLayout(row_play)
        
        side_layout.addStretch()

        # Canvas with fixed aspect ratio
        self.win = pg.GraphicsLayoutWidget()
        self.plot = self.win.addPlot(title="Fourier Epicycles")
        self.plot.setAspectLocked(True) # Very important for circles to be round
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        
        # Items for drawing
        self.path_curve = self.plot.plot()
        
        # We will keep a list of generic pyqtgraph plot items to draw circles/lines
        self.circle_items = []
        self.radius_lines = []
        
        # Helper circle data (unit circle) to draw quickly
        theta = np.linspace(0, 2*np.pi, 50)
        self.unit_circle_x = np.cos(theta)
        self.unit_circle_y = np.sin(theta)

        layout.addWidget(sidebar)
        layout.addWidget(self.win)
        self.apply_colors()

    def apply_colors(self):
        self.win.setBackground(GLOBAL_THEME['bg'])
        self.plot.setTitle("Fourier Epicycles", color=GLOBAL_THEME['text'].name())
        self.path_curve.setPen(pg.mkPen(GLOBAL_THEME['sig_main'], width=2))
        
        fill_c = QColor(GLOBAL_THEME['sig_main'])
        fill_c.setAlpha(80)
        line_c = QColor(GLOBAL_THEME['sig_main'])
        line_c.setAlpha(150)
        
        for c, l in zip(self.circle_items, self.radius_lines):
            c.setPen(pg.mkPen(fill_c, width=1))
            l.setPen(pg.mkPen(line_c, width=1))

    def update_speed(self):
        # We base the step on 2*pi / N. If speed > 1, we advance faster.
        if len(self.image_points) > 0:
            N = len(self.image_points)
            self.dt = self.spin_speed.value() * (2 * np.pi / N)
    
    def resample_contour_uniform(self, contour_points, num_samples):
        """Re-sample contour points uniformly by arc length.
        
        This ensures points are evenly distributed along the curve,
        producing cleaner DFT coefficients and a more faithful drawing.
        """
        # contour_points is Nx1x2 from OpenCV, flatten to Nx2
        pts = contour_points.reshape(-1, 2).astype(np.float64)
        
        # Close the contour by appending the first point at the end
        pts_closed = np.vstack([pts, pts[0:1]])
        
        # Compute cumulative arc length
        diffs = np.diff(pts_closed, axis=0)
        segment_lengths = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]
        
        if total_length == 0:
            return pts
        
        # Create uniform samples along the arc length (exclude last to avoid duplicate)
        uniform_distances = np.linspace(0, total_length, num_samples, endpoint=False)
        
        # Interpolate x and y separately
        resampled_x = np.interp(uniform_distances, cumulative_length, pts_closed[:, 0])
        resampled_y = np.interp(uniform_distances, cumulative_length, pts_closed[:, 1])
        
        return np.column_stack([resampled_x, resampled_y])

    def load_image(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file_name:
            return
            
        # 1. Read image with opencv
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        
        # 2. Resize if too large to avoid freezing
        max_dim = 800
        h, w = img.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        # 3. Canny edge detection (much better than simple threshold for detail)
        # Apply Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate slightly to close small gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 4. Find contours on the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            # Fallback to simple threshold if Canny finds nothing
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            self.lbl_info.setText("Error: No se encontró figura.")
            return
            
        # Pick the largest contour (NO approxPolyDP - keep all original detail)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 5. Uniform arc-length resampling
        # Use a good number of samples for fidelity (at least 1000, up to contour length)
        raw_point_count = len(largest_contour)
        num_samples = max(1000, min(raw_point_count, 4000))
        resampled_pts = self.resample_contour_uniform(largest_contour, num_samples)
        
        points = []
        # Flip Y axis because image Y goes down, but graph Y goes up
        for x, y in resampled_pts:
            points.append(complex(x, -y)) 
            
        self.image_points = np.array(points)
        N = len(self.image_points)
        self.lbl_info.setText(f"Puntos extraídos: {N} (remuestreados de {raw_point_count})")
        self.spin_circles.setMaximum(N)
        self.spin_circles.setValue(min(N, 500))
        
        # 6. Compute DFT
        self.compute_dft()

    def compute_dft(self):
        N = len(self.image_points)
        # Compute standard FFT
        X = np.fft.fft(self.image_points)
        freqs = np.fft.fftfreq(N, d=1/N)
        
        self.fourier_coefs = []
        for k, coef in zip(freqs, X):
            # Normalize amplitude by N
            amp = np.abs(coef) / N
            phase = np.angle(coef)
            freq = k 
            self.fourier_coefs.append((amp, freq, phase))
            
        # Sort by amplitude descending (largest circles first)
        self.fourier_coefs.sort(key=lambda x: x[0], reverse=True)
        
        self.reset_animation()

    def toggle_animation(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(20) # 50 fps
            
    def clear_canvas(self):
        for item in self.circle_items:
            self.plot.removeItem(item)
        for item in self.radius_lines:
            self.plot.removeItem(item)
        self.circle_items.clear()
        self.radius_lines.clear()

    def reset_animation(self):
        self.timer.stop()
        self.time = 0.0
        self.path_trace = []
        self.path_curve.setData([], [])
        self.clear_canvas()
        
        if len(self.image_points) == 0:
            return
            
        self.update_speed()
        
        # Pre-allocate circle and line visual items for the first N circles
        num_circles = min(self.spin_circles.value(), len(self.fourier_coefs))
        
        fill_c = QColor(GLOBAL_THEME['sig_main'])
        fill_c.setAlpha(80)
        line_c = QColor(GLOBAL_THEME['sig_main'])
        line_c.setAlpha(150)
        
        pen_circle = pg.mkPen(fill_c, width=1)
        pen_line = pg.mkPen(line_c, width=1)
        
        for _ in range(num_circles):
            c = self.plot.plot(pen=pen_circle)
            l = self.plot.plot(pen=pen_line)
            self.circle_items.append(c)
            self.radius_lines.append(l)
            
        # Auto-range the plot based on the first (DC) circle which is the offset
        if len(self.fourier_coefs) > 0:
            dc_offset = self.fourier_coefs[0] # The one with freq 0
            x_center = dc_offset[0] * np.cos(dc_offset[2]) # approx center
            y_center = dc_offset[0] * np.sin(dc_offset[2])
            margin = 300 # arbitrary margin
            self.plot.setXRange(x_center - margin, x_center + margin)
            self.plot.setYRange(y_center - margin, y_center + margin)
            
        self.animate_step()

    def animate_step(self):
        if len(self.fourier_coefs) == 0:
            return
            
        num_circles = min(self.spin_circles.value(), len(self.fourier_coefs))
        
        x, y = 0.0, 0.0
        
        for i in range(num_circles):
            prev_x, prev_y = x, y
            
            amp, freq, phase = self.fourier_coefs[i]
            
            # Current endpoint
            x += amp * np.cos(freq * self.time + phase)
            y += amp * np.sin(freq * self.time + phase)
            
            # Update circle visual
            # Scaled unit circle
            circ_x = prev_x + amp * self.unit_circle_x
            circ_y = prev_y + amp * self.unit_circle_y
            self.circle_items[i].setData(circ_x, circ_y)
            
            # Update radius line visual
            self.radius_lines[i].setData([prev_x, x], [prev_y, y])
            
        # Record the final tip position
        self.path_trace.append((x, y))
        
        # Don't let it grow infinitely, wrap around after 1 cycle (2*pi)
        # N elements -> 1 full cycle is time = 2*pi
        # Actually in DFT the frequency ranges means 1 cycle is 2*pi.
        if self.time > 2 * np.pi:
            self.time = 0.0
            self.path_trace = [] # Reset trace
            
        if len(self.path_trace) > 1:
            arr = np.array(self.path_trace)
            self.path_curve.setData(arr[:, 0], arr[:, 1])
            
        self.time += self.dt

class SettingsTabWidget(QtWidgets.QWidget):
    theme_updated = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        lbl = QtWidgets.QLabel("Ajustes Avanzados de Colores")
        lbl.setFont(QFont("Courier New", 14, QFont.Weight.Bold))
        layout.addWidget(lbl)

        # Helper method for items
        def add_color_picker(key, label_str):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label_str)
            btn = QtWidgets.QPushButton()
            btn.setFixedWidth(50)
            
            def update_btn_style():
                color = GLOBAL_THEME[key].name()
                btn.setStyleSheet(f"background-color: {color}; border: 1px solid white;")
            update_btn_style()

            def on_click():
                c = QtWidgets.QColorDialog.getColor(GLOBAL_THEME[key], self, f"Seleccionar Color para {label_str}")
                if c.isValid():
                    GLOBAL_THEME[key] = c
                    update_btn_style()
                    self.theme_updated.emit()
            
            btn.clicked.connect(on_click)
            row.addWidget(lbl)
            row.addWidget(btn)
            layout.addLayout(row)

        add_color_picker("bg", "Fondo Interfaz Pestañas")
        add_color_picker("bg2", "Fondo Controles Laterales")
        add_color_picker("text", "Letras y Bordes (General)")
        add_color_picker("grid", "Cuadrícula de Gráficas (Grid)")
        layout.addWidget(QtWidgets.QLabel("--------------"))
        add_color_picker("sig_main", "Señal Principal / Señal f(t) (Conv) / Epiciclos")
        add_color_picker("sig_sec", "Señal g(t) (Convolución)")
        add_color_picker("sig_res", "Resultado Convolución")

        layout.addStretch()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualizador Unificado: FFT y Convolución / Matias Vasques Yelorm")
        self.resize(1200, 750)

        # Global pg config
        pg.setConfigOption('background', '#000000') # we handle bg per plot
        pg.setConfigOption('foreground', '#ffffff') # handled dynamically by stylesheet

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.fft_tab = FFTTabWidget()
        self.conv_tab = ConvTabWidget()
        self.epi_tab = EpicyclesTabWidget()
        self.settings_tab = SettingsTabWidget()
        
        self.settings_tab.theme_updated.connect(self.apply_theme)
        
        self.tabs.addTab(self.fft_tab, "Análisis FFT")
        self.tabs.addTab(self.conv_tab, "Animación de Convolución")
        self.tabs.addTab(self.epi_tab, "Dibujo con Epiciclos")
        self.tabs.addTab(self.settings_tab, "Configuración")
        
        self.apply_theme() # Appy initial theme

    def apply_theme(self):
        # Update PG items
        self.fft_tab.apply_colors()
        self.conv_tab.apply_colors()
        self.epi_tab.apply_colors()
        
        bg = GLOBAL_THEME["bg"].name()
        bg2 = GLOBAL_THEME["bg2"].name()
        text = GLOBAL_THEME["text"].name()
        btn = GLOBAL_THEME["btn_bg"].name()
        btn_hover = GLOBAL_THEME["btn_hover"].name()
        
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {bg};
                color: {text};
                font-family: "Courier New", Courier, monospace;
            }}
            QPushButton {{
                background-color: {btn};
                color: {text};
                border: 1px solid {text};
                padding: 5px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {btn_hover};
            }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {bg2};
                color: {text};
                border: 1px solid {text};
                padding: 2px;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {text};
                height: 4px;
                background: {btn};
            }}
            QSlider::handle:horizontal {{
                background: {text};
                width: 14px;
                margin: -5px 0;
                border-radius: 2px;
            }}
            QTabWidget::pane {{
                border: 1px solid {text};
                background-color: {bg};
            }}
            QTabBar::tab {{
                background: #000000;
                color: gray;
                border: 1px solid {text};
                padding: 8px 20px;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: #222222;
                color: {text};
                font-weight: bold;
            }}
            QLabel {{
                color: {text};
                font-weight: bold;
            }}
            QScrollArea {{
                border: 1px solid {text};
            }}
        """)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
