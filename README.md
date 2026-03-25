# Visualizador Unificado: FFT, Convolución y Epiciclos

Una herramienta interactiva y visual desarrollada en Python para comprender conceptos matemáticos y de procesamiento de señales como la Transformada Rápida de Fourier (FFT), la Convolución y los Epiciclos de Fourier.

## 🚀 Características Principales

La aplicación está dividida en cuatro pestañas principales:

### 1. Análisis FFT (Transformada de Fourier)
- **Suma de Señales**: Combina múltiples señales en el dominio del tiempo para ver su forma resultante.
- **Tipos de Ondas Disponibles**: Seno, Coseno, Cuadrada, Triangular, Diente de Sierra, Sinc (Seno Cardinal), Exponencial, Logarítmica, Delta e Impulso Cuadrado.
- **Controles Precisos**: Ajusta en tiempo real la frecuencia, amplitud y desplazamiento (fase) de cada señal mediante deslizadores (sliders).
- **Límites de Tiempo**: Permite acotar cada señal a un rango de tiempo específico. Se incluye una **visualización mejorada** mediante una "señal fantasma" (línea punteada) que muestra la señal original completa para mayor claridad visual.
- **Modulación PAM**: Modula la señal resultante utilizando Pulse Amplitude Modulation (PAM).
    - **PAM Natural**: La señal conserva su forma original durante el pulso.
    - **PAM Instantánea (Sample & Hold)**: Muestra la señal al inicio de cada pulso y mantiene el valor constante.
    - **Controles Dinámicos**: Ajusta la frecuencia del tren de pulsos (1-200 Hz) y el ciclo de trabajo (duty cycle) de 1% a 99%.
- **Espectro de Frecuencia**: Muestra en tiempo real la magnitud de la FFT de la señal (incluyendo el efecto de la modulación PAM si está activa).
- **Sonificación (Audio)**: Escucha la "forma" de la onda. La amplitud de la señal se mapea a tonos audibles para comprender intuitivamente la forma de la señal compuesta.

### 2. Animación de Convolución
- **Proceso Paso a Paso**: Visualiza gráficamente cómo se calcula la convolución continua entre dos señales `f(t)` y `g(t)` en un **rango extendido de -5 a 5** para una visualización más completa.
- **Ilustración Dinámica**: Muestra la señal fija `f(τ)`, la señal invertida y desplazada `g(t-τ)`, su producto resaltado, y el área bajo la curva resultante.
- **Controles de Reproducción**: Pausa, reproduce o reinicia la animación para estudiar cada instante del cálculo.

### 3. Dibujo con Epiciclos de Fourier
- **Extrae Contornos**: Carga cualquier imagen en blanco y negro (PNG, JPG, etc.). La aplicación detectará automáticamente sus bordes utilizando Canny Edge Detection y un re-muestreo uniforme por longitud de arco para máxima fidelidad.
- **Dibuja con Círculos**: Calcula los coeficientes de la Transformada Discreta de Fourier (DFT) y los usa para recrear el contorno usando epiciclos giratorios (círculos acoplados).
- **Ajustes en Tiempo Real**: Controla la cantidad de círculos y la velocidad de dibujo. A más círculos, mayor precisión matemática en el dibujo resultante.

### 4. Configuración
- **Temas Personalizados**: Interfaz incorporada para cambiar los colores de casi cualquier elemento (fondos, líneas, ejes, cuadrículas) y crear tu propio esquema visual moderno.

---

## 🛠️ Requisitos e Instalación

### Dependencias
Este proyecto requiere Python 3.10 o superior y las siguientes librerías:

- `PyQt6` (Interfaz gráfica)
- `pyqtgraph` (Gráficos rápidos y eficientes)
- `numpy` (Cálculo numérico y FFT)
- `scipy` (Generación de ondas complejas)
- `opencv-python` (Carga y procesamiento de imágenes para epiciclos)
- `sounddevice` (Reproducción de audio)

### Instalación (Windows/Linux/macOS)

1. **Clona o descarga este repositorio.**
2. **Crea un entorno virtual (recomendado):**
   ```bash
   python -m venv .venv
   ```
3. **Activa el entorno virtual:**
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. **Instala los paquetes:**
   ```bash
   pip install PyQt6 pyqtgraph numpy scipy opencv-python sounddevice
   ```
5. **Ejecuta la aplicación:**
   ```bash
   python main.py
   ```

---

## 👨‍💻 Autor
Proyecto original por **Matias Vasques Yelorm**. Creado para hacer más accesibles y visuales las matemáticas detrás del análisis de señales.

## Imagenes
Las imagenes existentes en el repositorio para pruebas de epiciclos no son de autoria del autor y estan solo para probar el sitema si eres el creador original o lo conoces contactame para eliminar los archivos.

El sistema esta diceñado para meramente fines educativos y de experimentación.
