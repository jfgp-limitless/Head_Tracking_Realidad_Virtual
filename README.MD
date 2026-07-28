# Head Tracking CAD Viewer (bajo consumo de recursos)

Visor de **varios modelos CAD reales** (exportados desde SolidWorks u
otro software, en `.stl` o `.obj`) que se muestran **simultáneamente**
dentro de una escena tipo "cuarto de ingeniería" (piso + pared con
cuadrícula blanca sobre fondo azul), cuya perspectiva se controla
moviendo la cabeza frente a la webcam.

Construido solo con **OpenCV + NumPy**, sin motores 3D pesados (nada
de OpenGL, Panda3D, mediapipe, etc.), pensado para correr fluido en
equipos modestos.

## 1. Instalación

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

Requiere Python 3.8+ y una webcam. `tkinter` (selector de archivos)
viene incluido en la mayoría de instalaciones de Python; si falta en
Linux: `sudo apt install python3-tk`.

## 2. Ejecución

```bash
python main.py
```

Al iniciar se abre un explorador de archivos: **selecciona tus 3 (o
las que quieras) piezas CAD a la vez** (mantén `Ctrl` para elegir
varias). Se abre luego una ventana dividida: a la izquierda la
cámara (con el rectángulo verde sobre el rostro), a la derecha la
escena con tus modelos flotando dentro del "cuarto" con cuadrícula,
todos girando juntos según muevas la cabeza.

### Controles

| Tecla | Acción |
|-------|--------|
| `o` | Elegir/reemplazar los modelos CAD cargados (`.stl` / `.obj`) |
| `c` | Recalibrar la distancia "neutra" (útil si cambia de silla/distancia) |
| `r` | Reiniciar la vista a rotación 0 |
| `q` | Salir |

Si al iniciar cancelas el explorador de archivos, se muestran 3 cubos
de ejemplo para que puedas probar el sistema igual.

## 3. Estructura del código

| Archivo | Función |
|---|---|
| `camera.py` | Captura de video en tiempo real |
| `head_tracker.py` | Detección y seguimiento de la cabeza (Haar cascades) |
| `geometry.py` | Carga de `.stl`/`.obj`, decimación y generación del "cuarto" con cuadrícula |
| `renderer.py` | Proyección 3D y dibujo wireframe con depth cue |
| `main.py` | Arma la escena (varios modelos + cuarto) y aplica el head tracking en tiempo real |

## 4. Cómo se arma la escena

- Cada modelo cargado se **centra, normaliza y decima** (se reduce su
  cantidad de vértices/aristas) para que el conjunto de piezas no
  supere un presupuesto total de líneas (`MAX_TOTAL_EDGES` en
  `main.py`, por defecto 3000 repartidas entre todos los modelos).
- Los modelos se **distribuyen en fila** (eje X) dentro de la escena,
  cada uno con su color y su etiqueta (nombre del archivo).
- Se genera un **"cuarto"**: una grilla de piso + una pared trasera
  (función `build_room` en `geometry.py`), dibujada en blanco sobre
  fondo azul ingeniería, que da referencia de profundidad y refuerza
  la sensación de 3D.
- Piso, pared y modelos reciben **la misma rotación** (yaw/pitch/roll
  calculados del movimiento de tu cabeza) y la misma distancia de
  cámara, de modo que toda la escena gira como un solo bloque
  coherente.

## 5. Head tracking (resumen técnico)

- El rostro se detecta con `haarcascade_frontalface_default.xml`
  (clasificador incluido en OpenCV, mucho más liviano que un modelo
  de deep learning).
- **Posición horizontal/vertical** del rostro → yaw/pitch de la escena.
- **Tamaño del rostro** (más grande = más cerca) → distancia/zoom.
- **Ángulo entre ojos** (opcional) → leve roll (inclinación).
- Todo se suaviza con un promedio móvil exponencial para evitar
  temblores (jitter).

## 6. Formatos y modelos propios

- Soporta `.stl` (binario y ASCII) y `.obj` (solo vértices/caras, sin
  texturas ni materiales, para mantener el costo bajo).
- Si exportas desde SolidWorks, **STL** es la opción más simple y
  liviana; no necesitas simplificar el modelo a mano, el programa lo
  decima automáticamente al cargarlo.
- Modelos con demasiados triángulos tardarán un poco más en **cargar**
  (una sola vez, no por cuadro), pero el renderizado en vivo siempre
  respeta el presupuesto de líneas configurado.

## 7. Recomendaciones de rendimiento

Si el FPS mostrado en pantalla baja de ~15:

- Reduce `MAX_TOTAL_EDGES` en `main.py` (por ejemplo a 1500).
- Reduce la resolución de cámara (`CAM_W`, `CAM_H`).
- Desactiva la estimación de roll: `HeadTracker(detect_roll=False)`.
- Cierra otras aplicaciones que usen la cámara o CPU en segundo plano.

## 8. Pruebas y validación

- **Iluminación**: probar con luz frontal, lateral y contraluz (los
  cascades Haar son sensibles a poca luz).
- **Distancia**: probar a ~40 cm, ~80 cm y ~1.5 m, recalibrando (`c`)
  en cada caso.
- **Velocidad de movimiento**: mover la cabeza lento vs. rápido y
  ajustar `smoothing` en `HeadTracker` si se siente tembloroso o lento.
- **Estabilidad**: dejar corriendo varios minutos y observar que el
  FPS no se degrade.
- **Modelos reales**: probar con piezas simples y luego con piezas más
  complejas de tus 3 archivos para verificar que la decimación
  mantenga buen desempeño sin perder la forma reconocible.

## Posibles mejoras futuras

- Arreglar los modelos en una cuadrícula 2D (filas y columnas) en vez
  de una sola fila, si son más de 4-5 piezas.
- Sombreado plano por cara en vez de solo wireframe.
- Guardar la última selección de archivos para no tener que elegirlos
  de nuevo cada vez.