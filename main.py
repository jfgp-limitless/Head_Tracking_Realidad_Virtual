"""
main.py
Visor de varios modelos CAD (.stl / .obj) simultáneos, distribuidos en
una escena tipo "cuarto de ingeniería" (piso + pared con cuadrícula
blanca sobre fondo azul), cuya perspectiva se controla moviendo la
cabeza frente a la webcam. Solo usa OpenCV + NumPy (sin motores 3D
pesados) para funcionar bien en equipos con pocos recursos.

Controles:
  o - Elegir/agregar archivos CAD (.stl / .obj) desde cualquier carpeta
  c - Recalibrar distancia neutra (aléjese/acérquese y presione c)
  r - Reiniciar la vista (rotación en 0)
  q - Salir
"""

import os
import time
import cv2
import numpy as np

from camera import Camera
from HeadTracker import HeadTracker
from geometry import cube, build_room, load_cad_file
from renderer import project, draw_wireframe

CANVAS_W, CANVAS_H = 520, 520
CAM_W, CAM_H = 640, 480

MAX_YAW_DEG = 50
MAX_PITCH_DEG = 35
BASE_DISTANCE = 6.0
DISTANCE_RANGE = 2.0
FOCAL = 300

MAX_TOTAL_EDGES = 3000       # presupuesto total de aristas para los modelos CAD
OBJECT_TARGET_SIZE = 0.85     # tamaño normalizado de cada modelo en la escena

BG_COLOR = (110, 60, 15)      # azul "ingeniería" (BGR)
GRID_COLOR = (245, 245, 245)  # blanco

MODEL_COLORS = [
    (0, 210, 255),   # ámbar
    (255, 200, 0),   # celeste
    (120, 255, 120), # verde
    (255, 120, 255), # magenta
    (0, 140, 255),   # naranja
]

ROOM_VERTS, ROOM_EDGES = build_room(width=7.5, depth=8.0, floor_y=2.1, ceil_y=-2.6, step=0.7)


def pick_cad_files():
    """Abre un diálogo nativo para elegir uno o varios archivos CAD."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Seleccionar modelos CAD (.stl / .obj) - puede elegir varios",
            filetypes=[("Modelos CAD", "*.stl *.obj"), ("Todos los archivos", "*.*")],
        )
        root.destroy()
        return list(paths)
    except Exception as e:
        print(f"[Aviso] No se pudo abrir el explorador de archivos: {e}")
        return []


def compute_offsets(n, spacing=2.4):
    if n <= 1:
        return [np.array([0.0, 0.0, 0.0])]
    xs = np.linspace(-spacing * (n - 1) / 2, spacing * (n - 1) / 2, n)
    return [np.array([x, 0.0, 0.0]) for x in xs]


def build_scene(paths):
    """Carga todos los archivos indicados y arma la escena (lista de modelos)."""
    if not paths:
        # escena de demostración si aún no se ha seleccionado nada
        offsets = compute_offsets(3)
        return [
            {"name": f"Demo {i + 1}", "vertices": cube(1.2)[0] + offsets[i],
             "edges": cube(1.2)[1]}
            for i in range(3)
        ]

    budget = max(300, MAX_TOTAL_EDGES // max(len(paths), 1))
    offsets = compute_offsets(len(paths))
    models = []
    for i, p in enumerate(paths):
        try:
            v, e = load_cad_file(p, max_edges=budget)
            v = v * OBJECT_TARGET_SIZE + offsets[i]
            models.append({"name": os.path.basename(p), "vertices": v, "edges": e})
            print(f"[OK] Cargado: {os.path.basename(p)} ({len(e)} aristas tras decimación)")
        except Exception as ex:
            print(f"[Error] No se pudo cargar '{p}': {ex}")

    return models if models else build_scene([])


def main():
    cam = Camera(index=0, width=CAM_W, height=CAM_H, fps=20)
    tracker = HeadTracker(smoothing=0.35, detect_roll=True)

    print("=== Head Tracking CAD Viewer ===")
    print("Selecciona tus archivos CAD (.stl / .obj). Puedes elegir varios a la vez.")
    loaded_paths = pick_cad_files()
    models = build_scene(loaded_paths)

    prev_time = time.time()
    fps_smooth = 0.0
    manual_reset = False

    print("\nTeclas: o = agregar/cambiar modelos | c = recalibrar | r = reset vista | q = salir\n")

    while True:
        frame = cam.read()
        if frame is None:
            print("[Error] No se pudo leer el cuadro de la cámara.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hx, hy, hz, roll, found = tracker.update(gray)

        if found and tracker.last_box is not None:
            fx, fy, fw, fh = tracker.last_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        if manual_reset:
            yaw, pitch, roll_rad, distance = 0.0, 0.0, 0.0, BASE_DISTANCE
            manual_reset = False
        else:
            yaw = np.radians(hx * MAX_YAW_DEG)
            pitch = np.radians(-hy * MAX_PITCH_DEG)
            roll_rad = np.radians(np.clip(roll, -30, 30)) if found else 0.0
            distance = BASE_DISTANCE - hz * DISTANCE_RANGE

        # --- fondo azul + cuadrícula del "cuarto" ---
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR

        room_pts, room_depths = project(
            ROOM_VERTS, yaw, pitch, roll_rad, distance, CANVAS_W, CANVAS_H, focal=FOCAL
        )
        draw_wireframe(canvas, room_pts, room_depths, ROOM_EDGES, color=GRID_COLOR)

        # --- modelos CAD ---
        for i, m in enumerate(models):
            pts, depths = project(
                m["vertices"], yaw, pitch, roll_rad, distance,
                CANVAS_W, CANVAS_H, focal=FOCAL,
            )
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            draw_wireframe(canvas, pts, depths, m["edges"], color=color)
            cx, cy = pts.mean(axis=0)
            label = m["name"][:16]
            cv2.putText(canvas, label, (int(cx) - 30, int(cy) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        status = "Rostro detectado" if found else "Buscando rostro..."
        cv2.putText(canvas, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps
        cv2.putText(canvas, f"FPS: {fps_smooth:.1f}", (10, CANVAS_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cam_resized = cv2.resize(frame, (CANVAS_W, CANVAS_H))
        combined = np.hstack([cam_resized, canvas])
        cv2.imshow("Head Tracking CAD Viewer (q para salir)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            tracker.recalibrate()
        elif key == ord("r"):
            manual_reset = True
        elif key == ord("o"):
            new_paths = pick_cad_files()
            if new_paths:
                loaded_paths = new_paths  # reemplaza la escena por la nueva selección
                models = build_scene(loaded_paths)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()