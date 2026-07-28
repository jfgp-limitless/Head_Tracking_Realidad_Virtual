"""
renderer.py
Motor de renderizado 3D minimalista basado en proyección de perspectiva
manual (sin OpenGL), pensado para equipos con pocos recursos.
Dibuja los modelos como wireframe usando únicamente OpenCV + NumPy.
"""

import numpy as np
import cv2


def rotation_matrix(yaw, pitch, roll=0.0):
    """Ángulos en radianes. Devuelve la matriz de rotación combinada."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def project(vertices, yaw, pitch, roll, distance, canvas_w, canvas_h,
            focal=380, offset_x=0.0, offset_y=0.0):
    """
    Rota el modelo según los ángulos dados y proyecta sus vértices a
    coordenadas 2D con perspectiva simple (cámara fija en el origen,
    mirando hacia +Z).
    """
    R = rotation_matrix(yaw, pitch, roll)
    rotated = vertices @ R.T

    z = rotated[:, 2] + distance
    z = np.clip(z, 0.1, None)  # evita división por cero / puntos detrás de cámara

    x2d = focal * rotated[:, 0] / z + canvas_w / 2 + offset_x
    y2d = focal * rotated[:, 1] / z + canvas_h / 2 + offset_y

    points2d = np.stack([x2d, y2d], axis=1)
    return points2d, z


def draw_wireframe(canvas, points2d, depths, edges, color=(0, 220, 255)):
    """
    Dibuja las aristas del modelo variando levemente grosor/brillo según
    la profundidad (depth cue barato). Esto refuerza la sensación de 3D
    sin necesitar z-buffer ni sombreado real, manteniendo el costo bajo.
    """
    if len(depths) == 0:
        return canvas
    dmin, dmax = depths.min(), depths.max()
    drange = max(dmax - dmin, 1e-6)

    for (i, j) in edges:
        p1, p2 = points2d[i], points2d[j]
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue

        avg_depth = (depths[i] + depths[j]) / 2.0
        closeness = 1.0 - (avg_depth - dmin) / drange  # 1 = más cerca
        thickness = 1 + int(round(closeness * 2))
        shade = 0.5 + 0.5 * closeness
        b, g, r = color
        c = (int(b * shade), int(g * shade), int(r * shade))

        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(canvas, pt1, pt2, c, thickness, cv2.LINE_AA)

    return canvas