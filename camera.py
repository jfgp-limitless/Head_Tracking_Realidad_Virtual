"""
camera.py
Inicialización y captura de video en tiempo real, con resolución y
tasa de cuadros ajustadas para bajo consumo de recursos.
"""

import platform
import cv2


class Camera:
    def __init__(self, index=0, width=640, height=480, fps=20):
        backend = cv2.CAP_DSHOW if platform.system().lower() == "windows" else 0
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # MJPG reduce el uso de CPU en muchas webcams USB
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError(
                "No se pudo acceder a la cámara. Verifique la conexión, "
                "que no esté en uso por otra aplicación, y los permisos "
                "de cámara del sistema operativo."
            )

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self.cap.release()