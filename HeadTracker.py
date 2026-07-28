"""
head_tracker.py
Detección y seguimiento liviano de la cabeza usando clasificadores en
cascada de Haar (incluidos en OpenCV).
"""

import cv2
import numpy as np


class HeadTracker:
    def __init__(self, smoothing=0.35, detect_roll=True):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = (
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            if detect_roll else None
        )

        self.smoothing = smoothing
        self.ref_width = None

        self.hx, self.hy, self.hz, self.roll = 0.0, 0.0, 0.0, 0.0
        self.face_found = False
        self.last_box = None

    def recalibrate(self):
        self.ref_width = None

    def update(self, frame_gray):
        h, w = frame_gray.shape[:2]
        faces = self.face_cascade.detectMultiScale(
            frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) == 0:
            self.face_found = False
            return self.hx, self.hy, self.hz, self.roll, False

        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        self.last_box = (fx, fy, fw, fh)
        self.face_found = True

        if self.ref_width is None:
            self.ref_width = fw

        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        raw_hx = (cx - w / 2.0) / (w / 2.0)
        raw_hy = (cy - h / 2.0) / (h / 2.0)
        raw_hz = float(np.clip((fw - self.ref_width) / self.ref_width, -1.0, 1.0))

        raw_roll = 0.0
        if self.eye_cascade is not None:
            roi = frame_gray[fy:fy + fh, fx:fx + fw]
            eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 8, minSize=(15, 15))
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])[:2]
                (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
                p1 = (ex1 + ew1 / 2.0, ey1 + eh1 / 2.0)
                p2 = (ex2 + ew2 / 2.0, ey2 + eh2 / 2.0)
                raw_roll = float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))

        a = self.smoothing
        self.hx = a * raw_hx + (1 - a) * self.hx
        self.hy = a * raw_hy + (1 - a) * self.hy
        self.hz = a * raw_hz + (1 - a) * self.hz
        self.roll = a * raw_roll + (1 - a) * self.roll

        return self.hx, self.hy, self.hz, self.roll, True