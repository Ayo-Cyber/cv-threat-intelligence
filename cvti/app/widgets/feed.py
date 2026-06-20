"""Live video feed widget."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel


class FeedWidget(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background:#111; border-radius:6px;")
        self.setText("No feed — press Start")
        self.setStyleSheet("background:#111; color:#555; font-size:16px; border-radius:6px;")

    def update_frame(self, frame: np.ndarray) -> None:
        h, w, ch = frame.shape
        rgb = frame[:, :, ::-1].copy()  # BGR → RGB
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
