"""
Planograma de referencia para ShelfScan.

El planograma de referencia se define como la imagen fotografiada en condiciones controladas
que contiene la distribución ideal del estante. A partir de ese plano, se puede alinear
una imagen real y evaluar el cumplimiento espacial y la presencia de productos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PlanogramReference:
    reference_image_path: Path
    reference_corners: np.ndarray
    description: str = (
        "Imagen fotografiada en condiciones controladas que sirve como plantilla "
        "de distribución ideal del estante."
    )

    def load_image(self) -> np.ndarray:
        image = cv2.imread(str(self.reference_image_path))
        if image is None:
            raise FileNotFoundError(f"No se pudo abrir la imagen de referencia: {self.reference_image_path}")
        return image

    def reference_size(self) -> tuple[int, int]:
        if self.reference_corners.shape != (4, 2):
            raise ValueError("reference_corners debe ser un arreglo 4x2")
        width = int(max(np.linalg.norm(self.reference_corners[0] - self.reference_corners[1]),
                        np.linalg.norm(self.reference_corners[3] - self.reference_corners[2])))
        height = int(max(np.linalg.norm(self.reference_corners[0] - self.reference_corners[3]),
                         np.linalg.norm(self.reference_corners[1] - self.reference_corners[2])))
        return width, height

    def compute_alignment_matrix(self, real_corners: np.ndarray) -> np.ndarray:
        if real_corners.shape != (4, 2):
            raise ValueError("real_corners debe ser un arreglo 4x2")
        dst = np.array(
            [[0, 0], [self.reference_size()[0] - 1, 0],
             [self.reference_size()[0] - 1, self.reference_size()[1] - 1],
             [0, self.reference_size()[1] - 1]],
            dtype=np.float32,
        )
        src = np.array(real_corners, dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def warp_real_image(self, real_image: np.ndarray, real_corners: np.ndarray) -> np.ndarray:
        matrix = self.compute_alignment_matrix(real_corners)
        width, height = self.reference_size()
        return cv2.warpPerspective(real_image, matrix, (width, height))


def describe_planogram() -> str:
    return (
        "Un planograma de referencia es una imagen base tomada bajo condiciones controladas "
        "que representa la distribución ideal de los productos en el estante. "
        "Su propósito es servir como modelo para alinear imágenes reales y calcular "
        "métricas de cumplimiento espacial."
    )


if __name__ == "__main__":
    print(describe_planogram())
