"""
Perspective correction utility for ShelfScan.

This script implements manual four-point selection over a shelf image and applies
OpenCV homography-based warping using getPerspectiveTransform + warpPerspective.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

WINDOW_NAME = "Seleccionar 4 esquinas del estante"


def compute_destination_size(points: np.ndarray) -> tuple[int, int]:
    """Compute a destination rectangle size from four source points."""
    if points.shape != (4, 2):
        raise ValueError("points must be a 4x2 array")

    # Distancias horizontales y verticales
    width_top = np.linalg.norm(points[0] - points[1])
    width_bottom = np.linalg.norm(points[3] - points[2])
    height_left = np.linalg.norm(points[0] - points[3])
    height_right = np.linalg.norm(points[1] - points[2])

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    if width < 1 or height < 1:
        raise ValueError("Computed warp output size is invalid")

    return width, height


def warp_perspective_image(image: np.ndarray, src_points: np.ndarray) -> np.ndarray:
    """Warp the input image to a frontal view using four corner points."""
    src = np.array(src_points, dtype=np.float32)
    width, height = compute_destination_size(src)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped


def draw_selection(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    display = image.copy()
    for idx, pt in enumerate(points):
        cv2.circle(display, pt, 6, (0, 255, 0), -1)
        cv2.putText(
            display,
            str(idx + 1),
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if len(points) == 4:
        cv2.polylines(display, [np.array(points, dtype=np.int32)], True, (0, 255, 255), 2)
    return display


def select_points_interactive(image_path: Path) -> np.ndarray:
    """Allow the user to select four shelf corners by clicking on the image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    points: list[tuple[int, int]] = []

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, draw_selection(image, points))
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print(
        "Selecciona 4 esquinas del estante en el orden: superior izquierda, superior derecha, "
        "inferior derecha, inferior izquierda. Presiona 'r' para reiniciar, 'q' para salir."
    )

    while True:
        display = draw_selection(image, points)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(WINDOW_NAME)
            raise SystemExit("Selección cancelada por el usuario.")
        if key == ord("r"):
            points.clear()
        if len(points) == 4 and key in {13, 10, ord("w")}:
            cv2.destroyWindow(WINDOW_NAME)
            return np.array(points, dtype=np.float32)

    # Esta línea nunca se alcanza.


def parse_point_string(point_str: str) -> tuple[int, int]:
    x_str, y_str = point_str.split(",")
    return int(x_str), int(y_str)


def parse_points(args_points: Iterable[str]) -> np.ndarray:
    if len(args_points) != 4:
        raise ValueError("Se requieren exactamente 4 puntos en el formato x,y")
    pts = [parse_point_string(item) for item in args_points]
    return np.array(pts, dtype=np.float32)


def save_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise IOError(f"No se pudo guardar la imagen en: {output_path}")


def process_image(
    image_path: Path,
    points: np.ndarray | None = None,
    output_dir: Path | None = None,
    show: bool = False,
) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    if points is None:
        points = select_points_interactive(image_path)

    warped = warp_perspective_image(image, points)
    output_dir = output_dir or image_path.parent
    output_path = output_dir / f"{image_path.stem}_warped{image_path.suffix}"
    save_image(warped, output_path)

    print(f"Imagen corregida guardada en: {output_path}")
    if show:
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return output_path


def process_directory(input_dir: Path, output_dir: Path | None = None, show: bool = False) -> None:
    output_dir = output_dir or input_dir / "warped"
    images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg")) + sorted(input_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No se encontraron imágenes en el directorio: {input_dir}")

    for image_path in images:
        print(f"\nProcesando: {image_path.name}")
        try:
            process_image(image_path, output_dir=output_dir, show=show)
        except SystemExit as exc:
            print(f"Omitiendo {image_path.name}: {exc}")
        except Exception as exc:
            print(f"Error al procesar {image_path.name}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Corrección de perspectiva de estantes con selección de 4 puntos.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("-i", "--image", type=Path, help="Ruta de la imagen a procesar.")
    target.add_argument("-d", "--input-dir", type=Path, help="Directorio con imágenes JPG/JPEG/PNG a procesar.")
    parser.add_argument(
        "-p",
        "--points",
        nargs=4,
        help="Cuatro puntos x,y en orden TL, TR, BR, BL. Ej: 100,120 520,130 540,400 90,390",
    )
    parser.add_argument("-o", "--output-dir", type=Path, help="Directorio de salida para las imágenes corregidas.")
    parser.add_argument("--show", action="store_true", help="Mostrar la imagen corregida al terminar.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    points = None
    if args.points is not None:
        points = parse_points(args.points)

    if args.image:
        process_image(args.image, points=points, output_dir=args.output_dir, show=args.show)
    else:
        process_directory(args.input_dir, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
