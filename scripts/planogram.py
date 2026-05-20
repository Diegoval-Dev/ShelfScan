"""
Planograma de referencia para ShelfScan.

El planograma de referencia se define como la imagen fotografiada en condiciones controladas
que contiene la distribución ideal del estante. A partir de ese plano, se puede alinear
una imagen real y evaluar el cumplimiento espacial y la presencia de productos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class PlanogramZone:
    """Representa una zona específica en el planograma."""
    id: str
    expected_class: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = 1.0

    def contains_point(self, x: int, y: int) -> bool:
        """Verifica si un punto está dentro de la zona."""
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def area(self) -> int:
        """Calcula el área de la zona."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class PlanogramReference:
    reference_image_path: Path
    reference_corners: np.ndarray
    zones: List[PlanogramZone]
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

    def find_zone_for_detection(self, detection_bbox: Tuple[int, int, int, int]) -> Optional[PlanogramZone]:
        """Encuentra la zona del planograma que contiene el centro de una detección."""
        x1, y1, x2, y2 = detection_bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        for zone in self.zones:
            if zone.contains_point(center_x, center_y):
                return zone
        return None

    def calculate_compliance_score(self, detections: List[Dict]) -> Dict:
        """Calcula el score de cumplimiento basado en las detecciones."""
        zone_compliance = {}
        total_zones = len(self.zones)
        compliant_zones = 0

        for zone in self.zones:
            zone_detections = []
            for detection in detections:
                bbox = tuple(detection['bbox'])
                if zone.contains_point((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2):
                    zone_detections.append(detection)

            # Una zona es compliant si tiene al menos una detección de la clase esperada
            has_expected_product = any(d['class_name'] == zone.expected_class for d in zone_detections)
            zone_compliance[zone.id] = {
                'expected': zone.expected_class,
                'detected_products': [d['class_name'] for d in zone_detections],
                'compliant': has_expected_product,
                'confidence': max([d['confidence'] for d in zone_detections] + [0])
            }

            if has_expected_product:
                compliant_zones += 1

        overall_score = compliant_zones / total_zones if total_zones > 0 else 0

        return {
            'overall_compliance': overall_score,
            'compliant_zones': compliant_zones,
            'total_zones': total_zones,
            'zone_details': zone_compliance
        }

    def calculate_metrics(self, detections: List[Dict]) -> Dict:
        """Calcula métricas base: % de quiebre, share of shelf por categoría."""
        # Calcular share of shelf por categoría
        category_areas = {}
        total_shelf_area = sum(zone.area() for zone in self.zones)

        for zone in self.zones:
            category = zone.expected_class
            if category not in category_areas:
                category_areas[category] = 0
            category_areas[category] += zone.area()

        # Calcular productos detectados por categoría
        detected_by_category = {}
        for detection in detections:
            category = detection['class_name']
            if category not in detected_by_category:
                detected_by_category[category] = []
            detected_by_category[category].append(detection)

        # Calcular share of shelf actual (basado en detecciones)
        actual_share = {}
        for category, dets in detected_by_category.items():
            # Área aproximada cubierta por detecciones de esta categoría
            area_covered = sum(
                (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                for det in dets
                for bbox in [det['bbox']]
            )
            actual_share[category] = area_covered / total_shelf_area if total_shelf_area > 0 else 0

        # Calcular % de quiebre por categoría
        breakage_by_category = {}
        for category in category_areas.keys():
            expected_area = category_areas[category]
            actual_area = actual_share.get(category, 0) * total_shelf_area
            breakage = max(0, (expected_area - actual_area) / expected_area) if expected_area > 0 else 0
            breakage_by_category[category] = breakage

        # Score general (promedio de compliance y share of shelf alignment)
        compliance = self.calculate_compliance_score(detections)
        share_alignment = np.mean([
            1 - abs(category_areas.get(cat, 0) - actual_share.get(cat, 0) * total_shelf_area) / category_areas.get(cat, 1)
            for cat in category_areas.keys()
        ])

        general_score = (compliance['overall_compliance'] + share_alignment) / 2

        return {
            'breakage_by_category': breakage_by_category,
            'share_of_shelf_expected': {cat: area / total_shelf_area for cat, area in category_areas.items()},
            'share_of_shelf_actual': actual_share,
            'general_score': general_score,
            'total_shelf_area': total_shelf_area
        }


def load_planogram_from_json(json_path: Path) -> PlanogramReference:
    """Carga un planograma desde un archivo JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    zones = [
        PlanogramZone(
            id=zone['id'],
            expected_class=zone['expected_class'],
            bbox=tuple(zone['bbox']),
            confidence=zone.get('confidence', 1.0)
        )
        for zone in data['zones']
    ]

    return PlanogramReference(
        reference_image_path=Path(data['reference_image_path']),
        reference_corners=np.array(data['reference_corners']),
        zones=zones,
        description=data.get('description', '')
    )


def save_planogram_to_json(planogram: PlanogramReference, json_path: Path) -> None:
    """Guarda un planograma en un archivo JSON."""
    data = {
        'reference_image_path': str(planogram.reference_image_path),
        'reference_corners': planogram.reference_corners.tolist(),
        'zones': [
            {
                'id': zone.id,
                'expected_class': zone.expected_class,
                'bbox': list(zone.bbox),
                'confidence': zone.confidence
            }
            for zone in planogram.zones
        ],
        'description': planogram.description
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_visual_report(planogram: PlanogramReference, real_image: np.ndarray,
                        real_corners: np.ndarray, detections: List[Dict],
                        output_path: Path) -> None:
    """Genera un reporte visual integrado con las tres líneas de análisis."""
    # Warp real image to align with planogram
    warped_real = planogram.warp_real_image(real_image, real_corners)

    # Load reference image
    reference = planogram.load_image()

    # Calculate metrics
    compliance = planogram.calculate_compliance_score(detections)
    metrics = planogram.calculate_metrics(detections)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('ShelfScan - Reporte de Análisis Integrado', fontsize=16, fontweight='bold')

    # 1. Imagen de referencia con zonas
    axes[0, 0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Planograma de Referencia')
    for zone in planogram.zones:
        x1, y1, x2, y2 = zone.bbox
        color = 'green' if compliance['zone_details'][zone.id]['compliant'] else 'red'
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
        axes[0, 0].add_patch(rect)
        axes[0, 0].text(x1, y1-5, f"{zone.id}: {zone.expected_class}", fontsize=8, color=color)

    # 2. Imagen real alineada
    axes[0, 1].imshow(cv2.cvtColor(warped_real, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Imagen Real Alineada')

    # 3. Detecciones sobre imagen alineada
    annotated = warped_real.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{detection['class_name']}:{detection['confidence']:.2f}",
                   (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    axes[0, 2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Detecciones')

    # 4. Share of shelf comparison
    categories = list(metrics['share_of_shelf_expected'].keys())
    expected = [metrics['share_of_shelf_expected'][cat] for cat in categories]
    actual = [metrics['share_of_shelf_actual'].get(cat, 0) for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    axes[1, 0].bar(x - width/2, expected, width, label='Esperado', alpha=0.8)
    axes[1, 0].bar(x + width/2, actual, width, label='Actual', alpha=0.8)
    axes[1, 0].set_xlabel('Categoría')
    axes[1, 0].set_ylabel('Share of Shelf')
    axes[1, 0].set_title('Comparación Share of Shelf')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories, rotation=45, ha='right')
    axes[1, 0].legend()

    # 5. Breakage by category
    breakage = [metrics['breakage_by_category'][cat] for cat in categories]
    axes[1, 1].bar(categories, breakage, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Categoría')
    axes[1, 1].set_ylabel('% Quiebre')
    axes[1, 1].set_title('Porcentaje de Quiebre por Categoría')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. Summary metrics
    axes[1, 2].axis('off')
    summary_text = (
        f"Score General: {metrics['general_score']:.2f}\n"
        f"Cumplimiento: {compliance['overall_compliance']:.2f}\n"
        f"Zonas Compliant: {compliance['compliant_zones']}/{compliance['total_zones']}\n"
        f"Área Total Estante: {metrics['total_shelf_area']} px²\n"
    )
    axes[1, 2].text(0.1, 0.8, summary_text, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Reporte visual guardado en: {output_path}")


def describe_planogram() -> str:
    return (
        "Un planograma de referencia es una imagen base tomada bajo condiciones controladas "
        "que representa la distribución ideal de los productos en el estante. "
        "Su propósito es servir como modelo para alinear imágenes reales y calcular "
        "métricas de cumplimiento espacial."
    )


if __name__ == "__main__":
    print(describe_planogram())
