"""
Planogram Analysis Pipeline for ShelfScan.

Integrates perspective correction, detection, and planogram compliance analysis.
Generates comprehensive visual reports with all three analysis axes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from planogram import PlanogramReference, PlanogramZone, create_visual_report, save_planogram_to_json
from perspective import select_points_interactive, parse_points


def map_yolo_world_class(cls_id: int, class_names: Dict[int, str]) -> str:
    """Mapea clases de YOLO-World a categorías de supermercado."""
    yolo_class = class_names.get(cls_id, "").lower()

    # Mapeo de clases de YOLO-World a nuestras categorías
    mapping = {
        # bebidas
        'bottle': 'bebidas', 'wine': 'bebidas', 'beer': 'bebidas', 'drink': 'bebidas',
        'water': 'bebidas', 'soda': 'bebidas', 'juice': 'bebidas',

        # lacteos
        'milk': 'lacteos', 'cheese': 'lacteos', 'yogurt': 'lacteos', 'butter': 'lacteos',

        # snacks
        'chips': 'snacks', 'cookie': 'snacks', 'cracker': 'snacks', 'snack': 'snacks',
        'pretzel': 'snacks', 'popcorn': 'snacks', 'nuts': 'snacks',

        # cereales
        'cereal': 'cereales', 'oat': 'cereales', 'granola': 'cereales',

        # limpieza
        'soap': 'limpieza', 'detergent': 'limpieza', 'cleaner': 'limpieza',
        'sponge': 'limpieza', 'brush': 'limpieza',

        # enlatados
        'can': 'enlatados', 'tin': 'enlatados', 'jar': 'enlatados',
        'canister': 'enlatados', 'container': 'enlatados',

        # aceites
        'oil': 'aceites', 'bottle': 'aceites', 'container': 'aceites',
        'olive': 'aceites', 'vinegar': 'aceites',

        # higiene
        'toothbrush': 'higiene', 'toothpaste': 'higiene', 'shampoo': 'higiene',
        'deodorant': 'higiene', 'cream': 'higiene',

        # confiteria
        'candy': 'confiteria', 'chocolate': 'confiteria', 'gum': 'confiteria',
        'cake': 'confiteria', 'donut': 'confiteria', 'pastry': 'confiteria',
        'cookie': 'confiteria', 'sweet': 'confiteria', 'dessert': 'confiteria',
    }

    return mapping.get(yolo_class, 'zona_vacia')


def create_sample_planogram(reference_image_path: Path, output_json_path: Path) -> PlanogramReference:
    """Crea un planograma de ejemplo con zonas predefinidas."""
    # Cargar imagen de referencia
    img = cv2.imread(str(reference_image_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {reference_image_path}")

    h, w = img.shape[:2]

    # Definir esquinas de referencia (asumiendo imagen rectangular)
    reference_corners = np.array([
        [0, 0],          # superior izquierda
        [w-1, 0],        # superior derecha
        [w-1, h-1],      # inferior derecha
        [0, h-1]         # inferior izquierda
    ], dtype=np.float32)

    # Definir zonas de ejemplo (dividir el estante en una cuadrícula 3x3)
    zones = []
    zone_width = w // 3
    zone_height = h // 3

    categories = ['bebidas', 'lacteos', 'snacks', 'cereales', 'limpieza',
                 'enlatados', 'aceites', 'higiene', 'confiteria']

    zone_id = 0
    for row in range(3):
        for col in range(3):
            if zone_id < len(categories):
                x1 = col * zone_width
                y1 = row * zone_height
                x2 = (col + 1) * zone_width if col < 2 else w
                y2 = (row + 1) * zone_height if row < 2 else h

                zone = PlanogramZone(
                    id=f"Z{zone_id+1:02d}",
                    expected_class=categories[zone_id],
                    bbox=(x1, y1, x2, y2)
                )
                zones.append(zone)
                zone_id += 1

    # Crear planograma
    planogram = PlanogramReference(
        reference_image_path=reference_image_path,
        reference_corners=reference_corners,
        zones=zones,
        description="Planograma de ejemplo generado automáticamente"
    )

    # Guardar como JSON
    save_planogram_to_json(planogram, output_json_path)
    print(f"Planograma de ejemplo guardado en: {output_json_path}")

    return planogram


def process_single_image(image_path: Path, planogram: PlanogramReference,
                        points: np.ndarray | None, output_dir: Path,
                        model_path: str) -> Dict:
    """Procesa una imagen individual y genera análisis completo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar imagen
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    # Para imágenes ya corregidas, usar puntos por defecto
    if points is None:
        h, w = img.shape[:2]
        # Asumir que la imagen ya está corregida y usar puntos de esquina
        points = np.array([
            [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
        ], dtype=np.float32)

    # Corregir perspectiva (aunque ya esté corregida, para consistencia)
    warped = planogram.warp_real_image(img, points)

    # Ejecutar detección usando YOLO-World
    warped_path = output_dir / f"{image_path.stem}_warped{image_path.suffix}"
    cv2.imwrite(str(warped_path), warped)

    # Usar YOLO-World para detección con categorías de supermercado
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model.predict(
            source=warped,  # Pasar imagen directamente en lugar de path
            conf=0.1,  # Bajamos el threshold para detectar más objetos
            verbose=False,
            classes=None  # Detectar todas las clases
        )

        detections = []
        if results:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Mapear clases de YOLO-World a nuestras categorías
                class_name = map_yolo_world_class(cls_id, results[0].names)

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                })

        # Guardar imagen anotada
        annotated_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
        annotated = results[0].plot() if results else warped.copy()
        cv2.imwrite(str(annotated_path), annotated)

    except Exception as e:
        print(f"Error en detección: {e}")
        import traceback
        traceback.print_exc()  # Agregar traceback para debugging
        detections = []
        annotated_path = warped_path

    # Calcular métricas
    compliance = planogram.calculate_compliance_score(detections)
    metrics = planogram.calculate_metrics(detections)

    # Generar reporte visual
    report_path = output_dir / f"{image_path.stem}_report.png"
    create_visual_report(planogram, img, points, detections, report_path)

    # Guardar resultados como JSON
    results = {
        'image_path': str(image_path),
        'warped_path': str(warped_path),
        'annotated_path': str(annotated_path),
        'report_path': str(report_path),
        'perspective_points': points.tolist(),
        'detections': detections,
        'compliance': compliance,
        'metrics': metrics
    }

    results_path = output_dir / f"{image_path.stem}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Procesamiento completado para: {image_path.name}")
    print(f"  - Imagen corregida: {warped_path}")
    print(f"  - Imagen anotada: {annotated_path}")
    print(f"  - Reporte visual: {report_path}")
    print(f"  - Resultados JSON: {results_path}")
    print(".2f")
    return results


def run_batch_analysis(image_dir: Path, planogram_json: Path, output_dir: Path,
                      model_path: str, points_file: Path | None = None) -> List[Dict]:
    """Ejecuta análisis por lotes sobre múltiples imágenes."""
    # Cargar planograma
    planogram = load_planogram_from_json(planogram_json)

    # Cargar puntos de perspectiva si existen
    default_points = None
    if points_file and points_file.exists():
        with open(points_file, 'r') as f:
            points_data = json.load(f)
            default_points = np.array(points_data.get('points', []))

    # Encontrar imágenes
    image_extensions = {'.jpg', '.jpeg', '.png'}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions]

    if not images:
        print(f"No se encontraron imágenes en: {image_dir}")
        return []

    results = []
    for img_path in sorted(images):
        try:
            result = process_single_image(img_path, planogram, default_points,
                                        output_dir, model_path)
            results.append(result)
        except Exception as e:
            print(f"Error procesando {img_path.name}: {e}")
            continue

    # Generar resumen del lote
    generate_batch_summary(results, output_dir)

    return results


def generate_batch_summary(results: List[Dict], output_dir: Path) -> None:
    """Genera un resumen estadístico del análisis por lotes."""
    if not results:
        return

    summary = {
        'total_images': len(results),
        'average_compliance': np.mean([r['compliance']['overall_compliance'] for r in results]),
        'average_general_score': np.mean([r['metrics']['general_score'] for r in results]),
        'breakage_by_category': {},
        'compliance_distribution': []
    }

    # Agregar métricas por categoría
    all_categories = set()
    for result in results:
        for cat in result['metrics']['breakage_by_category'].keys():
            all_categories.add(cat)

    for cat in all_categories:
        breakage_values = [
            r['metrics']['breakage_by_category'].get(cat, 0)
            for r in results
        ]
        summary['breakage_by_category'][cat] = {
            'mean': np.mean(breakage_values),
            'std': np.std(breakage_values),
            'min': np.min(breakage_values),
            'max': np.max(breakage_values)
        }

    # Distribución de compliance
    compliance_scores = [r['compliance']['overall_compliance'] for r in results]
    summary['compliance_distribution'] = {
        'bins': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'counts': np.histogram(compliance_scores, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])[0].tolist()
    }

    # Guardar resumen
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Resumen del lote guardado en: {summary_path}")
    print(f"Imágenes procesadas: {summary['total_images']}")
    print(".2f")
def load_planogram_from_json(json_path: Path):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis de planograma completo para ShelfScan")
    parser.add_argument("--reference", "-r", type=Path,
                       help="Imagen de referencia del planograma")
    parser.add_argument("--planogram", "-p", type=Path,
                       help="Archivo JSON del planograma (se crea si no existe)")
    parser.add_argument("--image", "-i", type=Path,
                       help="Imagen individual para procesar")
    parser.add_argument("--batch-dir", "-b", type=Path,
                       help="Directorio con imágenes para procesamiento por lotes")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/planogram_results"),
                       help="Directorio de salida")
    parser.add_argument("--model", type=str, default="yolov8s-worldv2.pt",
                       help="Ruta al modelo YOLO (por defecto YOLO-World)")
    parser.add_argument("--points", nargs=4, help="Puntos de perspectiva (x,y x,y x,y x,y)")
    parser.add_argument("--points-file", type=Path,
                       help="Archivo JSON con puntos de perspectiva predefinidos")

    args = parser.parse_args()

    # Crear directorio de salida
    args.output.mkdir(parents=True, exist_ok=True)

    # Manejar puntos de perspectiva
    points = None
    if args.points:
        points = parse_points(args.points)

    # Crear o cargar planograma
    if args.reference and not args.planogram:
        # Crear planograma de ejemplo
        planogram_path = args.output / "planogram.json"
        planogram = create_sample_planogram(args.reference, planogram_path)
    elif args.planogram and args.planogram.exists():
        # Cargar planograma existente
        planogram = load_planogram_from_json(args.planogram)
    else:
        print("Error: Debe proporcionar una imagen de referencia (--reference) o un archivo de planograma existente (--planogram)")
        return

    # Procesar imagen individual
    if args.image:
        process_single_image(args.image, planogram, points, args.output, args.model)

    # Procesamiento por lotes
    elif args.batch_dir:
        run_batch_analysis(args.batch_dir, planogram.reference_image_path.parent / "planogram.json",
                          args.output, args.model, args.points_file)

    else:
        print("Debe especificar --image o --batch-dir para procesar imágenes")


if __name__ == "__main__":
    main()