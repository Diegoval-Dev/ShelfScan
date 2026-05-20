"""
Validación de planograma y correlación de cumplimiento vs. quiebre.

Este script usa resultados JSON generados por `scripts/planogram_analysis.py`
para calcular:
  - puntuación de cumplimiento por imagen
  - quiebre promedio detectado
  - correlación entre cumplimiento y quiebre
  - error absoluto vs. ground truth manual (conteo, share of shelf, quiebre)
  - reportes visuales finales

Formato esperado de truth CSV:
  image_name,category,manual_count,manual_share,manual_breakage

Donde:
  - image_name es el nombre de archivo sin extensión o con extensión
  - manual_count es un entero opcional
  - manual_share es un valor [0,1] opcional
  - manual_breakage es un valor [0,1] opcional
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float('nan')
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    if denom == 0:
        return float('nan')
    return float((x * y).sum() / denom)


def load_results(results_dir: Path) -> List[Dict]:
    results = []
    for json_path in sorted(results_dir.glob("*_results.json")):
        with open(json_path, 'r', encoding='utf-8') as f:
            results.append(json.load(f))
    return results


def normalize_image_name(path: str) -> str:
    p = Path(path)
    return p.name


def load_ground_truth_csv(gt_path: Path) -> Dict[str, List[Dict]]:
    ground_truth = {}
    with open(gt_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = normalize_image_name(row.get('image_name', '').strip())
            if not image_name:
                continue
            category = row.get('category', '').strip()
            if not category:
                continue

            entry = {
                'category': category,
                'manual_count': int(row['manual_count']) if row.get('manual_count') and row['manual_count'].isdigit() else None,
                'manual_share': parse_optional_float(row.get('manual_share')),
                'manual_breakage': parse_optional_float(row.get('manual_breakage'))
            }

            ground_truth.setdefault(image_name, []).append(entry)
    return ground_truth


def build_prediction_summary(result: Dict) -> Dict:
    detections = result.get('detections', [])
    metrics = result.get('metrics', {})

    counts_by_category: Dict[str, int] = {}
    for detection in detections:
        cat = detection.get('class_name', 'zona_vacia')
        counts_by_category[cat] = counts_by_category.get(cat, 0) + 1

    breakage_by_category = metrics.get('breakage_by_category', {})
    avg_breakage = float(np.mean(list(breakage_by_category.values()))) if breakage_by_category else 0.0

    return {
        'image_name': normalize_image_name(result.get('image_path', '')),
        'compliance': float(result.get('compliance', {}).get('overall_compliance', 0.0)),
        'avg_breakage': avg_breakage,
        'predicted_counts': counts_by_category,
        'predicted_share': metrics.get('share_of_shelf_actual', {}),
        'predicted_breakage': breakage_by_category,
    }


def validate_ground_truth(results: List[Dict], ground_truth: Dict[str, List[Dict]]) -> Dict:
    errors = []
    for result in results:
        image_name = normalize_image_name(result.get('image_path', result.get('image_name', '')))
        if image_name not in ground_truth:
            continue
        summary = build_prediction_summary(result)
        for row in ground_truth[image_name]:
            category = row['category']
            predicted_count = summary['predicted_counts'].get(category, 0)
            predicted_share = summary['predicted_share'].get(category, 0.0)
            predicted_breakage = summary['predicted_breakage'].get(category, 0.0)

            row_error = {
                'image_name': image_name,
                'category': category,
                'manual_count': row['manual_count'],
                'predicted_count': predicted_count,
                'count_error': None,
                'count_pct_error': None,
                'manual_share': row['manual_share'],
                'predicted_share': predicted_share,
                'share_error': None,
                'manual_breakage': row['manual_breakage'],
                'predicted_breakage': predicted_breakage,
                'breakage_error': None,
            }

            if row['manual_count'] is not None:
                row_error['count_error'] = abs(predicted_count - row['manual_count'])
                if row['manual_count'] > 0:
                    row_error['count_pct_error'] = abs(predicted_count - row['manual_count']) / row['manual_count']
            if row['manual_share'] is not None:
                row_error['share_error'] = abs(predicted_share - row['manual_share'])
            if row['manual_breakage'] is not None:
                row_error['breakage_error'] = abs(predicted_breakage - row['manual_breakage'])

            errors.append(row_error)

    if not errors:
        return {'errors': [], 'summary': {}}

    share_errors = [e['share_error'] for e in errors if e['share_error'] is not None]
    breakage_errors = [e['breakage_error'] for e in errors if e['breakage_error'] is not None]
    count_errors = [e['count_error'] for e in errors if e['count_error'] is not None]
    count_pct_errors = [e['count_pct_error'] for e in errors if e['count_pct_error'] is not None]

    summary = {
        'n_rows': len(errors),
        'mean_share_error': float(np.mean(share_errors)) if share_errors else None,
        'mean_breakage_error': float(np.mean(breakage_errors)) if breakage_errors else None,
        'mean_count_error': float(np.mean(count_errors)) if count_errors else None,
        'mean_count_pct_error': float(np.mean(count_pct_errors)) if count_pct_errors else None,
        'count_error_std': float(np.std(count_errors)) if count_errors else None,
    }

    return {'errors': errors, 'summary': summary}


def calculate_correlation(results: List[Dict]) -> Dict:
    compliances = []
    breakages = []
    for result in results:
        summary = build_prediction_summary(result)
        compliances.append(summary['compliance'])
        breakages.append(summary['avg_breakage'])

    x = np.array(compliances)
    y = np.array(breakages)
    if x.size == 0 or y.size == 0:
        correlation = float('nan')
        correlation_note = 'No hay datos suficientes.'
    elif np.allclose(x, x[0]) or np.allclose(y, y[0]):
        correlation = float('nan')
        correlation_note = 'No se puede calcular correlación: valores constantes en cumplimiento o quiebre.'
    else:
        correlation = pearson_correlation(x, y)
        correlation_note = 'Correlación calculada correctamente.'

    return {
        'mean_compliance': float(np.mean(compliances)) if compliances else None,
        'mean_avg_breakage': float(np.mean(breakages)) if breakages else None,
        'compliance_breakage_correlation': correlation,
        'correlation_note': correlation_note,
        'n_images': len(results)
    }


def create_validation_plots(metrics: Dict, validation: Dict, output_path: Path) -> None:
    plt.figure(figsize=(14, 10))

    # Scatter compliance vs breakage
    plt.subplot(2, 2, 1)
    plt.scatter(metrics['compliances'], metrics['avg_breakages'], c='blue', alpha=0.7)
    plt.xlabel('Score de Cumplimiento')
    plt.ylabel('Quiebre Promedio Detectado')
    plt.title('Cumplimiento vs. Quiebre Detectado')
    plt.grid(True)

    # Error absolute bars if ground truth exists
    if validation['summary']:
        categories = []
        values = []
        labels = []
        if validation['summary']['mean_share_error'] is not None:
            categories.append('Share error')
            values.append(validation['summary']['mean_share_error'])
            labels.append('Share error del estante')
        if validation['summary']['mean_breakage_error'] is not None:
            categories.append('Breakage error')
            values.append(validation['summary']['mean_breakage_error'])
            labels.append('Error % quiebre')
        if validation['summary']['mean_count_error'] is not None:
            categories.append('Count error')
            values.append(validation['summary']['mean_count_error'])
            labels.append('Error conteo')

        plt.subplot(2, 2, 2)
        plt.bar(labels, values, color=['orange', 'red', 'purple'][:len(values)], alpha=0.8)
        plt.title('Error Absoluto Promedio vs Ground Truth')
        plt.ylabel('Error promedio')
        plt.xticks(rotation=20)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

    # If there is a ground truth breakdown, show predicted vs manual share scatter
    if validation['errors']:
        manual_share = [e['manual_share'] for e in validation['errors'] if e['manual_share'] is not None]
        predicted_share = [e['predicted_share'] for e in validation['errors'] if e['manual_share'] is not None]
        if manual_share and predicted_share:
            plt.subplot(2, 2, 3)
            plt.scatter(manual_share, predicted_share, c='green', alpha=0.7)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlabel('Share manual')
            plt.ylabel('Share predicho')
            plt.title('Share of Shelf: manual vs predicho')
            plt.grid(True)

        manual_breakage = [e['manual_breakage'] for e in validation['errors'] if e['manual_breakage'] is not None]
        predicted_breakage = [e['predicted_breakage'] for e in validation['errors'] if e['manual_breakage'] is not None]
        if manual_breakage and predicted_breakage:
            plt.subplot(2, 2, 4)
            plt.scatter(manual_breakage, predicted_breakage, c='red', alpha=0.7)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlabel('Quiebre manual')
            plt.ylabel('Quiebre predicho')
            plt.title('Quiebre: manual vs predicho')
            plt.grid(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reporte de validación guardado en: {output_path}")


def save_validation_errors_csv(validation: Dict, output_path: Path) -> None:
    if not validation['errors']:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'image_name', 'category', 'manual_count', 'predicted_count', 'count_error',
        'count_pct_error', 'manual_share', 'predicted_share', 'share_error',
        'manual_breakage', 'predicted_breakage', 'breakage_error'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in validation['errors']:
            writer.writerow(row)
    print(f"Detalles de errores guardados en: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validación de planograma y correlación de cumplimiento vs quiebre detectado")
    parser.add_argument('--results-dir', '-r', type=Path, required=True,
                        help='Directorio con archivos *_results.json generados por planogram_analysis')
    parser.add_argument('--ground-truth', '-g', type=Path,
                        help='CSV con datos manuales para validación')
    parser.add_argument('--output-dir', '-o', type=Path, default=Path('data/planogram_validation'),
                        help='Directorio de salida para métricas y reportes')
    parser.add_argument('--report-name', type=str, default='validation_report.png',
                        help='Nombre del reporte visual de validación')
    parser.add_argument('--save-json', type=Path,
                        help='Archivo JSON opcional donde guardar métricas de validación')
    parser.add_argument('--save-errors', type=Path,
                        help='Archivo CSV opcional donde guardar los errores detallados vs ground truth')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results_dir)
    if not results:
        raise SystemExit(f"No se encontraron resultados en {args.results_dir}")

    summary = calculate_correlation(results)
    summary['images'] = [build_prediction_summary(r) for r in results]

    validation = {'errors': [], 'summary': {}}
    if args.ground_truth:
        validation = validate_ground_truth(results, load_ground_truth_csv(args.ground_truth))

    summary['validation'] = validation
    report_path = args.output_dir / args.report_name
    summary['plot_path'] = str(report_path)
    create_validation_plots({
        'compliances': [float(x['compliance']) for x in summary['images']],
        'avg_breakages': [float(x['avg_breakage']) for x in summary['images']],
    }, validation, report_path)

    json_path = args.save_json or args.output_dir / 'validation_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Resumen de validación guardado en: {json_path}")

    if args.save_errors and validation['errors']:
        save_validation_errors_csv(validation, args.save_errors)

    corr_value = summary['compliance_breakage_correlation']
    if np.isnan(corr_value):
        print(f"Correlación cumplimiento/quiebre: NaN ({summary.get('correlation_note', '')})")
    else:
        print(f"Correlación cumplimiento/quiebre: {corr_value:.4f}")

    if validation['summary']:
        print("Resumen de errores vs ground truth:")
        for key, value in validation['summary'].items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
