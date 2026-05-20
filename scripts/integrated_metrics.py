"""
Módulo de métricas integrado para ShelfScan.

Calcula métricas combinadas de detección, planograma y análisis temporal.
Genera reportes finales integrados.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def load_results_from_directory(results_dir: Path) -> List[Dict]:
    """Carga todos los resultados JSON de un directorio."""
    results = []
    for json_file in results_dir.glob("*_results.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            results.append(result)
    return results


def calculate_integrated_metrics(results: List[Dict]) -> Dict:
    """Calcula métricas integradas de todos los módulos."""
    if not results:
        return {}

    # Métricas de planograma
    compliance_scores = [r['compliance']['overall_compliance'] for r in results]
    general_scores = [r['metrics']['general_score'] for r in results]

    # Métricas de detección
    total_detections = sum(len(r['detections']) for r in results)
    avg_detections_per_image = total_detections / len(results)

    # Categorías detectadas
    all_categories = set()
    category_counts = {}
    for result in results:
        for detection in result['detections']:
            cat = detection['class_name']
            all_categories.add(cat)
            if cat not in category_counts:
                category_counts[cat] = 0
            category_counts[cat] += 1

    # Métricas de quiebre por categoría
    breakage_summary = {}
    for result in results:
        for cat, breakage in result['metrics']['breakage_by_category'].items():
            if cat not in breakage_summary:
                breakage_summary[cat] = []
            breakage_summary[cat].append(breakage)

    avg_breakage_by_category = {
        cat: np.mean(values) for cat, values in breakage_summary.items()
    }

    # Score de shelf health (métrica compuesta)
    shelf_health_score = np.mean([
        (compliance + (1 - breakage_avg)) / 2
        for compliance in compliance_scores
        for breakage_avg in [np.mean(list(avg_breakage_by_category.values()))]
    ])

    return {
        'total_images_processed': len(results),
        'planogram_metrics': {
            'avg_compliance_score': np.mean(compliance_scores),
            'avg_general_score': np.mean(general_scores),
            'compliance_std': np.std(compliance_scores),
            'general_score_std': np.std(general_scores)
        },
        'detection_metrics': {
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections_per_image,
            'unique_categories_detected': len(all_categories),
            'category_distribution': category_counts
        },
        'breakage_metrics': {
            'avg_breakage_by_category': avg_breakage_by_category,
            'overall_avg_breakage': np.mean(list(avg_breakage_by_category.values())),
            'most_problematic_category': max(avg_breakage_by_category.items(), key=lambda x: x[1])[0]
        },
        'shelf_health_score': shelf_health_score,
        'recommendations': generate_recommendations(avg_breakage_by_category, compliance_scores)
    }


def generate_recommendations(breakage_by_cat: Dict[str, float], compliance_scores: List[float]) -> List[str]:
    """Genera recomendaciones basadas en las métricas."""
    recommendations = []

    # Recomendaciones por quiebre
    high_breakage_cats = [cat for cat, breakage in breakage_by_cat.items() if breakage > 0.5]
    if high_breakage_cats:
        recommendations.append(f"Alto quiebre detectado en categorías: {', '.join(high_breakage_cats)}. Considerar reabastecimiento urgente.")

    # Recomendaciones por compliance
    avg_compliance = np.mean(compliance_scores)
    if avg_compliance < 0.3:
        recommendations.append("Cumplimiento del planograma muy bajo. Revisar disposición de productos.")
    elif avg_compliance < 0.7:
        recommendations.append("Cumplimiento del planograma moderado. Optimizar layout de estantería.")

    # Recomendaciones generales
    if not recommendations:
        recommendations.append("Sistema funcionando correctamente. Mantener estándares actuales.")

    return recommendations


def generate_integrated_report(metrics: Dict, output_path: Path) -> None:
    """Genera un reporte visual integrado con todas las métricas."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('ShelfScan - Reporte Integrado Final', fontsize=16, fontweight='bold')

    # 1. Score de salud del estante
    axes[0, 0].pie([metrics['shelf_health_score'], 1 - metrics['shelf_health_score']],
                   labels=['Saludable', 'Problemas'], autopct='%1.1f%%',
                   colors=['green', 'red'], startangle=90)
    axes[0, 0].set_title('Score de Salud del Estante')

    # 2. Cumplimiento del planograma
    compliance_data = metrics['planogram_metrics']
    axes[0, 1].bar(['Cumplimiento', 'Score General'],
                   [compliance_data['avg_compliance_score'], compliance_data['avg_general_score']],
                   color=['blue', 'orange'], alpha=0.7)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title('Métricas de Planograma')
    axes[0, 1].set_ylabel('Score')

    # 3. Quiebre por categoría
    breakage_data = metrics['breakage_metrics']['avg_breakage_by_category']
    categories = list(breakage_data.keys())
    breakage_values = list(breakage_data.values())

    axes[0, 2].barh(categories, breakage_values, color='red', alpha=0.7)
    axes[0, 2].set_xlabel('% Quiebre')
    axes[0, 2].set_title('Quiebre por Categoría')
    axes[0, 2].set_xlim(0, 1)

    # 4. Distribución de detecciones
    detection_data = metrics['detection_metrics']
    categories_det = list(detection_data['category_distribution'].keys())
    counts_det = list(detection_data['category_distribution'].values())

    axes[1, 0].bar(categories_det, counts_det, color='purple', alpha=0.7)
    axes[1, 0].set_xlabel('Categoría')
    axes[1, 0].set_ylabel('Detecciones')
    axes[1, 0].set_title('Distribución de Detecciones')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Estadísticas generales
    axes[1, 1].axis('off')
    stats_text = f"""
    Estadísticas Generales:

    Imágenes procesadas: {metrics['total_images_processed']}
    Detecciones totales: {detection_data['total_detections']}
    Categorías detectadas: {detection_data['unique_categories_detected']}
    Quiebre promedio: {metrics['breakage_metrics']['overall_avg_breakage']:.2f}
    Score de salud: {metrics['shelf_health_score']:.2f}
    """
    axes[1, 1].text(0.1, 0.8, stats_text, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # 6. Recomendaciones
    axes[1, 2].axis('off')
    recs_text = "Recomendaciones:\n\n" + "\n".join(f"• {rec}" for rec in metrics['recommendations'])
    axes[1, 2].text(0.1, 0.8, recs_text, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Reporte integrado guardado en: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cálculo de métricas integradas para ShelfScan")
    parser.add_argument("--results-dir", "-r", type=Path, required=True,
                       help="Directorio con resultados JSON")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/integrated_report.png"),
                       help="Archivo de salida para el reporte")
    parser.add_argument("--metrics-json", type=Path,
                       help="Guardar métricas como JSON")

    args = parser.parse_args()

    # Cargar resultados
    results = load_results_from_directory(args.results_dir)
    if not results:
        print(f"No se encontraron resultados en: {args.results_dir}")
        return

    # Calcular métricas integradas
    metrics = calculate_integrated_metrics(results)

    # Guardar métricas como JSON si se solicita
    if args.metrics_json:
        with open(args.metrics_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Métricas guardadas en: {args.metrics_json}")

    # Generar reporte visual
    generate_integrated_report(metrics, args.output)

    # Imprimir resumen
    print("\n" + "="*50)
    print("RESUMEN DE ANÁLISIS INTEGRADO")
    print("="*50)
    print(f"Imágenes procesadas: {metrics['total_images_processed']}")
    print(f"Avg compliance: {metrics['planogram_metrics']['avg_compliance_score']:.2f}")
    print(f"Avg general score: {metrics['planogram_metrics']['avg_general_score']:.2f}")
    print(f"Avg detecciones por imagen: {metrics['detection_metrics']['avg_detections_per_image']:.2f}")
    print(f"Categoría con más quiebre: {metrics['breakage_metrics']['most_problematic_category']}")
    print(f"Score de salud del estante: {metrics['shelf_health_score']:.2f}")
    print("\nRecomendaciones:")
    for rec in metrics['recommendations']:
        print(f"  • {rec}")


if __name__ == "__main__":
    main()