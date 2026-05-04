# ShelfScan — Módulo de Detección

Detección de productos y zonas vacías en estantes de supermercado usando YOLOv8.

## Categorías (10 clases)

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | bebidas | agua, refresco, jugos |
| 1 | lacteos | leche, yogurt, queso |
| 2 | snacks | papas, galletas, barras |
| 3 | cereales | cereales de caja, avena |
| 4 | limpieza | detergente, jabón, desinfectante |
| 5 | enlatados | atún, frijoles, sardinas |
| 6 | aceites | aceite, vinagre, salsas |
| 7 | higiene | shampoo, pasta dental, desodorante |
| 8 | confiteria | dulces, chocolate, chicle |
| 9 | zona_vacia | espacio vacío en el estante |

---

## Entrenamiento con GPU NVIDIA (para el compañero)

> **Este es el camino rápido si tienes GPU NVIDIA.**

### 1. Clonar y preparar entorno

```bash
git clone <repo-url>
cd ShelfScan
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verificar CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # debe ser True
```

### 2. Descargar datasets

```bash
# Requiere ~/.kaggle/kaggle.json (obtener en kaggle.com → Settings → API)
python scripts/download_dataset.py

# Open Images (Google, sin auth — descarga clases con pocas muestras)
python scripts/download_openimages.py
```

### 3. Auto-labelear imágenes sin anotar

```bash
python scripts/autolabel.py --input data/raw --output data/annotated
```

### 4. Augmentation y split

```bash
python scripts/augmentation.py
python scripts/split_dataset.py
```

### 5. Entrenar

```bash
python scripts/train.py
```

El script detecta CUDA automáticamente. El modelo entrena en GPU.
Al terminar guarda en `models/shelfscan_v1/weights/best.pt`.

### 6. Compartir el modelo entrenado

El archivo `best.pt` no se sube a git (binario grande).
Compartir por Google Drive u otro medio y pasarle el link a Diego.

Diego lo pone en: `models/shelfscan_v1/weights/best.pt`

---

## Uso del modelo (inferencia)

```bash
python scripts/inference.py ruta/imagen.jpg
# Output: JSON con class_name, confidence, bbox
python scripts/inference.py ruta/imagen.jpg models/shelfscan_v1/weights/best.pt
```

## Corrección de perspectiva

```bash
# Interactivo — click en 4 esquinas del estante
python scripts/perspective.py -i ruta/imagen.jpg

# Con puntos desde línea de comandos
python scripts/perspective.py -i ruta/imagen.jpg -p 100,120 520,130 540,400 90,390

# Directorio completo
python scripts/perspective.py -d data/raw -o data/warped
```

---

## Estructura

```
ShelfScan/
├── data/
│   ├── raw/              # imágenes sin anotar (ignorado en git)
│   ├── annotated/        # imágenes + labels YOLO (ignorado en git)
│   ├── augmented/        # salida de augmentation.py (ignorado en git)
│   └── dataset.yaml      # config YOLOv8
├── models/               # pesos entrenados (ignorado en git — compartir via Drive)
├── scripts/
│   ├── categories.py
│   ├── download_dataset.py
│   ├── download_openimages.py
│   ├── autolabel.py
│   ├── remap_labels.py
│   ├── augmentation.py
│   ├── split_dataset.py
│   ├── train.py
│   ├── inference.py
│   ├── perspective.py
│   └── planogram.py
├── notebooks/
│   └── entrega1_training.ipynb
└── requirements.txt
```

---

## Métricas objetivo

| Métrica | Entrega 1 (obtenido) | Entrega 2 objetivo | Entrega Final |
|---------|----------------------|--------------------|---------------|
| mAP@0.5 | 0.436 ✓ | > 0.55 | > 0.65 |
| mAP@0.5:0.95 | 0.263 | > 0.35 | > 0.45 |
| Precision | 0.780 | > 0.75 | > 0.70 |
| Recall | 0.425 | > 0.55 | > 0.65 |

## Dataset

- **Fuente:** Kaggle `humansintheloop/supermarket-shelves-dataset` + Open Images v7
- **Anotado con:** YOLO-World (auto-label) + makesense.ai (revisión)
- **Aumentado:** ~5× con rotación, brillo, blur, flip horizontal
