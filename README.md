# Crop-Pest-and-Disease-Detection-
I  developed  a classification model that identifies different types of crop pests and diseases from the images. The goal is to classify images into categories such as "healthy", "infected",
or "pest."
# Crop Pest & Disease Classification with ResNet-18  
 
**Date:** November 2025  

---

## Overview  

This project classifies agricultural crop images into three categories:  

* **Healthy** – `healthy`  
* **Disease-affected** – `disease-affected`  
* **Pest-infested** – `pest-infested`  

We use **transfer learning** with **ResNet-18** (pre-trained on ImageNet).  
The dataset is highly imbalanced (≈22 k images, 3 classes).  

---

## Dataset  

| Class               | # Images |
|---------------------|----------|
| `healthy`           | 1 369    |
| `disease-affected`  | 15 173   |
| `pest-infested`     | 5 770    |
| **Total**           | **22 312** |

*Source:* `crop_pest_and_desiase_classification.zip` (extracted to `/content/crop_pest_and_desiase_classification`).  

---

## 1. Pre-processing & Grouping  

```python
import os, shutil
from PIL import Image

extract_path = '/content/crop_pest_and_desiase_classification'
output_path  = '/content/organized_dataset'

# ---- class mapping -------------------------------------------------
healthy_classes = ['Cassava healthy', 'Maize healthy', 'Tomato healthy', 'Cashew healthy']
disease_classes = [
    'Cassava bacterial blight', 'Cassava mosaic', 'Maize streak virus', 'Maize leaf blight',
    'Maize leaf spot', 'Tomato leaf blight', 'Tomato septoria leaf spot', 'Tomato leaf curl',
    'Tomato verticulium wilt', 'Cashew anthracnose', 'Cashew red rust', 'Cashew gumosis'
]
pest_classes = [
    'Cassava green mite', 'Maize fall armyworm', 'Maize grasshoper', 'Maize leaf beetle',
    'Cashew leaf miner', 'Cassava brown spot'
]

# ---- create target folders -----------------------------------------
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(os.path.join(output_path, 'healthy'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'disease-affected'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'pest-infested'), exist_ok=True)

# ---- copy & verify images -----------------------------------------
valid_ext = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp')
for cls in os.listdir(extract_path):
    src = os.path.join(extract_path, cls)
    if not os.path.isdir(src): continue
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if not os.path.isfile(p) or not p.lower().endswith(valid_ext): continue
        try:
            with Image.open(p) as img: img.verify()
            with Image.open(p) as img: img.load()
            if cls in healthy_classes:
                shutil.copy(p, os.path.join(output_path, 'healthy', f))
            elif cls in disease_classes:
                shutil.copy(p, os.path.join(output_path, 'disease-affected', f))
            elif cls in pest_classes:
                shutil.copy(p, os.path.join(output_path, 'pest-infested', f))
        except Exception as e:
            print(f"Corrupted → {p} ({e})")
