# TFG — Psicología del Color en Productos de Consumo

**Autor:** Adrián Julve
**Universidad:** UFV — Business Analytics
**Curso:** 2024-2025

---

## Descripción del proyecto

Este TFG analiza la relación entre los colores dominantes de los productos de consumo y las emociones que generan en los consumidores, usando técnicas de scraping, procesamiento de imagen, análisis exploratorio de datos e ingeniería del dato.

Se trabaja con **tres fuentes de datos** obtenidas mediante scraping:

| Fuente | Descripción | Nº registros |
|--------|-------------|-------------|
| **Amazon Berkeley Objects (ABO)** | Catálogo de productos Amazon con imágenes | ~1.500 |
| **Mahou** | Productos de la marca Mahou (cerveza/bebidas) | ~200 |
| **Open Food Facts (OFF)** | Base de datos abierta de productos alimenticios | ~5.000 |

---

## Estructura del repositorio

```
TFG/
│
├── 📂 DATOS SCRAPPING/              # Datasets generados y gráficos de análisis
│   ├── Dataset_combinado_sin_emociones.csv   # Dataset unificado (3 fuentes)
│   ├── Dataset_con_emociones.csv             # Dataset final con emociones asignadas
│   ├── comparativa_modelos.csv               # Resultados comparativa de modelos ML
│   │
│   ├── 📂 abo_data/
│   │   └── dataset_abo.csv                   # Mini-dataset fuente ABO
│   │
│   ├── 📂 mahou_data/
│   │   └── dataset_mahou.csv                 # Mini-dataset fuente Mahou
│   │
│   ├── 📂 openfoodfacts_data/
│   │   └── dataset_openfoodfacts.csv         # Mini-dataset fuente Open Food Facts
│   │
│   └── 📂 graficos/                          # Gráficos del análisis exploratorio
│       ├── 00a_distribucion_fuente.png
│       ├── 00b_top15_categorias.png
│       ├── 01_histograma_L_capping.png
│       ├── 02_distribucion_emociones.png
│       ├── 03-05_boxplot_LAB.png
│       ├── 06_correlacion.png
│       ├── 07_paleta_cromatica.png
│       ├── 08_heatmap_emocion_categoria.png
│       ├── 09_scatter_L_a.png
│       ├── 10_alineacion_emocional.png
│       └── ... (gráficos por fuente: /abo, /mahou, /openfoodfacts, /analisis_dato)
│
├── 📂 Asignación de emociones/      # Metodología y datos de asignación de emociones
│   ├── dataset_con_emociones.csv
│   ├── metodologia_asignacion_emociones.docx
│   └── 📂 graficos/
│
├── 📂 Datos para trabajar/          # Versiones intermedias del dataset
│   ├── Dataset combinado sin emociones.csv
│   └── dataset_con_emociones.csv
│
├── 📂 Documentos Entregables/       # Documentación oficial del TFG
│   ├── Adrián Julve - TFG UFV - Anexo I - Solicitud autorización TFG.pdf
│   └── Anexo II - Adrián Julve Anteproyecto TFG Business Analytics Anexo II.pdf
│
├── 📂 Documentación Ingenieria del Dato/
│   └── ingenieria_del_dato_TFG.pdf
│
├── 📜 SCRIPT 1 - SCRAPING Y COLOR.py                        # Scraping de las 3 fuentes + extracción de color
├── 📜 SCRIPT 2, VARIABLES ECONOMICAS, EMOCIONAS E INGENIERIA DEL DATO.py  # Variables, emociones e ingeniería
├── 📜 Exploración Inicial de las 3 fuentes de datos.py      # EDA inicial de las 3 fuentes
├── 📜 Script Final TFG Ingeniería del dato.py               # Script final consolidado
├── 📓 Script Final TFG Ingeniería del dato Notebook.ipynb   # Versión notebook interactiva
│
├── 📜 SCRAPING AMAZON BERKELEY OBJECTS.py   # Scraper específico ABO
├── 📜 SCRAPING MAHOU.py                     # Scraper específico Mahou
├── 📜 SCRAPING OFF.py                       # Scraper específico Open Food Facts
├── 📜 ABO.py                                # Script auxiliar ABO
│
├── 📜 Asignar emociones a dataset final.py          # Lógica de asignación de emociones
├── 📜 COmprobar columnas de los ficheros.py         # Validación de estructura de datos
├── 📜 Combinar todos los scrappers y montar final dataset.py  # Unión de los 3 datasets
├── 📜 Resultados Ingenieria del dato.py             # Análisis de resultados
├── 📜 MODELOS TFG.py                                # Modelos de Machine Learning
├── 📜 SCRIPT PROFESIONAL.py                         # Versión optimizada del pipeline
│
├── 📄 Credentials-Color-Psychology.pdf              # Referencia bibliográfica
├── 📄 The color of emotion A metric for implicit color associations.pdf
├── 📄 defensa_script1.pdf
│
└── 📄 README.md                              # Este archivo
```

---

## Pipeline del proyecto

```
1. SCRAPING
   ├── SCRAPING AMAZON BERKELEY OBJECTS.py  →  dataset_abo.csv
   ├── SCRAPING MAHOU.py                    →  dataset_mahou.csv
   └── SCRAPING OFF.py                      →  dataset_openfoodfacts.csv
            ↓
2. UNIÓN DE FUENTES
   └── Combinar todos los scrappers y montar final dataset.py
            → Dataset_combinado_sin_emociones.csv
            ↓
3. EXPLORACIÓN INICIAL (EDA)
   └── Exploración Inicial de las 3 fuentes de datos.py
            ↓
4. ASIGNACIÓN DE EMOCIONES
   └── Asignar emociones a dataset final.py
            → Dataset_con_emociones.csv
            ↓
5. INGENIERÍA DEL DATO + MODELOS
   ├── SCRIPT 2, VARIABLES ECONOMICAS, EMOCIONES E INGENIERIA DEL DATO.py
   ├── MODELOS TFG.py
   └── Resultados Ingenieria del dato.py
            → comparativa_modelos.csv + gráficos
```

---

## Variables principales del dataset final

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `fuente` | Categórica | Origen del producto (ABO / Mahou / OFF) |
| `categoria` | Categórica | Categoría del producto |
| `mean_L` | Numérica | Luminosidad media (espacio CIE Lab) |
| `mean_a` | Numérica | Canal a* del color (verde-rojo) |
| `mean_b` | Numérica | Canal b* del color (azul-amarillo) |
| `contrast_L` | Numérica | Contraste de luminosidad |
| `emocion` | Categórica | Emoción asignada al color del producto |
| `alineacion_emocional` | Numérica | Score de alineación emoción-color |

---

## Referencias

- [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
- [Open Food Facts](https://world.openfoodfacts.org/)
- Solli, M., & Lenz, R. (2010). *Color and emotions – a psychological study on the influence of color on human emotion*
- Kaya, N., & Epps, H. H. (2004). *Relationship between color and emotion*
