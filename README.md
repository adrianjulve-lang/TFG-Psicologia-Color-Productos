# Análisis del Impacto del Color del Producto en las Emociones del Consumidor

**Adrián Julve Navarro · TFG Business Analytics · UFV 2025-2026**

---

## Índice

1. [Contexto y objetivo](#1-contexto-y-objetivo)
2. [Arquitectura del proyecto](#2-arquitectura-del-proyecto)
3. [Fuentes de datos](#3-fuentes-de-datos)
4. [Script 0 — Exploración inicial de las 3 fuentes](#4-script-0)
5. [Script 1 — Scraping y extracción de color](#5-script-1)
6. [Script 2 — Ingeniería del dato](#6-script-2)
7. [Datasets](#7-datasets)
8. [Modelos de Machine Learning](#8-modelos-de-machine-learning)
9. [Variables del sistema](#9-variables-del-sistema)
10. [Marco teórico y referencias](#10-marco-teórico-y-referencias)
11. [Requisitos técnicos](#11-requisitos-técnicos)

---

## 1. Contexto y objetivo

El color es el **primer estímulo visual** que procesa el cerebro al observar un producto: antes de leer el nombre, conocer el precio o analizar sus características, el consumidor ya ha generado una respuesta emocional basada exclusivamente en la información cromática.

Este TFG analiza esa relación de forma **cuantitativa y sistemática**. A partir de imágenes de productos de consumo obtenidas mediante web scraping, se extraen variables cromáticas y se emplean modelos de machine learning para:

- **Clasificar la emoción dominante** que transmite un producto (Random Forest, MLP).
- **Predecir el perfil emocional completo**: la intensidad con la que se manifiestan las 8 emociones simultáneamente (MultiOutput SVR — aportación original del TFG).

Este enfoque convierte la percepción del color, tradicionalmente subjetiva, en una herramienta cuantificable y aplicable en decisiones reales de diseño y marketing.

---

## 2. Arquitectura del proyecto

```
TFG/
├── Exploración Inicial de las 3 fuentes de datos.py   <- Script 0: reconocimiento sin descargar
├── SCRIPT 1 - SCRAPING Y COLOR.py                     <- Script 1: scraping + variables cromáticas
├── SCRIPT 2, VARIABLES ECONOMICAS, EMOCIONES E
│   INGENIERIA DEL DATO.py                             <- Script 2: ingeniería del dato completa
└── DATOS SCRAPPING/
    ├── Dataset_combinado_sin_emociones.csv             <- Output de Script 1
    ├── Dataset_con_emociones.csv                       <- Output de Script 2
    ├── abo_data/dataset_abo.csv
    ├── mahou_data/dataset_mahou.csv
    ├── openfoodfacts_data/dataset_openfoodfacts.csv
    └── graficos/
```

**Flujo de datos:**

```
3 Fuentes Web -> Script 0 (exploración) -> Script 1 (scraping + color)
-> Dataset sin emociones -> Script 2 (ingeniería del dato)
-> Dataset con emociones -> Modelos ML (Random Forest + MLP + SVR)
```

---

## 3. Fuentes de datos

| Característica | Amazon Berkeley Objects | Open Food Facts | Mahou San Miguel |
|---|---|---|---|
| **Tipo de acceso** | Descarga directa Amazon S3 | API REST pública | Web scraping Selenium |
| **Autenticación** | No requiere | No requiere | No requiere |
| **Formato** | JSON.gz + CSV.gz | JSON por petición | HTML dinámico |
| **Volumen estimado** | ~100.000+ productos | Millones de productos | ~81 productos |
| **Categorías** | Tecnología, hogar, moda... | Alimentación y bebidas | Cervezas, aguas... |
| **Condición fotográfica** | Fondo blanco estándar | Variable (consumidores) | Marketing profesional |
| **Sesgo cromático** | L* alto, b* bajo | L* medio, b* medio | Packaging premium |

---

## 4. Script 0 — Exploración inicial de las 3 fuentes

**Archivo:** `Exploración Inicial de las 3 fuentes de datos.py`

Fase de reconocimiento previo. **No descarga imágenes ni genera datasets.**
Se conecta a cada fuente, obtiene una muestra mínima y documenta qué variables existen, qué tipo tienen y cómo vienen estructuradas.

### Fuente 1 — Amazon Berkeley Objects (ABO)
1. Descarga el índice de imágenes (`images.csv.gz`) desde Amazon S3 y muestra columnas, tipos y porcentaje de nulos.
2. Descarga una muestra de 20 líneas de listings (`listings_0.json.gz`) y analiza todos los campos disponibles.
3. Muestra un producto completo de ejemplo para entender la estructura del JSON.

### Fuente 2 — Open Food Facts (API REST)
1. Lanza una petición de prueba con 5 productos de la categoría *cervezas*.
2. Lista los 100+ campos disponibles por producto e identifica los 7 que usará el proyecto (`code`, `product_name`, `brands`, `categories_tags`, `image_front_url`...).
3. Consulta cuántos productos hay en 10 categorías principales para estimar el volumen extraíble.

### Fuente 3 — Mahou San Miguel (Web Scraping)
1. Lista las 13 categorías del catálogo y sus rutas URL.
2. Verifica que la web responde (HTTP 200) y documenta por qué se necesita Selenium (contenido dinámico JavaScript).
3. Muestra la estructura exacta del dataset que generará el scraper (11 variables).

### Resumen comparativo
Tabla comparando las 3 fuentes en 9 dimensiones: tipo de acceso, volumen, sesgo cromático, velocidad de extracción, etc.

---

## 5. Script 1 — Scraping y extracción de color

**Archivo:** `SCRIPT 1 - SCRAPING Y COLOR.py`

Script principal de obtención de datos. Realiza el scraping completo de las tres fuentes y extrae las 7 variables cromáticas de cada imagen descargada.

### Función central: `calcular_color(ruta)`

Recibe la ruta de una imagen, la procesa pixel a pixel y devuelve 7 métricas cromáticas:

| Variable | Espacio | Descripción | Rango |
|---|---|---|---|
| `mean_R` | RGB | Media del canal Rojo | 0-255 |
| `mean_G` | RGB | Media del canal Verde | 0-255 |
| `mean_B` | RGB | Media del canal Azul | 0-255 |
| `mean_L` | CIELAB | Luminosidad media (oscuro a luminoso) | 0-100 |
| `mean_a` | CIELAB | Componente rojo-verde | -128 a 128 |
| `mean_b` | CIELAB | Componente amarillo-azul | -128 a 128 |
| `contrast_L` | CIELAB | Desviación típica de L* (contraste interno) | 0-100 |

El espacio **CIELAB** se usa porque distancias iguales corresponden a diferencias perceptivas iguales para el ojo humano, ideal para predecir respuestas emocionales.

### Parte 1 — Scraper Mahou San Miguel (Selenium)
Mahou usa JavaScript dinámico, se automatiza **Chrome en modo headless** con Selenium.
1. Abre Chrome en segundo plano y acepta cookies automáticamente.
2. Recorre 13 categorías del catálogo haciendo scroll automático (lazy loading).
3. Extrae nombre, URL de imagen (meta `og:image`) y calcula las 7 variables cromáticas.
4. Pausa de 3 segundos entre peticiones para no saturar el servidor.

### Parte 2 — Scraper Amazon Berkeley Objects (ABO)
1. Descarga archivos `.json.gz` desde Amazon S3 sin autenticación.
2. Extrae `item_id`, `item_name`, `product_type`, `main_image_id`.
3. Cruza con el índice de imágenes, descarga y calcula las 7 variables.

### Parte 3 — Scraper Open Food Facts (API REST)
1. Consulta la API por categorías (`en:beers`, `en:chocolates`, etc.).
2. Pagina en grupos de 100 productos con pausa de 1 segundo entre peticiones.
3. Extrae código de barras, nombre, marca, categorías y URL de imagen frontal.

**Output:** `Dataset_combinado_sin_emociones.csv` — ~10.000 productos × 11 columnas.

---

## 6. Script 2 — Ingeniería del dato

**Archivo:** `SCRIPT 2, VARIABLES ECONOMICAS, EMOCIONAS E INGENIERIA DEL DATO.py`

Pipeline de ingeniería del dato en 12 pasos. Transforma el dataset bruto en el dataset final listo para entrenar los modelos ML.

### Paso 1 — Gráficos del estado inicial
Genera `0a_distribucion_fuente.png` y `0b_distribucion_categoria.png` antes de ninguna transformación para documentar el punto de partida.

### Paso 2 — Tratamiento de nulos
- Menos del 1% de nulos: elimina las filas.
- 1% o más: sustituye por la **mediana** (más robusta que la media frente a outliers).

### Paso 3 — Eliminación de duplicados
Elimina filas con la misma URL de imagen, usándola como identificador único del producto.

### Paso 4 — Validación de rangos teóricos
Verifica que los valores están dentro de los rangos físicamente posibles. Documenta cualquier anomalía.

### Paso 5 — Outliers: IQR con capping
Aplica rango intercuartílico (Q1 - 1.5·IQR, Q3 + 1.5·IQR). Los colores extremos no se eliminan (son información válida), se limitan al borde del rango. Genera `1_antes_despues_outliers.png`.

### Paso 6 — Variables HSV

| Variable | Descripción | Rango |
|---|---|---|
| `hsv_h` | Tono: ángulo de color (0=rojo, 120=verde, 240=azul) | 0-360 |
| `hsv_s` | Saturación: pureza del color | 0-100% |
| `hsv_v` | Brillo: luminosidad en escala HSV | 0-100% |

### Paso 7 — Asignación de emoción (scoring gaussiano ponderado)

**Aportación metodológica central.** Asigna una emoción a cada producto sin etiquetado manual, usando funciones gaussianas en el espacio CIELAB calibradas con la literatura científica (Valdez & Mehrabian 1994, Gilbert et al. 2016, Russell 1980).

Las 8 emociones con sus centroides:

| Emoción | Centro L* | Centro C* | Peso L* | Peso C* | Peso H* |
|---|---|---|---|---|---|
| Ira | 22 | 10 | 0.50 | 0.35 | 0.15 |
| Tristeza | 42 | 4 | 0.65 | 0.30 | 0.05 |
| Romanticismo | 50 | 12 | 0.30 | 0.35 | 0.35 |
| Energía | 62 | 15 | 0.25 | 0.40 | 0.35 |
| Alegría | 80 | 16 | 0.30 | 0.35 | 0.35 |
| Relajación | 92 | 3 | 0.70 | 0.25 | 0.05 |
| Calma | 85 | 5 | 0.55 | 0.30 | 0.15 |
| Aburrimiento | 70 | 5 | 0.40 | 0.40 | 0.20 |

**Proceso:** Se calcula Croma C* y ángulo de tono H*. Para cada emoción se evalúa la gaussiana en L*, C* y H*, se suman con sus pesos y los 8 scores se normalizan con softmax. La emoción ganadora es la de mayor probabilidad. Si ninguna supera 0.22, se asigna "Neutro/Ambiguo".

Además de la emoción ganadora, se añaden las **8 columnas de score individual** que son los targets del modelo SVR.

### Paso 8 — Variables de negocio

| Variable | Descripción |
|---|---|
| `temperatura_color` | "Cálido" / "Frío" / "Neutro" según valores de a* y b* |
| `luminosidad_cat` | "Oscuro" / "Medio" / "Luminoso" según L* |
| `saturacion_cat` | "Apagado" / "Moderado" / "Intenso" según hsv_s |
| `coherencia_emocional` | 0-100: qué tan centrado está el producto en su zona emocional |
| `emocion_optima` | Emoción que debería transmitir esa categoría (diccionario de 90+ categorías) |
| `alineacion_emocional` | 1 si transmite la emoción óptima para su categoría, 0 si no |

### Paso 9 — Normalización Min-Max
Escala las 10 variables numéricas a [0, 1] con el sufijo `_norm` para que el SVR no se vea afectado por diferencias de escala.

### Paso 10 — 12 gráficos de análisis

| Gráfico | Descripción |
|---|---|
| `0a_distribucion_fuente.png` | Productos por fuente de datos |
| `0b_distribucion_categoria.png` | Top 15 categorías con más productos |
| `1_antes_despues_outliers.png` | L* antes y después del capping |
| `2_distribucion_emociones.png` | Distribución de las 8 emociones |
| `boxplot_mean_L/a/b.png` | Variables CIELAB por emoción (análisis bivariante) |
| `6_correlacion.png` | Matriz de correlación entre variables cromáticas |
| `7_paleta_colores_real.png` | Franja de color de todos los productos ordenada por tono |
| `8_mapa_calor_emocion_categoria.png` | Emoción predominante por categoría (%) |
| `9_scatter_separabilidad.png` | Separabilidad emocional en espacio CIELAB (L* vs a*) |
| `10_alineacion_por_categoria.png` | % de alineación emocional por categoría |
| `11_posicionamiento_cromatico.png` | Posicionamiento de cada fuente en L* vs b* |
| `12_contraste_por_emocion.png` | Contraste visual interno por emoción asignada |

**Output:** `Dataset_con_emociones.csv` — dataset final con 30+ columnas listo para ML.

---

## 7. Datasets

| Fichero | Descripción | Filas aprox. | Columnas |
|---|---|---|---|
| `Dataset_combinado_sin_emociones.csv` | Output de Script 1. Variables cromáticas brutas | ~10.000 | 11 |
| `Dataset_con_emociones.csv` | Output de Script 2. Dataset completo para ML | ~10.000 | 30+ |
| `abo_data/dataset_abo.csv` | Mini-dataset Amazon Berkeley Objects | ~8.000 | 11 |
| `mahou_data/dataset_mahou.csv` | Mini-dataset Mahou San Miguel | ~81 | 11 |
| `openfoodfacts_data/dataset_openfoodfacts.csv` | Mini-dataset Open Food Facts | ~2.000 | 11 |

---

## 8. Modelos de Machine Learning

**Features de entrada:** 10 variables cromáticas normalizadas (`mean_R_norm`, `mean_G_norm`, `mean_B_norm`, `mean_L_norm`, `mean_a_norm`, `mean_b_norm`, `contrast_L_norm`, `hsv_h_norm`, `hsv_s_norm`, `hsv_v_norm`).

### Modelo 1 — Random Forest Classifier
- **Target:** `emocion` (9 clases incluyendo Neutro/Ambiguo)
- Clasifica cada producto en su emoción dominante. Alta interpretabilidad mediante importancia de variables.

### Modelo 2 — Red Neuronal MLP (Perceptrón Multicapa)
- **Target:** `emocion` (mismas 9 clases)
- Mayor capacidad para detectar patrones no lineales en el espacio cromático.

### Modelo 3 — MultiOutput SVR (aportación original del TFG)
- **Target:** Los 8 scores emocionales en continuo
- Predice el **perfil emocional completo**: no solo la emoción dominante, sino la intensidad de las 8 emociones simultáneamente.
- Un packaging dorado puede ser alegre (0.42) + energético (0.31) + romántico (0.18). Un clasificador pierde toda esa riqueza; el SVR la preserva.
- **Métrica:** R2 y RMSE por emoción.

---

## 9. Variables del sistema

### Variables cromáticas (features del modelo)

| Variable | Espacio | Descripción | Rango |
|---|---|---|---|
| `mean_R/G/B` | RGB | Media de cada canal de color | 0-255 |
| `mean_L` | CIELAB | Luminosidad media | 0-100 |
| `mean_a` | CIELAB | Componente rojo-verde | -128 a 128 |
| `mean_b` | CIELAB | Componente amarillo-azul | -128 a 128 |
| `contrast_L` | CIELAB | Contraste interno (desv. típica de L*) | 0-100 |
| `hsv_h` | HSV | Tono (ángulo de color) | 0-360 |
| `hsv_s` | HSV | Saturación | 0-100% |
| `hsv_v` | HSV | Brillo | 0-100% |

### Variables emocionales (targets del modelo SVR)

`score_alegria` · `score_energia` · `score_calma` · `score_romanticismo` · `score_tristeza` · `score_ira` · `score_relajacion` · `score_aburrimiento`

### Variables de negocio

`temperatura_color` · `luminosidad_cat` · `saturacion_cat` · `coherencia_emocional` · `emocion_optima` · `alineacion_emocional`

---

## 10. Marco teórico y referencias

- **Valdez & Mehrabian (1994):** Cuantifica la relación entre luminosidad (L*) y respuestas emocionales pasivas (placer-arousal).
- **Gilbert et al. (2016):** Mapa de asociaciones implícitas color-emoción. Establece los pesos de Croma (C*) y ángulo de tono (H*) para emociones activas.
- **Russell (1980):** Modelo circunflejo del afecto. Las emociones se organizan en un espacio bidimensional continuo (valencia x activación), lo que justifica el enfoque de regresión del SVR.
- **Espacio CIELAB:** Estándar CIE 1976. Distancias iguales corresponden a diferencias perceptivas iguales para el ojo humano.

---

## 11. Requisitos técnicos

```bash
pip install requests pandas numpy matplotlib pillow scikit-image tqdm selenium webdriver-manager scikit-learn
```

Google Chrome debe estar instalado para el scraper de Mahou (ChromeDriver se descarga automáticamente con `webdriver-manager`).

**Orden de ejecución:** Script 0 → Script 1 → Script 2 → Script de modelos.

---

*TFG Business Analytics · Universidad Francisco de Vitoria · Adrián Julve Navarro · 2025-2026*
