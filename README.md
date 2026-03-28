# TFG — Psicología del Color en Productos de Consumo

**Autor:** Adrián Julve Navarro
**Universidad:** UFV — Grado en Business Analytics
**Curso:** 2024-2025

---

## ¿De qué trata este proyecto?

Este TFG investiga si el color de los productos de consumo está alineado con las emociones que se espera que transmitan según la psicología del color. La hipótesis de partida es que marcas y fabricantes diseñan su packaging de forma que el color dominante del producto evoque una emoción concreta (por ejemplo, una cerveza debería transmitir relajación, un refresco energía).

Para comprobarlo, se construye desde cero un dataset de más de 6.000 productos reales con sus colores extraídos automáticamente de imagen, se les asigna una emoción basada en modelos científicos del color, y se analiza cuántos productos están realmente alineados con la emoción óptima para su categoría.

---

## Fuentes de datos

Se trabaja con tres fuentes de datos complementarias, elegidas para cubrir distintos tipos de producto y condición fotográfica:

| Fuente | Tipo de acceso | Productos | Tipo de producto | Condición fotográfica |
|---|---|---|---|---|
| **Amazon Berkeley Objects (ABO)** | Descarga directa Amazon S3 (JSON.gz) | ~1.500 | Hogar, tecnología, cosmética, juguetes... | Fondo blanco estándar |
| **Open Food Facts (OFF)** | API REST pública | ~5.000 | Alimentación y bebidas (30+ categorías) | Variable (fotos de consumidores) |
| **Mahou San Miguel** | Web scraping con Selenium | ~200 | Cervezas, aguas, bebidas, merchandising | Fotografía de marketing |

Las tres fuentes se combinan en un **dataset unificado** con las mismas variables de color, lo que permite hacer comparativas entre ellas.

---

## Variables de color

Para cada producto se descarga su imagen y se calculan automáticamente 7 variables numéricas:

| Variable | Espacio de color | Descripción | Rango |
|---|---|---|---|
| `mean_R` | RGB | Media del canal Rojo pixel a pixel | 0 – 255 |
| `mean_G` | RGB | Media del canal Verde pixel a pixel | 0 – 255 |
| `mean_B` | RGB | Media del canal Azul pixel a pixel | 0 – 255 |
| `mean_L` | CIELAB | Luminosidad media (L*): qué tan claro/oscuro es el producto | 0 – 100 |
| `mean_a` | CIELAB | Componente a*: va del verde al rojo | -128 – +128 |
| `mean_b` | CIELAB | Componente b*: va del azul al amarillo | -128 – +128 |
| `contrast_L` | CIELAB | Desviación típica de L*: cuánto contraste interno tiene la imagen | 0 – 100 |

Se usa el espacio **CIELAB** porque está diseñado para aproximarse a la percepción humana del color: dos colores con la misma distancia numérica en CIELAB se perciben como igual de diferentes por el ojo humano, algo que no ocurre en RGB.

---

## Estructura del repositorio

```
TFG/
│
├── 📜 Exploración Inicial de las 3 fuentes de datos.py   ← SCRIPT 0 (explorar)
├── 📜 SCRIPT 1 - SCRAPING Y COLOR.py                     ← SCRIPT 1 (scraping)
├── 📜 SCRIPT 2, VARIABLES ECONOMICAS, EMOCIONAS
│      E INGENIERIA DEL DATO.py                           ← SCRIPT 2 (ingeniería)
├── 📜 SCRIPT 3 - CREACIÓN DE MODELOS.py                  ← SCRIPT 3 (modelos ML)
│
├── 📂 DATOS SCRAPPING/
│   ├── Dataset_combinado_sin_emociones.csv   ← Salida del Script 1
│   ├── Dataset_con_emociones.csv             ← Salida del Script 2
│   │
│   ├── 📂 abo_data/
│   │   └── dataset_abo.csv                  ← Mini-dataset Amazon Berkeley Objects
│   ├── 📂 mahou_data/
│   │   └── dataset_mahou.csv                ← Mini-dataset Mahou San Miguel
│   ├── 📂 openfoodfacts_data/
│   │   └── dataset_openfoodfacts.csv        ← Mini-dataset Open Food Facts
│   │
│   └── 📂 graficos/                         ← Gráficos generados por el Script 2
│       ├── 00a_distribucion_fuente.png
│       ├── 00b_top15_categorias.png
│       ├── 01_histograma_L_capping.png
│       ├── 02_distribucion_emociones.png
│       ├── 03_boxplot_mean_L.png
│       ├── 04_boxplot_mean_a.png
│       ├── 05_boxplot_mean_b.png
│       ├── 06_correlacion.png
│       ├── 07_paleta_cromatica.png
│       ├── 08_heatmap_emocion_categoria.png
│       ├── 09_scatter_L_a.png
│       ├── 10_alineacion_emocional.png
│       ├── 11_scatter_L_b_fuentes.png
│       ├── 12_boxplot_contrast_L.png
│       ├── 13_boxplot_fuentes_2x2.png
│       ├── 14_histogramas_fuentes_2x2.png
│       ├── 15_top12_categorias_fuente.png
│       ├── 📂 abo/            ← Gráficos específicos de la fuente ABO
│       ├── 📂 mahou/          ← Gráficos específicos de la fuente Mahou
│       └── 📂 openfoodfacts/  ← Gráficos específicos de OFF
│
└── 📂 Análisis del Dato/                                 ← Salidas del Script 3
    ├── resultados_analisis.json              ← Métricas de todos los modelos
    ├── resultados_analisis.xlsx              ← Métricas en formato Excel
    ├── 📂 graficos/                          ← 12 gráficos de análisis del dato (G1–G12)
    │   ├── G1_distribucion_emociones.png
    │   ├── G2_correlacion_features.png
    │   ├── G3_curva_aprendizaje_RF.png
    │   ├── G4_importancia_variables_RF.png
    │   ├── G5_matriz_confusion_RF.png
    │   ├── G6_curva_perdida_MLP.png
    │   ├── G7_matriz_confusion_MLP.png
    │   ├── G8_comparativa_F1_RF_vs_MLP.png
    │   ├── G9_pred_vs_real_scores.png
    │   ├── G10_perfiles_emocionales.png
    │   ├── G11_r2_por_emocion_SVR.png
    │   └── G12_resumen_comparativo.png
    └── 📂 modelos/                           ← Modelos entrenados serializados
        ├── modelo_random_forest.pkl
        ├── modelo_mlp.pkl
        ├── modelo_multioutput_svr.pkl
        └── label_encoder.pkl
```

---

## Pipeline completo

```
SCRIPT 0                    SCRIPT 1                         SCRIPT 2                       SCRIPT 3
─────────                   ────────                         ────────                       ────────
Explorar estructura   →     Scraping de las 3 fuentes   →   Ingeniería del dato        →   Análisis del dato
de las 3 fuentes            + Extracción de color            + Emociones                    + Modelos ML
(sin descargar nada)        ↓                                ↓                              ↓
                            Dataset_combinado_               Dataset_con_                   3 modelos .pkl
                            sin_emociones.csv                emociones.csv                  + 12 gráficos
```

---

## Descripción detallada de cada script

---

### Script 0 — `Exploración Inicial de las 3 fuentes de datos.py`

**Propósito:** Script de reconocimiento previo. Se ejecuta una sola vez antes de empezar a scrapearlo todo, para entender qué hay en cada fuente y cómo están estructurados los datos. No descarga imágenes ni genera ningún dataset.

**Qué hace:**

1. **Fuente ABO (Amazon Berkeley Objects):** Se conecta al bucket público de Amazon S3, descarga el índice de imágenes (`images.csv.gz`) y una muestra de 20 productos del primer archivo de listings (`listings_0.json.gz`). Muestra todos los campos disponibles, su tipo, un ejemplo y el porcentaje de nulos.

2. **Fuente Open Food Facts:** Llama a la API REST pública con una petición de prueba de 5 cervezas. Muestra los ~100 campos que devuelve la API por producto e identifica cuáles usaremos (código de barras, nombre, marca, categorías, URL de imagen). También consulta cuántos productos hay disponibles en 10 categorías distintas.

3. **Fuente Mahou San Miguel:** Hace una petición HTTP básica a la tienda online para comprobar que responde correctamente. Explica que el contenido es dinámico (JavaScript) y por eso el scraping real requiere Selenium. Lista las 13 categorías de la tienda que se cubrirán.

4. **Resumen comparativo:** Tabla que compara las tres fuentes en cuanto a tipo de acceso, autenticación, formato, volumen, variables, condición fotográfica y velocidad de extracción.

**Librerías:** `requests`, `gzip`, `json`, `pandas`

---

### Script 1 — `SCRIPT 1 - SCRAPING Y COLOR.py`

**Propósito:** Script principal de obtención de datos. Scraping de los tres orígenes, descarga de imágenes y extracción de las variables de color. Produce los cuatro CSV de datos.

#### Funciones comunes (usadas por los tres scrapers)

**`descargar_imagen(imagen_url, ruta_destino, headers)`**
Descarga una imagen de internet y la guarda en disco en formato JPEG con calidad 90. Si la imagen ya existe en la ruta indicada, no la vuelve a descargar (sistema de caché simple por ruta de archivo). Devuelve la ruta si tiene éxito o `None` si falla.

**`calcular_color(ruta)`**
Lee una imagen del disco, la convierte en una matriz NumPy y calcula las 7 variables de color:
- `mean_R`, `mean_G`, `mean_B`: media de cada canal RGB sobre todos los píxeles.
- Convierte la imagen de RGB a CIELAB (`skimage.color.rgb2lab`) y calcula `mean_L`, `mean_a`, `mean_b`.
- `contrast_L`: desviación típica de L* (mide el contraste interno de la imagen).

#### Scraper Mahou San Miguel

**Método:** Selenium con Chrome en modo headless (sin ventana visible), ventana virtual de 1920×1080.

**Proceso:**
1. Abre Chrome automáticamente, navega a la tienda de Mahou y acepta las cookies.
2. Para cada una de las **13 categorías** del catálogo (Mahou, San Miguel, Alhambra, Corona, Founders, Budweiser, Nómada, Brutus, Solán de Cabras, Sierra Natura, otras bebidas, cristalería, moda), recorre todas las páginas paginadas de la categoría (grupos de 12 productos) recogiendo las URLs de los productos. Hace scroll hasta el final de cada página para forzar la carga lazy de los elementos.
3. Entra en cada URL de producto, extrae el **nombre** del producto (buscando el elemento `h1`) y la **URL de la imagen** (primero desde la metaetiqueta `og:image`, si no desde selectores CSS directos).
4. Descarga la imagen con `descargar_imagen()` y calcula el color con `calcular_color()`.
5. Guarda el resultado en `dataset_mahou.csv` con las columnas: `fuente`, `categoria`, `nombre`, `imagen_url`, `mean_R`, `mean_G`, `mean_B`, `mean_L`, `mean_a`, `mean_b`, `contrast_L`.

**Resultado:** ~200 productos

#### Scraper Amazon Berkeley Objects (ABO)

**Método:** Descarga directa desde el bucket público de Amazon S3. No requiere autenticación. Usa caché en disco para no volver a descargar los archivos comprimidos si ya existen.

**Proceso:**
1. Descarga el índice de imágenes (`images.csv.gz`) desde S3, que contiene el mapeo entre `image_id` y la URL real de la imagen en `m.media-amazon.com`.
2. Itera sobre los 10 archivos de listings (`listings_0.json.gz` a `listings_9.json.gz`), cada uno con miles de productos en formato JSON-Lines (una línea = un producto en JSON). Los archivos se descomprimen en memoria.
3. Para cada producto, extrae el `product_type` y lo compara con un conjunto de **70 categorías de interés** (hogar, cocina, tecnología, cosmética, juguetes, deporte, mascotas...). Si no está en ese conjunto, se descarta.
4. Para los productos que pasan el filtro, extrae nombre (priorizando inglés), marca, categoría traducida al español (mediante un diccionario de mapeo) y el `main_image_id`.
5. Construye la URL de imagen en formato `https://m.media-amazon.com/images/I/{image_id}._SX512_.jpg`, descarga la imagen y calcula el color.
6. Para cuando llega a 10.000 productos.

**Resultado:** ~1.500 productos (los que tienen imagen disponible y categoría de interés)

#### Scraper Open Food Facts (OFF)

**Método:** API REST pública `https://world.openfoodfacts.org/api/v2/search`. Pausa de 1 segundo entre peticiones para no saturar el servidor.

**Proceso:**
1. Para cada una de las **30 categorías de alimentación** (bebidas, agua, zumos, refrescos, cervezas, vinos, lácteos, leche, yogures, quesos, cereales, galletas, chocolates, snacks, patatas fritas, conservas, salsas, aceites, pasta, arroz, pan, congelados, carne, pescado, frutas, verduras, legumbres, café, condimentos), hace peticiones paginadas de 50 productos por petición.
2. De cada producto extrae código de barras, nombre, marca, categorías y URL de imagen frontal (`image_front_url`).
3. Descarga la imagen, calcula el color y guarda la fila.
4. Para cuando llega a 30.000 productos en total entre todas las categorías.

**Resultado:** ~5.000 productos (los que tienen imagen disponible)

#### Combinación final

Una vez que los tres scrapers terminan, el script combina los tres DataFrames con `pd.concat()` y guarda el resultado en `Dataset_combinado_sin_emociones.csv`.

**Librerías:** `requests`, `pandas`, `numpy`, `matplotlib`, `PIL` (Pillow), `skimage`, `tqdm`, `selenium`, `webdriver_manager`

---

### Script 2 — `SCRIPT 2, VARIABLES ECONOMICAS, EMOCIONAS E INGENIERIA DEL DATO.py`

**Propósito:** Ingeniería del dato completa. Parte del dataset combinado sin emociones y lo enriquece con limpieza, nuevas variables, la variable objetivo (emoción), variables de negocio, normalización y los gráficos del análisis exploratorio.

#### Paso 1 — Carga de datos
Lee `Dataset_combinado_sin_emociones.csv` y documenta el número de filas y columnas iniciales.

#### Paso 2 — Gráficos iniciales (estado previo a transformaciones)
Genera dos gráficos del dataset en crudo para poder comparar antes/después:
- **Gráfico 0a:** Número de productos por fuente de datos (barras).
- **Gráfico 0b:** Top 15 categorías con más productos (barras horizontales).

#### Paso 3 — Tratamiento de nulos
Calcula el porcentaje de nulos en las 7 variables de color:
- Si es **< 1%**: elimina directamente las filas con nulos (son tan pocas que no sesgan).
- Si es **≥ 1%**: reemplaza los nulos por la **mediana** de cada variable (interpolación robusta frente a outliers).

#### Paso 4 — Eliminación de duplicados
Elimina filas con la misma `imagen_url`, ya que si dos productos tienen la misma URL de imagen son el mismo producto (independientemente de que tengan nombres distintos por error de scraping).

#### Paso 5 — Validación de rangos
Comprueba que todos los valores están dentro de sus rangos teóricos (RGB: 0-255, L*: 0-100, a* y b*: -128 a +128, contrast_L: 0-100). Informa de cuántos valores están fuera de rango en cada variable.

#### Paso 6 — Tratamiento de outliers (IQR con capping)
Usa el método **IQR (rango intercuartílico)** para cada variable: calcula Q1 y Q3, define los límites como `Q1 - 1.5×IQR` y `Q3 + 1.5×IQR` (intersectados con los rangos teóricos), y aplica **capping** (los valores fuera del límite se recortan al límite, no se eliminan). Se decide no eliminar outliers porque colores extremos son información válida y valiosa para el análisis.

#### Paso 7 — Variables HSV
Convierte las variables RGB a **HSV** (Hue, Saturation, Value) para añadir una representación más intuitiva del color:
- `hsv_h`: **Tono** — el color en sí, expresado como ángulo de 0° a 360° (0°=rojo, 120°=verde, 240°=azul).
- `hsv_s`: **Saturación** — qué tan "puro" o "vivo" es el color (0=gris, 100=color puro).
- `hsv_v`: **Brillo** — qué tan claro es el color (0=negro, 100=máximo brillo).

Estas variables capturan aspectos del color difíciles de interpretar directamente desde RGB y complementan al espacio CIELAB para el modelo.

#### Paso 8 — Variable objetivo: Emoción

Se asigna una emoción a cada producto mediante un **sistema de scoring gaussiano ponderado** en el espacio CIELAB, basado en tres referencias científicas:
- **Valdez & Mehrabian (1994):** pesos de L* para emociones pasivas.
- **Gilbert et al. (2016):** pesos de Croma C* y ángulo de tono H* para emociones activas.
- **Russell (1980):** modelo circunflejo (el espacio emocional es continuo y circular).

Para cada producto se calcula el **Croma** `C* = √(a*² + b*²)` (intensidad total del color) y el **ángulo de tono** `H* = arctan2(b*, a*)` (dirección del color en el plano a*b*). Luego se evalúa cada una de las 8 emociones posibles con una función gaussiana que mide la similitud del producto con el centroide de esa emoción en los tres ejes (L*, C*, H*). La emoción ganadora es la de mayor score. Si ninguna supera el umbral de confianza de 0.22, el producto se clasifica como "Neutro/Ambiguo".

**Las 8 emociones posibles:**

| Emoción | Perfil de color típico |
|---|---|
| **Alegría** | L* alto (~80), Croma medio-alto, tono amarillo-naranja |
| **Relajación** | L* muy alto (~92), Croma muy bajo, colores casi neutros |
| **Calma** | L* alto (~85), Croma bajo, tonos fríos (azul-verde) |
| **Energía** | L* medio-alto (~62), Croma alto, tonos cálidos |
| **Romanticismo** | L* medio (~50), Croma medio, tonos rosados |
| **Aburrimiento** | L* medio-alto (~70), Croma muy bajo, sin tono definido |
| **Tristeza** | L* medio-bajo (~42), Croma casi nulo, tonos fríos |
| **Ira** | L* muy bajo (~22), Croma bajo, tonos rojizos |

Además de la emoción ganadora, se calculan:
- `confianza_emocional`: probabilidad softmax de la emoción ganadora (entre 0 y 1).
- `score_alegria`, `score_calma`, ...: score individual de cada emoción (captura la ambigüedad).
- `distancia_centroide`: distancia euclidiana al centroide de la emoción asignada (qué tan "puro" es el producto dentro de su emoción).

#### Paso 9 — Variables de negocio
Variables pensadas para dar valor interpretativo al análisis desde una perspectiva de marketing:

- **`temperatura_color`**: Clasifica el color en `Cálido` (a* > 3 o b* > 10), `Frío` (a* < -3 o b* < -5) o `Neutro`.
- **`luminosidad_cat`**: Categoriza L* en tres grupos: `Oscuro` (0-40), `Medio` (40-70), `Luminoso` (70-100).
- **`saturacion_cat`**: Categoriza hsv_s en `Apagado` (0-25), `Moderado` (25-60), `Intenso` (60-100).
- **`coherencia_emocional`**: Puntuación de 0 a 100 que indica qué tan claramente el producto pertenece a su emoción asignada (inverso de la distancia al centroide, normalizado).
- **`emocion_optima`**: Emoción que debería transmitir el producto según su categoría (basado en criterios de psicología del color y marketing). Por ejemplo: Cervezas → Relajación, Refrescos → Energía, Chocolates → Romanticismo.
- **`alineacion_emocional`**: Variable binaria (0 o 1) que indica si la emoción real del producto coincide con la emoción óptima para su categoría. Es la variable de análisis central del TFG.

#### Paso 10 — Normalización Min-Max
Aplica normalización Min-Max a las 10 variables numéricas de color (`mean_R`, `mean_G`, `mean_B`, `mean_L`, `mean_a`, `mean_b`, `contrast_L`, `hsv_h`, `hsv_s`, `hsv_v`), generando columnas `*_norm` con valores en el rango [0, 1] para que el modelo no se vea afectado por diferencias de escala entre variables.

#### Paso 11 — Gráficos del análisis exploratorio
Genera 11 gráficos y los guarda en `DATOS SCRAPPING/graficos/`:

| Gráfico | Descripción |
|---|---|
| `01_histograma_L_capping.png` | Histograma de L* antes y después del tratamiento de outliers (comparativa) |
| `02_distribucion_emociones.png` | Distribución de las 8 emociones en el dataset completo |
| `03_boxplot_mean_L.png` | Boxplot de luminosidad L* por emoción |
| `04_boxplot_mean_a.png` | Boxplot del componente a* (rojo-verde) por emoción |
| `05_boxplot_mean_b.png` | Boxplot del componente b* (amarillo-azul) por emoción |
| `06_correlacion.png` | Matriz de correlación entre variables de color (L*, a*, b*, contrast_L, HSV) |
| `07_paleta_cromatica.png` | Paleta de colores real del dataset: un rectángulo por producto ordenados por tono |
| `08_heatmap_emocion_categoria.png` | Mapa de calor: % de productos con cada emoción en las 12 categorías principales |
| `09_scatter_L_a.png` | Scatter L* vs a* coloreado por emoción (demuestra separabilidad en CIELAB) |
| `10_alineacion_emocional.png` | Alineación emocional por categoría (% de productos con emoción óptima) |
| `11_scatter_L_b_fuentes.png` | Scatter L* vs b* diferenciando por fuente de datos |

**Librerías:** `pandas`, `numpy`, `matplotlib`, `sklearn` (MinMaxScaler)

---

### Script 3 — `SCRIPT 3 - CREACIÓN DE MODELOS.py`

**Propósito:** Análisis del dato completo. Parte del `Dataset_con_emociones.csv` y entrena tres modelos de machine learning para predecir la emoción de un producto a partir de sus variables de color. Genera 12 gráficos, guarda los modelos entrenados y exporta las métricas en JSON y Excel.

#### Variables de entrada (features)

El modelo recibe las **10 variables de color normalizadas** (Min-Max) calculadas en el Script 2:

| Variable | Descripción |
|---|---|
| `mean_R_norm`, `mean_G_norm`, `mean_B_norm` | Canales RGB normalizados |
| `mean_L_norm`, `mean_a_norm`, `mean_b_norm` | Variables CIELAB normalizadas |
| `contrast_L_norm` | Contraste de luminosidad normalizado |
| `hue_norm`, `saturation_norm`, `value_norm` | Tono, saturación y brillo HSV normalizados |

#### División train/test

- **80/20** estratificado (misma proporción de emociones en train y test).
- 4.465 productos totales → 3.572 entrenamiento / 893 test.
- Validación cruzada de **5 folds** para estimar la estabilidad de los modelos.

#### Modelo 1 — Random Forest Classifier

**Qué es:** Ensamble de árboles de decisión entrenados con bagging (muestreo bootstrap) y selección aleatoria de variables en cada nodo (`max_features="sqrt"`). La predicción final es la moda de los votos de todos los árboles.

**Por qué se elige:** Las relaciones color-emoción son no lineales (por ejemplo, Romanticismo requiere L* < 55 Y a* ≥ 4 simultáneamente). El Random Forest las captura sin suposiciones sobre la distribución. Además, `class_weight="balanced"` compensa el desbalance de clases sin necesidad de re-muestreo.

**Búsqueda de hiperparámetros:** `RandomizedSearchCV` con 25 iteraciones × 5-fold, optimizando F1 ponderado. Se exploran combinaciones de `n_estimators`, `max_depth`, `min_samples_leaf` y `max_features`.

**Resultados:**

| Métrica | Valor |
|---|---|
| Accuracy test | **89.03%** |
| F1 ponderado | **88.97%** |
| Precision | 89.06% |
| Recall | 89.03% |
| CV 5-fold (media ± std) | 89.05% ± 0.17% |

F1 por clase: Relajación (97.8%), Energía (91.8%), Tristeza (91.6%), Romanticismo (89.9%), Aburrimiento (88.8%), Alegría (88.1%), Ira (83.3%), Calma (81.6%), Neutro/Ambiguo (82.4%).

#### Modelo 2 — Red Neuronal MLP (Multilayer Perceptron)

**Qué es:** Red neuronal de capas densas entrenada con backpropagation y Adam. Captura relaciones muy no lineales entre las variables de color y las emociones.

**Por qué se elige:** Complementa al Random Forest con un tipo de modelo completamente diferente (conexionista vs. basado en reglas), lo que permite comparar si ambos enfoques convergen a los mismos resultados y valida la robustez del dataset.

**Entrenamiento:** Con `early_stopping=True` para detener el entrenamiento cuando la validación deja de mejorar, evitando el sobreajuste. En este caso convergió en **108 épocas**.

**Resultados:**

| Métrica | Valor |
|---|---|
| Accuracy test | **88.47%** |
| F1 ponderado | **88.45%** |
| Precision | 88.97% |
| Recall | 88.47% |
| CV 5-fold (media ± std) | 88.72% ± 1.34% |

F1 por clase: Tristeza (92.4%), Relajación (96.1%), Energía (90.4%), Romanticismo (89.7%), Aburrimiento (88.4%), Ira (84.6%), Calma (86.4%), Alegría (84.6%), Neutro/Ambiguo (81.8%).

#### Modelo 3 — MultiOutput SVR (modelo original ★)

**Qué es:** Un modelo de regresión de vectores de soporte (SVR) envuelto en `MultiOutputRegressor` para predecir simultáneamente los **8 scores emocionales** de un producto en lugar de clasificar en una sola emoción. Esto permite obtener el **perfil emocional completo** de cada producto (cuánto transmite de alegría, calma, energía, etc.).

**Por qué es original:** A diferencia de los dos clasificadores anteriores, este modelo no etiqueta el producto con una única emoción sino que devuelve un vector de 8 valores continuos. Esto captura la ambigüedad emocional: un producto puede transmitir a la vez relajación y calma en proporciones distintas, información que se pierde con la clasificación.

**Resultados:**

| Métrica | Valor |
|---|---|
| R² global | **0.9781** |
| RMSE global | **0.0093** |

R² por emoción: Relajación (0.988), Tristeza (0.987), Energía (0.985), Calma (0.984), Aburrimiento (0.982), Romanticismo (0.978), Alegría (0.968), Ira (0.955).

#### Gráficos generados (G1–G12)

| Gráfico | Descripción |
|---|---|
| `G1_distribucion_emociones.png` | Distribución de emociones en el dataset con línea de distribución uniforme |
| `G2_correlacion_features.png` | Heatmap de correlación entre las 10 variables cromáticas de entrada |
| `G3_curva_aprendizaje_RF.png` | Curva de aprendizaje del Random Forest (train vs. validación) |
| `G4_importancia_variables_RF.png` | Importancia de cada variable cromática según el Random Forest |
| `G5_matriz_confusion_RF.png` | Matriz de confusión del Random Forest (porcentaje por fila) |
| `G6_curva_perdida_MLP.png` | Curva de pérdida durante el entrenamiento de la red neuronal |
| `G7_matriz_confusion_MLP.png` | Matriz de confusión de la MLP |
| `G8_comparativa_F1_RF_vs_MLP.png` | Comparativa de F1 por clase entre Random Forest y MLP |
| `G9_pred_vs_real_scores.png` | Predicción vs. real de los 8 scores emocionales (modelo SVR) |
| `G10_perfiles_emocionales.png` | Perfil emocional predicho vs. real para productos de ejemplo |
| `G11_r2_por_emocion_SVR.png` | R² por emoción del modelo MultiOutput SVR |
| `G12_resumen_comparativo.png` | Resumen comparativo final de los tres modelos |

#### Salidas del Script 3

- **`resultados_analisis.json`** — Todas las métricas de los tres modelos en formato estructurado.
- **`resultados_analisis.xlsx`** — Las mismas métricas en Excel para presentación.
- **`modelos/modelo_random_forest.pkl`** — Random Forest entrenado y listo para usar.
- **`modelos/modelo_mlp.pkl`** — Red neuronal MLP entrenada.
- **`modelos/modelo_multioutput_svr.pkl`** — MultiOutput SVR entrenado.
- **`modelos/label_encoder.pkl`** — Codificador de etiquetas para decodificar las predicciones.

**Librerías:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`, `openpyxl`

---

## Dataset final (`Dataset_con_emociones.csv`)

El dataset final tiene las siguientes columnas:

| Columna | Tipo | Descripción |
|---|---|---|
| `fuente` | Categórica | Origen del producto: ABO / Mahou San Miguel / Open Food Facts |
| `categoria` | Categórica | Categoría del producto (ej. Cervezas, Hogar - Sofás...) |
| `nombre` | Texto | Nombre del producto |
| `imagen_url` | Texto | URL de la imagen usada para calcular el color |
| `mean_R` / `mean_G` / `mean_B` | Numérica | Media RGB pixel a pixel (0-255) |
| `mean_L` / `mean_a` / `mean_b` | Numérica | Variables CIELAB |
| `contrast_L` | Numérica | Contraste de luminosidad (desv. típica de L*) |
| `hsv_h` / `hsv_s` / `hsv_v` | Numérica | Tono, saturación y brillo (HSV) |
| `emocion` | Categórica | Emoción asignada (8 posibles + Neutro/Ambiguo) |
| `confianza_emocional` | Numérica [0-1] | Confianza del sistema de scoring en la emoción asignada |
| `score_alegria` / ... | Numérica | Score individual para cada una de las 8 emociones |
| `distancia_centroide` | Numérica | Distancia euclidiana al centroide de la emoción asignada |
| `temperatura_color` | Categórica | Cálido / Frío / Neutro |
| `luminosidad_cat` | Categórica | Oscuro / Medio / Luminoso |
| `saturacion_cat` | Categórica | Apagado / Moderado / Intenso |
| `coherencia_emocional` | Numérica [0-100] | Claridad de la señal emocional del producto |
| `emocion_optima` | Categórica | Emoción que debería transmitir según su categoría |
| `alineacion_emocional` | Binaria (0/1) | 1 si el producto transmite la emoción óptima |
| `mean_*_norm` | Numérica [0-1] | Versiones normalizadas Min-Max de las variables de color |

---

## Orden de ejecución

Para reproducir el proyecto desde cero:

```bash
# 1. Explorar la estructura de las fuentes (opcional, informativo)
python "Exploración Inicial de las 3 fuentes de datos.py"

# 2. Scraping + extracción de color (tarda varias horas por la descarga de imágenes)
python "SCRIPT 1 - SCRAPING Y COLOR.py"

# 3. Ingeniería del dato + emociones + gráficos
python "SCRIPT 2, VARIABLES ECONOMICAS, EMOCIONAS E INGENIERIA DEL DATO.py"

# 4. Entrenamiento de modelos ML + análisis del dato (tarda ~5-10 min)
python "SCRIPT 3 - CREACIÓN DE MODELOS.py"
```

## Dependencias

```
pandas
numpy
matplotlib
Pillow
scikit-image
scikit-learn
tqdm
requests
selenium
webdriver-manager
joblib
openpyxl
```

---

## Referencias

- Valdez, P., & Mehrabian, A. (1994). Effects of color on emotions. *Journal of Experimental Psychology: General*, 123(4), 394-409.
- Gilbert, A. N., Martin, R., & Kemp, S. E. (1996). Cross-modal correspondence between vision and olfaction: the color of smells. *The American Journal of Psychology*.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.
- [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
- [Open Food Facts API](https://world.openfoodfacts.org/api/v2)
