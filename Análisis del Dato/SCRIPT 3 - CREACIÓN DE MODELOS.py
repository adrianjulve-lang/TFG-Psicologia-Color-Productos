# ==============================================================================
# TFG - ADRIÁN JULVE NAVARRO
# SCRIPT 3: ANÁLISIS DEL DATO
# Universidad Francisco de Vitoria · Business Analytics 2025-26
# ==============================================================================
#
# MODELOS IMPLEMENTADOS:
#   1. Random Forest Classifier   — supervisado, clasificación de emoción
#   2. Red Neuronal MLP           — supervisado, clasificación de emoción
#   3. MultiOutput SVR ★ original — supervisado, predice el perfil emocional
#                                   completo (los 8 scores simultáneamente)
#
# MÉTRICAS (≥3 para la rúbrica):
#   · Accuracy                    — proporción global de aciertos
#   · Precision / Recall / F1     — por clase emocional
#   · Matriz de confusión         — dónde se equivoca cada modelo
#   · R² y RMSE                   — adecuación del modelo de regresión
#   · Validación cruzada (5-fold) — estabilidad del modelo
#
# GRÁFICOS GENERADOS (referenciables como Figura X en la memoria):
#   G1  — Distribución de emociones en el dataset
#   G2  — Heatmap de correlación entre features
#   G3  — Curva de aprendizaje Random Forest
#   G4  — Importancia de variables (Random Forest)
#   G5  — Matriz de confusión Random Forest
#   G6  — Curva de aprendizaje MLP
#   G7  — Matriz de confusión MLP
#   G8  — Comparativa de métricas F1 entre modelos
#   G9  — Predicción vs real de scores (modelo original)
#   G10 — Perfil emocional predicho vs real (radar/barras)
#   G11 — Error por emoción del modelo original
#   G12 — Resumen comparativo final de los tres modelos
# ==============================================================================

import sys #Importa el modulo sys que da acceso a información sobre Python y el ordenador donde se está ajustando 
import subprocess #Importa subprocess, que permite ejecutar comandos del sistema operativo desde dentro del propio script

# ── Instalador automático ──────────────────────────────────────────────────────
_REQUERIDOS = ["pandas", "numpy", "matplotlib", "scikit-learn", "joblib", "openpyxl"] # Crea una lista con los nombres de las librerías que el script necesita para funcioanr. 
for _pkg in _REQUERIDOS: # Comienza un bucle que dice: para cada libreía de la lista anterior..
    try:
        __import__(_pkg.replace("-", "_").split("scikit")[0] if "scikit" not in _pkg else "sklearn") # Intenta cargar esa librería. COmo scikit-learn tiene su propio nombre en python, hace un ajuste para ese casi. SI la librería ya esta instalada, continua
    except ImportError: # Si la librería no estaba instalada haz lo siguiente: 
        print(f"[INFO] Instalando {_pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg], # La instala automáticamente usando pip
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # stdout y stderr hacen que no muestre el proceso de instalación en pantalla.   

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble          import RandomForestClassifier
from sklearn.neural_network    import MLPClassifier
from sklearn.svm               import SVR
from sklearn.multioutput       import MultiOutputRegressor
from sklearn.model_selection   import train_test_split, StratifiedKFold, learning_curve, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
import joblib
import os
import json
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURACIÓN DE RUTAS  ← cambia solo este bloque
# ══════════════════════════════════════════════════════════════════════════════
RUTA_CSV      = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\Dataset_con_emociones.csv" # Aquí se determina la ruta del archivo CSV donde se encuentran los datos con las emociones ya calculadas 
RUTA_BASE     = r"C:\Users\34625\Desktop\4 Carrera\TFG\Análisis del Dato" # Guarda la carpeta principal del proyecto. Todo lo que genere el script se guarda dentro de ella. 
RUTA_GRAFICOS = os.path.join(RUTA_BASE, "graficos") # Aquí se construye la ruta de la subcarpeta donde se guardan los gráficos, uniendo RUTA_BASE + "graficos" con os.path.join
RUTA_MODELOS  = os.path.join(RUTA_BASE, "modelos")  # Lo mismo para los modelos entrenados
RUTA_JSON     = os.path.join(RUTA_BASE, "resultados_analisis.json") # Aqui se guarda el archivo JSON donde se guardarán los resultados numéricos del análisis. 
RUTA_EXCEL    = os.path.join(RUTA_BASE, "resultados_analisis.xlsx") # ruta donde se guardará el excel con los reusltados finales. 

os.makedirs(RUTA_GRAFICOS, exist_ok=True) # Crea la carpeta "gráficos" en el disco si no existe todavía
os.makedirs(RUTA_MODELOS,  exist_ok=True) # Igual que la anterior, crea la carpeta "modelos" si no existe. 

# Paleta de colores coherente con los scripts anteriores
COLORES_EMOCION = {
    "Alegría":        "#F4D03F",
    "Energía":        "#E67E22",
    "Calma":          "#5DADE2",
    "Romanticismo":   "#F1948A",
    "Tristeza":       "#85929E",
    "Ira":            "#C0392B",
    "Aburrimiento":   "#A9A9A9",
    "Relajación":     "#82E0AA",
    "Neutro/Ambiguo": "#BDC3C7",
}
COLOR_RF  = "#2E75B6"
COLOR_MLP = "#70AD47"
COLOR_SVR = "#E8A838"

print("=" * 65) # Estas lineas de aquí son más estéticas que otra cosa. 
print("  TFG — SCRIPT 3: ANÁLISIS DEL DATO")
print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y REVISIÓN DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Cargando dataset...")
df = pd.read_csv(RUTA_CSV, encoding="utf-8-sig") # lee el archivo CSV y lo guarda en una variable llamada "df", se usa el encoding para evitar que los caracteres especiales como tildes o ñ aparezcan mal. 
print(f"    → {len(df):,} filas | {len(df.columns)} columnas") # Muestra cuántas filas y columnas tiene la tabla. El : formatea el número con puntos de mil. 
print(f"    → Clases: {sorted(df['emocion'].unique())}")
print(f"    → Distribución:\n{df['emocion'].value_counts().to_string()}")

# Variables cromáticas normalizadas (features del modelo)
FEATURES = [
    "mean_R_norm", "mean_G_norm", "mean_B_norm", # Esta es la lista con los nombres exactos de las 10 columnas del CSV que contienen los valores de color normalizados. 
    "mean_L_norm", "mean_a_norm", "mean_b_norm", # SOn las variables que el modelo va a recibir para entrenarse. 
    "contrast_L_norm", "hue_norm", "saturation_norm", "value_norm"
]
FEATURES_LABEL = [
    "R (rojo)", "G (verde)", "B (azul)", # Lista paralela a FEATURES con nombres legubles en español. Se usa para que los gráficos muestren etiquetas comprensibles en luhar de los 
    "L* (luminosidad)", "a* (verde-rojo)", "b* (azul-amarillo)", # nombres técnicos del CSV
    "Contraste L*", "Tono (H)", "Saturación (S)", "Brillo (V)"
]

# Scores emocionales (targets del modelo original)
SCORE_COLS = [
    "score_ira", "score_tristeza", "score_romanticismo", # Lista con los nombres exactos de las 8 columnas del CSV que contienen la intensidad numérica de cada emoción
    "score_energía", "score_alegría", "score_relajación",
    "score_calma", "score_aburrimiento"
]
SCORE_LABELS = [
    "Ira", "Tristeza", "Romanticismo", "Energía", # Lista paralela a SCORE_COLS con los nombres bonitos de cada emoción, usados solo para poenr etiquetas entendibles en los gráficos. 
    "Alegría", "Relajación", "Calma", "Aburrimiento"
]

# Rellenar los 11 nulos de saturacion_cat con la moda
df["saturacion_cat"].fillna(df["saturacion_cat"].mode()[0], inplace=True) 

X = df[FEATURES].values # Pone como inputs del modelos las 10 columnas de color de la tabla. .values las convierte a formato numérico puro, que es lo que esperan los algoritmos de ML
y_scores = df[SCORE_COLS].values   # para el modelo de regresión SVR, extrae las 8 columnas de score emocionales y las guarda en "y_scores"

# Codificación del target de clasificación
le = LabelEncoder() # Crea un "codfiicador de etiquetas" y lo guarda en "le". Este objetivo aprenderá a convertir Alegria -> 0, Calma -> 1 etc. 
y_clase = le.fit_transform(df["emocion"]) # fit aprende qué etiquetas existen en la columna "emoción" y "transform" las convierte a numeros. EN lugar de alegría habra un 0 por ejemplo. Es lo que usarán RF y MLP
CLASES   = le.classes_ # Guarda la lista de emociones unicas tal como las aprendió el codificador en orden alfabético. 
N_CLASES = len(CLASES) #  CUenta cuántas emcociones distintas hay en total y guarda es enúmero. 
print(f"\n    → Features:  {len(FEATURES)} variables cromáticas normalizadas") # Muestra por pantalla que se van a usar 10 variables de color como entrada de los modelos. 
print(f"    → Target clf: {N_CLASES} clases → {list(CLASES)}") # cuenta cuántas emociones hay que clasificar y cuáles son
print(f"    → Target reg: {len(SCORE_COLS)} scores emocionales") # Muestra que el modelo de regresión SVR tiene que predecir 8 valores numéricos simultáneamente (uno por cada emoción). 


# ══════════════════════════════════════════════════════════════════════════════
# 2. DIVISIÓN TRAIN / TEST  (80 / 20, estratificada)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Dividiendo train/test (80/20 estratificado)...")
(X_train, X_test,
 y_train, y_test,
 ys_train, ys_test) = train_test_split(
    X, y_clase, y_scores,
    test_size=0.20,
    random_state=42,
    stratify=y_clase
)
print(f"    → Train: {len(X_train):,}  |  Test: {len(X_test):,}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 1 — Distribución de emociones en el dataset
# (se genera antes del modelado para documentar el punto de partida)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[G1] Distribución de emociones...")
vc = df["emocion"].value_counts()  # Cuenta cuántos productos hay de cada emoción, ordenados de mayor a menor
colores_barras = [COLORES_EMOCION.get(e, "#999") for e in vc.index]  # Asigna a cada emoción su color del diccionario. Si no tiene, usa gris (#999)


fig, ax = plt.subplots(figsize=(10, 5))  # Crea el lienzo del gráfico: 10 pulgadas de ancho y 5 de alto
barras = ax.barh(vc.index, vc.values, color=colores_barras, edgecolor="white", height=0.7)  # Dibuja barras horizontales: cada emoción con su color y borde blanco
for b, v in zip(barras, vc.values):  # Recorre cada barra junto a su valor numérico para añadirle una etiqueta
    ax.text(v + 15, b.get_y() + b.get_height() / 2,  # Posición del texto: justo a la derecha de la barra y centrado verticalmente
            f"{v:,}  ({v/len(df)*100:.1f}%)",  # Texto que muestra: número de productos y su porcentaje sobre el total
            va="center", fontsize=9, fontweight="bold", color="#2C3E50")  # Formato del texto: centrado, tamaño 9, negrita y azul oscuro
ax.set_xlabel("Número de productos", fontsize=11)  # Etiqueta del eje horizontal
ax.set_title("Distribución de emociones en el dataset\n"
             "(variable objetivo del análisis del dato)",
             fontsize=13, fontweight="bold")  # Título del gráfico en dos líneas, tamaño 13 y negrita
ax.set_xlim(0, vc.max() * 1.22)  # Deja un 22% de espacio extra a la derecha para que las etiquetas no se corten
ax.axvline(len(df) / N_CLASES, color="gray", linestyle="--",
           linewidth=1, label="Media esperada (distribución uniforme)")  # Línea gris discontinua que muestra dónde estarían las barras si todas las emociones fueran igual de frecuentes
ax.legend(fontsize=9)  # Muestra la leyenda que explica qué es la línea gris
plt.tight_layout()  # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G1_distribucion_emociones.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución en la carpeta de gráficos
plt.close()  # Cierra el gráfico y libera memoria para no ralentizar el script
print("    ✓ G1_distribucion_emociones.png")  # Confirma en pantalla que el gráfico se guardó correctamente

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 2 — Heatmap de correlación entre features
# ══════════════════════════════════════════════════════════════════════════════

print("[G2] Heatmap correlación features...")  # Avisa en pantalla de que va a generar el gráfico G2
corr = df[FEATURES].corr()  # Calcula la correlación entre todas las variables de color: mide cómo de relacionadas están entre sí (valor entre -1 y 1)


fig, ax = plt.subplots(figsize=(9, 7))  # Crea el lienzo del gráfico: 9 pulgadas de ancho y 7 de alto
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")  # Dibuja la matriz de correlación como un mapa de calor: rojo=correlación negativa, amarillo=sin correlación, verde=correlación positiva
plt.colorbar(im, ax=ax, shrink=0.8, label="Coeficiente de correlación de Pearson")  # Añade la barra de color lateral que explica qué significa cada tono
ax.set_xticks(range(len(FEATURES_LABEL)))  # Define las posiciones de las etiquetas en el eje horizontal
ax.set_yticks(range(len(FEATURES_LABEL)))  # Define las posiciones de las etiquetas en el eje vertical
ax.set_xticklabels(FEATURES_LABEL, rotation=40, ha="right", fontsize=9)  # Pone los nombres de las variables en el eje X, girados 40° para que no se solapen
ax.set_yticklabels(FEATURES_LABEL, fontsize=9)  # Pone los nombres de las variables en el eje Y
for i in range(len(FEATURES)):  # Bucle por cada fila de la matriz
    for j in range(len(FEATURES)):  # Bucle por cada columna de la matriz (recorre todas las celdas)
        val = corr.values[i, j]  # Obtiene el valor de correlación de esa celda concreta
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",  # Escribe el valor numérico centrado dentro de la celda, con 2 decimales
                fontsize=7.5, color="black" if abs(val) < 0.7 else "white",  # Si la correlación es fuerte (>0.7), el texto es blanco para que se lea sobre el color oscuro; si no, negro
                fontweight="bold" if abs(val) > 0.7 else "normal")  # Si la correlación es fuerte, el número aparece en negrita para destacarlo
ax.set_title("Correlación entre variables cromáticas\n"
             "(features del modelo de análisis del dato)",
             fontsize=13, fontweight="bold")  # Título del gráfico en dos líneas, tamaño 13 y negrita
plt.tight_layout()  # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G2_correlacion_features.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución en la carpeta de gráficos
plt.close()  # Cierra el gráfico y libera memoria para no ralentizar el script
print("    ✓ G2_correlacion_features.png")  # Confirma en pantalla que el gráfico se guardó correctamente

# ══════════════════════════════════════════════════════════════════════════════
# MODELO 1 — RANDOM FOREST CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
# ¿QUÉ ES?
#   Un Random Forest es un ensamble de árboles de decisión entrenados con dos
#   fuentes de aleatoriedad:
#     1. Bagging: cada árbol se entrena sobre una muestra bootstrap del dataset
#        (muestreo con reemplazo), reduciendo así la varianza del modelo.
#     2. Feature randomness: en cada nodo solo se considera un subconjunto
#        aleatorio de variables (max_features="sqrt" → √10 ≈ 3), lo que
#        descorrelaciona los árboles entre sí.
#   La predicción final es la moda de los votos de todos los árboles.
#
# ¿POR QUÉ PARA ESTE TFG?
#   · Las reglas de color-emoción son no lineales (ej.: Romanticismo requiere
#     L* < 55 Y a* ≥ 4 simultáneamente). El RF las captura sin suposiciones.
#   · Es robusto al ruido cromático (iluminación variable, fondos distintos).
#   · Ofrece importancia de variables directa: cuáles colores predicen mejor.
#   · class_weight="balanced" compensa el desbalance (Neutro/Ambiguo=22% vs
#     Ira=1.3%) sin necesidad de over/under-sampling.
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Entrenando Modelo 1: Random Forest (búsqueda de hiperparámetros)...")  # Avisa en pantalla de que empieza el entrenamiento del primer modelo
print("    [RandomizedSearchCV 25 iter × 5-fold — ~2-4 min]")  # Informa de que probará 25 combinaciones de parámetros y tardará entre 2 y 4 minutos


_param_rf = {
    "n_estimators":    [100, 200, 300, 500],       # Número de árboles del bosque: probará con 100, 200, 300 o 500 árboles
    "max_depth":       [None, 10, 15, 20, 25],     # Profundidad máxima de cada árbol: None = sin límite, o un número de niveles concreto
    "min_samples_leaf":[2, 3, 5, 8],               # Mínimo de productos que debe haber en cada hoja del árbol para que sea válida
    "max_features":    ["sqrt", "log2", 0.4, 0.6], # Cuántas variables usar en cada división: raíz cuadrada, logaritmo, o el 40-60% del total
}
# Diccionario con todas las combinaciones de configuración que se van a probar para encontrar el mejor Random Forest posible

_search_rf = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    # Crea el modelo Random Forest base con:
    #    - class_weight="balanced": compensa automáticamente si hay emociones con muchos más productos que otras
    #    - random_state=42: semilla fija para que los resultados sean reproducibles (siempre igual)
    #    - n_jobs=-1: usa todos los núcleos del procesador para ir más rápido
    _param_rf, n_iter=25, cv=cv, scoring="f1_weighted",
    # Le pasa el diccionario de parámetros y le dice:
    #    - n_iter=25: que pruebe solo 25 combinaciones aleatorias (no todas, que serían cientos)
    #    - cv=cv: que use la validación cruzada de 5 partes definida antes
    #    - scoring="f1_weighted": que elija la mejor combinación según el F1 ponderado
    random_state=42, n_jobs=-1, verbose=0
    # - random_state=42: misma semilla para reproducibilidad en la búsqueda
    #    - n_jobs=-1: paraleliza la búsqueda en todos los núcleos disponibles
    #    - verbose=0: no muestra mensajes intermedios durante la búsqueda
)
_search_rf.fit(X_train, y_train)  # Lanza la búsqueda: entrena y evalúa las 25 combinaciones usando los datos de entrenamiento
rf = _search_rf.best_estimator_   # Guarda en "rf" el modelo con la combinación de parámetros que mejor F1 obtuvo
print(f"    → Mejores hiperparámetros RF: {_search_rf.best_params_}")  # Muestra en pantalla cuál fue la combinación ganadora


y_pred_rf = rf.predict(X_test)  # Usa el mejor modelo para predecir las emociones del conjunto de prueba (datos que nunca ha visto)
acc_rf    = accuracy_score(y_test, y_pred_rf)   # Calcula el porcentaje de predicciones correctas sobre el total
f1_rf     = f1_score(y_test, y_pred_rf, average="weighted")  # Calcula el F1 ponderado: media entre precisión y recall, dando más peso a las emociones más frecuentes
prec_rf   = precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)  # Calcula la precisión: de las veces que predijo una emoción, ¿cuántas acertó?
rec_rf    = recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)     # Calcula el recall: de todos los productos de cada emoción, ¿cuántos detectó correctamente?
cv_rf     = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)  # Valida el modelo en 5 particiones distintas del entrenamiento para comprobar que es estable


print(f"    → Accuracy test:  {acc_rf:.4f}")   # Muestra el porcentaje de aciertos con 4 decimales
print(f"    → F1 ponderado:   {f1_rf:.4f}")    # Muestra el F1 con 4 decimales
print(f"    → CV 5-fold:      {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")  # Muestra la media y desviación del accuracy en las 5 validaciones: si la desviación es baja, el modelo es estable


# GRÁFICO 3 — Curva de aprendizaje Random Forest
print("[G3] Curva de aprendizaje Random Forest...")  # Avisa de que va a generar el gráfico G3
train_sizes, train_scores, val_scores = learning_curve(
    rf, X_train, y_train,               # Modelo y datos de entrenamiento que se van a evaluar
    train_sizes=np.linspace(0.1, 1.0, 8),  # Prueba con 8 tamaños distintos de datos: desde el 10% hasta el 100%
    cv=cv, scoring="accuracy", n_jobs=-1   # Usa validación cruzada de 5 partes y mide accuracy en paralelo
)
# Calcula cómo evoluciona el accuracy del modelo a medida que se le dan más datos de entrenamiento

fig, ax = plt.subplots(figsize=(8, 5))  # Crea el lienzo: 8 pulgadas de ancho y 5 de alto
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color=COLOR_RF,
        label="Accuracy entrenamiento", linewidth=2)
# Dibuja la línea azul del accuracy en entrenamiento: "o-" significa puntos circulares unidos por línea
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.15, color=COLOR_RF)
# Dibuja una banda semitransparente (alpha=0.15) alrededor de la línea que representa la variabilidad (±1 desviación típica)
ax.plot(train_sizes, val_scores.mean(axis=1), "s--", color="#C0392B",
        label="Accuracy validación (CV)", linewidth=2)
# Dibuja la línea roja del accuracy en validación: "s--" significa cuadrados unidos por línea discontinua
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.15, color="#C0392B")
# Banda semitransparente roja que muestra la variabilidad de la validación
ax.set_xlabel("Tamaño del conjunto de entrenamiento", fontsize=11)  # Etiqueta del eje X
ax.set_ylabel("Accuracy", fontsize=11)                              # Etiqueta del eje Y
ax.set_title("Curva de aprendizaje — Random Forest\n"
             "(convergencia train vs. validación)",
             fontsize=13, fontweight="bold")  # Título en dos líneas, tamaño 13 y negrita
ax.legend(fontsize=10)       # Muestra la leyenda que diferencia la línea de entrenamiento y la de validación
ax.set_ylim(0, 1.05)         # Fija el eje Y entre 0 y 1.05 para que el 100% de accuracy quede visible
ax.grid(alpha=0.3)           # Añade una rejilla de fondo muy suave (30% de opacidad) para facilitar la lectura
plt.tight_layout()           # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G3_curva_aprendizaje_RF.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución
plt.close()                  # Cierra el gráfico y libera memoria
print("    ✓ G3_curva_aprendizaje_RF.png")  # Confirma que el gráfico se guardó correctamente


# GRÁFICO 4 — Importancia de variables (Random Forest)
print("[G4] Importancia de variables Random Forest...")  # Avisa de que va a generar el gráfico G4
importancias = rf.feature_importances_  # Extrae del modelo entrenado la importancia de cada variable: cuánto contribuyó cada color a las decisiones del bosque
idx = np.argsort(importancias)[::-1]    # Ordena los índices de mayor a menor importancia para que la barra más alta quede a la izquierda


fig, ax = plt.subplots(figsize=(9, 5))  # Crea el lienzo: 9 pulgadas de ancho y 5 de alto
colores_imp = [COLOR_RF if v > importancias.mean() else "#A8C4E0"
               for v in importancias[idx]]
#  Asigna color azul oscuro a las variables más importantes que la media y azul claro a las que están por debajo
barras = ax.bar(range(len(FEATURES)), importancias[idx],
                color=colores_imp, edgecolor="white")
#  Dibuja las barras verticales ordenadas de mayor a menor importancia, con borde blanco entre ellas
ax.set_xticks(range(len(FEATURES)))  # Define las posiciones de las etiquetas en el eje X
ax.set_xticklabels([FEATURES_LABEL[i] for i in idx],
                   rotation=35, ha="right", fontsize=9)
#  Pone los nombres legibles de las variables en el eje X, en el orden de importancia, girados 35° para que no se solapen
ax.set_ylabel("Importancia (media de disminución de impureza Gini)", fontsize=10)  # Etiqueta técnica del eje Y: el criterio que usa el Random Forest para medir importancia
ax.set_title("Importancia de variables cromáticas\n"
             "en la predicción de emociones — Random Forest",
             fontsize=13, fontweight="bold")  # Título en dos líneas, tamaño 13 y negrita
ax.axhline(importancias.mean(), color="gray", linestyle="--",
           linewidth=1.2, label="Importancia media")
# Línea horizontal gris discontinua que marca la importancia media: las barras por encima son las variables más relevantes
for b, v in zip(barras, importancias[idx]):  # Recorre cada barra junto a su valor para añadirle etiqueta numérica
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002,
            f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    #  Escribe el valor de importancia encima de cada barra, centrado, con 3 decimales y en negrita
ax.legend(fontsize=10)  # Muestra la leyenda que explica qué es la línea gris
ax.set_xlim(-0.6, len(FEATURES) - 0.4)  # Ajusta el margen horizontal para que las barras no queden pegadas a los bordes
plt.tight_layout()  # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G4_importancia_variables_RF.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución
plt.close()   # Cierra el gráfico y libera memoria
print("    ✓ G4_importancia_variables_RF.png")  # Confirma que el gráfico se guardó correctamente

# GRÁFICO 5 — Matriz de confusión Random Forest
print("[G5] Matriz de confusión Random Forest...")  # Avisa en pantalla de que va a generar el gráfico G5
cm_rf = confusion_matrix(y_test, y_pred_rf)  # Crea la matriz de confusión: una tabla que cruza la emoción real de cada producto con la que predijo el modelo
cm_rf_pct = cm_rf.astype(float) / cm_rf.sum(axis=1, keepdims=True) * 100
# Convierte los números absolutos de la matriz a porcentajes por fila:
# divide cada celda entre el total de productos de esa emoción real y multiplica por 100
# Así en lugar de "124 productos" veremos "87.3%", más fácil de interpretar


fig, ax = plt.subplots(figsize=(10, 8))  # Crea el lienzo: 10 pulgadas de ancho y 8 de alto (más grande porque tiene más etiquetas)
im = ax.imshow(cm_rf_pct, cmap="Blues", vmin=0, vmax=100)  # Dibuja la matriz como mapa de calor en azules: blanco=0%, azul oscuro=100%
plt.colorbar(im, ax=ax, shrink=0.8, label="% de predicciones (por fila real)")  # Añade la barra lateral que explica la escala de colores
ax.set_xticks(range(N_CLASES))   # Define las posiciones de las etiquetas en el eje X (emociones predichas)
ax.set_yticks(range(N_CLASES))   # Define las posiciones de las etiquetas en el eje Y (emociones reales)
ax.set_xticklabels(CLASES, rotation=40, ha="right", fontsize=9)  # Pone los nombres de las emociones en el eje X girados 40° para que no se solapen
ax.set_yticklabels(CLASES, fontsize=9)  # Pone los nombres de las emociones en el eje Y
for i in range(N_CLASES):      # Bucle por cada fila de la matriz (emoción real)
    for j in range(N_CLASES):  # Bucle por cada columna de la matriz (emoción predicha): recorre todas las celdas una a una
        val = cm_rf_pct[i, j]  # Obtiene el porcentaje de esa celda concreta
        ax.text(j, i, f"{val:.1f}%\n({cm_rf[i,j]})",
                ha="center", va="center", fontsize=7.5,
                color="white" if val > 55 else "black",
                fontweight="bold" if i == j else "normal")
        # Escribe dentro de cada celda dos datos: el porcentaje (con 1 decimal) y entre paréntesis el número absoluto
        #    - Si el fondo es muy oscuro (>55%), el texto es blanco para que se lea bien; si no, negro
        #    - Si la celda está en la diagonal (i==j), significa que acertó: se pone en negrita para destacarlo
ax.set_xlabel("Emoción predicha", fontsize=11)  # Etiqueta del eje X: lo que dijo el modelo
ax.set_ylabel("Emoción real", fontsize=11)      # Etiqueta del eje Y: lo que era realmente
ax.set_title(f"Matriz de confusión — Random Forest\n"
             f"Accuracy = {acc_rf:.3f}  |  F1 ponderado = {f1_rf:.3f}",
             fontsize=13, fontweight="bold")
# Título en dos líneas: el nombre del gráfico y las dos métricas principales del modelo con 3 decimales
plt.tight_layout()  # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G5_matriz_confusion_RF.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución en la carpeta de gráficos
plt.close()   # Cierra el gráfico y libera memoria para no ralentizar el script
print("    ✓ G5_matriz_confusion_RF.png")  # Confirma que el gráfico se guardó correctamente


print(f"\n    INFORME COMPLETO Random Forest:")  # Imprime un encabezado antes del informe detallado
print(classification_report(y_test, y_pred_rf,
                             target_names=CLASES, zero_division=0))
# Imprime en pantalla un informe completo por cada emoción con cuatro métricas:
#    - precision: de las veces que predijo esa emoción, ¿qué % acertó?
#    - recall: de todos los productos de esa emoción, ¿qué % detectó?
#    - f1-score: media entre precisión y recall para esa emoción
#    - support: cuántos productos reales había de esa emoción en el test
#    zero_division=0 evita errores si alguna emoción no tuvo ninguna predicción


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 2 — RED NEURONAL ARTIFICIAL (MLP)
# ══════════════════════════════════════════════════════════════════════════════
# ¿QUÉ ES?
#   Un Perceptrón Multicapa (MLP) es una red neuronal de alimentación hacia
#   adelante compuesta por neuronas organizadas en capas:
#     · Capa de entrada: recibe las 10 variables cromáticas normalizadas.
#     · Capas ocultas (128→64→32): cada neurona aplica una función de
#       activación ReLU — f(x) = max(0, x) — sobre la suma ponderada de sus
#       entradas, aprendiendo representaciones intermedias del color.
#     · Capa de salida: 9 neuronas con softmax que devuelven probabilidades
#       para cada emoción (la clase con mayor probabilidad es la predicción).
#   El entrenamiento minimiza la entropía cruzada mediante el algoritmo Adam
#   (Adaptive Moment Estimation), que ajusta tasas de aprendizaje por
#   parámetro y converge más rápido que el gradiente estocástico clásico.
#
# ¿POR QUÉ PARA ESTE TFG?
#   · El espacio L*a*b* tiene fronteras de decisión no lineales y complejas.
#     La arquitectura 128→64→32 puede aprender jerarquías: la primera capa
#     detecta patrones básicos de luminosidad, las siguientes combinaciones
#     más abstractas de color-emoción.
#   · early_stopping evita el sobreajuste automáticamente, reservando el 10%
#     del train como validación interna.
#   · Complementa al RF: si ambos coinciden en un error, es un error del dato;
#     si discrepan, indica ambigüedad emocional real en ese producto.
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Entrenando Modelo 2: Red Neuronal MLP (búsqueda de hiperparámetros)...")  # Avisa en pantalla de que empieza el entrenamiento del segundo modelo
print("    [RandomizedSearchCV 12 iter × 5-fold — ~3-6 min]")  # Informa de que probará 12 combinaciones de parámetros y tardará entre 3 y 6 minutos


_param_mlp = {
    "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64), (64, 64, 32)],
    # Arquitecturas de la red neuronal a probar: cada tupla define cuántas neuronas tiene cada capa oculta
    # Por ejemplo (128, 64, 32) significa 3 capas con 128, 64 y 32 neuronas respectivamente
    "alpha":              [0.0001, 0.001, 0.01],
    # Parámetro de regularización L2: penaliza los pesos muy grandes para evitar sobreajuste
    # Valores más altos = modelo más conservador y generalizable
    "learning_rate_init": [0.0005, 0.001, 0.005],
    # Velocidad de aprendizaje inicial del algoritmo Adam: controla cuánto se ajustan los pesos en cada paso
    # Valores muy altos aprenden rápido pero pueden saltarse el óptimo; muy bajos aprenden lento pero con más precisión
}
_search_mlp = RandomizedSearchCV(
    MLPClassifier(activation="relu", solver="adam", max_iter=500,
    # Crea la red neuronal base con:
    # - activation="relu": función de activación ReLU, que introduce no-linealidad y es la más usada en redes modernas
    # - solver="adam": algoritmo de optimización que ajusta los pesos de forma adaptativa y eficiente
    # - max_iter=500: máximo 500 épocas de entrenamiento si no se activa el early stopping antes
                  early_stopping=True, validation_fraction=0.10,
    # - early_stopping=True: para automáticamente si la red deja de mejorar, evitando sobreajuste
    # - validation_fraction=0.10: reserva el 10% de los datos de entrenamiento para vigilar si mejora
                  n_iter_no_change=25, random_state=42, verbose=False),
    # - n_iter_no_change=25: si tras 25 épocas seguidas no mejora, detiene el entrenamiento
    # - random_state=42: semilla fija para que los resultados sean reproducibles
    # - verbose=False: no muestra mensajes intermedios durante el entrenamiento
    _param_mlp, n_iter=12, cv=cv, scoring="f1_weighted",
    # Pasa el diccionario de parámetros y configura la búsqueda:
    # - n_iter=12: prueba solo 12 combinaciones aleatorias de los parámetros definidos arriba
    # - cv=cv: usa la validación cruzada de 5 partes definida anteriormente
    # - scoring="f1_weighted": selecciona la mejor combinación según el F1 ponderado
    random_state=42, n_jobs=-1
    # - random_state=42: misma semilla para reproducibilidad en la búsqueda aleatoria
    # - n_jobs=-1: usa todos los núcleos del procesador para ejecutar en paralelo y ir más rápido
)
_search_mlp.fit(X_train, y_train)  # Lanza la búsqueda: entrena y evalúa las 12 combinaciones con los datos de entrenamiento
mlp = _search_mlp.best_estimator_  # Guarda en "mlp" la red neuronal con la combinación de parámetros que mejor F1 obtuvo
print(f"    → Mejores hiperparámetros MLP: {_search_mlp.best_params_}")  # Muestra en pantalla cuál fue la combinación ganadora


y_pred_mlp = mlp.predict(X_test)  # Usa la mejor red neuronal para predecir las emociones del conjunto de prueba (datos que nunca ha visto)
acc_mlp    = accuracy_score(y_test, y_pred_mlp)  # Calcula el porcentaje de predicciones correctas sobre el total
f1_mlp     = f1_score(y_test, y_pred_mlp, average="weighted")  # Calcula el F1 ponderado: media entre precisión y recall dando más peso a las emociones más frecuentes
prec_mlp   = precision_score(y_test, y_pred_mlp, average="weighted", zero_division=0)  # Calcula la precisión: de las veces que predijo una emoción, cuántas acertó realmente
rec_mlp    = recall_score(y_test, y_pred_mlp, average="weighted", zero_division=0)  # Calcula el recall: de todos los productos de cada emoción, cuántos detectó correctamente
cv_mlp     = cross_val_score(mlp, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)  # Valida el modelo en 5 particiones distintas para comprobar que su rendimiento es estable


print(f"    → Épocas entrenadas: {mlp.n_iter_}")  # Muestra cuántas épocas necesitó hasta que el early stopping lo detuvo
print(f"    → Accuracy test:     {acc_mlp:.4f}")  # Muestra el porcentaje de aciertos con 4 decimales
print(f"    → F1 ponderado:      {f1_mlp:.4f}")   # Muestra el F1 con 4 decimales
print(f"    → CV 5-fold:         {cv_mlp.mean():.4f} ± {cv_mlp.std():.4f}")  # Muestra la media y desviación del accuracy en las 5 validaciones: desviación baja indica modelo estable


# GRÁFICO 6 — Curva de pérdida (loss) durante el entrenamiento MLP
print("[G6] Curva de pérdida MLP...")  # Avisa en pantalla de que va a generar el gráfico G6
fig, ax = plt.subplots(figsize=(8, 5))  # Crea el lienzo: 8 pulgadas de ancho y 5 de alto
ax.plot(mlp.loss_curve_, color=COLOR_MLP, linewidth=2, label="Pérdida entrenamiento")
# Dibuja la curva de pérdida (error) a lo largo de todas las épocas de entrenamiento en verde
# Si la curva baja y se estabiliza, el modelo aprendió correctamente
if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
    # Comprueba si el modelo guardó las puntuaciones de validación interna durante el entrenamiento
    ax.plot(mlp.validation_scores_, color="#C0392B", linewidth=2,
            linestyle="--", label="Accuracy validación interna")
    # Si existen, dibuja también la curva de accuracy en validación en rojo discontinuo
    # Permite ver si el modelo mejora igual en entrenamiento que en datos no vistos
    ax2 = ax.twinx()
    # Crea un segundo eje Y a la derecha del gráfico para la escala del accuracy de validación
    # "twinx" significa que comparte el mismo eje X pero tiene su propio eje Y independiente
    ax2.plot(mlp.validation_scores_, color="#C0392B", linewidth=2,
             linestyle="--", alpha=0)
    # Dibuja la misma curva roja en el segundo eje pero completamente transparente (alpha=0)
    # Solo sirve para que el eje Y derecho tenga la escala correcta sin duplicar la línea visible
    ax2.set_ylabel("Accuracy validación", fontsize=10, color="#C0392B")  # Etiqueta del eje Y derecho en rojo
ax.set_xlabel("Época", fontsize=11)  # Etiqueta del eje X: cada época es una pasada completa por todos los datos de entrenamiento
ax.set_ylabel("Entropía cruzada (loss)", fontsize=11)  # Etiqueta del eje Y izquierdo: la entropía cruzada mide cuánto se equivoca la red
ax.set_title("Curva de pérdida durante el entrenamiento — Red Neuronal MLP\n"
             f"({mlp.n_iter_} épocas hasta early stopping)",
             fontsize=13, fontweight="bold")  # Título en dos líneas incluyendo el número de épocas que necesitó
ax.legend(fontsize=10)  # Muestra la leyenda que diferencia la curva de pérdida y la de accuracy
ax.grid(alpha=0.3)      # Añade una rejilla de fondo muy suave (30% de opacidad) para facilitar la lectura
plt.tight_layout()      # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G6_curva_perdida_MLP.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución en la carpeta de gráficos
plt.close()  # Cierra el gráfico y libera memoria para no ralentizar el script
print("    ✓ G6_curva_perdida_MLP.png")  # Confirma que el gráfico se guardó correctamente


# GRÁFICO 7 — Matriz de confusión MLP
print("[G7] Matriz de confusión MLP...")  # Avisa en pantalla de que va a generar el gráfico G7
cm_mlp = confusion_matrix(y_test, y_pred_mlp)  # Crea la matriz de confusión: tabla que cruza la emoción real con la predicha por la red neuronal
cm_mlp_pct = cm_mlp.astype(float) / cm_mlp.sum(axis=1, keepdims=True) * 100
# Convierte los valores absolutos a porcentajes por fila: divide cada celda entre el total de productos
# de esa emoción real y multiplica por 100, para que sea más fácil de interpretar visualmente


fig, ax = plt.subplots(figsize=(10, 8))  # Crea el lienzo: 10x8 pulgadas, más grande para acomodar todas las etiquetas
im = ax.imshow(cm_mlp_pct, cmap="Greens", vmin=0, vmax=100)  # Dibuja la matriz como mapa de calor en verdes: blanco=0%, verde oscuro=100%
plt.colorbar(im, ax=ax, shrink=0.8, label="% de predicciones (por fila real)")  # Añade la barra lateral que explica la escala de colores
ax.set_xticks(range(N_CLASES))  # Define las posiciones de las etiquetas en el eje X (emociones predichas)
ax.set_yticks(range(N_CLASES))  # Define las posiciones de las etiquetas en el eje Y (emociones reales)
ax.set_xticklabels(CLASES, rotation=40, ha="right", fontsize=9)  # Pone los nombres de las emociones en el eje X girados 40° para que no se solapen
ax.set_yticklabels(CLASES, fontsize=9)  # Pone los nombres de las emociones en el eje Y
for i in range(N_CLASES):      # Bucle por cada fila de la matriz (emoción real)
    for j in range(N_CLASES):  # Bucle por cada columna de la matriz (emoción predicha): recorre todas las celdas una a una
        val = cm_mlp_pct[i, j]  # Obtiene el porcentaje de esa celda concreta
        ax.text(j, i, f"{val:.1f}%\n({cm_mlp[i,j]})",
                ha="center", va="center", fontsize=7.5,
                color="white" if val > 55 else "black",
                fontweight="bold" if i == j else "normal")
        # Escribe dentro de cada celda el porcentaje y entre paréntesis el número absoluto
        # Si el fondo es muy oscuro (>55%), el texto es blanco para que se lea bien; si no, negro
        # Si la celda está en la diagonal (i==j) significa que acertó: se pone en negrita para destacarlo
ax.set_xlabel("Emoción predicha", fontsize=11)  # Etiqueta del eje X: lo que predijo el modelo
ax.set_ylabel("Emoción real", fontsize=11)      # Etiqueta del eje Y: lo que era realmente
ax.set_title(f"Matriz de confusión — Red Neuronal MLP\n"
             f"Accuracy = {acc_mlp:.3f}  |  F1 ponderado = {f1_mlp:.3f}",
             fontsize=13, fontweight="bold")  # Título en dos líneas con las dos métricas principales del modelo
plt.tight_layout()  # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G7_matriz_confusion_MLP.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución
plt.close()  # Cierra el gráfico y libera memoria para no ralentizar el script
print("    ✓ G7_matriz_confusion_MLP.png")  # Confirma que el gráfico se guardó correctamente


print(f"\n    INFORME COMPLETO Red Neuronal MLP:")  # Imprime un encabezado antes del informe detallado
print(classification_report(y_test, y_pred_mlp,
                             target_names=CLASES, zero_division=0))
# Imprime en pantalla un informe completo por cada emoción con cuatro métricas:
# - precision: de las veces que predijo esa emoción, que porcentaje acertó
# - recall: de todos los productos de esa emoción, que porcentaje detectó
# - f1-score: media entre precisión y recall para esa emoción concreta
# - support: cuántos productos reales había de esa emoción en el conjunto de test
# zero_division=0 evita errores si alguna emoción no tuvo ninguna predicción

# ══════════════════════════════════════════════════════════════════════════════
# MODELO 3 ★ ORIGINAL — MULTIOUTPUT SVR (Perfil Emocional Completo)
# ══════════════════════════════════════════════════════════════════════════════
# ¿QUÉ ES?
#   Un Support Vector Regressor (SVR) busca el hiperplano que mejor aproxima
#   los valores continuos minimizando el error dentro de un margen ε (epsilon).
#   A diferencia de los modelos de clasificación que devuelven UN label, este
#   modelo predice SIMULTÁNEAMENTE los 8 scores emocionales de cada imagen:
#   cuánto de Alegría, Energía, Calma, Tristeza... transmite ese producto.
#
#   MultiOutputRegressor envuelve un SVR independiente por cada uno de los
#   8 scores, entrenando 8 modelos en paralelo que comparten las mismas
#   features pero predicen targets distintos.
#
# ¿QUÉ LO HACE ORIGINAL?
#   Los modelos 1 y 2 responden: "¿qué emoción es la dominante?"
#   Este modelo responde: "¿cuál es el perfil emocional completo del producto?"
#   → Un producto puede transmitir 40% Aburrimiento + 30% Tristeza + 20% Calma.
#   → Esa información es imposible de capturar con una clasificación binaria.
#   → Es directamente aplicable en marketing: permite diseñar productos con
#     un mix emocional objetivo, no solo una emoción dominante.
#
# ¿POR QUÉ SVR Y NO RANDOM FOREST REGRESSOR?
#   · SVR con kernel RBF es especialmente eficaz en espacios de alta dimensión
#     donde las variables están en rangos similares (nuestras features ya están
#     normalizadas en [0,1]).
#   · El margen ε (epsilon) hace el modelo robusto a pequeñas variaciones de
#     color, lo que es ideal dado el ruido cromático del dataset.
#   · Ofrece un contraste metodológico claro con el RF (ensamble de árboles)
#     y el MLP (red neuronal), justificando la comparativa entre los tres.
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] Entrenando Modelo 3 ★ original: MultiOutput SVR...")  # Avisa de que empieza el tercer modelo, el más original del TFG
print("    (predice el perfil emocional completo — 8 scores simultáneamente)")  # Explica que este modelo no predice una sola emoción sino los 8 valores a la vez
print("    [búsqueda de hiperparámetros SVR vía proxy (score 1) — ~3-6 min]")  # Informa de que la búsqueda tardará entre 3 y 6 minutos


_param_svr = {
    "C":       [1, 5, 10, 50, 100],   # Parámetro de penalización: controla cuánto castiga el modelo los errores grandes; valores altos = más estricto
    "epsilon": [0.01, 0.02, 0.05, 0.1],  # Margen de tolerancia: diferencias menores que epsilon no se consideran error
    "gamma":   ["scale", "auto"],     # Define el alcance de influencia de cada punto; "scale" y "auto" son dos formas automáticas de calcularlo
}
_search_svr = RandomizedSearchCV(
    SVR(kernel="rbf"), _param_svr, n_iter=12, cv=3,
    # Crea el modelo SVR base con kernel "rbf" (función de base radial): permite aprender relaciones no lineales entre colores y emociones
    # Prueba 12 combinaciones aleatorias de los parámetros anteriores usando validación cruzada de 3 partes
    scoring="r2", random_state=42, n_jobs=-1, verbose=0
    # Selecciona la mejor combinación según el R² (cuánta variación explica el modelo)
    # random_state=42 para reproducibilidad, n_jobs=-1 para usar todos los núcleos del procesador
)
_search_svr.fit(X_train, ys_train[:, 0])   # Entrena la búsqueda usando solo el score de Ira como representativo de los 8
# Usar los 8 scores a la vez en la búsqueda sería muy lento, así que se usa uno solo como proxy (muestra) para encontrar los mejores parámetros
_best_svr = _search_svr.best_params_  # Guarda los parámetros ganadores de la búsqueda
print(f"    → Mejores hiperparámetros SVR: {_best_svr}")  # Muestra en pantalla cuál fue la combinación ganadora


svr_base = SVR(kernel="rbf", **_best_svr)  # Crea el SVR definitivo con los mejores parámetros encontrados; "**" desempaqueta el diccionario como argumentos individuales
mo_svr = MultiOutputRegressor(svr_base, n_jobs=-1)  # Envuelve el SVR para que pueda predecir los 8 scores a la vez: entrena internamente un SVR independiente por cada emoción
mo_svr.fit(X_train, ys_train)  # Entrena el modelo completo con los datos de entrenamiento y los 8 scores emocionales como objetivo


ys_pred = mo_svr.predict(X_test)  # Usa el modelo para predecir los 8 scores emocionales de cada producto del conjunto de prueba


# Normalizar para que los scores predichos sumen 1 (condición de distribución de probabilidad)
ys_pred_norm = np.clip(ys_pred, 0, 1)  # Recorta cualquier valor por debajo de 0 o por encima de 1, ya que los scores son proporciones y no pueden salirse de ese rango
ys_pred_norm = ys_pred_norm / ys_pred_norm.sum(axis=1, keepdims=True)
# Divide cada fila entre su suma total para que los 8 scores de cada producto sumen exactamente 1
# Es decir, los convierte en una distribución de probabilidad: cada score indica qué proporción de la emoción total representa


# Métricas por score
r2_por_score   = [r2_score(ys_test[:, i], ys_pred_norm[:, i])   for i in range(8)]  # Calcula el R² para cada una de las 8 emociones: cuánta variación del score real explica el modelo
rmse_por_score = [np.sqrt(mean_squared_error(ys_test[:, i], ys_pred_norm[:, i])) for i in range(8)]  # Calcula el RMSE para cada emoción: error medio en las mismas unidades que el score (cuánto se desvía de media)
r2_global   = np.mean(r2_por_score)   # Calcula la media del R² de las 8 emociones: métrica global del modelo
rmse_global = np.mean(rmse_por_score)  # Calcula la media del RMSE de las 8 emociones: error global medio del modelo


print(f"\n    → R² medio global:    {r2_global:.4f}")   # Muestra el R² global con 4 decimales: cuanto más cerca de 1, mejor
print(f"    → RMSE medio global:  {rmse_global:.4f}")   # Muestra el error medio global con 4 decimales: cuanto más cerca de 0, mejor
print(f"\n    R² y RMSE por score emocional:")  # Encabezado del desglose por emoción
for i, (label, r2, rmse) in enumerate(zip(SCORE_LABELS, r2_por_score, rmse_por_score)):
    print(f"      {label:<15}  R²={r2:.4f}  RMSE={rmse:.4f}")
    # Imprime el R² y RMSE de cada emoción individualmente
    # "<15" alinea el nombre de la emoción en 15 caracteres para que los números queden en columna


# GRÁFICO 9 — Predicción vs Real de scores (scatter por emoción)
print("[G9] Predicción vs Real de scores...")  # Avisa de que va a generar el gráfico G9
fig, axes = plt.subplots(2, 4, figsize=(14, 7))  # Crea una cuadrícula de 2 filas y 4 columnas para los 8 gráficos de dispersión, uno por emoción
axes = axes.flatten()  # Convierte la cuadrícula 2x4 en una lista plana de 8 ejes para recorrerlos más fácilmente con un bucle
color_list = list(COLORES_EMOCION.values())[:8]  # Toma los primeros 8 colores del diccionario de emociones para asignar uno a cada gráfico


for i, (label, color) in enumerate(zip(SCORE_LABELS, color_list)):  # Bucle que genera un gráfico por cada emoción junto a su color
    ax = axes[i]  # Selecciona el subgráfico correspondiente a esta emoción
    ax.scatter(ys_test[:, i], ys_pred_norm[:, i],
               alpha=0.3, s=8, color=color, edgecolors="none")
    # Dibuja un gráfico de dispersión: cada punto es un producto; eje X = score real, eje Y = score predicho
    # alpha=0.3 hace los puntos semitransparentes para ver la densidad; s=8 los hace pequeños para no solaparse
    lims = [0, max(ys_test[:, i].max(), ys_pred_norm[:, i].max()) * 1.05]  # Calcula los límites del gráfico: desde 0 hasta el valor máximo entre real y predicho, con un 5% extra de margen
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Predicción perfecta")
    # Dibuja una línea diagonal negra discontinua: representa la predicción perfecta (cuando predicho = real)
    # Cuanto más cerca estén los puntos de esta línea, mejor predice el modelo
    ax.set_title(f"{label}\nR²={r2_por_score[i]:.3f}  RMSE={rmse_por_score[i]:.4f}",
                 fontsize=9, fontweight="bold")  # Título con el nombre de la emoción y sus dos métricas principales
    ax.set_xlabel("Score real", fontsize=8)    # Etiqueta del eje X
    ax.set_ylabel("Score predicho", fontsize=8)  # Etiqueta del eje Y
    ax.set_xlim(lims)   # Aplica los límites calculados al eje X
    ax.set_ylim(lims)   # Aplica los mismos límites al eje Y para que la diagonal quede perfecta
    ax.tick_params(labelsize=7)  # Reduce el tamaño de los números de los ejes para que no se solapen
    ax.grid(alpha=0.2)  # Añade una rejilla muy suave de fondo


fig.suptitle("Modelo original ★ MultiOutput SVR\n"
             "Predicción vs. valor real de cada score emocional",
             fontsize=13, fontweight="bold")  # Título general que aparece encima de los 8 subgráficos
plt.tight_layout()  # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G9_pred_vs_real_scores.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución
plt.close()  # Cierra el gráfico y libera memoria
print("    ✓ G9_pred_vs_real_scores.png")  # Confirma que el gráfico se guardó correctamente


# GRÁFICO 10 — Perfil emocional predicho vs real (5 ejemplos representativos)
print("[G10] Perfiles emocionales predichos vs reales...")  # Avisa de que va a generar el gráfico G10


# Seleccionar 5 productos con alta confianza emocional para que el gráfico sea ilustrativo
idx_ejemplo = df["confianza_emocional"].nlargest(5).index.tolist()  # Selecciona los 5 productos con mayor confianza emocional como candidatos para el gráfico
test_indices_originales = df.index[df.index.isin(
    pd.RangeIndex(len(df))[len(X_train):]
)]
# Filtra esos candidatos para quedarse solo con los que pertenecen al conjunto de prueba
# pd.RangeIndex genera los índices del dataset y se comparan con los del train para aislar los del test
if len(test_indices_originales) >= 5:
    idx_ejemplo = test_indices_originales[:5].tolist()  # Si hay al menos 5 productos en el test, toma los 5 primeros
else:
    idx_ejemplo = list(range(min(5, len(X_test))))  # Si no hay suficientes, toma todos los disponibles hasta un máximo de 5


fig, axes = plt.subplots(1, 5, figsize=(16, 5))  # Crea una fila de 5 subgráficos, uno por producto de ejemplo
x_pos = np.arange(len(SCORE_LABELS))  # Posiciones numéricas del 0 al 7 para colocar las barras de las 8 emociones
width = 0.35  # Ancho de cada barra; al haber dos grupos (real y predicho) se necesita que sean estrechas para que quepan juntas


for plot_i, test_i in enumerate(range(min(5, len(X_test)))):  # Bucle que genera un subgráfico por cada uno de los 5 productos de ejemplo
    ax = axes[plot_i]  # Selecciona el subgráfico correspondiente a este producto
    real   = ys_test[test_i]       # Obtiene los 8 scores reales de este producto
    pred   = ys_pred_norm[test_i]  # Obtiene los 8 scores predichos (ya normalizados) de este producto
    nombre = df.iloc[len(X_train) + test_i]["nombre"][:18] + "..." \
             if len(df.iloc[len(X_train) + test_i]["nombre"]) > 18 \
             else df.iloc[len(X_train) + test_i]["nombre"]
    # Obtiene el nombre del producto y lo recorta a 18 caracteres si es demasiado largo, añadiendo "..." al final


    bars_r = ax.bar(x_pos - width/2, real, width,
                    label="Real", color="#5DADE2", alpha=0.85, edgecolor="white")
    # Dibuja las barras azules desplazadas a la izquierda del centro: representan los scores reales
    bars_p = ax.bar(x_pos + width/2, pred, width,
                    label="Predicho", color=COLOR_SVR, alpha=0.85, edgecolor="white")
    # Dibuja las barras doradas desplazadas a la derecha del centro: representan los scores predichos
    ax.set_xticks(x_pos)  # Define las posiciones de las etiquetas en el eje X
    ax.set_xticklabels(SCORE_LABELS, rotation=55, ha="right", fontsize=7)  # Pone los nombres de las emociones girados 55° para que no se solapen
    ax.set_ylim(0, 0.55)  # Fija el eje Y entre 0 y 0.55 para que todos los subgráficos tengan la misma escala y sean comparables
    ax.set_title(f"Producto {plot_i+1}\n{nombre}", fontsize=8, fontweight="bold")  # Título con el número y nombre del producto
    ax.set_ylabel("Score emocional" if plot_i == 0 else "", fontsize=8)  # Solo pone la etiqueta del eje Y en el primer subgráfico para no repetirla 5 veces
    if plot_i == 0:
        ax.legend(fontsize=7)  # Solo muestra la leyenda (real vs predicho) en el primer subgráfico
    ax.tick_params(labelsize=7)   # Reduce el tamaño de los números de los ejes
    ax.grid(axis="y", alpha=0.25)  # Añade una rejilla horizontal muy suave para facilitar la lectura de los valores


fig.suptitle("Modelo original ★ — Perfil emocional completo: predicción vs. real\n"
             "(los 8 scores emocionales de 5 productos del conjunto de test)",
             fontsize=12, fontweight="bold")  # Título general encima de los 5 subgráficos
plt.tight_layout()  # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G10_perfiles_emocionales.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución
plt.close()  # Cierra el gráfico y libera memoria
print("    ✓ G10_perfiles_emocionales.png")  # Confirma que el gráfico se guardó correctamente


# GRÁFICO 11 — Error por emoción del modelo original
print("[G11] Error por emoción del modelo original...")  # Avisa de que va a generar el gráfico G11
fig, ax = plt.subplots(figsize=(10, 5))  # Crea el lienzo: 10 pulgadas de ancho y 5 de alto
x_pos = np.arange(len(SCORE_LABELS))  # Posiciones numéricas del 0 al 7 para las 8 barras
colores_r2 = [COLORES_EMOCION.get(l, "#999") for l in SCORE_LABELS]  # Asigna a cada barra el color de su emoción
colores_rmse = [c + "99" if len(c) == 7 else c for c in colores_r2]
# Añade "99" al final del código hexadecimal de cada color para hacerlo semitransparente
# Los colores hexadecimales tienen 6 dígitos (#RRGGBB); con 8 (#RRGGBBAA) el último par controla la opacidad


bars = ax.bar(x_pos, r2_por_score, color=colores_r2, edgecolor="white", width=0.6)  # Dibuja una barra por emoción con su altura igual al R² obtenido en esa emoción
ax.set_xticks(x_pos)  # Define las posiciones de las etiquetas en el eje X
ax.set_xticklabels(SCORE_LABELS, rotation=30, ha="right", fontsize=10)  # Pone los nombres de las emociones girados 30° para que no se solapen
ax.set_ylabel("R² (coeficiente de determinación)", fontsize=11)  # Etiqueta del eje Y con el nombre técnico de la métrica
ax.set_title("Calidad de predicción por emoción — Modelo original MultiOutput SVR\n"
             "(R² más cercano a 1 = mejor predicción de ese score)",
             fontsize=13, fontweight="bold")  # Título en dos líneas explicando qué significa el R²
ax.axhline(0, color="gray", linewidth=0.8)  # Línea horizontal en R²=0 que marca el límite inferior: por debajo de 0 el modelo es peor que predecir siempre la media
ax.axhline(r2_global, color="#2C3E50", linestyle="--",
           linewidth=1.5, label=f"R² medio = {r2_global:.3f}")
# Línea horizontal discontinua oscura que indica el R² medio global de los 8 scores
for b, v, rmse in zip(bars, r2_por_score, rmse_por_score):  # Bucle que añade etiquetas numéricas encima de cada barra
    ax.text(b.get_x() + b.get_width()/2,
            max(v, 0) + 0.01,
            f"R²={v:.3f}\nRMSE={rmse:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
    # Escribe encima de cada barra el R² y el RMSE de esa emoción en dos líneas
    # max(v,0) evita que el texto quede por debajo del eje si el R² fuera negativo
ax.legend(fontsize=10)  # Muestra la leyenda que explica la línea discontinua del R² medio
ax.set_ylim(min(min(r2_por_score) - 0.1, -0.15), 1.0)  # Fija el eje Y dejando espacio por debajo del mínimo y llegando hasta 1.0 como máximo
plt.tight_layout()  # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G11_r2_por_emocion_SVR.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución
plt.close()  # Cierra el gráfico y libera memoria
print("    ✓ G11_r2_por_emocion_SVR.png")  # Confirma que el gráfico se guardó correctamente


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 8 — Comparativa de métricas F1 por clase entre RF y MLP
# ══════════════════════════════════════════════════════════════════════════════
print("[G8] Comparativa F1 por clase RF vs MLP...")  # Avisa de que va a generar el gráfico G8
rep_rf  = classification_report(y_test, y_pred_rf,
                                 target_names=CLASES, output_dict=True, zero_division=0)
rep_mlp = classification_report(y_test, y_pred_mlp,
                                 target_names=CLASES, output_dict=True, zero_division=0)
# Genera los informes completos de ambos modelos como diccionarios (output_dict=True)
# en lugar de texto, para poder extraer los valores numéricos del F1 por emoción
f1_rf_por_clase  = [rep_rf[c]["f1-score"]  for c in CLASES]   # Extrae el F1 de cada emoción del informe del Random Forest
f1_mlp_por_clase = [rep_mlp[c]["f1-score"] for c in CLASES]   # Extrae el F1 de cada emoción del informe del MLP


x_pos = np.arange(N_CLASES)  # Posiciones numéricas del 0 al 8 para las barras de cada emoción
width = 0.35  # Ancho de cada barra; al haber dos modelos por emoción deben ser estrechas para que quepan juntas
fig, ax = plt.subplots(figsize=(12, 6))  # Crea el lienzo: 12 pulgadas de ancho y 6 de alto
bars_rf  = ax.bar(x_pos - width/2, f1_rf_por_clase,  width, label="Random Forest",
                  color=COLOR_RF,  edgecolor="white", alpha=0.9)
# Dibuja las barras azules del Random Forest desplazadas a la izquierda del centro de cada grupo
bars_mlp = ax.bar(x_pos + width/2, f1_mlp_por_clase, width, label="Red Neuronal (MLP)",
                  color=COLOR_MLP, edgecolor="white", alpha=0.9)
# Dibuja las barras verdes del MLP desplazadas a la derecha del centro de cada grupo
ax.set_xticks(x_pos)  # Define las posiciones de las etiquetas en el eje X
ax.set_xticklabels(CLASES, rotation=30, ha="right", fontsize=10)  # Pone los nombres de las emociones girados 30°
ax.set_ylabel("F1-score", fontsize=11)  # Etiqueta del eje Y
ax.set_title("Comparativa de F1-score por clase emocional\n"
             "Random Forest vs. Red Neuronal MLP",
             fontsize=13, fontweight="bold")  # Título en dos líneas
ax.axhline(f1_rf,  color=COLOR_RF,  linestyle="--", linewidth=1.2, alpha=0.7,
           label=f"F1 medio RF = {f1_rf:.3f}")
# Línea horizontal azul discontinua que marca el F1 medio global del Random Forest
ax.axhline(f1_mlp, color=COLOR_MLP, linestyle="--", linewidth=1.2, alpha=0.7,
           label=f"F1 medio MLP = {f1_mlp:.3f}")
# Línea horizontal verde discontinua que marca el F1 medio global del MLP
ax.set_ylim(0, 1.1)   # Fija el eje Y entre 0 y 1.1 para dejar espacio a las etiquetas de las barras más altas
ax.legend(fontsize=10)        # Muestra la leyenda diferenciando los dos modelos y sus líneas de media
ax.grid(axis="y", alpha=0.25)  # Añade una rejilla horizontal muy suave para facilitar la lectura
for b, v in zip(bars_rf, f1_rf_por_clase):   # Bucle para añadir etiquetas numéricas encima de las barras azules
    if v > 0.03:  # Solo etiqueta si la barra es suficientemente alta para que quepa el número
        ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=8, color=COLOR_RF, fontweight="bold")
for b, v in zip(bars_mlp, f1_mlp_por_clase):  # Bucle para añadir etiquetas numéricas encima de las barras verdes
    if v > 0.03:  # Solo etiqueta si la barra es suficientemente alta para que quepa el número
        ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=8, color="#3D7A28", fontweight="bold")
plt.tight_layout()  # Ajusta los márgenes para que nada quede cortado
plt.savefig(os.path.join(RUTA_GRAFICOS, "G8_comparativa_F1_RF_vs_MLP.png"), dpi=150)  # Guarda el gráfico como PNG a 150 de resolución
plt.close()  # Cierra el gráfico y libera memoria
print("    ✓ G8_comparativa_F1_RF_vs_MLP.png")  # Confirma que el gráfico se guardó correctamente


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 12 — Resumen comparativo final de los tres modelos
# ══════════════════════════════════════════════════════════════════════════════
print("[G12] Resumen comparativo final...")  # Avisa de que va a generar el gráfico G12
fig = plt.figure(figsize=(14, 8))  # Crea el lienzo principal: 14 pulgadas de ancho y 8 de alto
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
# Divide el lienzo en una cuadrícula de 2 filas y 3 columnas para los 4 paneles del resumen
# hspace=0.45 es el espacio vertical entre filas; wspace=0.35 es el espacio horizontal entre columnas


# Panel A: Accuracy y F1 de RF y MLP
ax_a = fig.add_subplot(gs[0, :2])  # El panel A ocupa las dos primeras columnas de la primera fila
metricas  = ["Accuracy", "Precision", "Recall", "F1-score"]  # Nombres de las 4 métricas a comparar
vals_rf   = [acc_rf,  prec_rf,  rec_rf,  f1_rf]    # Valores de las 4 métricas del Random Forest
vals_mlp  = [acc_mlp, prec_mlp, rec_mlp, f1_mlp]   # Valores de las 4 métricas del MLP
x_m = np.arange(len(metricas))  # Posiciones numéricas del 0 al 3 para las 4 métricas
ax_a.bar(x_m - 0.2, vals_rf,  0.38, label="Random Forest", color=COLOR_RF,  edgecolor="white")  # Barras azules del RF desplazadas a la izquierda
ax_a.bar(x_m + 0.2, vals_mlp, 0.38, label="Red Neuronal (MLP)", color=COLOR_MLP, edgecolor="white")  # Barras verdes del MLP desplazadas a la derecha
ax_a.set_xticks(x_m)  # Define las posiciones de las etiquetas en el eje X
ax_a.set_xticklabels(metricas, fontsize=11)  # Pone los nombres de las métricas en el eje X
ax_a.set_ylim(0, 1.1)  # Fija el eje Y entre 0 y 1.1 para dejar margen a las etiquetas
ax_a.set_ylabel("Valor de la métrica", fontsize=10)  # Etiqueta del eje Y
ax_a.set_title("Métricas de clasificación — RF vs. MLP", fontsize=11, fontweight="bold")  # Título del panel A
ax_a.legend(fontsize=9)         # Leyenda que diferencia los dos modelos
ax_a.grid(axis="y", alpha=0.25)  # Rejilla horizontal suave
for x, v in zip(x_m - 0.2, vals_rf):   # Bucle para añadir etiquetas numéricas encima de las barras azules del RF
    ax_a.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=9,
              color=COLOR_RF, fontweight="bold")
for x, v in zip(x_m + 0.2, vals_mlp):  # Bucle para añadir etiquetas numéricas encima de las barras verdes del MLP
    ax_a.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=9,
              color="#3D7A28", fontweight="bold")


# Panel B: R² y RMSE del modelo original
ax_b = fig.add_subplot(gs[0, 2])  # El panel B ocupa la tercera columna de la primera fila
colores_b = [COLORES_EMOCION.get(l, "#999") for l in SCORE_LABELS]  # Asigna a cada barra el color de su emoción
barras_r2 = ax_b.barh(SCORE_LABELS, r2_por_score,
                       color=colores_b, edgecolor="white", height=0.6)
# Dibuja barras horizontales con el R² de cada emoción, una por fila, con el color de cada emoción
ax_b.axvline(0, color="black", linewidth=0.8)  # Línea vertical en 0: referencia mínima del R²
ax_b.axvline(r2_global, color="#2C3E50", linestyle="--",
             linewidth=1.2, label=f"Media: {r2_global:.3f}")
# Línea vertical discontinua oscura que indica el R² medio global de las 8 emociones
ax_b.set_xlabel("R²", fontsize=9)  # Etiqueta del eje X
ax_b.set_title("R² por score — Modelo ★\nMultiOutput SVR", fontsize=11, fontweight="bold")  # Título del panel B
ax_b.legend(fontsize=8)  # Leyenda que explica la línea discontinua
for b, v in zip(barras_r2, r2_por_score):  # Bucle que añade el valor numérico a la derecha de cada barra
    ax_b.text(max(v, 0) + 0.005, b.get_y() + b.get_height()/2,
              f"{v:.3f}", va="center", fontsize=8, fontweight="bold")


# Panel C: CV 5-fold comparativa
ax_c = fig.add_subplot(gs[1, :2])  # El panel C ocupa las dos primeras columnas de la segunda fila
ax_c.boxplot([cv_rf, cv_mlp], labels=["Random Forest", "Red Neuronal (MLP)"],
             patch_artist=True,
             boxprops=dict(facecolor="#EBF5FB"),
             medianprops=dict(color="black", linewidth=2))
# Dibuja un diagrama de caja (boxplot) para cada modelo mostrando la distribución del accuracy en los 5 folds
# patch_artist=True rellena las cajas de color; boxprops define el color de relleno; medianprops define la línea de la mediana
ax_c.set_ylabel("Accuracy (validación cruzada 5-fold)", fontsize=10)  # Etiqueta del eje Y
ax_c.set_title("Estabilidad de modelos — Validación cruzada 5-fold\n"
               "(cada punto = 1 fold)", fontsize=11, fontweight="bold")  # Título del panel C explicando qué representa cada punto
ax_c.grid(axis="y", alpha=0.25)  # Rejilla horizontal suave


# Panel D: tabla resumen textual
ax_d = fig.add_subplot(gs[1, 2])  # El panel D ocupa la tercera columna de la segunda fila
ax_d.axis("off")  # Desactiva los ejes del panel porque solo va a contener una tabla de texto, no un gráfico
tabla = [
    ["Modelo",       "Acc",        "F1",        "Métrica clave"],         # Fila de cabecera de la tabla
    ["Random Forest", f"{acc_rf:.3f}", f"{f1_rf:.3f}", "Importancia vars."],  # Fila del RF con sus métricas
    ["MLP",          f"{acc_mlp:.3f}", f"{f1_mlp:.3f}", "Curva de pérdida"],   # Fila del MLP con sus métricas
    ["MultiOut SVR ★", "—",         f"R²={r2_global:.3f}", "Perfil 8 scores"],  # Fila del SVR: sin Accuracy porque es un modelo de regresión, no clasificación
]
t = ax_d.table(cellText=tabla[1:], colLabels=tabla[0],
               loc="center", cellLoc="center")
# Crea la tabla visual dentro del panel: cellText son las filas de datos, colLabels es la fila de cabecera
# loc="center" centra la tabla en el panel; cellLoc="center" centra el texto dentro de cada celda
t.auto_set_font_size(False)  # Desactiva el ajuste automático del tamaño de fuente para poder fijarlo manualmente
t.set_fontsize(9)   # Fija el tamaño de letra de todas las celdas a 9
t.scale(1, 1.8)     # Escala la tabla: ancho sin cambio (1) y altura de cada fila aumentada (1.8) para que no quede apretada
for j in range(4):  # Bucle que aplica formato especial a la fila de cabecera y a la fila del modelo estrella
    t[(0, j)].set_facecolor("#2E75B6")                        # Fondo azul corporativo para la celda de cabecera de la columna j
    t[(0, j)].set_text_props(color="white", fontweight="bold")  # Texto blanco y negrita en la cabecera
    t[(3, j)].set_facecolor("#FFF3CD")                        # Fondo amarillo suave para destacar la fila del modelo SVR estrella
ax_d.set_title("Resumen final", fontsize=11, fontweight="bold")  # Título del panel D


fig.suptitle("Análisis del Dato — Resumen comparativo de los tres modelos\n"
             "TFG: Impacto del Color de Producto en las Emociones del Consumidor",
             fontsize=13, fontweight="bold", y=1.01)
# Título general de todo el gráfico G12, con el nombre del TFG, colocado ligeramente por encima (y=1.01) para no solaparse con los paneles
plt.savefig(os.path.join(RUTA_GRAFICOS, "G12_resumen_comparativo.png"),
            dpi=150, bbox_inches="tight")
# Guarda el gráfico como PNG a 150 de resolución; bbox_inches="tight" incluye el título superior en la imagen sin cortarlo
plt.close()  # Cierra el gráfico y libera memoria
print("    ✓ G12_resumen_comparativo.png")  # Confirma que el gráfico se guardó correctamente

# ══════════════════════════════════════════════════════════════════════════════
# GUARDADO DE MODELOS Y RESULTADOS JSON
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6] Guardando modelos entrenados...")  # Avisa en pantalla de que empieza el guardado de los modelos entrenados
joblib.dump(rf,     os.path.join(RUTA_MODELOS, "modelo_random_forest.pkl"))  # Guarda el modelo Random Forest en un archivo .pkl: serializa toda la "inteligencia" aprendida para poder reutilizarla sin reentrenar
joblib.dump(mlp,    os.path.join(RUTA_MODELOS, "modelo_mlp.pkl"))            # Guarda la Red Neuronal MLP entrenada en otro archivo .pkl
joblib.dump(mo_svr, os.path.join(RUTA_MODELOS, "modelo_multioutput_svr.pkl"))  # Guarda el modelo SVR multioutput entrenado en otro archivo .pkl
joblib.dump(le,     os.path.join(RUTA_MODELOS, "label_encoder.pkl"))         # Guarda también el codificador de etiquetas: es necesario para convertir los números de vuelta a nombres de emociones al hacer predicciones futuras
print("    ✓ Modelos guardados en", RUTA_MODELOS)  # Confirma que los 4 archivos se guardaron correctamente


resultados = {
    "fecha":       datetime.now().isoformat(),   # Fecha y hora exacta de ejecución en formato estándar internacional (ej: 2025-05-14T10:32:07)
    "n_muestras":  int(len(df)),                 # Número total de productos del dataset
    "n_train":     int(len(X_train)),            # Número de productos usados para entrenar los modelos
    "n_test":      int(len(X_test)),             # Número de productos usados para evaluar los modelos
    "clases":      list(CLASES),                 # Lista de las emociones que clasifican los modelos
    "random_forest": {
        "accuracy":          round(float(acc_rf), 4),           # Porcentaje global de aciertos del RF redondeado a 4 decimales
        "f1_ponderado":      round(float(f1_rf), 4),            # F1 ponderado del RF redondeado a 4 decimales
        "precision":         round(float(prec_rf), 4),          # Precisión del RF redondeada a 4 decimales
        "recall":            round(float(rec_rf), 4),           # Recall del RF redondeado a 4 decimales
        "cv_media":          round(float(cv_rf.mean()), 4),     # Media del accuracy en los 5 folds de validación cruzada
        "cv_std":            round(float(cv_rf.std()), 4),      # Desviación típica del accuracy en los 5 folds: mide la estabilidad del modelo
        "f1_por_clase":      {c: round(rep_rf[c]["f1-score"], 4) for c in CLASES},  # Diccionario con el F1 individual de cada emoción
    },
    "mlp": {
        "accuracy":          round(float(acc_mlp), 4),          # Porcentaje global de aciertos del MLP redondeado a 4 decimales
        "f1_ponderado":      round(float(f1_mlp), 4),           # F1 ponderado del MLP redondeado a 4 decimales
        "precision":         round(float(prec_mlp), 4),         # Precisión del MLP redondeada a 4 decimales
        "recall":            round(float(rec_mlp), 4),          # Recall del MLP redondeado a 4 decimales
        "cv_media":          round(float(cv_mlp.mean()), 4),    # Media del accuracy en los 5 folds de validación cruzada
        "cv_std":            round(float(cv_mlp.std()), 4),     # Desviación típica del accuracy en los 5 folds
        "epocas_entrenadas": int(mlp.n_iter_),                  # Número de épocas que necesitó la red hasta que el early stopping la detuvo
        "f1_por_clase":      {c: round(rep_mlp[c]["f1-score"], 4) for c in CLASES},  # Diccionario con el F1 individual de cada emoción
    },
    "multioutput_svr": {
        "r2_global":         round(float(r2_global), 4),        # R² medio global de las 8 emociones redondeado a 4 decimales
        "rmse_global":       round(float(rmse_global), 4),      # RMSE medio global de las 8 emociones redondeado a 4 decimales
        "r2_por_score":      {l: round(float(v), 4) for l, v in zip(SCORE_LABELS, r2_por_score)},    # Diccionario con el R² individual de cada emoción
        "rmse_por_score":    {l: round(float(v), 4) for l, v in zip(SCORE_LABELS, rmse_por_score)},  # Diccionario con el RMSE individual de cada emoción
    },
}
# El diccionario "resultados" agrupa todas las métricas de los tres modelos en una estructura organizada lista para exportar


with open(RUTA_JSON, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
# Abre el archivo JSON en modo escritura ("w") y vuelca el diccionario "resultados" en él
# ensure_ascii=False permite guardar tildes y caracteres especiales correctamente
# indent=2 formatea el JSON con sangría de 2 espacios para que sea legible al abrirlo con un editor de texto
print(f"    ✓ JSON guardado en {RUTA_JSON}")  # Confirma que el archivo JSON se guardó correctamente


# ── Excel (fácil de abrir) ────────────────────────────────────────────────────
print("\n[7] Generando Excel de resultados...")  # Avisa de que empieza la generación del archivo Excel
with pd.ExcelWriter(RUTA_EXCEL, engine="openpyxl") as writer:
    # Abre un escritor de Excel en la ruta definida usando openpyxl como motor
    # El bloque "with" garantiza que el archivo se cierre y guarde correctamente al terminar


    # Hoja 1: Resumen comparativo
    df_resumen = pd.DataFrame([
        {
            "Modelo":           "Random Forest",
            "Accuracy":         round(acc_rf,   4),   # Accuracy del RF con 4 decimales
            "Precision":        round(prec_rf,  4),   # Precisión del RF con 4 decimales
            "Recall":           round(rec_rf,   4),   # Recall del RF con 4 decimales
            "F1 ponderado":     round(f1_rf,    4),   # F1 del RF con 4 decimales
            "CV media":         round(float(cv_rf.mean()),  4),  # Media de validación cruzada del RF
            "CV std":           round(float(cv_rf.std()),   4),  # Desviación de validación cruzada del RF
            "Mejores params":   str(_search_rf.best_params_),    # Los mejores parámetros encontrados convertidos a texto
        },
        {
            "Modelo":           "Red Neuronal MLP",
            "Accuracy":         round(acc_mlp,  4),   # Accuracy del MLP con 4 decimales
            "Precision":        round(prec_mlp, 4),   # Precisión del MLP con 4 decimales
            "Recall":           round(rec_mlp,  4),   # Recall del MLP con 4 decimales
            "F1 ponderado":     round(f1_mlp,   4),   # F1 del MLP con 4 decimales
            "CV media":         round(float(cv_mlp.mean()), 4),  # Media de validación cruzada del MLP
            "CV std":           round(float(cv_mlp.std()),  4),  # Desviación de validación cruzada del MLP
            "Mejores params":   str(_search_mlp.best_params_),   # Los mejores parámetros del MLP convertidos a texto
        },
        {
            "Modelo":           "MultiOutput SVR ★",
            "Accuracy":         "—",   # El SVR no tiene Accuracy porque es un modelo de regresión, no de clasificación
            "Precision":        "—",   # Igual: métrica no aplicable a este modelo
            "Recall":           "—",   # Igual: métrica no aplicable a este modelo
            "F1 ponderado":     "—",   # Igual: métrica no aplicable a este modelo
            "CV media":         "—",   # Igual: no se calculó validación cruzada de clasificación para el SVR
            "CV std":           "—",   # Igual: no aplicable
            "Mejores params":   str(_best_svr),  # Los mejores parámetros del SVR convertidos a texto
        },
    ])
    df_resumen.to_excel(writer, sheet_name="Resumen", index=False)  # Guarda la tabla anterior en la hoja "Resumen" del Excel; index=False evita que se añada una columna extra con números de fila


    # Hoja 2: F1 por clase — Random Forest
    rep_rf_rows = [
        {"Clase": c, "Precision": round(rep_rf[c]["precision"],4),
         "Recall": round(rep_rf[c]["recall"],4), "F1-score": round(rep_rf[c]["f1-score"],4),
         "Soporte": int(rep_rf[c]["support"])} for c in CLASES
    ]
    # Construye una lista de diccionarios con las 4 métricas del RF para cada emoción
    # "Soporte" es el número de productos reales de esa emoción que había en el conjunto de prueba
    pd.DataFrame(rep_rf_rows).to_excel(writer, sheet_name="RF_por_clase", index=False)  # Guarda la tabla en la hoja "RF_por_clase" del Excel


    # Hoja 3: F1 por clase — MLP
    rep_mlp_rows = [
        {"Clase": c, "Precision": round(rep_mlp[c]["precision"],4),
         "Recall": round(rep_mlp[c]["recall"],4), "F1-score": round(rep_mlp[c]["f1-score"],4),
         "Soporte": int(rep_mlp[c]["support"])} for c in CLASES
    ]
    # Exactamente igual que la hoja anterior pero con los datos del MLP en lugar del RF
    pd.DataFrame(rep_mlp_rows).to_excel(writer, sheet_name="MLP_por_clase", index=False)  # Guarda la tabla en la hoja "MLP_por_clase" del Excel


    # Hoja 4: R² y RMSE por score — SVR
    svr_rows = [
        {"Score emocional": l, "R²": round(r2, 4), "RMSE": round(rmse, 4)}
        for l, r2, rmse in zip(SCORE_LABELS, r2_por_score, rmse_por_score)
    ]
    # Construye una fila por cada emoción con su nombre, R² y RMSE del modelo SVR
    svr_rows.append({"Score emocional": "MEDIA GLOBAL",
                     "R²": round(r2_global, 4), "RMSE": round(rmse_global, 4)})
    # Añade al final una fila extra con los valores medios globales del modelo
    pd.DataFrame(svr_rows).to_excel(writer, sheet_name="SVR_por_score", index=False)  # Guarda la tabla en la hoja "SVR_por_score" del Excel


    # Hoja 5: Validación cruzada detallada
    cv_rows = [{"Fold": i+1, "Accuracy RF": round(a,4), "Accuracy MLP": round(b,4)}
               for i, (a, b) in enumerate(zip(cv_rf, cv_mlp))]
    # Construye una fila por cada uno de los 5 folds con el accuracy del RF y del MLP en ese fold
    # Permite ver fold a fold si los modelos fueron consistentes o tuvieron altibajos
    pd.DataFrame(cv_rows).to_excel(writer, sheet_name="CV_5fold", index=False)  # Guarda la tabla en la hoja "CV_5fold" del Excel


print(f"    ✓ Excel guardado en {RUTA_EXCEL}")  # Confirma que el archivo Excel con las 5 hojas se guardó correctamente


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{'='*65}
  ✓  ANÁLISIS DEL DATO COMPLETADO
{'='*65}
# Imprime una línea de 65 "=" como separador visual de inicio del resumen final


  MODELO 1 — Random Forest
    Accuracy   : {acc_rf:.4f}    # Accuracy del RF con 4 decimales
    F1 ponder. : {f1_rf:.4f}    # F1 ponderado del RF con 4 decimales
    CV 5-fold  : {cv_rf.mean():.4f} ± {cv_rf.std():.4f}  # Media y desviación del accuracy en los 5 folds del RF


  MODELO 2 — Red Neuronal MLP
    Accuracy   : {acc_mlp:.4f}   # Accuracy del MLP con 4 decimales
    F1 ponder. : {f1_mlp:.4f}   # F1 ponderado del MLP con 4 decimales
    CV 5-fold  : {cv_mlp.mean():.4f} ± {cv_mlp.std():.4f}  # Media y desviación del accuracy en los 5 folds del MLP
    Épocas     : {mlp.n_iter_}  # Número de épocas que entrenó la red hasta que el early stopping la detuvo


  MODELO 3 ★ — MultiOutput SVR (perfil emocional)
    R² medio   : {r2_global:.4f}   # R² medio global de las 8 emociones: cuanto más cerca de 1, mejor predice el modelo
    RMSE medio : {rmse_global:.4f} # Error medio global: cuanto más cerca de 0, menor es el error de predicción


  GRÁFICOS generados (12):
    G1  Distribución de emociones          # Cuántos productos hay de cada emoción
    G2  Heatmap correlación features       # Relaciones entre las variables de color
    G3  Curva de aprendizaje RF            # Cómo mejora el RF con más datos
    G4  Importancia de variables RF        # Qué variables de color son más relevantes
    G5  Matriz de confusión RF             # Dónde acierta y dónde falla el RF
    G6  Curva de pérdida MLP               # Cómo aprendió la red neuronal época a época
    G7  Matriz de confusión MLP            # Dónde acierta y dónde falla el MLP
    G8  Comparativa F1 RF vs MLP           # Comparación del F1 por emoción entre ambos modelos
    G9  Predicción vs real scores          # Dispersión entre valores reales y predichos del SVR
    G10 Perfiles emocionales (5 productos) # Los 8 scores de 5 productos reales vs predichos
    G11 R² por emoción SVR                 # Calidad de predicción del SVR por cada emoción
    G12 Resumen comparativo final          # Panel completo con todas las métricas de los tres modelos


  Carpeta gráficos : {RUTA_GRAFICOS}  # Ruta donde se guardaron los 12 gráficos PNG
  Carpeta modelos  : {RUTA_MODELOS}   # Ruta donde se guardaron los 4 archivos .pkl
  JSON resultados  : {RUTA_JSON}      # Ruta del archivo JSON con todas las métricas
  Excel resultados : {RUTA_EXCEL}     # Ruta del archivo Excel con las 5 hojas de resultados
{'='*65}
""")
# Cierra el bloque del print con la línea final de "=" y muestra todo el resumen de una vez en consola
