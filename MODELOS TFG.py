# ==============================================================
# TFG - ADRIÁN JULVE NAVARRO
# SCRIPT 3: ANÁLISIS DEL DATO
# Universidad Francisco de Vitoria — Business Analytics 2025-26
#
# Este script construye y compara tres modelos predictivos para
# clasificar la emoción que transmite el color de un producto:
#
#   Modelo A — Random Forest      (línea base interpretable)
#   Modelo B — XGBoost            (boosting intermedio)
#   Modelo C — Red Neuronal MLP   (modelo avanzado)
#
# Estructura del script:
#   1. Carga y preparación del dataset
#   2. División train/test
#   3. Modelo A: Random Forest
#   4. Modelo B: XGBoost
#   5. Modelo C: Red Neuronal MLP
#   6. Comparativa final de los tres modelos
#   7. Guardado de resultados
# ==============================================================


# --------------------------------------------------------------
# LIBRERÍAS
# Importamos todas las herramientas que vamos a necesitar.
# Si alguna falla al importar, ejecuta en tu terminal:
#   pip install scikit-learn xgboost matplotlib seaborn pandas
# --------------------------------------------------------------

import sys, subprocess

_PAQUETES = {"pandas": "pandas", "numpy": "numpy", "matplotlib": "matplotlib",
             "seaborn": "seaborn", "sklearn": "scikit-learn", "xgboost": "xgboost"}
_faltantes = []
for _mod, _pkg in _PAQUETES.items():
    try:
        __import__(_mod)
    except ImportError:
        _faltantes.append(_pkg)
if _faltantes:
    print(f"[INFO] Instalando: {', '.join(_faltantes)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + _faltantes)
    print("[INFO] Hecho. Continúa...\n")

import pandas as pd                        # Para trabajar con tablas de datos (DataFrames)
import numpy as np                         # Para operaciones matemáticas con arrays
import matplotlib.pyplot as plt            # Para crear gráficos
import matplotlib                          # Para configurar el backend de matplotlib
matplotlib.use("Agg")                      # Usamos "Agg" para guardar gráficos sin abrir ventanas
import seaborn as sns                      # Para gráficos más bonitos (matrices de confusión, etc.)
import os                                  # Para manejar rutas y carpetas del sistema operativo
import warnings                            # Para suprimir avisos innecesarios
warnings.filterwarnings("ignore")          # Silenciamos todos los warnings para que la consola quede limpia

# Herramientas de scikit-learn para dividir datos y evaluar modelos
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,          # Porcentaje de predicciones correctas
    f1_score,                # Media armónica entre precisión y recall (ideal para clases desbalanceadas)
    classification_report,   # Informe completo por clase (precisión, recall, f1 por emoción)
    confusion_matrix,        # Matriz que muestra errores por clase
    roc_auc_score            # Área bajo la curva ROC (capacidad de discriminación del modelo)
)
from sklearn.preprocessing import LabelEncoder  # Convierte etiquetas de texto a números (necesario para los modelos)

# Modelo A: Random Forest
from sklearn.ensemble import RandomForestClassifier

# Modelo B: XGBoost
from xgboost import XGBClassifier

# Modelo C: Red Neuronal MLP (Multi-Layer Perceptron)
from sklearn.neural_network import MLPClassifier


# ==============================================================
# 0. CONFIGURACIÓN DE RUTAS
# Cambia RUTA_CSV y RUTA_GRAFICOS a las rutas de tu ordenador.
# ==============================================================

RUTA_CSV      = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\Dataset_con_emociones.csv"
RUTA_GRAFICOS = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\graficos\analisis_dato"
RUTA_RESULTADOS = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING"

# Creamos la carpeta de gráficos si no existe
os.makedirs(RUTA_GRAFICOS, exist_ok=True)


# ==============================================================
# 1. CARGA Y PREPARACIÓN DEL DATASET
# ==============================================================

print("=" * 60)
print("SCRIPT 3 — ANÁLISIS DEL DATO")
print("=" * 60)

# Cargamos el CSV final generado por el Script 2
df = pd.read_csv(RUTA_CSV, encoding="utf-8-sig")
print(f"\nDataset cargado: {len(df)} filas, {len(df.columns)} columnas")

# -------------------------------------------------------
# VARIABLES DE ENTRADA (features)
# Usamos las 10 variables normalizadas (_norm) más los
# 8 scores emocionales continuos generados por el sistema
# gaussiano. Esto da al modelo 18 features de entrada.
#
# Las variables _norm están en escala [0,1] (Min-Max).
# Los scores son probabilidades [0,1] por emoción.
# -------------------------------------------------------
FEATURES = [
    # Variables de color normalizadas
    "mean_R_norm", "mean_G_norm", "mean_B_norm",
    "mean_L_norm", "mean_a_norm", "mean_b_norm",
    "contrast_L_norm", "hue_norm", "saturation_norm", "value_norm",
    # Scores gaussianos por emoción (capturan ambigüedad emocional)
    "score_ira", "score_tristeza", "score_romanticismo",
    "score_energia", "score_alegria", "score_relajacion",
    "score_calma", "score_aburrimiento",
]

# Variable objetivo: la emoción asignada
TARGET = "emocion"

# Comprobamos que todas las columnas necesarias existen
columnas_faltantes = [c for c in FEATURES + [TARGET] if c not in df.columns]
if columnas_faltantes:
    print(f"\nATENCIÓN: Faltan estas columnas en el CSV: {columnas_faltantes}")
    print("Comprueba que el CSV fue generado con el sistema gaussiano (Script 2 v2).")
    # Si faltan los scores, trabajamos solo con las 10 variables _norm
    FEATURES = [f for f in FEATURES if f in df.columns]
    print(f"Continuando con {len(FEATURES)} features disponibles.")

# Eliminamos filas donde falte la emoción o alguna feature
df = df.dropna(subset=[TARGET] + FEATURES).reset_index(drop=True)
print(f"Filas válidas para modelado: {len(df)}")

# Mostramos la distribución de la variable objetivo
print("\nDistribución de emociones (variable objetivo):")
print(df[TARGET].value_counts().to_string())

# -------------------------------------------------------
# CODIFICACIÓN DE LA VARIABLE OBJETIVO
# Los modelos de ML necesitan números, no texto.
# LabelEncoder convierte "Alegría" → 0, "Calma" → 1, etc.
# Guardamos el encoder para poder revertir la conversión
# después y mostrar los nombres reales en los gráficos.
# -------------------------------------------------------
le = LabelEncoder()                          # Creamos el codificador
le.fit(df[TARGET])                           # Lo entrenamos con todas las clases posibles
df["emocion_cod"] = le.transform(df[TARGET]) # Creamos columna numérica

print(f"\nClases codificadas: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Preparamos las matrices X (features) e y (objetivo)
X = df[FEATURES].values   # Matriz de variables de entrada: shape (n_productos, n_features)
y = df["emocion_cod"].values  # Vector de etiquetas numéricas


# ==============================================================
# 2. DIVISIÓN TRAIN / TEST
#
# Dividimos el dataset en dos partes:
#   - Train (80%): el modelo aprende con estos datos
#   - Test  (20%): evaluamos el modelo con datos que nunca vio
#
# stratify=y asegura que la proporción de cada emoción
# sea la misma en train y en test (importante con clases
# desbalanceadas). random_state=42 fija la semilla para
# que los resultados sean reproducibles.
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      # 20% para test
    random_state=42,     # Semilla fija: siempre la misma división
    stratify=y           # Mantiene proporciones de cada clase
)

print(f"\nDivisión train/test:")
print(f"  Train: {len(X_train)} productos ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test)} productos  ({len(X_test)/len(X)*100:.1f}%)")


# ==============================================================
# FUNCIÓN AUXILIAR: guardar_metricas_graficos
# Esta función la llamamos para cada modelo y genera
# automáticamente todos los gráficos exigidos por la rúbrica:
#   - Matriz de confusión
#   - Importancia de variables (si el modelo la proporciona)
#   - Curva de aprendizaje
# ==============================================================

def calcular_metricas(nombre, modelo, X_tr, y_tr, X_te, y_te, clases):
    """
    Entrena el modelo, calcula las métricas y devuelve un resumen.

    Parámetros:
        nombre   : nombre del modelo (para los gráficos)
        modelo   : objeto del clasificador ya configurado
        X_tr, y_tr : datos de entrenamiento
        X_te, y_te : datos de test
        clases     : lista de nombres de las clases (emociones)

    Devuelve:
        dict con todas las métricas calculadas
    """
    print(f"\n{'='*60}")
    print(f"  {nombre}")
    print(f"{'='*60}")

    # --- ENTRENAMIENTO ---
    # El modelo aprende los patrones de los datos de entrenamiento.
    # .fit() es la función de aprendizaje de todos los modelos sklearn.
    print(f"  Entrenando...")
    modelo.fit(X_tr, y_tr)
    print(f"  Entrenamiento completado.")

    # --- PREDICCIÓN ---
    # Aplicamos el modelo entrenado a los datos de test.
    # y_pred son las predicciones del modelo (números).
    y_pred = modelo.predict(X_te)

    # --- PROBABILIDADES ---
    # predict_proba devuelve la probabilidad de cada clase.
    # La necesitamos para calcular el AUC-ROC.
    y_proba = modelo.predict_proba(X_te)

    # --- MÉTRICAS ---

    # 1. Accuracy: porcentaje de predicciones correctas
    acc = accuracy_score(y_te, y_pred)

    # 2. F1-Score macro: media del F1 de cada clase con igual peso
    # Usamos 'macro' porque nuestras clases pueden estar desbalanceadas
    # y no queremos que las clases mayoritarias dominen la métrica.
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

    # 3. AUC-ROC multiclase (One-vs-Rest)
    # Mide la capacidad del modelo para distinguir cada emoción
    # del resto. Un valor de 1.0 es perfecto, 0.5 es aleatorio.
    try:
        auc = roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")  # Si falla por clases sin ejemplos en test

    # 4. Validación cruzada (5-fold) sobre train
    # Divide el train en 5 partes, entrena 5 veces y promedia.
    # Da una estimación más robusta del rendimiento real.
    cv_scores = cross_val_score(modelo, X_tr, y_tr, cv=5, scoring="accuracy")
    cv_media = cv_scores.mean()
    cv_std   = cv_scores.std()

    print(f"\n  RESULTADOS:")
    print(f"    Accuracy  (test)  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    F1-Score  (macro) : {f1:.4f}")
    print(f"    AUC-ROC   (macro) : {auc:.4f}")
    print(f"    CV Accuracy (5-fold): {cv_media:.4f} ± {cv_std:.4f}")

    # Informe detallado por clase
    print(f"\n  INFORME POR EMOCIÓN:")
    print(classification_report(y_te, y_pred,
                                 target_names=clases,
                                 zero_division=0))

    return {
        "nombre":   nombre,
        "modelo":   modelo,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "accuracy": acc,
        "f1_macro": f1,
        "auc_roc":  auc,
        "cv_media": cv_media,
        "cv_std":   cv_std,
    }


def grafico_confusion(nombre, y_te, y_pred, clases, carpeta):
    """
    Genera y guarda la matriz de confusión como heatmap.
    Las filas son la emoción real, las columnas la predicha.
    Los colores más oscuros indican más productos en esa celda.
    La diagonal principal son los aciertos.
    """
    cm = confusion_matrix(y_te, y_pred)

    # Normalizamos por filas: cada celda muestra % sobre el total real
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel izquierdo: valores absolutos (número de productos)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title(f"{nombre}\nMatriz de Confusión — Valores Absolutos", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Emoción Predicha")
    axes[0].set_ylabel("Emoción Real")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].tick_params(axis="y", rotation=0)

    # Panel derecho: porcentajes normalizados por fila
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=clases, yticklabels=clases,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"{nombre}\nMatriz de Confusión — Proporción por Clase", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Emoción Predicha")
    axes[1].set_ylabel("Emoción Real")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    nombre_archivo = nombre.lower().replace(" ", "_").replace("/", "_")
    ruta = os.path.join(carpeta, f"confusion_{nombre_archivo}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {ruta}")


def grafico_importancia(nombre, modelo, feature_names, carpeta, top_n=15):
    """
    Genera el gráfico de importancia de variables (Feature Importance).
    Solo disponible para Random Forest y XGBoost.
    Muestra qué variables tienen más peso en las predicciones.
    """
    if not hasattr(modelo, "feature_importances_"):
        return  # La red neuronal no tiene feature_importances_

    importancias = modelo.feature_importances_   # Array con la importancia de cada feature
    indices = np.argsort(importancias)[::-1]      # Ordenamos de mayor a menor
    top_idx = indices[:top_n]                     # Nos quedamos con las top_n más importantes

    fig, ax = plt.subplots(figsize=(10, 6))
    colores = ["#2E75B6" if i < 3 else "#5B9BD5" if i < 6 else "#BDD7EE" for i in range(top_n)]
    ax.barh(
        [feature_names[i] for i in top_idx[::-1]],   # Nombres en el eje Y (invertidos para que el mayor quede arriba)
        importancias[top_idx[::-1]],                  # Valores en el eje X
        color=colores[::-1], edgecolor="white"
    )
    ax.set_title(f"{nombre}\nImportancia de Variables (Top {top_n})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importancia relativa")
    ax.axvline(importancias[top_idx].mean(), color="red", linestyle="--",
               alpha=0.6, label="Media")
    ax.legend()
    plt.tight_layout()
    nombre_archivo = nombre.lower().replace(" ", "_").replace("/", "_")
    ruta = os.path.join(carpeta, f"importancia_{nombre_archivo}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {ruta}")


# ==============================================================
# 3. MODELO A — RANDOM FOREST
#
# MARCO TEÓRICO:
# Random Forest es un método de ensemble que construye múltiples
# árboles de decisión independientes sobre subconjuntos aleatorios
# del dataset (bagging) y de las variables (feature sampling).
# La predicción final es la clase más votada entre todos los árboles.
#
# Justificación de uso:
#   - Robusto frente al overfitting gracias al bagging
#   - No requiere escalado de variables (trabaja con rangos)
#   - Proporciona feature importance directamente
#   - Fácil de interpretar y defender académicamente
#   - Breiman (2001) demostró su superioridad frente a árboles simples
#
# Hiperparámetros elegidos:
#   n_estimators=300 : 300 árboles. Más árboles → más estable pero
#                      más lento. 300 es un equilibrio probado.
#   max_depth=15     : Profundidad máxima de cada árbol. Limita el
#                      overfitting sin sacrificar capacidad.
#   min_samples_leaf=3: Cada hoja necesita al menos 3 muestras.
#                      Evita hojas con un solo producto.
#   class_weight='balanced': Da más peso a clases con pocos productos.
#                      Fundamental para datasets desbalanceados.
# ==============================================================

print("\n\n" + "=" * 60)
print("MODELO A — RANDOM FOREST")
print("=" * 60)

rf_modelo = RandomForestClassifier(
    n_estimators=300,           # 300 árboles de decisión
    max_depth=15,               # Profundidad máxima por árbol
    min_samples_leaf=3,         # Mínimo 3 muestras por hoja
    class_weight="balanced",    # Compensa el desbalance de clases
    random_state=42,            # Semilla para reproducibilidad
    n_jobs=-1                   # Usa todos los núcleos del procesador
)

# Entrenamos y evaluamos el modelo
res_rf = calcular_metricas(
    nombre="Random Forest",
    modelo=rf_modelo,
    X_tr=X_train, y_tr=y_train,
    X_te=X_test,  y_te=y_test,
    clases=le.classes_
)

# Generamos los gráficos
grafico_confusion("Random Forest", y_test, res_rf["y_pred"], le.classes_, RUTA_GRAFICOS)
grafico_importancia("Random Forest", rf_modelo, FEATURES, RUTA_GRAFICOS)


# ==============================================================
# 4. MODELO B — XGBOOST
#
# MARCO TEÓRICO:
# XGBoost (eXtreme Gradient Boosting) es un algoritmo de boosting
# que construye árboles de forma secuencial: cada árbol nuevo
# aprende a corregir los errores del anterior. Usa el gradiente
# del error para guiar la construcción de cada árbol.
#
# A diferencia de Random Forest (bagging en paralelo),
# XGBoost construye los árboles en serie, lo que lo hace más
# preciso pero también más sensible a los hiperparámetros.
#
# Justificación de uso:
#   - Estado del arte en competiciones de ML con datos tabulares
#   - Regularización L1 y L2 incorporada (evita overfitting)
#   - Chen & Guestrin (2016) demostraron su eficiencia superior
#   - Complementa al RF: donde uno falla, el otro suele acertar
#
# Hiperparámetros elegidos:
#   n_estimators=400  : 400 rondas de boosting
#   max_depth=6       : Árboles más pequeños que en RF (boosting
#                       funciona mejor con árboles poco profundos)
#   learning_rate=0.05: Paso de aprendizaje conservador.
#                       Más pequeño → más robusto, más lento.
#   subsample=0.8     : Cada árbol usa el 80% aleatorio de filas
#   colsample_bytree=0.8: Cada árbol usa el 80% aleatorio de features
# ==============================================================

print("\n\n" + "=" * 60)
print("MODELO B — XGBOOST")
print("=" * 60)

n_clases = len(le.classes_)   # Número de emociones distintas

xgb_modelo = XGBClassifier(
    n_estimators=400,           # 400 árboles secuenciales
    max_depth=6,                # Profundidad pequeña (ideal para boosting)
    learning_rate=0.05,         # Paso de aprendizaje (eta)
    subsample=0.8,              # 80% de filas por árbol
    colsample_bytree=0.8,       # 80% de features por árbol
    objective="multi:softprob", # Clasificación multiclase con probabilidades
    num_class=n_clases,         # Número de clases
    eval_metric="mlogloss",     # Métrica de evaluación interna: log loss multiclase
    use_label_encoder=False,    # Desactivamos el encoder interno (usamos el nuestro)
    random_state=42,
    n_jobs=-1
)

res_xgb = calcular_metricas(
    nombre="XGBoost",
    modelo=xgb_modelo,
    X_tr=X_train, y_tr=y_train,
    X_te=X_test,  y_te=y_test,
    clases=le.classes_
)

grafico_confusion("XGBoost", y_test, res_xgb["y_pred"], le.classes_, RUTA_GRAFICOS)
grafico_importancia("XGBoost", xgb_modelo, FEATURES, RUTA_GRAFICOS)


# ==============================================================
# 5. MODELO C — RED NEURONAL MLP
#
# MARCO TEÓRICO:
# El Perceptrón Multicapa (MLP) es una red neuronal artificial
# con capas de neuronas conectadas entre sí. Cada neurona aplica
# una función de activación no lineal a la suma ponderada de sus
# entradas. El aprendizaje ocurre mediante backpropagation:
# el error se propaga hacia atrás y los pesos se ajustan usando
# descenso de gradiente estocástico (Adam optimizer).
#
# Arquitectura elegida: (256, 128, 64)
#   - Capa de entrada: 18 neuronas (una por feature)
#   - Capa oculta 1:  256 neuronas con activación ReLU
#   - Capa oculta 2:  128 neuronas con activación ReLU
#   - Capa oculta 3:   64 neuronas con activación ReLU
#   - Capa de salida:   9 neuronas (una por emoción) con softmax
#
# Justificación de uso:
#   - Captura relaciones no lineales entre variables de color
#   - La función ReLU (max(0,x)) evita el problema del gradiente
#     desvanecido que afectaba a las redes antiguas (LeCun, 1989)
#   - Tres capas es suficiente para aproximar cualquier función
#     continua (teorema de aproximación universal)
#
# Hiperparámetros:
#   hidden_layer_sizes=(256,128,64): arquitectura piramidal
#   activation='relu'  : función de activación más eficiente
#   solver='adam'      : optimizador adaptativo (Kingma, 2014)
#   alpha=0.001        : regularización L2 para evitar overfitting
#   max_iter=500       : máximo de épocas de entrenamiento
#   early_stopping=True: para el entrenamiento si deja de mejorar
# ==============================================================

print("\n\n" + "=" * 60)
print("MODELO C — RED NEURONAL MLP")
print("=" * 60)

mlp_modelo = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # Tres capas ocultas en pirámide
    activation="relu",                  # Rectified Linear Unit: f(x) = max(0, x)
    solver="adam",                      # Optimizador Adam (adaptativo, eficiente)
    alpha=0.001,                        # Regularización L2: penaliza pesos grandes
    learning_rate_init=0.001,           # Tasa de aprendizaje inicial
    max_iter=500,                       # Máximo 500 épocas de entrenamiento
    early_stopping=True,               # Para si la validación no mejora en 20 épocas
    validation_fraction=0.1,            # 10% del train como validación interna
    n_iter_no_change=20,               # Paciencia: espera 20 épocas sin mejora
    random_state=42,
    verbose=False                       # No imprime progreso época por época
)

res_mlp = calcular_metricas(
    nombre="Red Neuronal MLP",
    modelo=mlp_modelo,
    X_tr=X_train, y_tr=y_train,
    X_te=X_test,  y_te=y_test,
    clases=le.classes_
)

grafico_confusion("Red Neuronal MLP", y_test, res_mlp["y_pred"], le.classes_, RUTA_GRAFICOS)
# La red neuronal no tiene feature_importances_, así que no llamamos a grafico_importancia


# ==============================================================
# 6. COMPARATIVA FINAL DE LOS TRES MODELOS
#
# Generamos todos los gráficos comparativos que exige la rúbrica:
#   - Tabla comparativa de métricas
#   - Gráfico de barras: Accuracy / F1 / AUC-ROC
#   - Curva de aprendizaje (Learning Curve) del mejor modelo
# ==============================================================

print("\n\n" + "=" * 60)
print("COMPARATIVA FINAL — LOS TRES MODELOS")
print("=" * 60)

# Construimos el DataFrame comparativo
resultados = [res_rf, res_xgb, res_mlp]

tabla_comparativa = pd.DataFrame([
    {
        "Modelo":    r["nombre"],
        "Accuracy":  round(r["accuracy"], 4),
        "F1-Macro":  round(r["f1_macro"], 4),
        "AUC-ROC":   round(r["auc_roc"], 4),
        "CV Acc. ±σ": f"{r['cv_media']:.4f} ± {r['cv_std']:.4f}",
    }
    for r in resultados
])

print("\nTABLA COMPARATIVA:")
print(tabla_comparativa.to_string(index=False))

# Guardamos la tabla como CSV
tabla_comparativa.to_csv(
    os.path.join(RUTA_RESULTADOS, "comparativa_modelos.csv"),
    index=False, encoding="utf-8-sig"
)
print("\n✓ Tabla guardada en comparativa_modelos.csv")


# --- GRÁFICO COMPARATIVO DE LAS 3 MÉTRICAS ---
# Un gráfico de barras agrupadas mostrando Accuracy, F1 y AUC-ROC
# para los tres modelos. Permite ver de un vistazo cuál gana en qué.

nombres    = [r["nombre"] for r in resultados]
accuracies = [r["accuracy"] for r in resultados]
f1s        = [r["f1_macro"] for r in resultados]
aucs       = [r["auc_roc"]  for r in resultados]

x = np.arange(len(nombres))   # Posiciones en el eje X
ancho = 0.25                   # Ancho de cada barra

fig, ax = plt.subplots(figsize=(12, 6))

# Tres grupos de barras: una por métrica
barras_acc = ax.bar(x - ancho, accuracies, ancho, label="Accuracy",  color="#2E75B6", edgecolor="white")
barras_f1  = ax.bar(x,         f1s,        ancho, label="F1-Macro",  color="#ED7D31", edgecolor="white")
barras_auc = ax.bar(x + ancho, aucs,       ancho, label="AUC-ROC",   color="#70AD47", edgecolor="white")

# Añadimos el valor numérico encima de cada barra
for barra in list(barras_acc) + list(barras_f1) + list(barras_auc):
    altura = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,  # Posición X centrada
        altura + 0.005,                           # Justo encima de la barra
        f"{altura:.3f}",                          # Valor con 3 decimales
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )

ax.set_ylim(0, 1.12)                              # Eje Y de 0 a 1.12 para que quepan los textos
ax.set_xticks(x)
ax.set_xticklabels(nombres, fontsize=11)
ax.set_ylabel("Puntuación (0-1)", fontsize=11)
ax.set_title("Comparativa de Modelos — Accuracy, F1-Macro y AUC-ROC", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="Línea base (aleatorio)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "comparativa_metricas.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfico: comparativa_metricas.png")


# --- GRÁFICO: VALIDACIÓN CRUZADA (CV) ---
# Muestra la distribución de las 5 puntuaciones de CV para cada modelo.
# Un boxplot estrecho indica un modelo estable; uno ancho indica varianza alta.

fig, ax = plt.subplots(figsize=(10, 5))
cv_data  = []
cv_labels = []
for r in resultados:
    # Recalculamos los 5 scores de CV para el boxplot
    scores = cross_val_score(r["modelo"], X_train, y_train, cv=5, scoring="accuracy")
    cv_data.append(scores)
    cv_labels.append(r["nombre"])

bp = ax.boxplot(cv_data, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=2))
colores_box = ["#2E75B6", "#ED7D31", "#70AD47"]
for patch, color in zip(bp["boxes"], colores_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticklabels(cv_labels, fontsize=11)
ax.set_ylabel("Accuracy (5-fold CV)", fontsize=11)
ax.set_title("Estabilidad de los Modelos — Validación Cruzada 5-Fold", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "comparativa_cv.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfico: comparativa_cv.png")


# --- GRÁFICO: DISTRIBUCIÓN DE ERRORES POR EMOCIÓN ---
# Para cada modelo, mostramos el F1-Score por emoción individual.
# Esto revela qué emociones son más difíciles de predecir.

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
colores_modelos = ["#2E75B6", "#ED7D31", "#70AD47"]

for ax, r, color in zip(axes, resultados, colores_modelos):
    # Calculamos el F1 por clase
    f1_por_clase = f1_score(y_test, r["y_pred"],
                             labels=list(range(len(le.classes_))),
                             average=None, zero_division=0)
    indices_orden = np.argsort(f1_por_clase)  # Ordenamos de menor a mayor

    ax.barh(
        [le.classes_[i] for i in indices_orden],
        f1_por_clase[indices_orden],
        color=color, alpha=0.8, edgecolor="white"
    )
    ax.set_xlim(0, 1.05)
    ax.set_title(f"{r['nombre']}\nF1 por Emoción", fontsize=11, fontweight="bold")
    ax.set_xlabel("F1-Score")
    ax.axvline(f1_por_clase.mean(), color="black", linestyle="--",
               alpha=0.5, linewidth=1, label=f"Media: {f1_por_clase.mean():.2f}")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("F1-Score por Emoción — Comparativa de Modelos", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "f1_por_emocion.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfico: f1_por_emocion.png")


# --- IDENTIFICAR EL MEJOR MODELO ---
# El mejor modelo es el que tiene mayor F1-Macro en test,
# que es la métrica más justa para datasets desbalanceados.

mejor = max(resultados, key=lambda r: r["f1_macro"])
print(f"\n{'='*60}")
print(f"MEJOR MODELO: {mejor['nombre']}")
print(f"  Accuracy : {mejor['accuracy']:.4f} ({mejor['accuracy']*100:.2f}%)")
print(f"  F1-Macro : {mejor['f1_macro']:.4f}")
print(f"  AUC-ROC  : {mejor['auc_roc']:.4f}")
print(f"{'='*60}")


# ==============================================================
# 7. CURVA DE APRENDIZAJE DEL MEJOR MODELO
#
# La curva de aprendizaje muestra cómo evoluciona el rendimiento
# del modelo a medida que aumenta el tamaño del conjunto de
# entrenamiento. Permite diagnosticar:
#   - Underfitting: ambas curvas bajas (el modelo no aprende)
#   - Overfitting:  train alta, test baja (el modelo memoriza)
#   - Buen ajuste:  ambas curvas altas y convergentes
# ==============================================================

from sklearn.model_selection import learning_curve

print(f"\nGenerando curva de aprendizaje para: {mejor['nombre']}...")

# Calculamos scores para diferentes tamaños de train
# train_sizes=[0.1, 0.2, ..., 1.0]: desde el 10% hasta el 100% del train
train_sizes_abs, train_scores, test_scores = learning_curve(
    mejor["modelo"],
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 puntos entre 10% y 100%
    cv=5,                                     # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1
)

# Calculamos media y desviación típica para el área de confianza
train_media = train_scores.mean(axis=1)
train_std   = train_scores.std(axis=1)
test_media  = test_scores.mean(axis=1)
test_std    = test_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

# Curva de entrenamiento (el modelo ve estos datos)
ax.plot(train_sizes_abs, train_media, "o-", color="#2E75B6",
        label="Accuracy en Entrenamiento", linewidth=2)
ax.fill_between(train_sizes_abs,
                train_media - train_std,
                train_media + train_std,
                alpha=0.15, color="#2E75B6")  # Banda de confianza ±1σ

# Curva de validación (datos que el modelo no vio)
ax.plot(train_sizes_abs, test_media, "o-", color="#ED7D31",
        label="Accuracy en Validación (CV)", linewidth=2)
ax.fill_between(train_sizes_abs,
                test_media - test_std,
                test_media + test_std,
                alpha=0.15, color="#ED7D31")

ax.set_xlabel("Número de productos en entrenamiento", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title(f"Curva de Aprendizaje — {mejor['nombre']}", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "curva_aprendizaje_mejor_modelo.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfico: curva_aprendizaje_mejor_modelo.png")


# ==============================================================
# RESUMEN FINAL
# ==============================================================

print("\n" + "=" * 60)
print("SCRIPT 3 COMPLETADO")
print("=" * 60)
print(f"\nGráficos generados en: {RUTA_GRAFICOS}")
print("\nLista de gráficos:")
print("  · confusion_random_forest.png       — Matriz de confusión RF")
print("  · confusion_xgboost.png             — Matriz de confusión XGBoost")
print("  · confusion_red_neuronal_mlp.png    — Matriz de confusión MLP")
print("  · importancia_random_forest.png     — Feature importance RF")
print("  · importancia_xgboost.png           — Feature importance XGBoost")
print("  · comparativa_metricas.png          — Accuracy / F1 / AUC comparados")
print("  · comparativa_cv.png                — Estabilidad por CV 5-fold")
print("  · f1_por_emocion.png                — F1 por emoción en los 3 modelos")
print("  · curva_aprendizaje_mejor_modelo.png— Curva de aprendizaje del ganador")
print(f"\nMejor modelo: {mejor['nombre']}")
print(f"  Accuracy: {mejor['accuracy']*100:.2f}%")
print(f"  F1-Macro: {mejor['f1_macro']:.4f}")
print(f"  AUC-ROC:  {mejor['auc_roc']:.4f}")
print("\n✓ Listo para el Análisis de Negocio (Pilar 3).")
input("\nPulsa Enter para cerrar...")