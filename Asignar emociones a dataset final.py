# ============================================================
# ASIGNACIÓN DE EMOCIONES POR COLOR CIELAB - TFG
# Basado en: Gilbert, Fridlund & Lucchina (2016).
# "The color of emotion: A metric for implicit color associations"
# Food Quality and Preference, 52, 203-210.
#
# El paper asignó colores CIELAB a 20 emociones concretas
# mediante un estudio con 194 participantes. Este script
# asigna a cada producto del dataset la emoción cuyo color
# de referencia es más cercano en el espacio CIELAB,
# usando distancia euclidiana (ΔE).
#
# INSTALAR: pip install pandas numpy matplotlib seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Rutas ──────────────────────────────────────────────────
RUTA_CSV    = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\dataset_limpio.csv"
RUTA_SALIDA = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\dataset_con_emociones.csv"
RUTA_GRAFICOS = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\graficos"
os.makedirs(RUTA_GRAFICOS, exist_ok=True)


# ── Tabla de referencia de emociones ───────────────────────
# Valores L*a*b* medios por emoción derivados de:
# Gilbert, Fridlund & Lucchina (2016), Fig. 1 (Mondrians)
# L*: luminosidad (0=negro, 100=blanco)
# a*: rojo(+) / verde(-) 
# b*: amarillo(+) / azul(-)
#
# Agrupación por valencia emocional (útil para marketing):
#   NEGATIVA_ALTA_ACTIVACION: angry, tense, irritated, anxious
#   POSITIVA_ALTA_ACTIVACION: energized, alert, happy, romantic
#   POSITIVA_BAJA_ACTIVACION: healthy, refreshed, relaxed, calm, soothed
#   NEGATIVA_BAJA_ACTIVACION: sad, bored, tired, sleepy
#   NEUTRAL_SENSORIAL: hungry, thirsty, sensual

EMOCIONES_REFERENCIA = {
    # Emociones negativas de alta activación
    "Angry":    {"L": 32, "a": 22, "b": 8,   "valencia": "Negativa - Alta activación",  "color_hex": "#8B0000"},
    "Tense":    {"L": 40, "a": 18, "b": 6,   "valencia": "Negativa - Alta activación",  "color_hex": "#A52A2A"},
    "Irritated":{"L": 48, "a": 14, "b": 10,  "valencia": "Negativa - Alta activación",  "color_hex": "#CD5C5C"},
    "Anxious":  {"L": 50, "a": 12, "b": 12,  "valencia": "Negativa - Alta activación",  "color_hex": "#D2691E"},

    # Emociones positivas de alta activación
    "Energized":{"L": 70, "a": 8,  "b": 30,  "valencia": "Positiva - Alta activación",  "color_hex": "#FFD700"},
    "Alert":    {"L": 78, "a": 5,  "b": 35,  "valencia": "Positiva - Alta activación",  "color_hex": "#FFFF00"},
    "Happy":    {"L": 82, "a": 4,  "b": 25,  "valencia": "Positiva - Alta activación",  "color_hex": "#FFE135"},
    "Romantic": {"L": 62, "a": 22, "b": 5,   "valencia": "Positiva - Alta activación",  "color_hex": "#FF69B4"},

    # Emociones positivas de baja activación
    "Healthy":  {"L": 65, "a": -10,"b": 15,  "valencia": "Positiva - Baja activación",  "color_hex": "#228B22"},
    "Refreshed":{"L": 74, "a": -6, "b": 3,   "valencia": "Positiva - Baja activación",  "color_hex": "#B0E0E6"},
    "Relaxed":  {"L": 70, "a": -8, "b": -5,  "valencia": "Positiva - Baja activación",  "color_hex": "#87CEEB"},
    "Calm":     {"L": 74, "a": -12,"b": -10, "valencia": "Positiva - Baja activación",  "color_hex": "#ADD8E6"},
    "Soothed":  {"L": 80, "a": -7, "b": -6,  "valencia": "Positiva - Baja activación",  "color_hex": "#B0C4DE"},

    # Emociones negativas de baja activación
    "Sad":      {"L": 35, "a": 2,  "b": -14, "valencia": "Negativa - Baja activación",  "color_hex": "#191970"},
    "Bored":    {"L": 42, "a": 0,  "b": -4,  "valencia": "Negativa - Baja activación",  "color_hex": "#696969"},
    "Tired":    {"L": 30, "a": 0,  "b": -10, "valencia": "Negativa - Baja activación",  "color_hex": "#2F4F4F"},
    "Sleepy":   {"L": 44, "a": -2, "b": -7,  "valencia": "Negativa - Baja activación",  "color_hex": "#708090"},

    # Neutral / sensorial
    "Hungry":   {"L": 55, "a": 14, "b": 22,  "valencia": "Neutral - Sensorial",         "color_hex": "#D2691E"},
    "Thirsty":  {"L": 65, "a": -5, "b": -2,  "valencia": "Neutral - Sensorial",         "color_hex": "#87CEEB"},
    "Sensual":  {"L": 52, "a": 18, "b": 3,   "valencia": "Neutral - Sensorial",         "color_hex": "#C71585"},
}

# Convertir a DataFrame para trabajar más fácil
df_ref = pd.DataFrame(EMOCIONES_REFERENCIA).T.reset_index()
df_ref.columns = ["emocion", "L_ref", "a_ref", "b_ref", "valencia", "color_hex"]
df_ref[["L_ref", "a_ref", "b_ref"]] = df_ref[["L_ref", "a_ref", "b_ref"]].astype(float)


# ── Función principal: distancia euclidiana en CIELAB ──────
def asignar_emocion(mean_L, mean_a, mean_b):
    """
    Calcula la distancia ΔE entre el color del producto
    y cada color de referencia de emoción. Devuelve la
    emoción más cercana y la distancia (ΔE).
    ΔE < 10: diferencia pequeña (colores muy similares)
    ΔE 10-25: diferencia moderada
    ΔE > 25: colores bastante diferentes
    """
    distancias = []
    for _, fila in df_ref.iterrows():
        delta_L = mean_L - fila["L_ref"]
        delta_a = mean_a - fila["a_ref"]
        delta_b = mean_b - fila["b_ref"]
        deltaE = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
        distancias.append((fila["emocion"], fila["valencia"], deltaE))

    distancias.sort(key=lambda x: x[2])
    emocion_1, valencia_1, dE_1 = distancias[0]
    emocion_2, valencia_2, dE_2 = distancias[1]
    emocion_3, valencia_3, dE_3 = distancias[2]

    return emocion_1, valencia_1, round(dE_1, 2), emocion_2, round(dE_2, 2), emocion_3, round(dE_3, 2)


# ── Pipeline ───────────────────────────────────────────────
print("=" * 55)
print("ASIGNACIÓN DE EMOCIONES POR COLOR CIELAB")
print("Fuente: Gilbert, Fridlund & Lucchina (2016)")
print("=" * 55)

print("\nCargando dataset...")
df = pd.read_csv(RUTA_CSV, encoding="utf-8-sig")
print(f"  → {len(df)} productos cargados")

print("\nCalculando emoción más cercana para cada producto...")

resultados = df.apply(
    lambda fila: asignar_emocion(fila["mean_L"], fila["mean_a"], fila["mean_b"]),
    axis=1,
    result_type="expand"
)
resultados.columns = [
    "emocion_1", "valencia_1", "deltaE_1",
    "emocion_2", "deltaE_2",
    "emocion_3", "deltaE_3"
]

df = pd.concat([df, resultados], axis=1)

# ── Estadísticas de calidad de asignación ─────────────────
print(f"\nDistancia ΔE media (emoción asignada): {df['deltaE_1'].mean():.1f}")
print(f"  ΔE < 10 (alta confianza):  {(df['deltaE_1'] < 10).sum()} productos ({(df['deltaE_1'] < 10).mean()*100:.1f}%)")
print(f"  ΔE 10-25 (confianza media): {((df['deltaE_1'] >= 10) & (df['deltaE_1'] < 25)).sum()} productos")
print(f"  ΔE > 25 (baja confianza):  {(df['deltaE_1'] >= 25).sum()} productos")

# ── Distribución de emociones asignadas ───────────────────
print("\nDistribución de emociones asignadas (top 10):")
print(df["emocion_1"].value_counts().head(10).to_string())

print("\nDistribución por valencia emocional:")
print(df["valencia_1"].value_counts().to_string())

# ── Guardar CSV ────────────────────────────────────────────
df.to_csv(RUTA_SALIDA, index=False, encoding="utf-8-sig")
print(f"\n✓ Dataset con emociones guardado en:")
print(f"  {RUTA_SALIDA}")

# ── Gráfico 1: distribución de emociones ──────────────────
plt.figure(figsize=(12, 6))
conteo = df["emocion_1"].value_counts()
colores_barras = [EMOCIONES_REFERENCIA[e]["color_hex"] for e in conteo.index]
plt.bar(conteo.index, conteo.values, color=colores_barras, edgecolor="white", linewidth=0.5)
plt.title("Distribución de emociones asignadas por color CIELAB\n(Gilbert et al., 2016)",
          fontsize=14, fontweight="bold")
plt.xlabel("Emoción")
plt.ylabel("Número de productos")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "distribucion_emociones.png"), dpi=150)
plt.close()
print("\nGráfico guardado: distribucion_emociones.png")

# ── Gráfico 2: distribución por valencia ──────────────────
plt.figure(figsize=(9, 5))
colores_valencia = {
    "Positiva - Alta activación": "#FFD700",
    "Positiva - Baja activación": "#87CEEB",
    "Negativa - Alta activación": "#8B0000",
    "Negativa - Baja activación": "#696969",
    "Neutral - Sensorial":        "#D2691E",
}
conteo_val = df["valencia_1"].value_counts()
plt.bar(
    conteo_val.index,
    conteo_val.values,
    color=[colores_valencia.get(v, "#AAAAAA") for v in conteo_val.index],
    edgecolor="white"
)
plt.title("Distribución por valencia emocional", fontsize=14, fontweight="bold")
plt.xlabel("Valencia")
plt.ylabel("Número de productos")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "distribucion_valencia.png"), dpi=150)
plt.close()
print("Gráfico guardado: distribucion_valencia.png")

# ── Gráfico 3: emoción por categoría de producto ──────────
if "categoria" in df.columns:
    top_cats = df["categoria"].value_counts().head(10).index
    df_top = df[df["categoria"].isin(top_cats)]

    pivot = pd.crosstab(df_top["categoria"], df_top["emocion_1"])
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    pivot_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="none")
    ax.set_title("Distribución de emociones por categoría de producto (%)\n(Top 10 categorías)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("% de productos")
    ax.legend(title="Emoción", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, "emociones_por_categoria.png"), dpi=150)
    plt.close()
    print("Gráfico guardado: emociones_por_categoria.png")

# ── Gráfico 4: emociones por fuente ───────────────────────
if "fuente" in df.columns:
    pivot_f = pd.crosstab(df["fuente"], df["emocion_1"])
    pivot_f_pct = pivot_f.div(pivot_f.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_f_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="none")
    ax.set_title("Distribución de emociones por fuente de datos (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fuente")
    ax.set_ylabel("% de productos")
    ax.legend(title="Emoción", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, "emociones_por_fuente.png"), dpi=150)
    plt.close()
    print("Gráfico guardado: emociones_por_fuente.png")

print("""
CITA PARA EL TFG:
  Gilbert, A.N., Fridlund, A.J., & Lucchina, L.A. (2016).
  The color of emotion: A metric for implicit color associations.
  Food Quality and Preference, 52, 203-210.
  https://doi.org/10.1016/j.foodqual.2016.04.007
""")
print("=" * 55)
print("PROCESO COMPLETADO")
print("=" * 55)