# ============================================================
# INGENIERÍA DEL DATO - TFG
# Autor: Adrián Julve Navarro
#
# Este script realiza el análisis exploratorio y limpieza
# del dataset combinado de imágenes de productos.
#
# Para ejecutarlo: abre la terminal en VSCode y escribe:
#   python ingenieria_dato.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── 1. CARGAR EL DATASET ───────────────────────────────────
# Cambia esta ruta a donde tengas el CSV combinado
RUTA_CSV = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\dataset_final_combinado.csv"
RUTA_SALIDA = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\graficos"

# Crear carpeta de gráficos si no existe
os.makedirs(RUTA_SALIDA, exist_ok=True)

print("Cargando el dataset...")
df = pd.read_csv(RUTA_CSV, encoding="utf-8-sig")
print(f"Dataset cargado correctamente.")


# ============================================================
# PASO 1: VISTA GENERAL DEL DATASET
# ============================================================
print("\n" + "="*55)
print("PASO 1: VISTA GENERAL")
print("="*55)

print(f"\nNúmero de filas (productos):  {len(df)}")
print(f"Número de columnas:           {len(df.columns)}")
print(f"\nColumnas disponibles:")
for col in df.columns:
    print(f"  - {col} ({df[col].dtype})")

print(f"\nPrimeras 5 filas del dataset:")
print(df.head())


# ============================================================
# PASO 2: DISTRIBUCIÓN POR FUENTE Y CATEGORÍA
# ============================================================
print("\n" + "="*55)
print("PASO 2: DISTRIBUCIÓN POR FUENTE Y CATEGORÍA")
print("="*55)

# Por fuente de datos
print("\nProductos por fuente de datos:")
print(df["fuente"].value_counts().to_string())

# Por categoría (top 15)
print("\nTop 15 categorías con más productos:")
print(df["categoria"].value_counts().head(15).to_string())

# Gráfico de barras por fuente
plt.figure(figsize=(8, 5))
df["fuente"].value_counts().plot(kind="bar", color=["#2E75B6", "#1F4E79", "#70AD47"])
plt.title("Número de productos por fuente de datos", fontsize=14, fontweight="bold")
plt.xlabel("Fuente")
plt.ylabel("Número de productos")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, "distribucion_fuente.png"), dpi=150)
plt.close()
print("\nGráfico guardado: distribucion_fuente.png")

# Gráfico de barras por categoría (top 15)
plt.figure(figsize=(12, 6))
df["categoria"].value_counts().head(15).plot(kind="barh", color="#2E75B6")
plt.title("Top 15 categorías con más productos", fontsize=14, fontweight="bold")
plt.xlabel("Número de productos")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, "distribucion_categoria.png"), dpi=150)
plt.close()
print("Gráfico guardado: distribucion_categoria.png")


# ============================================================
# PASO 3: DETECCIÓN Y TRATAMIENTO DE VALORES NULOS
# ============================================================
print("\n" + "="*55)
print("PASO 3: VALORES NULOS")
print("="*55)

nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)

resumen_nulos = pd.DataFrame({
    "Nulos": nulos,
    "% sobre total": pct_nulos
})
print("\nValores nulos por columna:")
print(resumen_nulos.to_string())

# Eliminamos filas con nulos en variables de color (son imprescindibles)
variables_color = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]
filas_antes = len(df)
df = df.dropna(subset=variables_color)
filas_despues = len(df)
print(f"\nFilas eliminadas por nulos en variables de color: {filas_antes - filas_despues}")
print(f"Filas restantes: {filas_despues}")


# ============================================================
# PASO 4: DETECCIÓN Y ELIMINACIÓN DE DUPLICADOS
# ============================================================
print("\n" + "="*55)
print("PASO 4: DUPLICADOS")
print("="*55)

duplicados = df.duplicated(subset=["imagen_url"]).sum()
print(f"\nFilas duplicadas (misma URL de imagen): {duplicados}")

df = df.drop_duplicates(subset=["imagen_url"])
df = df.reset_index(drop=True)
print(f"Filas tras eliminar duplicados: {len(df)}")


# ============================================================
# PASO 5: VALIDACIÓN DE RANGOS DE COLOR
# ============================================================
print("\n" + "="*55)
print("PASO 5: VALIDACIÓN DE RANGOS")
print("="*55)

# Rangos teóricos de cada variable
rangos = {
    "mean_R":     (0, 255),
    "mean_G":     (0, 255),
    "mean_B":     (0, 255),
    "mean_L":     (0, 100),
    "mean_a":     (-128, 128),
    "mean_b":     (-128, 128),
    "contrast_L": (0, 100),
}

print("\nValidación de rangos teóricos:")
for var, (vmin, vmax) in rangos.items():
    fuera = ((df[var] < vmin) | (df[var] > vmax)).sum()
    estado = "✓ OK" if fuera == 0 else f"⚠ {fuera} valores fuera de rango"
    print(f"  {var:<12}: [{vmin}, {vmax}]  →  {estado}")


# ============================================================
# PASO 6: ESTADÍSTICOS DESCRIPTIVOS
# ============================================================
print("\n" + "="*55)
print("PASO 6: ESTADÍSTICOS DESCRIPTIVOS DE VARIABLES DE COLOR")
print("="*55)

descripcion = df[variables_color].describe().round(3)
print(descripcion.to_string())

# Guardar también en CSV para incluir en la memoria
descripcion.to_csv(
    os.path.join(RUTA_SALIDA, "estadisticos_descriptivos.csv"),
    encoding="utf-8-sig"
)
print("\nTabla guardada: estadisticos_descriptivos.csv")


# ============================================================
# PASO 7: DETECCIÓN DE OUTLIERS (MÉTODO IQR)
# ============================================================
print("\n" + "="*55)
print("PASO 7: DETECCIÓN DE OUTLIERS")
print("="*55)

# El método IQR (rango intercuartílico) detecta valores
# que están muy por encima o por debajo de la mayoría
print("\nOutliers detectados por variable (método IQR):")
for var in variables_color:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    outliers = ((df[var] < limite_inf) | (df[var] > limite_sup)).sum()
    pct = round(outliers / len(df) * 100, 2)
    print(f"  {var:<12}: {outliers} outliers ({pct}%)")


# ============================================================
# PASO 8: HISTOGRAMAS DE VARIABLES DE COLOR
# ============================================================
print("\n" + "="*55)
print("PASO 8: HISTOGRAMAS DE DISTRIBUCIÓN DE COLOR")
print("="*55)

nombres_variables = {
    "mean_R":     "Canal Rojo — RGB (0-255)",
    "mean_G":     "Canal Verde — RGB (0-255)",
    "mean_B":     "Canal Azul — RGB (0-255)",
    "mean_L":     "Luminosidad L* — CIELAB (0-100)",
    "mean_a":     "Componente a* — CIELAB (-128 a 128)",
    "mean_b":     "Componente b* — CIELAB (-128 a 128)",
    "contrast_L": "Contraste L* — Desv. típica (0-100)",
}

colores_hist = ["#C0392B", "#27AE60", "#2980B9", "#8E44AD", "#E67E22", "#16A085", "#2C3E50"]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Distribución de variables de color", fontsize=16, fontweight="bold")
axes = axes.flatten()

for i, (var, titulo) in enumerate(nombres_variables.items()):
    axes[i].hist(df[var].dropna(), bins=40, color=colores_hist[i], edgecolor="white", alpha=0.85)
    axes[i].set_title(titulo, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Valor")
    axes[i].set_ylabel("Frecuencia")
    media = df[var].mean()
    axes[i].axvline(media, color="black", linestyle="--", linewidth=1.2, label=f"Media: {media:.1f}")
    axes[i].legend(fontsize=8)

# Ocultar el subplot sobrante (tenemos 7 variables y 9 celdas)
axes[7].set_visible(False)
axes[8].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, "histogramas_color.png"), dpi=150)
plt.close()
print("Gráfico guardado: histogramas_color.png")


# ============================================================
# PASO 9: MATRIZ DE CORRELACIÓN
# ============================================================
print("\n" + "="*55)
print("PASO 9: MATRIZ DE CORRELACIÓN")
print("="*55)

correlacion = df[variables_color].corr().round(2)
print("\nMatriz de correlación entre variables de color:")
print(correlacion.to_string())

# Visualización como mapa de calor
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(correlacion, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label="Correlación de Pearson")

# Etiquetas
ax.set_xticks(range(len(variables_color)))
ax.set_yticks(range(len(variables_color)))
ax.set_xticklabels(variables_color, rotation=45, ha="right")
ax.set_yticklabels(variables_color)

# Valores dentro de cada celda
for i in range(len(variables_color)):
    for j in range(len(variables_color)):
        valor = correlacion.iloc[i, j]
        color_texto = "white" if abs(valor) > 0.6 else "black"
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                fontsize=9, color=color_texto, fontweight="bold")

ax.set_title("Matriz de correlación — Variables de color", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, "correlacion.png"), dpi=150)
plt.close()
print("Gráfico guardado: correlacion.png")

# Correlaciones fuertes (|r| > 0.7)
print("\nCorrelaciones fuertes detectadas (|r| > 0.7):")
encontradas = False
for i in range(len(variables_color)):
    for j in range(i + 1, len(variables_color)):
        v1 = variables_color[i]
        v2 = variables_color[j]
        r = correlacion.loc[v1, v2]
        if abs(r) > 0.7:
            tipo = "positiva" if r > 0 else "negativa"
            print(f"  {v1} ↔ {v2}: r = {r:.2f} (correlación {tipo} fuerte)")
            encontradas = True
if not encontradas:
    print("  No se han encontrado correlaciones fuertes entre variables distintas.")


# ============================================================
# PASO 10: GUARDAR DATASET LIMPIO FINAL
# ============================================================
print("\n" + "="*55)
print("PASO 10: GUARDAR DATASET LIMPIO")
print("="*55)

RUTA_DATASET_LIMPIO = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\dataset_limpio.csv"
df.to_csv(RUTA_DATASET_LIMPIO, index=False, encoding="utf-8-sig")

print(f"\n✓ Dataset limpio guardado en:")
print(f"  {RUTA_DATASET_LIMPIO}")
print(f"\n  Filas finales:   {len(df)}")
print(f"  Columnas:        {len(df.columns)}")
print(f"  Fuentes:         {df['fuente'].nunique()}")
print(f"  Categorías:      {df['categoria'].nunique()}")

print("\n" + "="*55)
print("INGENIERÍA DEL DATO COMPLETADA")
print(f"Gráficos guardados en: {RUTA_SALIDA}")
print("="*55)