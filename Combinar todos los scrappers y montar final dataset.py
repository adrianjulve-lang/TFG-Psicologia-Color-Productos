# ============================================================
# UNIFICADOR DE DATASETS - TFG
# Combina los tres CSVs de scraping en un único dataset final
# con columnas homogéneas: fuente, categoria, nombre,
# imagen_url y variables de color RGB + CIELAB
#
# INSTALAR: pip install pandas
# ============================================================

import pandas as pd
import os

# ── Rutas de entrada ───────────────────────────────────────
# Pon aquí la carpeta donde tienes los tres CSV
CARPETA = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING"

PATH_ABO  = os.path.join(CARPETA, "dataset_abo.csv")
PATH_MAHOU = os.path.join(CARPETA, "dataset_mahou.csv")
PATH_OFF  = os.path.join(CARPETA, "dataset_openfoodfacts.csv")

# ── Ruta de salida ─────────────────────────────────────────
PATH_FINAL = os.path.join(CARPETA, "dataset_final_combinado.csv")

# ── Columnas que queremos en el dataset final ──────────────
COLUMNAS_FINALES = [
    "fuente",
    "categoria",
    "nombre",
    "imagen_url",
    "mean_R",
    "mean_G",
    "mean_B",
    "mean_L",
    "mean_a",
    "mean_b",
    "contrast_L",
]


def main():
    print("=" * 55)
    print("UNIFICADOR DE DATASETS - TFG")
    print("=" * 55)

    dfs = []

    # ── ABO ────────────────────────────────────────────────
    print("\nCargando dataset_abo.csv...")
    df_abo = pd.read_csv(PATH_ABO, encoding="utf-8-sig")
    df_abo = df_abo[["fuente", "categoria", "nombre", "imagen_url",
                     "mean_R", "mean_G", "mean_B",
                     "mean_L", "mean_a", "mean_b", "contrast_L"]]
    print(f"  → {len(df_abo)} filas")
    dfs.append(df_abo)

    # ── MAHOU ──────────────────────────────────────────────
    print("Cargando dataset_mahou.csv...")
    df_mahou = pd.read_csv(PATH_MAHOU, encoding="utf-8-sig")
    df_mahou = df_mahou[["fuente", "categoria", "nombre", "imagen_url",
                          "mean_R", "mean_G", "mean_B",
                          "mean_L", "mean_a", "mean_b", "contrast_L"]]
    print(f"  → {len(df_mahou)} filas")
    dfs.append(df_mahou)

    # ── OPEN FOOD FACTS ────────────────────────────────────
    print("Cargando dataset_openfoodfacts.csv...")
    df_off = pd.read_csv(PATH_OFF, encoding="utf-8-sig")
    df_off = df_off[["fuente", "categoria", "nombre", "imagen_url",
                     "mean_R", "mean_G", "mean_B",
                     "mean_L", "mean_a", "mean_b", "contrast_L"]]
    print(f"  → {len(df_off)} filas")
    dfs.append(df_off)

    # ── Combinar ───────────────────────────────────────────
    print("\nCombinando datasets...")
    df_final = pd.concat(dfs, ignore_index=True)

    # Eliminar filas con valores nulos en columnas de color
    antes = len(df_final)
    df_final = df_final.dropna(subset=["mean_R", "mean_G", "mean_B",
                                        "mean_L", "mean_a", "mean_b",
                                        "contrast_L"])
    despues = len(df_final)
    if antes != despues:
        print(f"  Filas eliminadas por nulos: {antes - despues}")

    # Eliminar duplicados por imagen_url
    antes = len(df_final)
    df_final = df_final.drop_duplicates(subset=["imagen_url"])
    despues = len(df_final)
    if antes != despues:
        print(f"  Duplicados eliminados: {antes - despues}")

    # Resetear índice
    df_final = df_final.reset_index(drop=True)

    # ── Guardar ────────────────────────────────────────────
    df_final.to_csv(PATH_FINAL, index=False, encoding="utf-8-sig")

    # ── Resumen ────────────────────────────────────────────
    print(f"\n✓ Dataset final guardado en:")
    print(f"  {PATH_FINAL}")
    print(f"\n  Total filas:    {len(df_final)}")
    print(f"  Columnas:       {list(df_final.columns)}")
    print(f"\nDistribución por fuente:")
    print(df_final["fuente"].value_counts().to_string())
    print(f"\nDistribución por categoría (top 15):")
    print(df_final["categoria"].value_counts().head(15).to_string())
    print(f"\nEstadísticas de color:")
    for col in ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]:
        print(f"  {col:<12}: media={df_final[col].mean():.2f}  "
              f"min={df_final[col].min():.2f}  "
              f"max={df_final[col].max():.2f}")


if __name__ == "__main__":
    main()