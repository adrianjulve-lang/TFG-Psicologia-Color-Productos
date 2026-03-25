# ============================================================
# SCRAPER - OPEN FOOD FACTS (API pública)
# Web: https://world.openfoodfacts.org
#
# Open Food Facts es una base de datos abierta de productos
# alimentarios con más de 4 millones de productos de 150 países.
# Sus datos son de libre uso bajo licencia ODbL (Open Database License).
# Es perfectamente citable en un TFG como fuente académica.
#
# Esta fuente es IDEAL para el TFG porque:
#   - Las imágenes son siempre del packaging del producto (frente)
#   - La categoría viene dada directamente por la base de datos
#   - No hay problemas legales ni de robots.txt
#   - Citable: https://world.openfoodfacts.org
#
# NOTA LEGAL:
#   Uso de la API conforme a los términos de Open Food Facts.
#   Se respeta un delay de 1s entre peticiones para no saturar
#   el servidor (límite oficial: no hay límite estricto para
#   uso razonable académico, pero recomiendan ser moderados).
#
# INSTALAR:
#   pip install requests pillow scikit-image tqdm pandas
# ============================================================

import os
import time
import re
import io
import requests
import pandas as pd
import numpy as np
from PIL import Image
from skimage import color as skcolor
from tqdm import tqdm

# ── Configuración ──────────────────────────────────────────
API_BASE   = "https://world.openfoodfacts.org/api/v2/search"
DELAY      = 1.0      # segundos entre peticiones a la API
PAGE_SIZE  = 50       # productos por petición (máximo recomendado: 50)
MAX_PROD   = 10000     # total de productos a obtener
OUTPUT_DIR = "openfoodfacts_data"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "imagenes")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "dataset_openfoodfacts.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)

# User-Agent obligatorio según las normas de la API de Open Food Facts
# Identifica tu aplicación para que puedan contactarte si hay problemas
HEADERS = {
    "User-Agent": "TFG-ColorAnalysis/1.0 (universidad; uso-academico)"
}

# ── Categorías a descargar ─────────────────────────────────
# Se filtran por país España (en:spain) para obtener productos
# relevantes para el mercado español.
# Las categorías usan el formato de tags de Open Food Facts.
# Referencia: https://world.openfoodfacts.org/categories
CATEGORIAS = {
    "Bebidas":              "en:beverages",
    "Agua":                 "en:waters",
    "Zumos":                "en:fruit-juices",
    "Refrescos":            "en:sodas",
    "Cervezas":             "en:beers",
    "Vinos":                "en:wines",
    "Lácteos":              "en:dairy-products",
    "Leche":                "en:milks",
    "Yogures":              "en:yogurts",
    "Quesos":               "en:cheeses",
    "Cereales y desayuno":  "en:breakfast-cereals",
    "Galletas":             "en:biscuits",
    "Chocolates":           "en:chocolates",
    "Snacks":               "en:snacks",
    "Patatas fritas":       "en:chips-and-crisps",
    "Dulces y caramelos":   "en:confectioneries",
    "Conservas":            "en:canned-foods",
    "Salsas":               "en:sauces",
    "Aceites":              "en:oils",
    "Pasta":                "en:pastas",
    "Arroz":                "en:rices",
    "Pan y bollería":       "en:breads",
    "Congelados":           "en:frozen-foods",
    "Carne":                "en:meats",
    "Pescado":              "en:fishes",
    "Frutas":               "en:fruits",
    "Verduras":             "en:vegetables",
    "Legumbres":            "en:legumes",
    "Café e infusiones":    "en:coffees",
    "Condimentos":          "en:condiments",
}

# Campos que pedimos a la API — solo los necesarios para ahorrar ancho de banda
CAMPOS_API = ",".join([
    "code",
    "product_name",
    "brands",
    "categories_tags",
    "image_front_url",
    "image_url",
    "countries_tags",
])


# ── Funciones de API ───────────────────────────────────────

def buscar_productos(categoria_tag, max_productos):
    """
    Llama a la API de Open Food Facts para obtener productos
    de una categoría específica vendidos en España.
    Devuelve una lista de dicts con los datos de cada producto.
    """
    productos = []
    pagina = 1

    while len(productos) < max_productos:
        params = {
            "categories_tags":  categoria_tag,
            "countries_tags":   "en:spain",
            "fields":           CAMPOS_API,
            "page_size":        PAGE_SIZE,
            "page":             pagina,
            "sort_by":          "unique_scans_n",  # primero los más escaneados = más populares
            "json":             1,
        }

        try:
            resp = requests.get(API_BASE, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            datos = resp.json()
        except Exception as e:
            print(f"    ⚠ Error en API (página {pagina}): {e}")
            break

        items = datos.get("products", [])
        if not items:
            break

        for item in items:
            # Nos quedamos solo con los que tienen imagen del packaging
            imagen_url = (item.get("image_front_url") or
                          item.get("image_url") or "")
            if not imagen_url:
                continue

            # Nombre del producto
            nombre = item.get("product_name", "").strip()
            if not nombre:
                nombre = "Sin nombre"

            # Marca
            marca = item.get("brands", "").strip()

            productos.append({
                "codigo":     item.get("code", ""),
                "nombre":     nombre,
                "marca":      marca,
                "imagen_url": imagen_url,
            })

        # Comprobamos si hay más páginas
        total_disponible = datos.get("count", 0)
        if pagina * PAGE_SIZE >= total_disponible:
            break

        pagina += 1
        time.sleep(DELAY)

    return productos[:max_productos]


# ── Funciones de color ─────────────────────────────────────

def descargar_imagen(imagen_url, nombre_archivo):
    """Descarga imagen y la guarda como JPEG. Devuelve ruta local."""
    ruta = os.path.join(IMAGES_DIR, nombre_archivo)
    if os.path.exists(ruta):
        return ruta
    try:
        resp = requests.get(imagen_url, headers=HEADERS, timeout=20)
        img  = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(ruta, "JPEG", quality=90)
        return ruta
    except Exception:
        return None


def calcular_color(ruta):
    """Calcula métricas RGB y CIELAB de una imagen."""
    img = Image.open(ruta).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)

    mean_R = float(arr[:, :, 0].mean())
    mean_G = float(arr[:, :, 1].mean())
    mean_B = float(arr[:, :, 2].mean())

    lab        = skcolor.rgb2lab(arr / 255.0)
    mean_L     = float(lab[:, :, 0].mean())
    mean_a     = float(lab[:, :, 1].mean())
    mean_b_val = float(lab[:, :, 2].mean())
    contrast_L = float(lab[:, :, 0].std())

    return mean_R, mean_G, mean_B, mean_L, mean_a, mean_b_val, contrast_L


# ── Pipeline principal ─────────────────────────────────────

def main():
    print("=" * 60)
    print("SCRAPER OPEN FOOD FACTS — Productos españoles")
    print("Fuente: world.openfoodfacts.org (licencia ODbL)")
    print("=" * 60)

    # Repartimos el máximo de productos entre las categorías
    max_por_categoria = max(10, MAX_PROD // len(CATEGORIAS))

    # FASE 1: Obtener datos de la API
    print(f"\n[1/3] Descargando metadatos de la API...")
    print(f"  Categorías: {len(CATEGORIAS)}")
    print(f"  Máx. por categoría: {max_por_categoria}")
    print(f"  Total objetivo: {MAX_PROD} productos\n")

    todos_los_productos = []

    for nombre_cat, tag_cat in CATEGORIAS.items():
        print(f"  [{nombre_cat}]...", end=" ", flush=True)
        productos = buscar_productos(tag_cat, max_por_categoria)

        for p in productos:
            p["categoria"] = nombre_cat
        todos_los_productos.extend(productos)

        print(f"{len(productos)} productos")
        time.sleep(DELAY)

    # Deduplicar por código de producto (un mismo producto puede aparecer
    # en varias categorías)
    vistos = set()
    todos_unicos = []
    for p in todos_los_productos:
        clave = p["codigo"] or p["imagen_url"]
        if clave not in vistos:
            vistos.add(clave)
            todos_unicos.append(p)

    print(f"\n  Total productos únicos obtenidos: {len(todos_unicos)}")

    # FASE 2: Descargar imágenes y calcular color
    print(f"\n[2/3] Descargando imágenes y calculando RGB + CIELAB...")
    filas_finales = []

    for i, prod in enumerate(tqdm(todos_unicos, desc="Procesando", unit="img")):
        # Nombre de archivo limpio
        nombre_limpio  = re.sub(r'[^a-zA-Z0-9]', '_', prod["nombre"])[:40]
        codigo_limpio  = re.sub(r'[^a-zA-Z0-9]', '_', prod["codigo"])[:15]
        nombre_archivo = f"off_{i:04d}_{codigo_limpio}_{nombre_limpio}.jpg"

        # Descargamos la imagen
        ruta = descargar_imagen(prod["imagen_url"], nombre_archivo)
        if ruta is None:
            continue

        # Calculamos color
        try:
            mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = calcular_color(ruta)
        except Exception:
            continue

        filas_finales.append({
            "fuente":       "Open Food Facts",
            "categoria":    prod["categoria"],
            "nombre":       prod["nombre"],
            "marca":        prod["marca"],
            "codigo":       prod["codigo"],
            "imagen_local": nombre_archivo,
            "imagen_url":   prod["imagen_url"],
            "mean_R":       round(mean_R, 4),
            "mean_G":       round(mean_G, 4),
            "mean_B":       round(mean_B, 4),
            "mean_L":       round(mean_L, 4),
            "mean_a":       round(mean_a, 4),
            "mean_b":       round(mean_b, 4),
            "contrast_L":   round(contrast_L, 4),
        })

    # FASE 3: Guardar CSV
    print(f"\n[3/3] Guardando dataset...")
    df = pd.DataFrame(filas_finales)

    if len(df) == 0:
        print("⚠ No se procesó ningún producto.")
        print("  Comprueba tu conexión a internet e inténtalo de nuevo.")
    else:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        print(f"\n✓ Dataset guardado en: {OUTPUT_CSV}")
        print(f"  Productos procesados: {len(df)}")
        print(f"  Categorías:          {df['categoria'].nunique()}")
        print(f"  Marcas únicas:       {df['marca'].nunique()}")
        print(f"  Imágenes en:         {IMAGES_DIR}/")
        print(f"\nDistribución por categoría:")
        print(df["categoria"].value_counts().to_string())

        print(f"\nEstadísticas de color (medias globales):")
        for col in ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]:
            print(f"  {col:<12}: {df[col].mean():.2f}")

    print("""
CITA PARA EL TFG:
  Open Food Facts contributors (2024). Open Food Facts database.
  Open Database License (ODbL). https://world.openfoodfacts.org
""")


if __name__ == "__main__":
    main()