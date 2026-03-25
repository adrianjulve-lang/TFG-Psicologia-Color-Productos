# ============================================================
# DESCARGADOR - AMAZON BERKELEY OBJECTS (ABO) DATASET
# Dataset: https://amazon-berkeley-objects.s3.amazonaws.com
# Licencia: CC BY-NC 4.0 (uso académico permitido)
#
# El dataset ABO contiene 147.702 productos con 398.212 imágenes
# de catálogo en alta resolución, metadatos multilingüe (nombre,
# marca, tipo de producto, color, material, dimensiones) y modelos
# 3D. Es ideal para TFGs y publicaciones académicas.
#
# CÓMO FUNCIONA:
#   1) Descarga el archivo de metadatos de imágenes desde S3 (~3MB)
#   2) Descarga el archivo de listings de productos desde S3 (~70MB)
#   3) Filtra por categorías de producto de consumo relevantes
#   4) Descarga las imágenes en alta resolución
#   5) Calcula RGB y CIELAB para cada imagen
#   6) Guarda el dataset final en CSV
#
# NO necesita Selenium ni scraping. Todo es descarga directa
# desde los servidores públicos de Amazon/AWS.
#
# INSTALAR:
#   pip install requests pillow scikit-image tqdm pandas
#
# CITA PARA EL TFG:
#   Collins, J. et al. (2022). ABO: Dataset and Benchmarks for
#   Real-World 3D Object Understanding. CVPR 2022.
#   https://amazon-berkeley-objects.s3.amazonaws.com/index.html
#   Licencia: CC BY-NC 4.0
# ============================================================

import os
import io
import gzip
import json
import time
import re
import requests
import pandas as pd
import numpy as np
from PIL import Image
from skimage import color as skcolor
from tqdm import tqdm

# ── Configuración ──────────────────────────────────────────
OUTPUT_DIR  = "abo_data"
IMAGES_DIR  = os.path.join(OUTPUT_DIR, "imagenes")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "dataset_abo.csv")
CACHE_DIR   = os.path.join(OUTPUT_DIR, "cache")   # metadatos descargados

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# URLs de los archivos de metadatos en S3 (públicos, sin autenticación)
# El dataset completo de listings está partido en varios archivos .json.gz
# Usamos el archivo de imágenes (pequeño, ~3MB) como índice de URLs
URL_IMAGES_META  = "https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz"
URL_LISTINGS_BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/"

# Archivos de listings disponibles (0 al 9)
# Cada uno tiene ~7MB comprimido y ~70MB descomprimido
# Para el TFG con 1.000 productos, el archivo 0 es suficiente
# Si quieres más variedad, añade más números: [0, 1, 2, ...]
LISTING_FILES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # todos los archivos disponibles

MAX_PROD    = 10000   # máximo de productos a procesar
# Resolución de imagen: SX256 (pequeña/rápida), SX512 (media), SX1000 (alta)
IMG_RESOLUCION = "SX512"

HEADERS = {
    "User-Agent": "TFG-ColorAnalysis/1.0 (uso-academico)"
}

# ── Categorías de producto de consumo a incluir ────────────
# Filtro sobre el campo product_type del dataset ABO
# Referencia de tipos: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
CATEGORIAS_INTERES = {
    # Moda y complementos
    "SHIRT", "PANTS", "DRESS", "JACKET", "COAT", "SWEATER", "SKIRT",
    "SHOES", "BOOTS", "SNEAKERS", "SANDALS", "HANDBAG", "BACKPACK",
    "WALLET", "BELT", "WATCH", "SUNGLASSES", "HAT", "SCARF", "GLOVES",
    "JEWELRY", "NECKLACE", "BRACELET", "EARRING", "RING",
    # Hogar y mobiliario
    "SOFA", "CHAIR", "TABLE", "BED", "LAMP", "DESK", "SHELF",
    "CABINET", "MIRROR", "RUG", "CURTAIN", "PILLOW", "BLANKET",
    "VASE", "PICTURE_FRAME", "CLOCK",
    # Cocina y menaje
    "MUG", "CUP", "PLATE", "BOWL", "POT", "PAN", "KNIFE",
    "CUTTING_BOARD", "KETTLE", "TOASTER", "BLENDER", "COFFEE_MAKER",
    "WINE_GLASS", "BOTTLE", "STORAGE_CONTAINER",
    # Tecnología
    "LAPTOP", "TABLET", "HEADPHONES", "SPEAKER", "CAMERA",
    "KEYBOARD", "MOUSE", "MONITOR", "PHONE_CASE",
    # Cosmética e higiene
    "PERFUME", "LOTION", "SHAMPOO", "TOOTHBRUSH", "HAIR_DRYER",
    "MAKEUP", "LIPSTICK", "NAIL_POLISH",
    # Juguetes y ocio
    "TOY", "PUZZLE", "BOARD_GAME", "DOLL", "ACTION_FIGURE",
    "BOOK", "NOTEBOOK", "PEN",
    # Deportes
    "YOGA_MAT", "WATER_BOTTLE", "GYM_BAG", "BIKE", "HELMET",
    # Mascotas
    "PET_BED", "PET_BOWL", "PET_TOY",
}

# Mapeo de product_type → categoría legible para el dataset
MAPA_CATEGORIAS = {
    # Moda
    "SHIRT": "Moda - Camisetas", "PANTS": "Moda - Pantalones",
    "DRESS": "Moda - Vestidos", "JACKET": "Moda - Chaquetas",
    "COAT": "Moda - Abrigos", "SWEATER": "Moda - Jerseys",
    "SKIRT": "Moda - Faldas", "SHOES": "Moda - Zapatos",
    "BOOTS": "Moda - Botas", "SNEAKERS": "Moda - Zapatillas",
    "SANDALS": "Moda - Sandalias", "HANDBAG": "Moda - Bolsos",
    "BACKPACK": "Moda - Mochilas", "WALLET": "Moda - Carteras",
    "BELT": "Moda - Cinturones", "WATCH": "Moda - Relojes",
    "SUNGLASSES": "Moda - Gafas de sol", "HAT": "Moda - Sombreros",
    "SCARF": "Moda - Bufandas", "GLOVES": "Moda - Guantes",
    "JEWELRY": "Moda - Joyería", "NECKLACE": "Moda - Collares",
    "BRACELET": "Moda - Pulseras", "EARRING": "Moda - Pendientes",
    "RING": "Moda - Anillos",
    # Hogar
    "SOFA": "Hogar - Sofás", "CHAIR": "Hogar - Sillas",
    "TABLE": "Hogar - Mesas", "BED": "Hogar - Camas",
    "LAMP": "Hogar - Lámparas", "DESK": "Hogar - Escritorios",
    "SHELF": "Hogar - Estantes", "CABINET": "Hogar - Armarios",
    "MIRROR": "Hogar - Espejos", "RUG": "Hogar - Alfombras",
    "CURTAIN": "Hogar - Cortinas", "PILLOW": "Hogar - Cojines",
    "BLANKET": "Hogar - Mantas", "VASE": "Hogar - Jarrones",
    "PICTURE_FRAME": "Hogar - Marcos", "CLOCK": "Hogar - Relojes pared",
    # Cocina
    "MUG": "Cocina - Tazas", "CUP": "Cocina - Vasos",
    "PLATE": "Cocina - Platos", "BOWL": "Cocina - Boles",
    "POT": "Cocina - Ollas", "PAN": "Cocina - Sartenes",
    "KNIFE": "Cocina - Cuchillos", "CUTTING_BOARD": "Cocina - Tablas",
    "KETTLE": "Cocina - Teteras", "TOASTER": "Cocina - Tostadoras",
    "BLENDER": "Cocina - Batidoras", "COFFEE_MAKER": "Cocina - Cafeteras",
    "WINE_GLASS": "Cocina - Copas", "BOTTLE": "Cocina - Botellas",
    "STORAGE_CONTAINER": "Cocina - Recipientes",
    # Tecnología
    "LAPTOP": "Tecnología - Portátiles", "TABLET": "Tecnología - Tablets",
    "HEADPHONES": "Tecnología - Auriculares", "SPEAKER": "Tecnología - Altavoces",
    "CAMERA": "Tecnología - Cámaras", "KEYBOARD": "Tecnología - Teclados",
    "MOUSE": "Tecnología - Ratones", "MONITOR": "Tecnología - Monitores",
    "PHONE_CASE": "Tecnología - Fundas móvil",
    # Cosmética
    "PERFUME": "Cosmética - Perfumes", "LOTION": "Cosmética - Cremas",
    "SHAMPOO": "Cosmética - Champús", "TOOTHBRUSH": "Cosmética - Cepillos",
    "HAIR_DRYER": "Cosmética - Secadores", "MAKEUP": "Cosmética - Maquillaje",
    "LIPSTICK": "Cosmética - Labiales", "NAIL_POLISH": "Cosmética - Esmaltes",
    # Juguetes
    "TOY": "Juguetes - General", "PUZZLE": "Juguetes - Puzzles",
    "BOARD_GAME": "Juguetes - Juegos de mesa", "DOLL": "Juguetes - Muñecas",
    "ACTION_FIGURE": "Juguetes - Figuras", "BOOK": "Ocio - Libros",
    "NOTEBOOK": "Ocio - Cuadernos", "PEN": "Ocio - Bolígrafos",
    # Deporte
    "YOGA_MAT": "Deporte - Esterillas", "WATER_BOTTLE": "Deporte - Botellas",
    "GYM_BAG": "Deporte - Bolsas", "BIKE": "Deporte - Bicis",
    "HELMET": "Deporte - Cascos",
    # Mascotas
    "PET_BED": "Mascotas - Camas", "PET_BOWL": "Mascotas - Comederos",
    "PET_TOY": "Mascotas - Juguetes",
}


# ── Funciones de descarga de metadatos ────────────────────

def descargar_con_cache(url, nombre_cache):
    """Descarga un archivo y lo guarda en cache para no repetir."""
    ruta_cache = os.path.join(CACHE_DIR, nombre_cache)
    if os.path.exists(ruta_cache):
        print(f"    (usando cache: {nombre_cache})")
        with open(ruta_cache, "rb") as f:
            return f.read()
    print(f"    Descargando {nombre_cache}...")
    resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
    resp.raise_for_status()
    datos = resp.content
    with open(ruta_cache, "wb") as f:
        f.write(datos)
    return datos


def cargar_metadatos_imagenes():
    """
    Carga el CSV de metadatos de imágenes.
    Columnas: image_id, height, width, path
    La URL de cada imagen es:
    https://m.media-amazon.com/images/I/[image_id]._[RESOLUCION]_.jpg
    """
    datos = descargar_con_cache(URL_IMAGES_META, "images.csv.gz")
    with gzip.open(io.BytesIO(datos), "rt", encoding="utf-8") as f:
        df = pd.read_csv(f)
    print(f"    → {len(df)} imágenes en el índice")
    return df


def cargar_listings(num_archivo):
    """
    Carga un archivo de listings de productos.
    Cada línea es un JSON con los metadatos de un producto.
    """
    url   = URL_LISTINGS_BASE + f"listings_{num_archivo}.json.gz"
    cache = f"listings_{num_archivo}.json.gz"
    datos = descargar_con_cache(url, cache)

    productos = []
    with gzip.open(io.BytesIO(datos), "rt", encoding="utf-8") as f:
        for linea in f:
            try:
                productos.append(json.loads(linea.strip()))
            except Exception:
                continue
    print(f"    → {len(productos)} productos en listings_{num_archivo}")
    return productos


def construir_url_imagen(image_id, resolucion="SX512"):
    """
    Construye la URL de la imagen en Amazon CDN.
    Resoluciones disponibles: SX256, SX512, SX1000, AC_US400
    """
    return f"https://m.media-amazon.com/images/I/{image_id}._{resolucion}_.jpg"


def extraer_nombre(listing):
    """Extrae el nombre del producto en inglés (o el primero disponible)."""
    nombres = listing.get("item_name", [])
    for n in nombres:
        if n.get("language_tag", "").startswith("en"):
            return n.get("value", "Sin nombre").strip()
    if nombres:
        return nombres[0].get("value", "Sin nombre").strip()
    return "Sin nombre"


def extraer_product_type(listing):
    """Extrae el tipo de producto (product_type)."""
    tipos = listing.get("product_type", [])
    if tipos:
        return tipos[0].get("value", "").upper()
    return ""


def extraer_color_declarado(listing):
    """Extrae el color declarado por Amazon (si está disponible)."""
    colores = listing.get("color", [])
    for c in colores:
        if c.get("language_tag", "").startswith("en"):
            vals = c.get("standardized_values", [])
            if vals:
                return ", ".join(vals)
            return c.get("value", "")
    return ""


def extraer_color_code(listing):
    """Extrae el código HEX de color si está disponible."""
    codigos = listing.get("color_code", [])
    return codigos[0] if codigos else ""


def extraer_marca(listing):
    """Extrae la marca del producto."""
    marcas = listing.get("brand", [])
    for m in marcas:
        if m.get("language_tag", "").startswith("en"):
            return m.get("value", "").strip()
    if marcas:
        return marcas[0].get("value", "").strip()
    return ""


# ── Funciones de color ─────────────────────────────────────

def descargar_imagen(imagen_url, nombre_archivo):
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
    print("DESCARGADOR - AMAZON BERKELEY OBJECTS (ABO)")
    print("Licencia: CC BY-NC 4.0 (uso académico permitido)")
    print("=" * 60)

    # FASE 1: Cargar metadatos de imágenes
    print("\n[1/4] Cargando índice de imágenes...")
    df_imgs = cargar_metadatos_imagenes()
    # Crear diccionario image_id → path para búsqueda rápida
    img_dict = dict(zip(df_imgs["image_id"], df_imgs.get("path", df_imgs.get("image_id", df_imgs["image_id"]))))

    # FASE 2: Cargar listings y filtrar por categoría
    print(f"\n[2/4] Cargando listings de productos (archivos {LISTING_FILES})...")
    productos_filtrados = []

    for num in LISTING_FILES:
        print(f"  Archivo listings_{num}...")
        try:
            listings = cargar_listings(num)
        except Exception as e:
            print(f"    ⚠ Error al cargar listings_{num}: {e}")
            continue

        for prod in listings:
            product_type = extraer_product_type(prod)
            if product_type in CATEGORIAS_INTERES:
                main_image_id = prod.get("main_image_id", "")
                if not main_image_id:
                    continue
                productos_filtrados.append({
                    "item_id":        prod.get("item_id", ""),
                    "product_type":   product_type,
                    "categoria":      MAPA_CATEGORIAS.get(product_type, product_type),
                    "nombre":         extraer_nombre(prod),
                    "marca":          extraer_marca(prod),
                    "color_declarado": extraer_color_declarado(prod),
                    "color_code":     extraer_color_code(prod),
                    "main_image_id":  main_image_id,
                })

        if len(productos_filtrados) >= MAX_PROD:
            break

    # Deduplicar por item_id
    vistos = set()
    unicos = []
    for p in productos_filtrados:
        if p["item_id"] not in vistos:
            vistos.add(p["item_id"])
            unicos.append(p)

    productos_filtrados = unicos[:MAX_PROD]
    print(f"\n  Productos de consumo encontrados: {len(productos_filtrados)}")

    if len(productos_filtrados) == 0:
        print("  ⚠ No se encontraron productos. Prueba a añadir más archivos")
        print("    en la lista LISTING_FILES = [0, 1, 2, 3, ...]")
        return

    # Mostrar distribución por categoría
    df_preview = pd.DataFrame(productos_filtrados)
    print(f"\n  Distribución por categoría (top 10):")
    print(df_preview["categoria"].value_counts().head(10).to_string())

    # FASE 3: Descargar imágenes
    print(f"\n[3/4] Descargando imágenes y calculando RGB + CIELAB...")
    filas_finales = []

    for i, prod in enumerate(tqdm(productos_filtrados, desc="Procesando", unit="prod")):
        image_id      = prod["main_image_id"]
        imagen_url    = construir_url_imagen(image_id, IMG_RESOLUCION)
        nombre_limpio = re.sub(r'[^a-zA-Z0-9]', '_', prod["nombre"])[:35]
        nombre_archivo = f"abo_{i:04d}_{prod['item_id']}_{nombre_limpio}.jpg"

        ruta = descargar_imagen(imagen_url, nombre_archivo)
        if ruta is None:
            continue

        try:
            mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = calcular_color(ruta)
        except Exception:
            continue

        filas_finales.append({
            "fuente":           "Amazon Berkeley Objects",
            "item_id":          prod["item_id"],
            "categoria":        prod["categoria"],
            "product_type":     prod["product_type"],
            "nombre":           prod["nombre"],
            "marca":            prod["marca"],
            "color_declarado":  prod["color_declarado"],
            "color_code_hex":   prod["color_code"],
            "imagen_local":     nombre_archivo,
            "imagen_url":       imagen_url,
            "mean_R":           round(mean_R, 4),
            "mean_G":           round(mean_G, 4),
            "mean_B":           round(mean_B, 4),
            "mean_L":           round(mean_L, 4),
            "mean_a":           round(mean_a, 4),
            "mean_b":           round(mean_b, 4),
            "contrast_L":       round(contrast_L, 4),
        })

    # FASE 4: Guardar CSV
    print(f"\n[4/4] Guardando dataset...")
    df = pd.DataFrame(filas_finales)

    if len(df) == 0:
        print("⚠ No se procesó ningún producto.")
    else:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n✓ Dataset guardado en: {OUTPUT_CSV}")
        print(f"  Productos procesados:  {len(df)}")
        print(f"  Categorías:            {df['categoria'].nunique()}")
        print(f"  Marcas únicas:         {df['marca'].nunique()}")
        print(f"  Con color declarado:   {(df['color_declarado'] != '').sum()}")
        print(f"  Con código HEX:        {(df['color_code_hex'] != '').sum()}")
        print(f"  Imágenes en:           {IMAGES_DIR}/")
        print(f"\nDistribución por categoría:")
        print(df["categoria"].value_counts().to_string())
        print(f"\nEstadísticas de color calculado:")
        for col in ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]:
            print(f"  {col:<12}: {df[col].mean():.2f}")

    print("""
CITA PARA EL TFG:
  Collins, J., Goel, S., et al. (2022). ABO: Dataset and Benchmarks
  for Real-World 3D Object Understanding. CVPR 2022.
  Dataset: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
  Licencia: Creative Commons CC BY-NC 4.0
""")


if __name__ == "__main__":
    main()