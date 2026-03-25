"""
=============================================================================
TFG: Análisis del Impacto de los Colores de Producto en las Emociones del Consumidor
Universidad Francisco de Vitoria + SAS
=============================================================================
Script único que:
1. Descarga imágenes de productos de Amazon Berkeley Objects (ABO)
2. Descarga imágenes de productos de Open Food Facts (OFF)
3. Descarga imágenes de productos de Mahou San Miguel (Selenium)
4. Unifica los 3 datasets
5. Realiza ingeniería del dato completa con 15 gráficos EDA
=============================================================================
"""

import sys
import subprocess

# --- Comprobador de dependencias ---
_PAQUETES = {
    "requests":   "requests",
    "numpy":      "numpy",
    "pandas":     "pandas",
    "matplotlib": "matplotlib",
    "PIL":        "Pillow",
    "tqdm":       "tqdm",
    "skimage":    "scikit-image",
    "sklearn":    "scikit-learn",
    "selenium":   "selenium",
}
_faltantes = []
for _modulo, _paquete in _PAQUETES.items():
    try:
        __import__(_modulo)
    except ImportError:
        _faltantes.append(_paquete)

if _faltantes:
    print(f"[INFO] Instalando paquetes que faltan: {', '.join(_faltantes)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + _faltantes)
    print("[INFO] Instalación completada. Reinicia el script si algo falla.\n")
# ------------------------------------

import os
import re
import gzip
import json
import time
import math
import random
import hashlib
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from skimage import color as skcolor
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================

CARPETA = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING"

CSV_MAHOU     = os.path.join(CARPETA, "mahou_data",          "dataset_mahou.csv")
CSV_ABO       = os.path.join(CARPETA, "abo_data",            "dataset_abo.csv")
CSV_OFF       = os.path.join(CARPETA, "openfoodfacts_data",  "dataset_openfoodfacts.csv")
CSV_COMBINADO = os.path.join(CARPETA, "Dataset_combinado_sin_emociones.csv")
CSV_FINAL     = os.path.join(CARPETA, "Dataset_con_emociones.csv")
CARPETA_GRAFICOS = os.path.join(CARPETA, "graficos")

IMG_MAHOU = os.path.join(CARPETA, "mahou_data",         "imagenes")
IMG_ABO   = os.path.join(CARPETA, "abo_data",           "imagenes")
CACHE_ABO = os.path.join(CARPETA, "abo_data",           "cache")
IMG_OFF   = os.path.join(CARPETA, "openfoodfacts_data", "imagenes")

# Crear todas las carpetas necesarias al inicio
for carpeta in [IMG_MAHOU, IMG_ABO, CACHE_ABO, IMG_OFF, CARPETA_GRAFICOS]:
    os.makedirs(carpeta, exist_ok=True)

# =============================================================================
# PARTE 1 — FUNCIONES COMPARTIDAS
# =============================================================================

def descargar_imagen(imagen_url: str, ruta_destino: str, headers: dict) -> str | None:
    """
    Descarga una imagen desde una URL y la guarda como JPEG.
    Sistema de caché: si el archivo ya existe en disco, devuelve la ruta sin descargar.
    Retorna la ruta del archivo guardado o None si falla.
    """
    # Caché: si ya existe en disco, no volver a descargar
    if os.path.exists(ruta_destino):
        return ruta_destino

    try:
        respuesta = requests.get(imagen_url, headers=headers, timeout=20)
        respuesta.raise_for_status()

        # Convertir a RGB con PIL y guardar como JPEG quality=90
        img = Image.open(BytesIO(respuesta.content)).convert("RGB")
        img.save(ruta_destino, format="JPEG", quality=90)
        return ruta_destino

    except Exception:
        # Si falla cualquier cosa, devolver None y continuar
        return None


def calcular_color(ruta: str) -> tuple | None:
    """
    Calcula las variables de color de una imagen en los espacios RGB y CIELAB.
    Devuelve una tupla de 7 valores: (mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L)
    o None si la imagen no se puede procesar.
    """
    try:
        img = Image.open(ruta).convert("RGB")
        arr = np.array(img, dtype=np.float32)

        # Medias de cada canal RGB (escala 0-255)
        mean_R = float(np.mean(arr[:, :, 0]))
        mean_G = float(np.mean(arr[:, :, 1]))
        mean_B = float(np.mean(arr[:, :, 2]))

        # Convertir a CIELAB (skimage espera valores en [0, 1])
        arr_norm = arr / 255.0
        lab = skcolor.rgb2lab(arr_norm)

        mean_L = float(np.mean(lab[:, :, 0]))   # Luminosidad [0, 100]
        mean_a = float(np.mean(lab[:, :, 1]))   # Rojo-verde [-128, 128]
        mean_b = float(np.mean(lab[:, :, 2]))   # Amarillo-azul [-128, 128]

        # Contraste como desviación típica de L* en todos los píxeles
        contrast_L = float(np.std(lab[:, :, 0]))

        # Redondear a 4 decimales
        return (
            round(mean_R, 4), round(mean_G, 4), round(mean_B, 4),
            round(mean_L, 4), round(mean_a, 4), round(mean_b, 4),
            round(contrast_L, 4)
        )

    except Exception:
        return None


# =============================================================================
# PARTE 2 — SCRAPER AMAZON BERKELEY OBJECTS (ABO)
# =============================================================================

# Categorías de interés (se descartan todas las demás, especialmente Moda)
CATEGORIAS_INTERES_ABO = {
    "LAMP", "MIRROR", "VASE", "PICTURE_FRAME", "CLOCK",
    "MUG", "CUP", "PLATE", "BOWL", "POT", "PAN", "KNIFE",
    "CUTTING_BOARD", "KETTLE", "TOASTER", "BLENDER", "COFFEE_MAKER",
    "WINE_GLASS", "BOTTLE", "STORAGE_CONTAINER",
    "LAPTOP", "TABLET", "HEADPHONES", "SPEAKER", "CAMERA",
    "KEYBOARD", "MOUSE", "MONITOR", "PHONE_CASE",
    "PERFUME", "LOTION", "SHAMPOO", "TOOTHBRUSH", "HAIR_DRYER",
    "MAKEUP", "LIPSTICK", "NAIL_POLISH",
    "TOY", "PUZZLE", "BOARD_GAME", "DOLL", "ACTION_FIGURE",
    "BOOK", "NOTEBOOK", "PEN",
    "YOGA_MAT", "WATER_BOTTLE", "GYM_BAG", "BIKE", "HELMET",
    "PET_BED", "PET_BOWL", "PET_TOY",
    "CANDLE", "SOAP", "DIFFUSER",
    "SUNSCREEN", "FACE_MASK", "SERUM",
    "PROTEIN_POWDER", "SUPPLEMENT",
    "DOG_FOOD", "CAT_FOOD",
    "CLEANING_PRODUCT", "DETERGENT",
}

# Mapa de código de categoría → nombre en español
MAPA_CATEGORIAS_ABO = {
    "LAMP": "Hogar - Lámparas", "MIRROR": "Hogar - Espejos",
    "VASE": "Hogar - Jarrones", "PICTURE_FRAME": "Hogar - Marcos",
    "CLOCK": "Hogar - Relojes pared",
    "MUG": "Cocina - Tazas", "CUP": "Cocina - Vasos",
    "PLATE": "Cocina - Platos", "BOWL": "Cocina - Boles",
    "POT": "Cocina - Ollas", "PAN": "Cocina - Sartenes",
    "KNIFE": "Cocina - Cuchillos", "CUTTING_BOARD": "Cocina - Tablas",
    "KETTLE": "Cocina - Teteras", "TOASTER": "Cocina - Tostadoras",
    "BLENDER": "Cocina - Batidoras", "COFFEE_MAKER": "Cocina - Cafeteras",
    "WINE_GLASS": "Cocina - Copas", "BOTTLE": "Cocina - Botellas",
    "STORAGE_CONTAINER": "Cocina - Recipientes",
    "LAPTOP": "Tecnología - Portátiles", "TABLET": "Tecnología - Tablets",
    "HEADPHONES": "Tecnología - Auriculares", "SPEAKER": "Tecnología - Altavoces",
    "CAMERA": "Tecnología - Cámaras", "KEYBOARD": "Tecnología - Teclados",
    "MOUSE": "Tecnología - Ratones", "MONITOR": "Tecnología - Monitores",
    "PHONE_CASE": "Tecnología - Fundas móvil",
    "PERFUME": "Cosmética - Perfumes", "LOTION": "Cosmética - Cremas",
    "SHAMPOO": "Cosmética - Champús", "TOOTHBRUSH": "Cosmética - Cepillos",
    "HAIR_DRYER": "Cosmética - Secadores", "MAKEUP": "Cosmética - Maquillaje",
    "LIPSTICK": "Cosmética - Labiales", "NAIL_POLISH": "Cosmética - Esmaltes",
    "CANDLE": "Hogar - Velas", "SOAP": "Cosmética - Jabones",
    "DIFFUSER": "Hogar - Difusores",
    "SUNSCREEN": "Cosmética - Protectores solares",
    "FACE_MASK": "Cosmética - Mascarillas", "SERUM": "Cosmética - Sérums",
    "TOY": "Juguetes - General", "PUZZLE": "Juguetes - Puzzles",
    "BOARD_GAME": "Juguetes - Juegos de mesa",
    "DOLL": "Juguetes - Muñecas", "ACTION_FIGURE": "Juguetes - Figuras",
    "BOOK": "Ocio - Libros", "NOTEBOOK": "Ocio - Cuadernos",
    "PEN": "Ocio - Bolígrafos",
    "YOGA_MAT": "Deporte - Esterillas", "WATER_BOTTLE": "Deporte - Botellas",
    "GYM_BAG": "Deporte - Bolsas", "BIKE": "Deporte - Bicis",
    "HELMET": "Deporte - Cascos",
    "PET_BED": "Mascotas - Camas", "PET_BOWL": "Mascotas - Comederos",
    "PET_TOY": "Mascotas - Juguetes",
    "PROTEIN_POWDER": "Deporte - Proteínas",
    "SUPPLEMENT": "Deporte - Suplementos",
    "DOG_FOOD": "Mascotas - Comida perro",
    "CAT_FOOD": "Mascotas - Comida gato",
    "CLEANING_PRODUCT": "Hogar - Limpieza",
    "DETERGENT": "Hogar - Detergentes",
}

MAX_PROD_ABO = 15000
BASE_URL_ABO_LISTINGS = "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/"
BASE_URL_ABO_IMG = "https://m.media-amazon.com/images/I/"

HEADERS_ABO = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def _descargar_listing_abo(idx: int) -> list:
    """
    Descarga y cachea un archivo de listings ABO (listings_N.json.gz).
    Devuelve la lista de productos del archivo.
    """
    nombre = f"listings_{idx}.json.gz"
    ruta_cache = os.path.join(CACHE_ABO, nombre)

    # Si ya está en caché, leer desde disco
    if os.path.exists(ruta_cache):
        print(f"  [ABO] Listing {idx} leído desde caché.")
        with gzip.open(ruta_cache, "rt", encoding="utf-8") as f:
            productos = [json.loads(linea) for linea in f if linea.strip()]
        return productos

    # Si no está, descargar de S3
    url = BASE_URL_ABO_LISTINGS + nombre
    try:
        print(f"  [ABO] Descargando listing {idx} desde S3...")
        r = requests.get(url, headers=HEADERS_ABO, timeout=60)
        r.raise_for_status()
        # Guardar en caché
        with open(ruta_cache, "wb") as f:
            f.write(r.content)
        # Parsear el contenido (formato JSONL dentro del .gz)
        with gzip.open(BytesIO(r.content), "rt", encoding="utf-8") as f:
            productos = [json.loads(linea) for linea in f if linea.strip()]
        print(f"  [ABO] Listing {idx} descargado: {len(productos)} productos.")
        return productos
    except Exception as e:
        print(f"  [ABO] Error descargando listing {idx}: {e}")
        return []


def _extraer_texto_abo(campo, idioma_pref="en") -> str:
    """
    Extrae el valor de texto de un campo multilingüe de ABO.
    Prioriza el idioma especificado (por defecto inglés).
    """
    if not campo:
        return ""
    if isinstance(campo, str):
        return campo
    if isinstance(campo, list):
        # Buscar primero en el idioma preferido
        for item in campo:
            if isinstance(item, dict) and item.get("language_tag", "").startswith(idioma_pref):
                return item.get("value", "")
        # Fallback: primer item disponible
        for item in campo:
            if isinstance(item, dict):
                return item.get("value", "")
    return ""


def scraper_abo():
    """
    Scraper de Amazon Berkeley Objects (ABO).
    Descarga los 10 archivos de listings, filtra por categorías de interés,
    descarga imágenes y calcula variables de color.
    Guarda el resultado en dataset_abo.csv.
    """
    print("\n" + "="*60)
    print("SCRAPER ABO — Amazon Berkeley Objects")
    print("="*60)

    # Recopilar todos los productos de los 10 archivos de listings
    print("\n[ABO] Fase 1: Recopilando productos de los 10 archivos de listings...")
    todos_los_productos = {}  # item_id → datos del producto (deduplicación)

    for idx in range(10):
        productos_raw = _descargar_listing_abo(idx)

        for prod in productos_raw:
            # Extraer tipo/categoría del producto
            tipo = prod.get("product_type", "")
            if isinstance(tipo, list) and tipo:
                tipo = tipo[0].get("value", "") if isinstance(tipo[0], dict) else str(tipo[0])

            # Filtrar solo las categorías de interés
            tipo_upper = str(tipo).upper()
            if tipo_upper not in CATEGORIAS_INTERES_ABO:
                continue

            item_id = prod.get("item_id", "")
            if not item_id or item_id in todos_los_productos:
                continue

            # Extraer main_image_id
            main_image_id = prod.get("main_image_id", "")
            if not main_image_id:
                continue

            # Extraer nombre (priorizar inglés)
            nombre = _extraer_texto_abo(prod.get("item_name", ""))

            # Extraer marca
            marca_raw = prod.get("brand", "")
            marca = _extraer_texto_abo(marca_raw) if isinstance(marca_raw, list) else str(marca_raw)

            todos_los_productos[item_id] = {
                "item_id": item_id,
                "tipo": tipo_upper,
                "categoria": MAPA_CATEGORIAS_ABO.get(tipo_upper, tipo_upper),
                "nombre": nombre,
                "marca": marca,
                "main_image_id": main_image_id,
            }

    print(f"\n[ABO] Total productos válidos encontrados (deduplicados): {len(todos_los_productos)}")

    # Limitar al máximo definido
    lista_productos = list(todos_los_productos.values())
    if len(lista_productos) > MAX_PROD_ABO:
        lista_productos = lista_productos[:MAX_PROD_ABO]
        print(f"[ABO] Limitando a {MAX_PROD_ABO} productos.")

    # Fase 2: Descargar imágenes y calcular color
    print(f"\n[ABO] Fase 2: Descargando imágenes y calculando color de {len(lista_productos)} productos...")

    filas = []
    for prod in tqdm(lista_productos, desc="ABO imágenes"):
        try:
            main_image_id = prod["main_image_id"]
            imagen_url = f"{BASE_URL_ABO_IMG}{main_image_id}._SX512_.jpg"

            # Nombre de archivo basado en el main_image_id
            nombre_archivo = re.sub(r"[^\w.-]", "_", main_image_id) + ".jpg"
            ruta_img = os.path.join(IMG_ABO, nombre_archivo)

            ruta_guardada = descargar_imagen(imagen_url, ruta_img, HEADERS_ABO)
            if not ruta_guardada:
                continue

            colores = calcular_color(ruta_guardada)
            if not colores:
                continue

            mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = colores

            filas.append({
                "fuente": "ABO",
                "categoria": prod["categoria"],
                "nombre": prod["nombre"],
                "imagen_url": imagen_url,
                "mean_R": mean_R, "mean_G": mean_G, "mean_B": mean_B,
                "mean_L": mean_L, "mean_a": mean_a, "mean_b": mean_b,
                "contrast_L": contrast_L,
            })

        except Exception as e:
            # Nunca parar por un error en un producto individual
            continue

    df_abo = pd.DataFrame(filas)
    df_abo.to_csv(CSV_ABO, index=False, encoding="utf-8-sig")
    print(f"\n[ABO] ✓ Dataset guardado: {CSV_ABO} ({len(df_abo)} filas)")

    # --- Gráficos EDA de ABO ---
    if not df_abo.empty:
        carpeta_abo = os.path.join(CARPETA_GRAFICOS, "abo")
        os.makedirs(carpeta_abo, exist_ok=True)
        cols_color = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]

        # Gráfico ABO-1: Boxplots de cada variable de color por categoría
        print("[ABO] Generando gráficos EDA...")
        for col in cols_color:
            categorias = df_abo["categoria"].unique()
            datos_box = [df_abo[df_abo["categoria"] == cat][col].dropna().values for cat in categorias]
            fig, ax = plt.subplots(figsize=(max(12, len(categorias) * 0.45), 7))
            ax.boxplot(datos_box, patch_artist=True)
            ax.set_xticks(range(1, len(categorias) + 1))
            ax.set_xticklabels(categorias, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"Distribución de {col} por categoría — ABO", fontsize=13)
            ax.set_ylabel(col)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(carpeta_abo, f"abo_boxplot_{col}.png"), dpi=150)
            plt.close()

        # Gráfico ABO-2: Histogramas de cada variable de color (panel 2×4)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, col in zip(axes.flatten(), cols_color):
            ax.hist(df_abo[col].dropna(), bins=40, color="#5DADE2", edgecolor="none", alpha=0.85)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            ax.grid(alpha=0.3)
        axes.flatten()[-1].set_visible(False)  # el 8º subplot queda vacío
        fig.suptitle("Distribución de variables de color — ABO", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_abo, "abo_histogramas_color.png"), dpi=150)
        plt.close()

        # Gráfico ABO-3: Número de productos por categoría
        conteo_cat = df_abo["categoria"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(10, max(6, len(conteo_cat) * 0.28)))
        conteo_cat.plot(kind="barh", ax=ax, color="#5DADE2")
        ax.set_title("Número de productos por categoría — ABO", fontsize=13)
        ax.set_xlabel("Número de productos")
        for i, v in enumerate(conteo_cat.values):
            ax.text(v + 1, i, str(v), va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_abo, "abo_productos_por_categoria.png"), dpi=150)
        plt.close()

        print(f"[ABO] ✓ Gráficos EDA guardados en: {carpeta_abo}")

    return df_abo


# =============================================================================
# PARTE 3 — SCRAPER OPEN FOOD FACTS (API REST)
# =============================================================================

CATEGORIAS_OFF = {
    "Bebidas": "en:beverages",
    "Agua": "en:waters",
    "Zumos": "en:fruit-juices",
    "Refrescos": "en:sodas",
    "Cervezas": "en:beers",
    "Vinos": "en:wines",
    "Lácteos": "en:dairy-products",
    "Leche": "en:milks",
    "Yogures": "en:yogurts",
    "Quesos": "en:cheeses",
    "Cereales y desayuno": "en:breakfast-cereals",
    "Galletas": "en:biscuits",
    "Chocolates": "en:chocolates",
    "Snacks": "en:snacks",
    "Patatas fritas": "en:chips-and-crisps",
    "Dulces y caramelos": "en:confectioneries",
    "Conservas": "en:canned-foods",
    "Salsas": "en:sauces",
    "Aceites": "en:oils",
    "Pasta": "en:pastas",
    "Arroz": "en:rices",
    "Pan y bollería": "en:breads",
    "Congelados": "en:frozen-foods",
    "Carne": "en:meats",
    "Pescado": "en:fishes",
    "Frutas": "en:fruits",
    "Verduras": "en:vegetables",
    "Legumbres": "en:legumes",
    "Café e infusiones": "en:coffees",
    "Condimentos": "en:condiments",
    "Sopas": "en:soups",
    "Helados": "en:ice-creams",
    "Mermeladas": "en:jams",
    "Bebidas vegetales": "en:plant-based-beverages",
    "Comida infantil": "en:baby-foods",
}

MAX_PROD_OFF = 30000
PAGE_SIZE_OFF = 50
DELAY_OFF = 1.0  # Segundos entre peticiones para no sobrecargar la API

HEADERS_OFF = {
    "User-Agent": "TFG-ColorAnalysis/1.0 (universidad; uso-academico)"
}


def scraper_openfoodfacts():
    """
    Scraper de Open Food Facts usando su API REST v2.
    Descarga productos de 35 categorías de alimentación,
    calcula variables de color y guarda el dataset.
    """
    print("\n" + "="*60)
    print("SCRAPER OPEN FOOD FACTS")
    print("="*60)

    codigos_vistos = set()   # Deduplicación por código de barras
    urls_vistas    = set()   # Deduplicación adicional por URL de imagen
    filas          = []
    total_descargados = 0

    for nombre_cat, tag_cat in CATEGORIAS_OFF.items():
        if total_descargados >= MAX_PROD_OFF:
            print(f"\n[OFF] Límite de {MAX_PROD_OFF} productos alcanzado. Deteniendo.")
            break

        print(f"\n[OFF] Categoría: {nombre_cat} ({tag_cat})")
        pagina = 1
        productos_cat = 0
        max_por_cat = MAX_PROD_OFF // len(CATEGORIAS_OFF) + 200  # Distribuir el límite

        while total_descargados < MAX_PROD_OFF and productos_cat < max_por_cat:
            try:
                params = {
                    "categories_tags": tag_cat,
                    "fields": "code,product_name,image_url,image_front_url",
                    "sort_by": "unique_scans_n",
                    "page_size": PAGE_SIZE_OFF,
                    "page": pagina,
                }
                r = requests.get(
                    "https://world.openfoodfacts.org/api/v2/search",
                    params=params,
                    headers=HEADERS_OFF,
                    timeout=30
                )
                r.raise_for_status()
                datos = r.json()

                productos = datos.get("products", [])
                if not productos:
                    break  # No hay más resultados para esta categoría

                for prod in productos:
                    # Filtrar sin imagen o sin nombre
                    imagen_url = prod.get("image_front_url") or prod.get("image_url", "")
                    nombre = prod.get("product_name", "").strip()
                    codigo = prod.get("code", "")

                    if not imagen_url or not nombre:
                        continue

                    # Deduplicar por código de barras
                    if codigo and codigo in codigos_vistos:
                        continue
                    # Deduplicar por URL de imagen como fallback
                    if imagen_url in urls_vistas:
                        continue

                    if codigo:
                        codigos_vistos.add(codigo)
                    urls_vistas.add(imagen_url)

                    # Descargar imagen
                    nombre_archivo = hashlib.md5(imagen_url.encode()).hexdigest() + ".jpg"
                    ruta_img = os.path.join(IMG_OFF, nombre_archivo)
                    ruta_guardada = descargar_imagen(imagen_url, ruta_img, HEADERS_OFF)
                    if not ruta_guardada:
                        continue

                    colores = calcular_color(ruta_guardada)
                    if not colores:
                        continue

                    mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = colores

                    filas.append({
                        "fuente": "OpenFoodFacts",
                        "categoria": nombre_cat,
                        "nombre": nombre,
                        "imagen_url": imagen_url,
                        "mean_R": mean_R, "mean_G": mean_G, "mean_B": mean_B,
                        "mean_L": mean_L, "mean_a": mean_a, "mean_b": mean_b,
                        "contrast_L": contrast_L,
                    })

                    total_descargados += 1
                    productos_cat += 1

                    if total_descargados % 500 == 0:
                        print(f"  [OFF] {total_descargados} productos totales procesados...")

                    if total_descargados >= MAX_PROD_OFF:
                        break

                pagina += 1
                time.sleep(DELAY_OFF)  # Respetar el rate limit de la API

            except Exception as e:
                print(f"  [OFF] Error en página {pagina} de {nombre_cat}: {e}")
                time.sleep(3)
                break

        print(f"  [OFF] {nombre_cat}: {productos_cat} productos añadidos.")

    df_off = pd.DataFrame(filas)
    df_off.to_csv(CSV_OFF, index=False, encoding="utf-8-sig")
    print(f"\n[OFF] ✓ Dataset guardado: {CSV_OFF} ({len(df_off)} filas)")

    # --- Gráficos EDA de Open Food Facts ---
    if not df_off.empty:
        carpeta_off = os.path.join(CARPETA_GRAFICOS, "openfoodfacts")
        os.makedirs(carpeta_off, exist_ok=True)
        cols_color = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]

        print("[OFF] Generando gráficos EDA...")

        # Gráfico OFF-1: Boxplots de cada variable de color por categoría
        for col in cols_color:
            categorias = df_off["categoria"].unique()
            datos_box = [df_off[df_off["categoria"] == cat][col].dropna().values for cat in categorias]
            fig, ax = plt.subplots(figsize=(max(12, len(categorias) * 0.5), 7))
            ax.boxplot(datos_box, patch_artist=True)
            ax.set_xticks(range(1, len(categorias) + 1))
            ax.set_xticklabels(categorias, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"Distribución de {col} por categoría — Open Food Facts", fontsize=13)
            ax.set_ylabel(col)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(carpeta_off, f"off_boxplot_{col}.png"), dpi=150)
            plt.close()

        # Gráfico OFF-2: Histogramas de cada variable de color (panel 2×4)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, col in zip(axes.flatten(), cols_color):
            ax.hist(df_off[col].dropna(), bins=40, color="#2ECC71", edgecolor="none", alpha=0.85)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            ax.grid(alpha=0.3)
        axes.flatten()[-1].set_visible(False)
        fig.suptitle("Distribución de variables de color — Open Food Facts", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_off, "off_histogramas_color.png"), dpi=150)
        plt.close()

        # Gráfico OFF-3: Número de productos por categoría
        conteo_cat = df_off["categoria"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(10, max(6, len(conteo_cat) * 0.35)))
        conteo_cat.plot(kind="barh", ax=ax, color="#2ECC71")
        ax.set_title("Número de productos por categoría — Open Food Facts", fontsize=13)
        ax.set_xlabel("Número de productos")
        for i, v in enumerate(conteo_cat.values):
            ax.text(v + 1, i, str(v), va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_off, "off_productos_por_categoria.png"), dpi=150)
        plt.close()

        print(f"[OFF] ✓ Gráficos EDA guardados en: {carpeta_off}")

    return df_off


# =============================================================================
# PARTE 4 — SCRAPER MAHOU SAN MIGUEL (Selenium)
# =============================================================================

CATEGORIAS_MAHOU = {
    "Cerveza Mahou":        "/tienda/marcas/mahou/producto",
    "Cerveza San Miguel":   "/tienda/marcas/cervezas-san-miguel/producto",
    "Cerveza Alhambra":     "/tienda/marcas/cervezas-alhambra/producto",
    "Corona":               "/tienda/marcas/corona",
    "Founders Brewing":     "/tienda/marcas/founders/producto",
    "Budweiser":            "/tienda/marcas/budweiser",
    "Nómada Brewing":       "/tienda/marcas/nomada",
    "Brutus":               "/tienda/marcas/brutus/producto",
    "Agua Solán de Cabras": "/tienda/marcas/solan-de-cabras/producto",
    "Agua Sierra Natura":   "/tienda/marcas/sierra-natura",
    "Malta y otras bebidas":"/tienda/otras-bebidas",
    "Cristalería y hogar":  "/tienda/regalos-cerveceros/cristaleria-y-hogar",
    "Moda y accesorios":    "/tienda/regalos-cerveceros/moda-y-accesorios",
}

BASE_URL_MAHOU = "https://www.mahou-sanmiguel.com"
DELAY_MAHOU = 3  # Segundos entre páginas


def scraper_mahou():
    """
    Scraper de la tienda online de Mahou San Miguel usando Selenium.
    Recorre todas las categorías, extrae URLs de productos y descarga imágenes.
    """
    print("\n" + "="*60)
    print("SCRAPER MAHOU SAN MIGUEL (Selenium)")
    print("="*60)

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException
    except ImportError:
        print("[MAHOU] ERROR: selenium no instalado. Instala con: pip install selenium")
        return pd.DataFrame()

    # Configurar Chrome en modo headless
    opciones = Options()
    opciones.add_argument("--headless=new")
    opciones.add_argument("--no-sandbox")
    opciones.add_argument("--disable-dev-shm-usage")
    opciones.add_argument("--window-size=1920,1080")
    opciones.add_argument("--lang=es-ES")
    opciones.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=opciones)
    wait = WebDriverWait(driver, 15)

    headers_mahou = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Aceptar cookies al inicio
    try:
        driver.get(BASE_URL_MAHOU + "/tienda")
        time.sleep(3)
        # Buscar botón de aceptar cookies por XPath con texto "acepto"
        botones_cookies = driver.find_elements(
            By.XPATH,
            "//*[contains(translate(text(), 'ACEPTO', 'acepto'), 'acepto')]"
        )
        for btn in botones_cookies:
            try:
                btn.click()
                print("[MAHOU] Cookies aceptadas.")
                time.sleep(2)
                break
            except Exception:
                continue
    except Exception as e:
        print(f"[MAHOU] Aviso al aceptar cookies: {e}")

    # Recopilar URLs de productos de todas las categorías
    print("\n[MAHOU] Fase 1: Recopilando URLs de productos...")
    urls_productos = {}  # URL → nombre_categoria (deduplicación con dict)

    for nombre_cat, ruta_cat in CATEGORIAS_MAHOU.items():
        print(f"\n  [MAHOU] Categoría: {nombre_cat}")
        inicio = 0
        tamano = 12

        while True:
            url_pagina = f"{BASE_URL_MAHOU}{ruta_cat}?start={inicio}&sz={tamano}"
            try:
                driver.get(url_pagina)
                time.sleep(DELAY_MAHOU)

                # Scroll para forzar lazy loading
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(1)

                # Extraer URLs de productos (patrón /tienda/p/)
                enlaces = driver.find_elements(By.XPATH, "//a[contains(@href, '/tienda/p/')]")
                urls_pagina = [a.get_attribute("href") for a in enlaces if a.get_attribute("href")]

                if not urls_pagina:
                    break  # No hay más productos en esta categoría

                # Contar cuántas URLs nuevas se añaden en esta página
                nuevas = 0
                for url in urls_pagina:
                    if url not in urls_productos:
                        urls_productos[url] = nombre_cat
                        nuevas += 1

                print(f"    Página {inicio//tamano + 1}: {len(urls_pagina)} encontradas, {nuevas} nuevas")

                # Si ninguna URL es nueva, el sitio repite la última página → parar
                if nuevas == 0:
                    break

                inicio += tamano

            except Exception as e:
                print(f"    Error en página: {e}")
                break

    # Deduplicar con dict.fromkeys() para mantener orden de inserción
    urls_unicas = list(dict.fromkeys(urls_productos.keys()))
    print(f"\n[MAHOU] Total URLs de productos únicas: {len(urls_unicas)}")

    # Fase 2: Visitar cada producto y extraer información
    print("\n[MAHOU] Fase 2: Extrayendo datos de cada producto...")
    filas = []

    for url_prod in tqdm(urls_unicas, desc="MAHOU productos"):
        try:
            nombre_cat = urls_productos.get(url_prod, "Desconocido")
            driver.get(url_prod)
            time.sleep(2)

            # Extraer nombre del producto (4 selectores CSS en cascada)
            nombre = ""
            for selector in ["h1.product-name", "h1.product-detail-name", "h1[itemprop='name']", "h1"]:
                try:
                    el = driver.find_element(By.CSS_SELECTOR, selector)
                    nombre = el.text.strip()
                    if nombre:
                        break
                except NoSuchElementException:
                    continue

            if not nombre:
                nombre = "Producto sin nombre"

            # Extraer imagen: estrategia 1 = meta og:image
            imagen_url = ""
            try:
                meta_og = driver.find_element(By.XPATH, "//meta[@property='og:image']")
                imagen_url = meta_og.get_attribute("content") or ""
            except NoSuchElementException:
                pass

            # Estrategia 2: selectores CSS filtrando por "demandware" o "mahou" en la URL
            if not imagen_url:
                for selector_img in ["img.primary-image", "img.product-image", ".product-detail img", "img"]:
                    try:
                        imgs = driver.find_elements(By.CSS_SELECTOR, selector_img)
                        for img in imgs:
                            src = img.get_attribute("src") or ""
                            if "demandware" in src or "mahou" in src:
                                imagen_url = src
                                break
                        if imagen_url:
                            break
                    except Exception:
                        continue

            if not imagen_url:
                continue

            # Limpiar URL de imagen: quitar parámetros de query string
            imagen_url = re.sub(r"\?.*$", "", imagen_url)

            # Si la URL es relativa, añadir el dominio base
            if imagen_url.startswith("/"):
                imagen_url = BASE_URL_MAHOU + imagen_url

            # Descargar imagen
            nombre_archivo = hashlib.md5(imagen_url.encode()).hexdigest() + ".jpg"
            ruta_img = os.path.join(IMG_MAHOU, nombre_archivo)
            ruta_guardada = descargar_imagen(imagen_url, ruta_img, headers_mahou)
            if not ruta_guardada:
                continue

            colores = calcular_color(ruta_guardada)
            if not colores:
                continue

            mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = colores

            filas.append({
                "fuente": "Mahou",
                "categoria": nombre_cat,
                "nombre": nombre,
                "imagen_url": imagen_url,
                "mean_R": mean_R, "mean_G": mean_G, "mean_B": mean_B,
                "mean_L": mean_L, "mean_a": mean_a, "mean_b": mean_b,
                "contrast_L": contrast_L,
            })

        except Exception as e:
            continue  # Nunca parar por error en un producto

    driver.quit()
    print("[MAHOU] Chrome cerrado.")

    df_mahou = pd.DataFrame(filas)
    df_mahou.to_csv(CSV_MAHOU, index=False, encoding="utf-8-sig")
    print(f"\n[MAHOU] ✓ Dataset guardado: {CSV_MAHOU} ({len(df_mahou)} filas)")

    # --- Gráficos EDA de Mahou ---
    if not df_mahou.empty:
        carpeta_mahou = os.path.join(CARPETA_GRAFICOS, "mahou")
        os.makedirs(carpeta_mahou, exist_ok=True)
        cols_color = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]

        print("[MAHOU] Generando gráficos EDA...")

        # Gráfico MAHOU-1: Boxplots de cada variable de color por categoría
        for col in cols_color:
            categorias = df_mahou["categoria"].unique()
            datos_box = [df_mahou[df_mahou["categoria"] == cat][col].dropna().values for cat in categorias]
            fig, ax = plt.subplots(figsize=(max(10, len(categorias) * 0.7), 6))
            ax.boxplot(datos_box, patch_artist=True)
            ax.set_xticks(range(1, len(categorias) + 1))
            ax.set_xticklabels(categorias, rotation=40, ha="right", fontsize=8)
            ax.set_title(f"Distribución de {col} por categoría — Mahou", fontsize=13)
            ax.set_ylabel(col)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(carpeta_mahou, f"mahou_boxplot_{col}.png"), dpi=150)
            plt.close()

        # Gráfico MAHOU-2: Histogramas de cada variable de color (panel 2×4)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, col in zip(axes.flatten(), cols_color):
            ax.hist(df_mahou[col].dropna(), bins=20, color="#E74C3C", edgecolor="none", alpha=0.85)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            ax.grid(alpha=0.3)
        axes.flatten()[-1].set_visible(False)
        fig.suptitle("Distribución de variables de color — Mahou San Miguel", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_mahou, "mahou_histogramas_color.png"), dpi=150)
        plt.close()

        # Gráfico MAHOU-3: Número de productos por categoría
        conteo_cat = df_mahou["categoria"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(10, max(5, len(conteo_cat) * 0.4)))
        conteo_cat.plot(kind="barh", ax=ax, color="#E74C3C")
        ax.set_title("Número de productos por categoría — Mahou San Miguel", fontsize=13)
        ax.set_xlabel("Número de productos")
        for i, v in enumerate(conteo_cat.values):
            ax.text(v + 0.2, i, str(v), va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_mahou, "mahou_productos_por_categoria.png"), dpi=150)
        plt.close()

        print(f"[MAHOU] ✓ Gráficos EDA guardados en: {carpeta_mahou}")

    return df_mahou


# =============================================================================
# PARTE 5 — UNIFICACIÓN DE LOS 3 DATASETS
# =============================================================================

COLUMNAS_ESTANDAR = [
    "fuente", "categoria", "nombre", "imagen_url",
    "mean_R", "mean_G", "mean_B",
    "mean_L", "mean_a", "mean_b",
    "contrast_L"
]


def unificar_datasets():
    """
    Lee los 3 CSVs individuales, los une, limpia duplicados
    y guarda Dataset_combinado_sin_emociones.csv.
    """
    print("\n" + "="*60)
    print("UNIFICACIÓN DE DATASETS")
    print("="*60)

    dfs = []

    for ruta, nombre in [(CSV_MAHOU, "Mahou"), (CSV_ABO, "ABO"), (CSV_OFF, "OpenFoodFacts")]:
        if os.path.exists(ruta):
            df = pd.read_csv(ruta, encoding="utf-8-sig")
            print(f"  {nombre}: {len(df)} filas leídas")
            dfs.append(df)
        else:
            print(f"  {nombre}: archivo no encontrado ({ruta}), omitiendo.")

    if not dfs:
        print("ERROR: No se encontró ningún CSV para unificar.")
        return pd.DataFrame()

    # Concatenar todos los datasets
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total tras concat: {len(df)} filas")

    # Mantener solo las 11 columnas estándar
    for col in COLUMNAS_ESTANDAR:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNAS_ESTANDAR]

    # Eliminar nulos en variables de color
    cols_color = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]
    df = df.dropna(subset=cols_color)

    # Eliminar duplicados por URL de imagen
    df = df.drop_duplicates(subset=["imagen_url"]).reset_index(drop=True)

    print(f"  Total tras limpieza: {len(df)} filas")
    df.to_csv(CSV_COMBINADO, index=False, encoding="utf-8-sig")
    print(f"\n✓ Dataset combinado guardado: {CSV_COMBINADO} ({len(df)} filas)")
    return df


# =============================================================================
# PARTE 6 — INGENIERÍA DEL DATO
# =============================================================================

# Colores por emoción para los gráficos
COLORES_EMOCION = {
    "Ira":            "#C0392B",
    "Tristeza":       "#5D6D7E",
    "Romanticismo":   "#A569BD",
    "Energía":        "#E67E22",
    "Alegría":        "#F1C40F",
    "Relajación":     "#AED6F1",
    "Calma":          "#A9DFBF",
    "Aburrimiento":   "#BDC3C7",
    "Neutro/Ambiguo": "#95A5A6",
}

ORDEN_EMOCIONES = [
    "Ira", "Tristeza", "Romanticismo", "Energía",
    "Alegría", "Relajación", "Calma", "Aburrimiento", "Neutro/Ambiguo"
]

# Centroides emocionales v2 — espacio (L*, C*, H*)
# Fuente: Gilbert et al. (2016) + Valdez & Mehrabian (1994), adaptados a e-commerce
# Formato: (µ_L, σ_L, µ_C, σ_C, µ_H, σ_H, w_L, w_C, w_H)
CENTROIDES = {
    "Ira":            (22,   8,   10,   6,   0.10,  0.60,  0.50, 0.35, 0.15),
    "Tristeza":       (42,  10,    4,   4,   0.00,  1.50,  0.65, 0.30, 0.05),
    "Romanticismo":   (50,  12,   12,   7,   0.35,  0.55,  0.30, 0.35, 0.35),
    "Energía":        (62,  10,   15,   8,   0.25,  0.60,  0.25, 0.40, 0.35),
    "Alegría":        (80,   8,   16,   7,   0.85,  0.55,  0.30, 0.35, 0.35),
    "Relajación":     (92,   6,    3,   3,   0.00,  1.80,  0.70, 0.25, 0.05),
    "Calma":          (85,   7,    5,   4,  -0.40,  1.20,  0.55, 0.30, 0.15),
    "Aburrimiento":   (70,  12,    5,   4,   0.00,  2.00,  0.40, 0.40, 0.20),
}

# Emoción óptima esperada por categoría de producto
EMOCION_OPTIMA = {
    "Cerveza Mahou": "Relajación", "Cerveza San Miguel": "Relajación",
    "Cerveza Alhambra": "Relajación", "Corona": "Relajación",
    "Founders Brewing": "Relajación", "Budweiser": "Relajación",
    "Nómada Brewing": "Relajación", "Brutus": "Relajación",
    "Agua Solán de Cabras": "Relajación", "Agua Sierra Natura": "Relajación",
    "Malta y otras bebidas": "Energía", "Cristalería y hogar": "Calma",
    "Moda y accesorios": "Alegría",
    "Bebidas": "Energía", "Agua": "Relajación", "Zumos": "Alegría",
    "Refrescos": "Energía", "Cervezas": "Relajación", "Vinos": "Romanticismo",
    "Lácteos": "Calma", "Leche": "Calma", "Yogures": "Calma",
    "Quesos": "Romanticismo", "Cereales y desayuno": "Energía",
    "Galletas": "Energía", "Chocolates": "Romanticismo", "Snacks": "Energía",
    "Patatas fritas": "Energía", "Dulces y caramelos": "Alegría",
    "Conservas": "Calma", "Salsas": "Energía", "Aceites": "Calma",
    "Pasta": "Romanticismo", "Arroz": "Calma", "Pan y bollería": "Alegría",
    "Congelados": "Calma", "Carne": "Romanticismo", "Pescado": "Calma",
    "Frutas": "Alegría", "Verduras": "Calma", "Legumbres": "Calma",
    "Café e infusiones": "Romanticismo", "Condimentos": "Energía",
    "Sopas": "Calma", "Helados": "Alegría", "Mermeladas": "Alegría",
    "Bebidas vegetales": "Calma", "Comida infantil": "Alegría",
    "Hogar - Lámparas": "Relajación", "Hogar - Espejos": "Calma",
    "Hogar - Jarrones": "Calma", "Hogar - Marcos": "Calma",
    "Hogar - Relojes pared": "Calma", "Hogar - Velas": "Romanticismo",
    "Hogar - Difusores": "Relajación", "Hogar - Limpieza": "Calma",
    "Hogar - Detergentes": "Calma",
    "Cocina - Tazas": "Calma", "Cocina - Vasos": "Calma",
    "Cocina - Platos": "Calma", "Cocina - Boles": "Calma",
    "Cocina - Ollas": "Calma", "Cocina - Sartenes": "Calma",
    "Cocina - Cuchillos": "Ira", "Cocina - Tablas": "Calma",
    "Cocina - Teteras": "Calma", "Cocina - Tostadoras": "Energía",
    "Cocina - Batidoras": "Energía", "Cocina - Cafeteras": "Romanticismo",
    "Cocina - Copas": "Relajación", "Cocina - Botellas": "Calma",
    "Cocina - Recipientes": "Calma",
    "Tecnología - Portátiles": "Calma", "Tecnología - Tablets": "Calma",
    "Tecnología - Auriculares": "Calma", "Tecnología - Altavoces": "Energía",
    "Tecnología - Cámaras": "Calma", "Tecnología - Teclados": "Calma",
    "Tecnología - Ratones": "Calma", "Tecnología - Monitores": "Calma",
    "Tecnología - Fundas móvil": "Alegría",
    "Cosmética - Perfumes": "Romanticismo", "Cosmética - Cremas": "Relajación",
    "Cosmética - Champús": "Energía", "Cosmética - Cepillos": "Calma",
    "Cosmética - Secadores": "Calma", "Cosmética - Maquillaje": "Alegría",
    "Cosmética - Labiales": "Romanticismo", "Cosmética - Esmaltes": "Alegría",
    "Cosmética - Jabones": "Relajación", "Cosmética - Protectores solares": "Relajación",
    "Cosmética - Mascarillas": "Calma", "Cosmética - Sérums": "Calma",
    "Juguetes - General": "Alegría", "Juguetes - Puzzles": "Calma",
    "Juguetes - Juegos de mesa": "Alegría", "Juguetes - Muñecas": "Alegría",
    "Juguetes - Figuras": "Energía",
    "Ocio - Libros": "Calma", "Ocio - Cuadernos": "Calma",
    "Ocio - Bolígrafos": "Calma",
    "Deporte - Esterillas": "Calma", "Deporte - Botellas": "Energía",
    "Deporte - Bolsas": "Energía", "Deporte - Bicis": "Energía",
    "Deporte - Cascos": "Energía", "Deporte - Proteínas": "Energía",
    "Deporte - Suplementos": "Energía",
    "Mascotas - Camas": "Calma", "Mascotas - Comederos": "Calma",
    "Mascotas - Juguetes": "Alegría", "Mascotas - Comida perro": "Energía",
    "Mascotas - Comida gato": "Calma",
}


def rgb_a_hsv(r, g, b):
    """
    Convierte valores RGB (0-255) a HSV.
    H en [0, 360], S en [0, 1], V en [0, 1].
    """
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    if delta == 0:
        H = 0.0
    elif cmax == r:
        H = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        H = 60 * (((b - r) / delta) + 2)
    else:
        H = 60 * (((r - g) / delta) + 4)

    S = 0.0 if cmax == 0 else delta / cmax
    V = cmax
    return H, S, V


# =============================================================================
# SISTEMA DE ASIGNACIÓN EMOCIONAL v2 — Scoring Gaussiano
# Basado en: Valdez & Mehrabian (1994), Gilbert et al. (2016), Russell (1980)
# Sustituye el sistema de reglas jerárquicas por puntuaciones continuas en el
# espacio (L*, C*, H*). Elimina los "agujeros" del sistema anterior y produce
# una confianza_emocional interpretable como probabilidad softmax.
# =============================================================================

def derivar_CH(a, b):
    """Calcula Croma C* y ángulo de tono H* (radianes) a partir de a*, b*."""
    C = math.sqrt(a ** 2 + b ** 2)
    H = math.atan2(b, a)  # [-π, π]
    return C, H


def _score_gaussiano(valor, mu, sigma):
    """Gaussiana no normalizada: 1.0 en el centro, decrece hacia los extremos."""
    if sigma <= 0:
        return 1.0 if valor == mu else 0.0
    return math.exp(-0.5 * ((valor - mu) / sigma) ** 2)


def _distancia_angular(h1, h2):
    """Distancia circular mínima entre dos ángulos en radianes."""
    diff = abs(h1 - h2)
    return min(diff, 2 * math.pi - diff)


def asignar_emocion_v2(L, a, b, umbral_confianza=0.22):
    """
    Asigna emoción mediante scoring gaussiano ponderado en (L*, C*, H*).
    Devuelve (emocion, confianza, scores_dict).
    Si ninguna emoción supera umbral_confianza → "Neutro/Ambiguo".
    """
    C, H = derivar_CH(a, b)
    scores_raw = {}
    for emocion, params in CENTROIDES.items():
        mu_L, sigma_L, mu_C, sigma_C, mu_H, sigma_H, w_L, w_C, w_H = params
        s_L = _score_gaussiano(L, mu_L, sigma_L)
        s_C = _score_gaussiano(C, mu_C, sigma_C)
        dist_H = _distancia_angular(H, mu_H)
        s_H = math.exp(-0.5 * (dist_H / sigma_H) ** 2)
        scores_raw[emocion] = w_L * s_L + w_C * s_C + w_H * s_H

    total = sum(scores_raw.values())
    scores_prob = {e: v / total for e, v in scores_raw.items()} if total > 0 \
                  else {e: 1 / len(CENTROIDES) for e in CENTROIDES}

    emocion_ganadora = max(scores_prob, key=scores_prob.get)
    confianza = scores_prob[emocion_ganadora]
    if confianza < umbral_confianza:
        emocion_ganadora = "Neutro/Ambiguo"

    return emocion_ganadora, round(confianza, 4), scores_prob


def aplicar_emocion_v2(df, umbral=0.22):
    """
    Aplica asignar_emocion_v2 a todo el DataFrame.
    Añade: emocion, confianza_emocional y columnas score_* por emoción.
    """
    resultados = df.apply(
        lambda r: asignar_emocion_v2(r["mean_L"], r["mean_a"], r["mean_b"],
                                      umbral_confianza=umbral),
        axis=1
    )
    df = df.copy()
    df["emocion"]             = resultados.apply(lambda x: x[0])
    df["confianza_emocional"] = resultados.apply(lambda x: x[1])
    scores_df = resultados.apply(lambda x: pd.Series(x[2]))
    scores_df.columns = [f"score_{e.lower().replace('/', '_').replace(' ', '_')}"
                         for e in scores_df.columns]
    df = pd.concat([df, scores_df], axis=1)

    print(f"\n  Emociones asignadas (v2 gaussiana):")
    print(df["emocion"].value_counts().to_string())
    pct_ambiguo = (df["emocion"] == "Neutro/Ambiguo").mean() * 100
    print(f"  Neutros/Ambiguos: {pct_ambiguo:.1f}%  (umbral={umbral})")
    print(f"  Confianza media:  {df['confianza_emocional'].mean():.3f}")
    return df


def temperatura_color(a, b):
    """Clasifica el color como Cálido, Frío o Neutro."""
    if b > 8 or a > 5:
        return "Cálido"
    if b < -5 or a < -5:
        return "Frío"
    return "Neutro"


def ingenieria_del_dato():
    """
    Función principal de ingeniería del dato:
    - Carga el dataset combinado
    - Realiza limpieza y validación
    - Aplica transformaciones (HSV, emociones, variables de negocio)
    - Genera 15 gráficos EDA
    - Guarda el dataset final con emociones
    """
    print("\n" + "="*60)
    print("INGENIERÍA DEL DATO")
    print("="*60)

    # Leer dataset combinado
    if not os.path.exists(CSV_COMBINADO):
        print(f"ERROR: No se encontró {CSV_COMBINADO}. Ejecuta primero unificar_datasets().")
        return

    df = pd.read_csv(CSV_COMBINADO, encoding="utf-8-sig")
    print(f"\nDataset cargado: {len(df)} filas, {len(df.columns)} columnas")

    cols_color = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]

    # ------------------------------------------------------------------
    # 6.1 GRÁFICOS INICIALES (antes de cualquier limpieza)
    # ------------------------------------------------------------------
    print("\n[EDA] Generando gráficos iniciales...")

    # Gráfico 0a: Distribución por fuente
    fig, ax = plt.subplots(figsize=(8, 5))
    conteo_fuente = df["fuente"].value_counts()
    ax.bar(conteo_fuente.index, conteo_fuente.values, color=["#3498DB", "#E74C3C", "#2ECC71"])
    ax.set_title("Distribución de productos por fuente", fontsize=14)
    ax.set_xlabel("Fuente")
    ax.set_ylabel("Número de productos")
    for i, v in enumerate(conteo_fuente.values):
        ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "00a_distribucion_fuente.png"), dpi=150)
    plt.close()

    # Gráfico 0b: Top 15 categorías
    fig, ax = plt.subplots(figsize=(10, 7))
    top_cat = df["categoria"].value_counts().head(15)
    top_cat.sort_values().plot(kind="barh", ax=ax, color="#5DADE2")
    ax.set_title("Top 15 categorías con más productos", fontsize=14)
    ax.set_xlabel("Número de productos")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "00b_top15_categorias.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 6.2 LIMPIEZA Y VALIDACIÓN
    # ------------------------------------------------------------------
    print("\n[Limpieza] Iniciando limpieza y validación...")

    # Eliminar Moda de ABO (categorías que empiecen por "Moda -" de ABO)
    mask_moda = (df["fuente"] == "ABO") & (df["categoria"].str.startswith("Moda -", na=False))
    n_moda = mask_moda.sum()
    if n_moda > 0:
        df = df[~mask_moda].reset_index(drop=True)
        print(f"  Eliminadas {n_moda} filas de Moda en ABO.")

    # Nulos: si pct < 1% → dropna, si pct >= 1% → fillna con mediana
    for col in cols_color:
        pct_nulo = df[col].isna().mean() * 100
        if pct_nulo > 0:
            if pct_nulo < 1:
                df = df.dropna(subset=[col])
                print(f"  {col}: {pct_nulo:.2f}% nulos → eliminadas filas")
            else:
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
                print(f"  {col}: {pct_nulo:.2f}% nulos → imputados con mediana ({mediana:.4f})")

    # Duplicados por imagen_url
    n_antes = len(df)
    df = df.drop_duplicates(subset=["imagen_url"]).reset_index(drop=True)
    print(f"  Duplicados eliminados: {n_antes - len(df)}")

    # Validación de rangos teóricos
    rangos = {
        "mean_R": (0, 255), "mean_G": (0, 255), "mean_B": (0, 255),
        "mean_L": (0, 100), "mean_a": (-128, 128), "mean_b": (-128, 128),
        "contrast_L": (0, 100)
    }
    for col, (vmin, vmax) in rangos.items():
        n_out = ((df[col] < vmin) | (df[col] > vmax)).sum()
        if n_out > 0:
            df = df[(df[col] >= vmin) & (df[col] <= vmax)]
            print(f"  {col}: {n_out} valores fuera de rango [{vmin},{vmax}] eliminados")

    df = df.reset_index(drop=True)

    # Outliers: capping IQR × 1.5 con clip(), sin superar rangos teóricos
    L_antes = df["mean_L"].copy()
    for col, (vmin, vmax) in rangos.items():
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = max(Q1 - 1.5 * IQR, vmin)
        upper = min(Q3 + 1.5 * IQR, vmax)
        df[col] = df[col].clip(lower=lower, upper=upper)

    # Gráfico 1: Histograma L* antes/después del capping
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(L_antes, bins=50, alpha=0.6, color="#E74C3C", label="Antes del capping")
    ax.hist(df["mean_L"], bins=50, alpha=0.6, color="#2ECC71", label="Después del capping")
    ax.set_title("Distribución de L* antes y después del capping IQR", fontsize=14)
    ax.set_xlabel("L* (Luminosidad)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "01_histograma_L_capping.png"), dpi=150)
    plt.close()

    # Cap de 300 productos por categoría
    df = (df.groupby("categoria", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), 300), random_state=42))
            .reset_index(drop=True))
    print(f"  Después del cap por categoría: {len(df)} filas")

    # ------------------------------------------------------------------
    # 6.3 TRANSFORMACIONES
    # ------------------------------------------------------------------
    print("\n[Transformaciones] Aplicando transformaciones...")

    # Conversión a HSV
    hsv = df.apply(lambda r: rgb_a_hsv(r["mean_R"], r["mean_G"], r["mean_B"]), axis=1)
    df["hue"]        = hsv.apply(lambda x: round(x[0], 4))
    df["saturation"] = hsv.apply(lambda x: round(x[1], 4))
    df["value"]      = hsv.apply(lambda x: round(x[2], 4))

    # Asignación de emoción (sistema gaussiano v2)
    df = aplicar_emocion_v2(df, umbral=0.22)
    # Convertir confianza [0,1] a escala [0,100] para coherencia_emocional
    df["coherencia_emocional"] = (df["confianza_emocional"] * 100).round(2)

    # Variables de negocio
    df["temperatura_color"] = df.apply(
        lambda r: temperatura_color(r["mean_a"], r["mean_b"]), axis=1
    )
    df["luminosidad_cat"] = pd.cut(
        df["mean_L"], bins=[0, 40, 70, 100],
        labels=["Oscuro", "Medio", "Luminoso"]
    )
    df["saturacion_cat"] = pd.cut(
        df["saturation"], bins=[0, 0.25, 0.6, 1.0],
        labels=["Apagado", "Moderado", "Intenso"]
    )

    # Alineación emocional: si la emoción del producto coincide con la óptima de la categoría
    df["emocion_optima"] = df["categoria"].map(EMOCION_OPTIMA).fillna("Desconocida")
    df["alineacion_emocional"] = df["emocion"] == df["emocion_optima"]

    # Normalización Min-Max de las variables de color y HSV
    cols_normalizar = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b",
                       "contrast_L", "hue", "saturation", "value"]
    scaler = MinMaxScaler()
    datos_norm = scaler.fit_transform(df[cols_normalizar])
    for i, col in enumerate(cols_normalizar):
        df[f"{col}_norm"] = datos_norm[:, i].round(4)

    print(f"  Variables normalizadas añadidas (sufijo _norm)")

    # ------------------------------------------------------------------
    # 6.4 LOS 15 GRÁFICOS DE EDA
    # ------------------------------------------------------------------
    print("\n[EDA] Generando los 15 gráficos de análisis...")

    # — Gráfico 2: Distribución de las 8 emociones
    fig, ax = plt.subplots(figsize=(10, 6))
    conteo = df["emocion"].value_counts().reindex(ORDEN_EMOCIONES, fill_value=0)
    colores_barras = [COLORES_EMOCION[e] for e in ORDEN_EMOCIONES]
    bars = ax.bar(ORDEN_EMOCIONES, conteo.values, color=colores_barras, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, conteo.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Distribución de emociones en el dataset", fontsize=14)
    ax.set_xlabel("Emoción")
    ax.set_ylabel("Número de productos")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "02_distribucion_emociones.png"), dpi=150)
    plt.close()

    # — Gráficos 3, 4, 5: Boxplots de mean_L, mean_a, mean_b por emoción
    for num_graf, (col, titulo, unidad) in enumerate([
        ("mean_L", "Luminosidad (L*) por emoción", "L* [0-100]"),
        ("mean_a", "Componente a* (rojo-verde) por emoción", "a* [-128, 128]"),
        ("mean_b", "Componente b* (amarillo-azul) por emoción", "b* [-128, 128]"),
    ], start=3):
        fig, ax = plt.subplots(figsize=(12, 6))
        datos_box = [df[df["emocion"] == e][col].dropna().values for e in ORDEN_EMOCIONES]
        bp = ax.boxplot(datos_box, patch_artist=True, widths=0.6)
        for patch, emocion in zip(bp["boxes"], ORDEN_EMOCIONES):
            patch.set_facecolor(COLORES_EMOCION[emocion])
            patch.set_alpha(0.75)
        ax.set_xticklabels(ORDEN_EMOCIONES, rotation=15)
        ax.set_title(titulo, fontsize=14)
        ax.set_ylabel(unidad)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(CARPETA_GRAFICOS, f"0{num_graf}_boxplot_{col}.png"), dpi=150)
        plt.close()

    # — Gráfico 6: Matriz de correlación
    fig, ax = plt.subplots(figsize=(10, 8))
    cols_corr = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b",
                 "contrast_L", "hue", "saturation", "value"]
    corr = df[cols_corr].corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols_corr)))
    ax.set_yticks(range(len(cols_corr)))
    ax.set_xticklabels(cols_corr, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols_corr, fontsize=9)
    # Valores numéricos en cada celda
    for i in range(len(cols_corr)):
        for j in range(len(cols_corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(corr.values[i, j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Matriz de correlación de variables de color", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "06_correlacion.png"), dpi=150)
    plt.close()

    # — Gráfico 7: Paleta cromática real (franjas de color ordenadas por tono HSV)
    df_ordenado = df.sort_values("hue").reset_index(drop=True)
    n = len(df_ordenado)
    altura = max(1, n // 500)  # Franjas de altura adaptativa
    paleta = np.zeros((altura, n, 3), dtype=np.uint8)
    for i, row in df_ordenado.iterrows():
        paleta[:, i, 0] = int(row["mean_R"])
        paleta[:, i, 1] = int(row["mean_G"])
        paleta[:, i, 2] = int(row["mean_B"])
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.imshow(paleta, aspect="auto")
    ax.set_title("Paleta cromática real del dataset (ordenada por tono HSV)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "07_paleta_cromatica.png"), dpi=150)
    plt.close()

    # — Gráfico 8: Heatmap emoción × categoría (top 12 categorías)
    top12_cats = df["categoria"].value_counts().head(12).index.tolist()
    df_top = df[df["categoria"].isin(top12_cats)]
    pivot = df_top.groupby(["categoria", "emocion"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot_pct.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot_pct.columns)))
    ax.set_yticks(range(len(pivot_pct.index)))
    ax.set_xticklabels(pivot_pct.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(pivot_pct.index, fontsize=9)
    for i in range(len(pivot_pct.index)):
        for j in range(len(pivot_pct.columns)):
            val = pivot_pct.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="black" if val < 60 else "white")
    plt.colorbar(im, ax=ax, label="% de productos")
    ax.set_title("Distribución de emociones por categoría (top 12)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "08_heatmap_emocion_categoria.png"), dpi=150)
    plt.close()

    # — Gráfico 9: Scatter L* vs a* coloreado por emoción
    fig, ax = plt.subplots(figsize=(10, 7))
    for emocion in ORDEN_EMOCIONES:
        sub = df[df["emocion"] == emocion]
        ax.scatter(sub["mean_L"], sub["mean_a"],
                   c=COLORES_EMOCION[emocion], label=emocion,
                   alpha=0.4, s=15, edgecolors="none")
    ax.set_xlabel("L* (Luminosidad)")
    ax.set_ylabel("a* (Rojo-Verde)")
    ax.set_title("Dispersión L* vs a* por emoción", fontsize=14)
    ax.legend(markerscale=2, fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "09_scatter_L_a.png"), dpi=150)
    plt.close()

    # — Gráfico 10: Alineación emocional por categoría
    alin_cat = df.groupby("categoria")["alineacion_emocional"].mean() * 100
    alin_cat = alin_cat.sort_values()
    colores_alin = ["#E74C3C" if v < 10 else "#E67E22" if v < 25 else "#27AE60"
                    for v in alin_cat.values]
    fig, ax = plt.subplots(figsize=(12, max(6, len(alin_cat) // 3)))
    ax.barh(alin_cat.index, alin_cat.values, color=colores_alin)
    ax.axvline(10, color="#E74C3C", linestyle="--", alpha=0.7, label="10%")
    ax.axvline(25, color="#E67E22", linestyle="--", alpha=0.7, label="25%")
    ax.set_xlabel("% de productos con emoción alineada")
    ax.set_title("Alineación emocional por categoría", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "10_alineacion_emocional.png"), dpi=150)
    plt.close()

    # — Gráfico 11: Scatter L* vs b* por fuente con centroides anotados
    fuentes = df["fuente"].unique()
    colores_fuente = {"ABO": "#3498DB", "OpenFoodFacts": "#E74C3C", "Mahou": "#27AE60"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for fuente in fuentes:
        sub = df[df["fuente"] == fuente]
        color = colores_fuente.get(fuente, "#95A5A6")
        ax.scatter(sub["mean_L"], sub["mean_b"], c=color, label=fuente,
                   alpha=0.3, s=12, edgecolors="none")
        # Centroide de la fuente con anotación
        cx, cy = sub["mean_L"].mean(), sub["mean_b"].mean()
        ax.scatter(cx, cy, c=color, s=150, marker="*", edgecolors="black", zorder=5)
        ax.annotate(
            f"{fuente}\n(L*={cx:.1f}, b*={cy:.1f})",
            xy=(cx, cy), xytext=(5, 5), textcoords="offset points", fontsize=8
        )
    ax.set_xlabel("L* (Luminosidad)")
    ax.set_ylabel("b* (Amarillo-Azul)")
    ax.set_title("Dispersión L* vs b* por fuente con centroides", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "11_scatter_L_b_fuentes.png"), dpi=150)
    plt.close()

    # — Gráfico 12: Boxplot contrast_L por emoción con medias anotadas
    fig, ax = plt.subplots(figsize=(12, 6))
    datos_box = [df[df["emocion"] == e]["contrast_L"].dropna().values for e in ORDEN_EMOCIONES]
    bp = ax.boxplot(datos_box, patch_artist=True, widths=0.6)
    for patch, emocion in zip(bp["boxes"], ORDEN_EMOCIONES):
        patch.set_facecolor(COLORES_EMOCION[emocion])
        patch.set_alpha(0.75)
    # Anotar medias encima de cada caja
    for i, (emocion, datos) in enumerate(zip(ORDEN_EMOCIONES, datos_box)):
        if len(datos) > 0:
            media = np.mean(datos)
            ax.text(i + 1, media + 0.3, f"{media:.1f}", ha="center", fontsize=8, color="#2C3E50")
    ax.set_xticklabels(ORDEN_EMOCIONES, rotation=15)
    ax.set_title("Contraste (L*) por emoción", fontsize=14)
    ax.set_ylabel("contrast_L (desv. típica de L*)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "12_boxplot_contrast_L.png"), dpi=150)
    plt.close()

    # — Gráfico 13: Boxplots L*, a*, b*, contrast_L entre las 3 fuentes (2x2)
    fuentes_lista = df["fuente"].unique().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, titulo) in zip(axes.flatten(), [
        ("mean_L", "L* (Luminosidad)"), ("mean_a", "a* (Rojo-Verde)"),
        ("mean_b", "b* (Amarillo-Azul)"), ("contrast_L", "Contraste L*")
    ]):
        datos_box = [df[df["fuente"] == f][col].dropna().values for f in fuentes_lista]
        bp = ax.boxplot(datos_box, patch_artist=True)
        colores_box = [colores_fuente.get(f, "#95A5A6") for f in fuentes_lista]
        for patch, c in zip(bp["boxes"], colores_box):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticklabels(fuentes_lista, rotation=10)
        ax.set_title(titulo)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Comparativa de variables de color por fuente", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "13_boxplot_fuentes_2x2.png"), dpi=150)
    plt.close()

    # — Gráfico 14: Histogramas superpuestos por fuente (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flatten(), ["mean_L", "mean_a", "mean_b", "contrast_L"]):
        for fuente in fuentes_lista:
            sub = df[df["fuente"] == fuente][col].dropna()
            ax.hist(sub, bins=40, alpha=0.5, label=fuente,
                    color=colores_fuente.get(fuente, "#95A5A6"), edgecolor="none")
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Histogramas de variables de color por fuente", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "14_histogramas_fuentes_2x2.png"), dpi=150)
    plt.close()

    # — Gráfico 15: Top 12 categorías por fuente (3 subplots horizontales)
    n_fuentes = len(fuentes_lista)
    fig, axes = plt.subplots(1, n_fuentes, figsize=(6 * n_fuentes, 8))
    if n_fuentes == 1:
        axes = [axes]
    for ax, fuente in zip(axes, fuentes_lista):
        sub = df[df["fuente"] == fuente]["categoria"].value_counts().head(12)
        sub.sort_values().plot(kind="barh", ax=ax, color=colores_fuente.get(fuente, "#95A5A6"))
        ax.set_title(f"Top 12 categorías — {fuente}", fontsize=11)
        ax.set_xlabel("Nº de productos")
        ax.tick_params(axis="y", labelsize=8)
    plt.suptitle("Top 12 categorías por fuente de datos", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, "15_top12_categorias_fuente.png"), dpi=150)
    plt.close()

    print(f"\n[EDA] ✓ 15 gráficos guardados en: {CARPETA_GRAFICOS}")

    # ------------------------------------------------------------------
    # 6.5 GUARDAR DATASET FINAL
    # ------------------------------------------------------------------
    df.to_csv(CSV_FINAL, index=False, encoding="utf-8-sig")
    print(f"\n✓ Dataset final guardado: {CSV_FINAL}")

    # Resumen final
    n_alineados = df["alineacion_emocional"].sum()
    pct_alineados = n_alineados / len(df) * 100 if len(df) > 0 else 0
    coherencia_media = df["coherencia_emocional"].mean() if "coherencia_emocional" in df.columns else 0

    print("\n" + "="*60)
    print("RESUMEN FINAL DEL DATASET")
    print("="*60)
    print(f"  Total de filas:         {len(df):,}")
    print(f"  Total de columnas:      {len(df.columns)}")
    print(f"  Emociones únicas:       {df['emocion'].nunique()}")
    print(f"  Coherencia media:       {coherencia_media:.2f}/100")
    print(f"  % alineación emocional: {pct_alineados:.1f}%")
    print(f"  Distribución emociones:")
    for emocion, count in df["emocion"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {emocion:<15}: {count:>5} ({pct:.1f}%)")
    print("="*60)

    return df


# =============================================================================
# PARTE 7 — BLOQUE __main__
# =============================================================================

if __name__ == "__main__":
    # Controlar qué scrapers ejecutar
    EJECUTAR_MAHOU = True
    EJECUTAR_ABO   = True
    EJECUTAR_OFF   = True

    print("="*60)
    print("TFG — Análisis de Color en Productos de Consumo")
    print("Universidad Francisco de Vitoria + SAS")
    print("="*60)

    if EJECUTAR_MAHOU:
        scraper_mahou()

    if EJECUTAR_ABO:
        scraper_abo()

    if EJECUTAR_OFF:
        scraper_openfoodfacts()

    unificar_datasets()
    ingenieria_del_dato()

    print("\n✓ ¡Proceso completado con éxito!")