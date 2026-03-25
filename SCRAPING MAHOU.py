# ============================================================
# SCRAPER - TIENDA MAHOU SAN MIGUEL (versión Selenium)
# Web: https://www.mahou-sanmiguel.com/tienda
# Plataforma: Salesforce Commerce Cloud (Demandware)
#
# Instalar dependencias:
#   pip install selenium webdriver-manager pillow scikit-image tqdm pandas
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

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ── Configuración ──────────────────────────────────────────
BASE_URL   = "https://www.mahou-sanmiguel.com"
DELAY      = 3
MAX_PROD   = 300
OUTPUT_DIR = "mahou_data"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "imagenes")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "dataset_mahou.csv")
os.makedirs(IMAGES_DIR, exist_ok=True)

HEADERS_REQUESTS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Cuando haces requests.get(url), algunas webs detectan si no pareces un navegador real y pueden bloquearte o devolverte contenido raro. 
# Entonces el script hace que requests “parezca” Chrome normal o Safari o Mozilla...

# ── Categorías verificadas desde el HTML de la tienda ─────      Diccionario montado por variable que se entiende : su ruta en internet sirve para luego 
# Demandware pagina con ?start=N&sz=12                            recorrer todas las categorias en bucle
CATEGORIAS = {
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

ITEMS_POR_PAGINA = 12

# ── Driver Selenium ────────────────────────────────────────

def crear_driver(): # Esto genera el navegador Chrome que selenium va a controlar. Selenium abre un navegador real por detrás. 
    opciones = Options()  # opciones es la variable que guarda el objeto Options(). OPtions() sirve para configurar Chrome
    opciones.add_argument("--headless=new")  # Indica que se ejecute Chrome sin abrir ventana visible
    opciones.add_argument("--no-sandbox") # Esto es una opción de seguridad que evita ciertos problemas cuando se ejecuta en servidores Linux.
    opciones.add_argument("--disable-dev-shm-usage") # Otra opción para evitar errores relacionados con memoria compartida, sobre todo en entornos tipo Docker o Linux.
    opciones.add_argument("--window-size=1920,1080") # Esto define el tamaño de la ventana del navegador. Aunque esté en modo headless el navegador tiene tamaño
    opciones.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ) # Esto le dice al navegador que se identifique como un Chrome normal. El User-Agent es como el “DNI” que el navegador envía a la web. 
      # Sin esto, algunas webs pueden detectar que es automatización.
    opciones.add_argument("--lang=es-ES") # Le dice al navegador que use idioma español.
    opciones.add_experimental_option("excludeSwitches", ["enable-logging"]) # Esto evita que Chrome muestre ciertos mensajes molestos en consola.
    servicio = Service(ChromeDriverManager().install()) #Busca la versión correcta del ChromeDriver. Si no está instalada, la descarga automáticamente.
                                                        # Service() Crea un objeto servicio con esa ruta.
    return webdriver.Chrome(service=servicio, options=opciones)
    # webdriver.Chrome(...) crea el navegador real.

def aceptar_cookies(driver): # Intenta encontrar el botón de “Aceptar cookies” en la página y hacer clic en él. SI no lo encuentra no pasa nada y sigue
    try:
        boton = WebDriverWait(driver, 6).until( # Si la condición se cumple antes de 6 segundos, sigue. Si pasan 6 segundos y no se cumple, lanza error.
            EC.element_to_be_clickable( # EC = Expected condition  “Que exista el elemento y que se pueda hacer clic en él”.
                (By.XPATH, "//button[contains(translate(., 'ACEPTARCOOKIES', 'aceptarcookies'), 'acepto')]")
            )
        ) # Esto es un truco para que funcione aunque el botón diga: Acepto, ACEPTO, acepto etc. 
        boton.click() # Hace clic real sobre el botón.
        time.sleep(1) # Espera 1 segundo
    except Exception:
        pass

# ── Scraping ───────────────────────────────────────────────

def extraer_urls_categoria(driver, path_categoria): # Lo que hace es entrar en una categoría de la tienda, revisa todas las páginas, extrae las URLS de todos los 
                                                    # productos y devuelve una lista con todas las URLS. 
    """
    Demandware pagina con ?start=0&sz=12, start=12, start=24...
    Los productos tienen URLs /tienda/p/nombre.html
    """
    urls_producto = []
    start = 0

    while True:
        url = f"{BASE_URL}{path_categoria}?start={start}&sz={ITEMS_POR_PAGINA}" # driver es el navegador Selenium
        driver.get(url) # carga la página
        time.sleep(DELAY) # espera 3 segundos

        # Scroll para cargar lazy loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # ejecuta JavaScript dentro del navegador y se desplaza hasta el final de la página
        time.sleep(1.5)

        elementos = driver.find_elements(By.CSS_SELECTOR, "a[href]") # Devuelve una lista con todos los elementos "a" que tengan un atributo "href"
        nuevas = []
        for el in elementos:
            href = el.get_attribute("href") or "" # Obtiene la URL. Si es None usa un string vacío 
            if "/tienda/p/" in href and href not in urls_producto and href not in nuevas: # Solo queremos URLs que contengan /tienda/p/. y que no estén duplicadas
                nuevas.append(href)

        if not nuevas:
            break

        urls_producto.extend(nuevas) # Añadir las nuevas url a la lista 
        start += ITEMS_POR_PAGINA

        if len(nuevas) < ITEMS_POR_PAGINA: # Si esta página tiene menos de 12 productos es la última asi que para
            break
        if len(urls_producto) >= MAX_PROD:  # Si ya llegó al límite máximo de urls para
            break

    return list(dict.fromkeys(urls_producto)) # Usa un diccionario usando las urls como claves. Un diccionario no permite claves repetidas. Luego a lista
# otra vez, de esta manera se omite y evita duplicados. 


# Entra en la página de un producto y devuelve un diccionario con: nombre, categoría, url de imagen, url del producto. Si falla o no hay imagen, devuelve None.
def extraer_datos_producto(driver, url_producto, nombre_categoria):
    try:
        driver.get(url_producto)
        time.sleep(DELAY)

        nombre = None
        for selector in ["h1.product-name", "h1.product-detail-name",
                          "h1[itemprop='name']", "h1"]:  # Esto es una lista de selectores CSS. ¿Por qué varios? Porque el HTML puede cambiar según el producto
                          #  o la web, entonces el script prueba varias “formas de encontrar el título”.
            try:
                el = driver.find_element(By.CSS_SELECTOR, selector)
                nombre = el.text.strip()
                if nombre:
                    break
            except Exception:
                continue
        if not nombre:
            nombre = "Sin nombre"  # Dentro del bucle, intenta encontrar el elemento. 

        imagen_url = None

        # og:image con alta resolución — eliminamos parámetros de redimensionado. SI falla sigue
        try:
            og = driver.find_element(By.XPATH, "//meta[@property='og:image']")
            imagen_url = og.get_attribute("content")
            if imagen_url:
                imagen_url = re.sub(r'\?.*$', '', imagen_url)
        except Exception:
            pass

        # Fallback: imagen del bloque de producto. Solo entra aquí si no encontró og:image en el bloque anterior. Busca una etiqueta <img>
        # Esto intenta evitar coger iconos, logos o imágenes raras. Se queda con URLs que “parecen” de la tienda.
        if not imagen_url:
            for selector in [
                ".product-primary-image img",
                ".product-image img",
                "img.primary-image",
                ".product-images img"
            ]:
                try:
                    img = driver.find_element(By.CSS_SELECTOR, selector)
                    src = (img.get_attribute("data-src") or
                           img.get_attribute("src") or "")
                    if "demandware" in src or "mahou" in src:
                        imagen_url = re.sub(r'\?.*$', '', src)
                        break
                except Exception:
                    continue

        if not imagen_url: # SI no hay imagen se descarta el producto 
            return None
        if not imagen_url.startswith("http"):
            imagen_url = BASE_URL + imagen_url

        return {   # Esto es lo que devuelve de manera estructurada. 
            "nombre":       nombre,
            "categoria":    nombre_categoria,
            "imagen_url":   imagen_url,
            "url_producto": url_producto,
        }
    except Exception as e:
        print(f"    ⚠ Error en {url_producto}: {e}")
        return None

# ── Color ──────────────────────────────────────────────────

def descargar_imagen(imagen_url, nombre_archivo):
    ruta = os.path.join(IMAGES_DIR, nombre_archivo)
    if os.path.exists(ruta):
        return ruta
    try:
        resp = requests.get(imagen_url, headers=HEADERS_REQUESTS, timeout=20)
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

# ── Main ───────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("SCRAPER TIENDA MAHOU SAN MIGUEL (Selenium)")
    print("=" * 55)
    print("Iniciando Chrome en modo invisible...")

    driver = crear_driver()

    try:
        driver.get(BASE_URL + "/tienda")
        time.sleep(2)
        aceptar_cookies(driver)

        print("\n[1/3] Recopilando URLs de productos...")
        todos_los_productos = []

        for nombre_cat, path_cat in CATEGORIAS.items():
            print(f"  Categoría: {nombre_cat}")
            urls = extraer_urls_categoria(driver, path_cat)
            print(f"    → {len(urls)} productos encontrados")
            for url in urls:
                todos_los_productos.append({"url": url, "categoria": nombre_cat})
            if len(todos_los_productos) >= MAX_PROD:
                todos_los_productos = todos_los_productos[:MAX_PROD]
                break

        vistos = set()
        todos_unicos = []
        for item in todos_los_productos:
            if item["url"] not in vistos:
                vistos.add(item["url"])
                todos_unicos.append(item)
        todos_los_productos = todos_unicos
        print(f"\n  Total URLs únicas: {len(todos_los_productos)}")

        print("\n[2/3] Extrayendo nombre e imagen de cada producto...")
        registros = []
        for item in tqdm(todos_los_productos, desc="Scrapeando", unit="prod"):
            datos = extraer_datos_producto(driver, item["url"], item["categoria"])
            if datos:
                registros.append(datos)
        print(f"  → {len(registros)} productos con datos válidos")

    finally:
        driver.quit()
        print("  Chrome cerrado.")

    print("\n[3/3] Descargando imágenes y calculando RGB + CIELAB...")
    filas_finales = []

    for i, reg in enumerate(tqdm(registros, desc="Procesando", unit="img")):
        nombre_limpio  = re.sub(r'[^a-zA-Z0-9]', '_', reg["nombre"])[:40]
        nombre_archivo = f"mahou_{i:04d}_{nombre_limpio}.jpg"
        ruta = descargar_imagen(reg["imagen_url"], nombre_archivo)
        if ruta is None:
            continue
        try:
            mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = calcular_color(ruta)
        except Exception:
            continue
        filas_finales.append({
            "fuente":       "Mahou San Miguel",
            "categoria":    reg["categoria"],
            "nombre":       reg["nombre"],
            "imagen_local": nombre_archivo,
            "imagen_url":   reg["imagen_url"],
            "url_producto": reg["url_producto"],
            "mean_R":       round(mean_R, 4),
            "mean_G":       round(mean_G, 4),
            "mean_B":       round(mean_B, 4),
            "mean_L":       round(mean_L, 4),
            "mean_a":       round(mean_a, 4),
            "mean_b":       round(mean_b, 4),
            "contrast_L":   round(contrast_L, 4),
        })

    df = pd.DataFrame(filas_finales)
    if len(df) == 0:
        print("\n⚠ No se procesó ningún producto.")
        print("  Posible causa: la tienda bloquea el headless browser.")
        print("  Prueba a poner headless=False para ver qué pasa en el navegador.")
    else:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n✓ Dataset guardado en: {OUTPUT_CSV}")
        print(f"  Productos: {len(df)}")
        print(f"  Categorías: {df['categoria'].nunique()}")
        print("\nDistribución por categoría:")
        print(df["categoria"].value_counts().to_string())


if __name__ == "__main__":
    main()