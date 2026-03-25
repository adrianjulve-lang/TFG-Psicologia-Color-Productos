# ============================================================
# TFG - ADRIÁN JULVE NAVARRO
# Script completo: scraping + limpieza + análisis de color
# ============================================================

import os
import time
import re
import io
import gzip
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

CARPETA = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING"

CSV_MAHOU  = CARPETA + r"\mahou_data\dataset_mahou.csv"
CSV_ABO    = CARPETA + r"\abo_data\dataset_abo.csv"
CSV_OFF    = CARPETA + r"\openfoodfacts_data\dataset_openfoodfacts.csv"
CSV_FINAL  = CARPETA + r"\dataset_final_combinado.csv"
CSV_LIMPIO = CARPETA + r"\Dataset combinado sin emociones.csv"

CARPETA_GRAFICOS = CARPETA + r"\graficos"
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)


# ============================================================
# FUNCIONES DE COLOR (compartidas por los tres scrapers)
# ============================================================

def descargar_imagen(imagen_url, ruta_destino, headers): #Descarga una imagen de internet y la guarda en el ordenador. SI ya estaba descargada, no la vuelve a descargar
    if os.path.exists(ruta_destino): # Comprueba si el archivo ya está en el ordenador. Si es así, devuelve la ruta de destino y sale de la función
        return ruta_destino
    try:
        resp = requests.get(imagen_url, headers=headers, timeout=20) # Es la descarga real, hace una petición a internet. Si en 20 segundos no hay respuesta, sigue
        img  = Image.open(io.BytesIO(resp.content)).convert("RGB") # Convierte los bytes que se han descargado y lo convierte en un formato que pueda leer Python 
        # Además abre la imagen y la convierte en RGB por si viene en RGBA o blanco y negro 
        img.save(ruta_destino, "JPEG", quality=90) # Giarda la imagen en la ruta que se da con formato JPEG y calidad de 90 (0 - 100) para que se vea bien y no ocupe mucho
        return ruta_destino
    except Exception:
        return None


def calcular_color(ruta): # Recobe una imagen, la descompone pixel por pixel y devuelve 7 numeros que describen el color de esa imagen
    img = Image.open(ruta).convert("RGB") # Abre la imagen y la ocnvierte en RGB para asegurarnos de que está en ese formato
    arr = np.asarray(img, dtype=np.float32) # Convierte la imagen en una matriz de números usando NumPy. Una imagen digital no es más que una tabla de números 
    # SI tienes una imagen de 100 x 100 pixesles, tu matriz será de (100,100,3). Ese 3 es por las capas, cada una pertenece a R, G y B corresponfientemente. 
    mean_R = float(arr[:, :, 0].mean()) # Todas las filas (:) y todas las columnas (:) y la capa 0 (R). Es una tabla con el valor rojo de cada pixel y .mean calcula 
    # la media de todos esos valores
    mean_G = float(arr[:, :, 1].mean())
    mean_B = float(arr[:, :, 2].mean())
    lab        = skcolor.rgb2lab(arr / 255.0) # Primero arr / 255.0: divide todos los valores de la matriz entre 255. Esto es porque la función de conversión 
    # espera valores entre 0 y 1, no entre 0 y 255. Es simplemente normalizar la escala. Luego skcolor.rgb2lab(...): convierte la imagen del espacio de color RGB al 
    # espacio CIELAB. El resultado lab es otra matriz del mismo tamaño (100, 100, 3) pero ahora las tres capas son L*, a* y b* en vez de R, G, B.
    mean_L     = float(lab[:, :, 0].mean()) # Luminosidad
    mean_a     = float(lab[:, :, 1].mean()) # rojo - verde 
    mean_b_val = float(lab[:, :, 2].mean()) # amarillo - azul
    contrast_L = float(lab[:, :, 0].std()) # La desviación típica mide cuánto varían los valores entre sí. Si todos los píxeles tienen L* parecido (imagen uniforme), 
    # el resultado es bajo. Si hay píxeles muy oscuros y muy claros mezclados (imagen con mucho contraste), el resultado es alto.
    return mean_R, mean_G, mean_B, mean_L, mean_a, mean_b_val, contrast_L


# ============================================================
# PARTE 1 — MAHOU SAN MIGUEL (Selenium)
# ============================================================

def scraper_mahou():
    print("\n" + "="*55)
    print("SCRAPER MAHOU SAN MIGUEL")
    print("="*55) # Banner para que quede bonito en el terminal, insignificante para la ejecución del escraping

    HEADERS_MAHOU = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"} # Diccionario que se usa para 
    # que la web piense que es un navegador de Chrome normal y no lo bloquee. 
    BASE_URL = "https://www.mahou-sanmiguel.com" # Dirección web de la que necesitamos la información
    DELAY    = 3 # Pausa de 3 segundos entre petición y petición para no saturar a los servidores y ser responsable
    IMGS_DIR = CARPETA + r"\mahou_data\imagenes" # Lugar donde se van a guardar las imágenes descargadas. 

    CATEGORIAS = { # Se crea un diccionario en el que la clave es el nombre legible de la categoría y el valor la ruta en internet
        # Más adelante el código recorre este diccionario y combina BASE_URL con cada ruta para construir la URL completa. 
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

    # Abrir Chrome en modo invisible
    opciones = Options() # Crea un objeto de configuración para Chrome. Cada argument se añadirá a esta opción
    opciones.add_argument("--headless=new") # Chrome se abre sin ventana visible, es decir, funciona en segundo plano
    opciones.add_argument("--no-sandbox") # No sandbox desactiva la capa de seguridad que a veces da problemas en algunos sistemas
    opciones.add_argument("--disable-dev-shm-usage") # Evita errores de memoria en Linux o Docker. No es estrictamente necesario, pero lo incuyo por si acaso. 
    opciones.add_argument("--window-size=1920,1080") # Esto determina el tamaño de la ventana virtual que se abrirá para el escraping
    opciones.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36") # Hace que Chrome se identifique
    # como un navegador normal
    opciones.add_argument("--lang=es-ES") # Se dicta que se utilice el español en la página web
    opciones.add_experimental_option("excludeSwitches", ["enable-logging"]) # Lo que hace esto básicamente es eliminar mensajes molestos de Chrome en la consola. 
    servicio = Service(ChromeDriverManager().install()) # Busca en tu ordenador el programa de ChromeDriver que Selenium necesita para controlar Chrome.
    # Si no está instalado o está desactualizado, lo descarga automáticamente. 
    driver   = webdriver.Chrome(service=servicio, options=opciones) # Esta línea es la que abre Chrome de verdad con toda la configuración hecha anteriormente. 
    # El resultado se guarda en "driver" que es el objeto con el que se controlará el navegador más adelante

    def aceptar_cookies(driver): # Recibe en la función driver que es el navegador que hemos configurado y arrancado justo arriba
        try:
            boton = WebDriverWait(driver, 6).until( # Espera máximo 6 segundos a que se cumpla una condición
                EC.element_to_be_clickable((By.XPATH, "//button[contains(translate(., 'ACEPTARCOOKIES', 'aceptarcookies'), 'acepto')]"))
            ) # La condición a cumplirse es la siguiente: "Ec.element_to_be_clickable" dice que se espere hasta que el elemtno exista en la página y se pueda
            # hacer click en él. # XPath es es un lenguaje para encontrar elementos dentro del html de una página. 
            # En resumen, este código dice que se encuentre cualquier botón cuyo texto contenga "acepto" sin importar las mayúsculas. 
            boton.click() # Una vez encontrado el botón, haz click en él. 
            time.sleep(1) # Espera 1 segundo después de hacer click
        except Exception:
            pass

    def extraer_urls_categoria(driver, path_categoria): # Esta función entra en una categoría de la tienda de Mahou, va pasando página por página y recoge las URLS
        # de todos los productos que encuentra. Recibe dos cosas, driver (el navegador Chrome) y path_categoría que es la lista de categorías de antes. 
        urls_producto = [] # Esto es una lista vacía donde se irán acumulando todas las URLS encontradas. 
        start = 0 # COmo la página de Mahou efunciona con páginas, start = 0 es la primera página, start = 12 es la segunda etc. 
        while True:
            url = f"{BASE_URL}{path_categoria}?start={start}&sz=12" # Construye la URL completa de la página actual. La `f` delante de las comillas indica que es un 
            # **f-string**, que permite meter variables dentro del texto usando llaves {}. El ?start=0&sz=12 es la parte de las páginas. start indica desde qué producto
            #  empezar y sz=12 indica cuántos mostrar por página
            driver.get(url)
            time.sleep(DELAY) # El navegador va a esa url y espera el delay que eran 3 segundos que definimos antes para que la página cargue antes de entrar a ella
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # Ejecuta JavaScript dentro del navegador para hacer scroll hasta el final de la 
            # página. Muchas webs modernas usan lazy loading , las imágenes y elementos solo se cargan cuando llegas a ellos con el scroll. Si no hicieras esto, algunos 
            # productos podrían no aparecer en el HTML. 
            time.sleep(1.5) # Espera 1.5 segundos para que los elementos que acaban de cargarse estén disponibles.
            elementos = driver.find_elements(By.CSS_SELECTOR, "a[href]") # Busca en la web todos los elementos que sean un enlace a con atributo
            nuevas = []
            for el in elementos:
                href = el.get_attribute("href") or "" # Extrae la url del enlace. Si no hay URL en el resultado, usa un string vacío para evitar errores
                if "/tienda/p/" in href and href not in urls_producto and href not in nuevas:
                    nuevas.append(href) # Si la URL contiene /tienda/p, que es el patrón de las páginas de Mahou y no está en la lista, métela
            if not nuevas:
                break
            urls_producto.extend(nuevas) #Añade todas las URLS sacadas en este bucle a la lista general de URLS. 
            start += 12 # Avanza al siguiente grupo de 12 para seguir con el bucl3. 
            if len(nuevas) < 12 or len(urls_producto) >= 300: # Si esta página tenía menos de 12 productos, es la última página de la categoría o si ya tenemos 300 para.
                break
        return list(dict.fromkeys(urls_producto)) # crea un diccionario con las urls y como el diccionario no permite duplicados, es una manera alternativa de checkear
    # Resumen esquemático: Empieza en la página 1 -> Carga la página en Chrome -> Hace scroll hasta abajo -> Busca todos los enlaces de la página -> Filtra solo los 
    # que son de producto -> SI no hay productos nuevos, para si los hay los añade a la lista y avanza a la isguiente página -> Página incompleto ya hay 300 urls? Para 

    def extraer_datos_producto(driver, url_producto, nombre_categoria): # Esta función entra en la página de un producto concreto y extrae su nombre y la URL de su imagen.
    # Recibe el navegador (driver), la URL del producto y el nombre de la categoría a la que pertenece.
        try:
            driver.get(url_producto) # El navegador va a la página del producto y espera 3 segundos a que cargue la categoría
            time.sleep(DELAY)
            nombre = None # Empieza con un nombre vacío, si no encuentra nada, quedará como None. 
            for selector in ["h1.product-name", "h1.product-detail-name", "h1[itemprop='name']", "h1"]: # Prueba varios selectores CSS para encontrar el título del 
                # producto. Prueba varios porque el HTML puede cambiar según el producto. Va de más específico a más genérico.
                try:
                    el = driver.find_element(By.CSS_SELECTOR, selector) # Intenta encontrar el titulo del producto y prueba varios porque el HTML puede cambiar
                    nombre = el.text.strip() # Extrae el texto del elemento y elimina espacios sobrantes. 
                    if nombre:
                        break # Si encuentra algo, break para el bucle, si no sigue
                except Exception:
                    continue
            if not nombre:
                nombre = "Sin nombre" # Si después de probar los cuatro selectores no encuentra nada, lo pone "Sin nombre" para no dejarlo vacío. 
            imagen_url = None
            try:
                og = driver.find_element(By.XPATH, "//meta[@property='og:image']") #Busca la etiqueta meta og:image en el HTML. Es una etiqueta especial que las webs
                # usan para indicar la imagen principal del contenido. Suele tener la mejor calidad.
                imagen_url = og.get_attribute("content") # Extrae la URL de la imagen de esa etiqueta
                if imagen_url:
                    imagen_url = re.sub(r'\?.*$', '', imagen_url) # Elimina los parámetros que vengan detrás de ?. Por ejemplo: "foto.jpg?width=200&height=200" se queda
                    # en "foto.jpg". 
            except Exception:
                pass # Si no encuentra la url, pasa al método alternativo de abajo. 
            if not imagen_url: # Si el método og:image no funcionó, prueba buscar directamente una etiqueta img en la página.
                for selector in [".product-primary-image img", ".product-image img", "img.primary-image", ".product-images img"]:
                    try:
                        img = driver.find_element(By.CSS_SELECTOR, selector) # Busca el elemento imagen con ese selector.
                        src = (img.get_attribute("data-src") or img.get_attribute("src") or "") # intenta primero data-src porque algunas webs guardan ahí la URL real 
                        # para el lazy loading, y si no existe prueba src. El or "" evita errores si ambos son None.
                        if "demandware" in src or "mahou" in src: # filtra para quedarse solo con imágenes que sean claramente del servidor de Mahou y no iconos o logos aleatorios. 
                            imagen_url = re.sub(r'\?.*$', '', src) # Igual que antes, elimina los parámetros del ? para obtener la imagen original.
                            break
                    except Exception:
                        continue
            if not imagen_url:
                return None # Si después de los dos métodos no hay imagen, descarta el producto completamente porque no nos sirve 
            if not imagen_url.startswith("http"):
                imagen_url = BASE_URL + imagen_url # Si la url de la imagen no empieza con "http" significa que es una ruta relativa. POr ello le añade BASE_URL
                # para comvertirla en una ruta permitida
            return {"nombre": nombre, "categoria": nombre_categoria, "imagen_url": imagen_url, "url_producto": url_producto} # Si todo va bien devuelve un diccionario
        # con los cuatro datos del producto. Se usa un diccionario porque así es más fácil acceder a cada dayo por su nombre más adelante
        except Exception as e:
            print(f"    Error en {url_producto}: {e}") # Si algo salió muy mal, imprime el error para saber qué producto falló y devuelve None para que el programa 
            # continúe con el siguiente sin romperse.
            return None

    try:
        driver.get(BASE_URL + "/tienda")
        time.sleep(2)
        aceptar_cookies(driver) # Abre la tienda de Mahou, espera 2 segundos y acepta las cookies llamando a la función creada anteriormente

        print("\n[1/3] Recopilando URLs de productos...")
        todos = []
        for nombre_cat, path_cat in CATEGORIAS.items(): # Recorre el diccionario de categorías de antes. En cada vuelta, nombre_cat es el nombre
            print(f"  {nombre_cat}...", end=" ", flush=True) # Imprime el nombre de la categoría sin salto de línea para que el número de productos quede en la misma línea.
            urls = extraer_urls_categoria(driver, path_cat) # Llama a la función anterior para obtener todas las URLS de esta categoría
            print(f"{len(urls)} productos") # Imprime cuántos productos encontró en esa categoría
            for url in urls:
                todos.append({"url": url, "categoria": nombre_cat}) # Añade cada URL a la lista general junto con su categoría

        # Eliminar duplicados
        vistos = set() # Un set es como una lista pero no permite elementos repetidos. Se usa para llevar registro de URLs ya vistas.
        todos_unicos = []
        for item in todos:
            if item["url"] not in vistos: # Si la url no la hemos visto antes le añade al set de vistos y el producto a la lista de únicos. 
                vistos.add(item["url"])
                todos_unicos.append(item)

        print(f"\n[2/3] Extrayendo datos de {len(todos_unicos)} productos...")
        registros = [] # Lista donde se guardarán los datos completos de cada producto (nombre, imagen, categoría).
        for item in tqdm(todos_unicos, desc="Scrapeando", unit="prod"): # tqdm muestra una barra de progreso en la consola para saber cuánto queda
            datos = extraer_datos_producto(driver, item["url"], item["categoria"]) # Para cada url única llama a la función de extraer datos de producto 
            if datos:
                registros.append(datos) # Si datos no es None, lo añade a registros. 

    finally:
        driver.quit() # Cuando acaba cierra Chrome para no ocupar memoria

    print(f"\n[3/3] Descargando imágenes y calculando color...")
    filas = [] # Lista donde se guardará una fila completa por cada producto con todos sus datos y variables de color.
    for i, reg in enumerate(tqdm(registros, desc="Procesando", unit="img")): # enumerate añade un número i a cada elemento del bucle. tqdm pone la barra de progreso.
        nombre_limpio  = re.sub(r'[^a-zA-Z0-9]', '_', reg["nombre"])[:40] # Limpia el nombre del producto para usarlo como nombre de archivo.
        # re.sub sustituye cualquier carácter que NO sea letra o número por un guion bajo. [:40] lo recorta a 40 caracteres máximo.
        nombre_archivo = f"mahou_{i:04d}_{nombre_limpio}.jpg" # Construye el nombre del archivo. {i:04d} formatea el número con 4 dígitos: 0001, 0002, etc.
        ruta = descargar_imagen(reg["imagen_url"], os.path.join(IMGS_DIR, nombre_archivo), HEADERS_MAHOU) # Descarga la imagen llamando a la función de antes.
        if ruta is None:
            continue # Si la descarga falló salta este punto y pasa al siguiente
        try:
            mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = calcular_color(ruta) # Calcula las 7 variables de color llamando a la función de antes.
        except Exception:
            continue # Si el cálculo de color falla salta este producto. 
        filas.append({ # Esto Añade una fila a la lista con todos los datos del producto.
            "fuente":     "Mahou San Miguel",
            "categoria":  reg["categoria"],
            "nombre":     reg["nombre"],
            "imagen_url": reg["imagen_url"],
            "mean_R":     round(mean_R, 4), "mean_G": round(mean_G, 4), "mean_B": round(mean_B, 4),
            "mean_L":     round(mean_L, 4), "mean_a": round(mean_a, 4), "mean_b": round(mean_b, 4),
            "contrast_L": round(contrast_L, 4),
        })

    df = pd.DataFrame(filas) # Convierte la lista de diccionarios en un DataFrame de pandas, que es básicamente una tabla tipo Excel.
    df.to_csv(CSV_MAHOU, index=False, encoding="utf-8-sig") # Guarda esa tabla como CSV en la ruta que definimos al principio
    # index=False evita que pandas añada una columna extra de números. encoding="utf-8-sig" para que los caracteres especiales como acentos se guarden bien.
    print(f"\n✓ Mahou guardado: {len(df)} productos")
    return df # Devuelve el DataFrame por si otra parte del código lo necesita.

# Resumen esquemático: Va a la tienda de Mahou -> Acepta cookies -> Por cada categoría recoge todas las URLs ->
# Elimina duplicados -> Por cada URL entra al producto y saca nombre e imagen ->
# Cierra Chrome -> Descarga cada imagen y calcula su color -> Guarda todo en CSV


# ============================================================
# PARTE 2 — AMAZON BERKELEY OBJECTS (descarga directa S3)
# ============================================================

def scraper_abo():
    print("\n" + "="*55) # Esto no aporta nada útil a la ejecución del escrping, simplemente adorna el output en el terminal
    print("DESCARGADOR AMAZON BERKELEY OBJECTS (ABO)")
    print("="*55)

    HEADERS_ABO = {"User-Agent": "TFG-ColorAnalysis/1.0 (uso-academico)"} 
    IMGS_DIR    = CARPETA + r"\abo_data\imagenes"  # Aquí se detalla el lugar donde se van a guardar las imágenes
    CACHE_DIR   = CARPETA + r"\abo_data\cache" # Determina dónde se va a guardar el cache
    MAX_PROD    = 10000 # Se indica que se va a utilizar como máximo 10.000 imágenes

    CATEGORIAS_INTERES = { # Estas son todas las categorías que resultan de interés para analizar. Son categorías que están en la web y que nos conviene analizar. 
        # Generamos un set con valores únicos, esto nos ayudará a ver luego si está la categoría disponible. 
        "SHIRT", "PANTS", "DRESS", "JACKET", "COAT", "SWEATER", "SKIRT",
        "SHOES", "BOOTS", "SNEAKERS", "SANDALS", "HANDBAG", "BACKPACK",
        "WALLET", "BELT", "WATCH", "SUNGLASSES", "HAT", "SCARF", "GLOVES",
        "JEWELRY", "NECKLACE", "BRACELET", "EARRING", "RING",
        "SOFA", "CHAIR", "TABLE", "BED", "LAMP", "DESK", "SHELF",
        "CABINET", "MIRROR", "RUG", "CURTAIN", "PILLOW", "BLANKET",
        "VASE", "PICTURE_FRAME", "CLOCK",
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
    }

    MAPA_CATEGORIAS = { # Generamos un diccionario para que ese valor como "SHIRT" que es util para el código, se convierta en una categoría más utilizada en 
        # nuestro día a día
        "SHIRT": "Moda - Camisetas", "PANTS": "Moda - Pantalones", "DRESS": "Moda - Vestidos",
        "JACKET": "Moda - Chaquetas", "COAT": "Moda - Abrigos", "SWEATER": "Moda - Jerseys",
        "SKIRT": "Moda - Faldas", "SHOES": "Moda - Zapatos", "BOOTS": "Moda - Botas",
        "SNEAKERS": "Moda - Zapatillas", "SANDALS": "Moda - Sandalias", "HANDBAG": "Moda - Bolsos",
        "BACKPACK": "Moda - Mochilas", "WALLET": "Moda - Carteras", "BELT": "Moda - Cinturones",
        "WATCH": "Moda - Relojes", "SUNGLASSES": "Moda - Gafas de sol", "HAT": "Moda - Sombreros",
        "SCARF": "Moda - Bufandas", "GLOVES": "Moda - Guantes", "JEWELRY": "Moda - Joyería",
        "NECKLACE": "Moda - Collares", "BRACELET": "Moda - Pulseras", "EARRING": "Moda - Pendientes",
        "RING": "Moda - Anillos",
        "SOFA": "Hogar - Sofás", "CHAIR": "Hogar - Sillas", "TABLE": "Hogar - Mesas",
        "BED": "Hogar - Camas", "LAMP": "Hogar - Lámparas", "DESK": "Hogar - Escritorios",
        "SHELF": "Hogar - Estantes", "CABINET": "Hogar - Armarios", "MIRROR": "Hogar - Espejos",
        "RUG": "Hogar - Alfombras", "CURTAIN": "Hogar - Cortinas", "PILLOW": "Hogar - Cojines",
        "BLANKET": "Hogar - Mantas", "VASE": "Hogar - Jarrones", "PICTURE_FRAME": "Hogar - Marcos",
        "CLOCK": "Hogar - Relojes pared",
        "MUG": "Cocina - Tazas", "CUP": "Cocina - Vasos", "PLATE": "Cocina - Platos",
        "BOWL": "Cocina - Boles", "POT": "Cocina - Ollas", "PAN": "Cocina - Sartenes",
        "KNIFE": "Cocina - Cuchillos", "CUTTING_BOARD": "Cocina - Tablas", "KETTLE": "Cocina - Teteras",
        "TOASTER": "Cocina - Tostadoras", "BLENDER": "Cocina - Batidoras", "COFFEE_MAKER": "Cocina - Cafeteras",
        "WINE_GLASS": "Cocina - Copas", "BOTTLE": "Cocina - Botellas", "STORAGE_CONTAINER": "Cocina - Recipientes",
        "LAPTOP": "Tecnología - Portátiles", "TABLET": "Tecnología - Tablets", "HEADPHONES": "Tecnología - Auriculares",
        "SPEAKER": "Tecnología - Altavoces", "CAMERA": "Tecnología - Cámaras", "KEYBOARD": "Tecnología - Teclados",
        "MOUSE": "Tecnología - Ratones", "MONITOR": "Tecnología - Monitores", "PHONE_CASE": "Tecnología - Fundas móvil",
        "PERFUME": "Cosmética - Perfumes", "LOTION": "Cosmética - Cremas", "SHAMPOO": "Cosmética - Champús",
        "TOOTHBRUSH": "Cosmética - Cepillos", "HAIR_DRYER": "Cosmética - Secadores", "MAKEUP": "Cosmética - Maquillaje",
        "LIPSTICK": "Cosmética - Labiales", "NAIL_POLISH": "Cosmética - Esmaltes",
        "TOY": "Juguetes - General", "PUZZLE": "Juguetes - Puzzles", "BOARD_GAME": "Juguetes - Juegos de mesa",
        "DOLL": "Juguetes - Muñecas", "ACTION_FIGURE": "Juguetes - Figuras", "BOOK": "Ocio - Libros",
        "NOTEBOOK": "Ocio - Cuadernos", "PEN": "Ocio - Bolígrafos",
        "YOGA_MAT": "Deporte - Esterillas", "WATER_BOTTLE": "Deporte - Botellas", "GYM_BAG": "Deporte - Bolsas",
        "BIKE": "Deporte - Bicis", "HELMET": "Deporte - Cascos",
        "PET_BED": "Mascotas - Camas", "PET_BOWL": "Mascotas - Comederos", "PET_TOY": "Mascotas - Juguetes",
    }

    def descargar_con_cache(url, nombre_cache): # Esta función tiene como objetivo utilizar el caché para comprobar que no se descargan elementos dos veces. 
        ruta_cache = os.path.join(CACHE_DIR, nombre_cache) # En esta ruta se guardará el archivo. 
        if os.path.exists(ruta_cache): 
            print(f"    (usando cache: {nombre_cache})") # Si la ruta ya existe, muestra un mensaje indicando que se usa el caché. 
            with open(ruta_cache, "rb") as f: # Abre el archivo cache y leelo en binario "rb", sirve para leer el contenido tal cual. Se usa el with para que 
                # se cierre el archivo nada más abrirlo. 
                return f.read()
        print(f"    Descargando {nombre_cache}...") # Si el archvo no existe, se imprime en pantalla que se va a descargar. 
        resp = requests.get(url, headers=HEADERS_ABO, timeout=60, stream=True) # Aquí se hace una petición para obtener lo que hay en esa url con el "request.get"
        # headers se usa para que la web te acepte como un navegador, si en 60 segundos no responde, se cancela y da error. 
        resp.raise_for_status() # Si la web respondió con error (por ejemplo 404, 500), esta línea lanza una excepción y el programa se para o entra en manejo de errores.
        datos = resp.content # Guarda en datos "resp.content" que es el contenido real del archivo o página.
        with open(ruta_cache, "wb") as f: # Guarda en el ordenador lo que se descargó. "wb" significa escribir en binario, escribirlo tal cual lo entiende el ordenador
            # con 0 y 1, si el archivo no existía lo crea y si ya existe, lo sobreescribe
            f.write(datos) 
        return datos

    def extraer_nombre(listing): # Esta función sirve para extraer el nombre de un conjunto de productos, será un diccionario. 
        nombres = listing.get("item_name", []) # Aquí se pide lo que haya en el campo item_name y si no hay nada, devuelveme el campo vacío. 
        for n in nombres:
            if n.get("language_tag", "").startswith("en"): #Aquí se pide que se reconozca el idioma, si no se puede, se devuelve una cadena vacía. 
                # Si empieza por en
                return n.get("value", "Sin nombre").strip() # Si encuentra un nombre en inglés, devuelve el texto del nombre, si no existe devuelve "Sin Nombre"
            # El strip() elimina espacios al principio y al final. 
        if nombres: # SI no encuentras ningun nombre en inglés entra en este bucle
            return nombres[0].get("value", "Sin nombre").strip() # Si había nombres pero ninguno en inglés, coge el primero y lo devuelve limpiando los espacios antes y depsués
        return "Sin nombre" # Si la lista estaba vacía desde el principio devuelve el valor "Sin Nombre"

    def extraer_product_type(listing): # EL objetivo de esta función es sacar el tipo de producto y devolverlo con mayúsculas. 
        tipos = listing.get("product_type", []) # Trata de obtener del diccionario el valor asociado a "product_type". Si existe, lo guarda en types, si no lista vacía. 
        return tipos[0].get("value", "").upper() if tipos else "" # Si tipos está vacía, devyelve una cadena vacía . Intentar obtener el primer elemento de la lista
    # intenta sacar el valor de la clave "value". SI no existe, devuelve la cadena vacía. Lo que devuelve lo convierte en mayúsculas. 

    def extraer_marca(listing): # Del diccionario de productos trata de extraer la marca priorizando el valor en inglés. 
        marcas = listing.get("brand", []) # Si existe la clave "brand", guarda su contenido en marcas , si no existe guarda una lista vacía. 
        for m in marcas:
            if m.get("language_tag", "").startswith("en"): # para cada marca, intenta averiguar el idioma, si no devuelve un campo vacío. Además, comprueba si empieza 
                # por "en", inglés. 
                return m.get("value", "").strip() # Si el texto está en inglés, obtiene el texto con m.get(), y le quita los espacios alante y detrás.
        if marcas: # Si no encontraste ninguna marca en inglés, haz lo siguiente
            return marcas[0].get("value", "").strip() # COge la primera da igual el idioma en el que esté y quitale los espacios alante y atrás. 
        return ""

    # Fase 1: índice de imágenes
    print("\n[1/4] Cargando índice de imágenes...") # Indica que se va a ejecutar el primer paso de los cuatro que hay 
    URL_IMAGES_META = "https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz" # Esta es la URL de la que se van a obtener las imágenes. 
    datos = descargar_con_cache(URL_IMAGES_META, "images.csv.gz") # Si el archivo ya está cargado en la tabla caché lo usa, si no , lo descarga
    with gzip.open(io.BytesIO(datos), "rt", encoding="utf-8") as f:  # Bytes10 convierte el archivo datos en algo que se comporta como un archivo en la memoria 
        # griz.open abre ese archivo comprimirdo. encoding="utf-8" indica cómo interpretar los caracteres. as f significa que el archivo abierto se llamará f.
        # with asegura que se cierre automáticamente al terminar.  En resumen, aquí se está descomprimiendo el archivo en memoria y preparándolo para leerlo como texto.
        df_imgs = pd.read_csv(f) #  Lee el archvo y lo convierte en una tabla etsructurada 
    print(f"    → {len(df_imgs)} imágenes en el índice") # Indica el núemro de filas que tiene la tabla. Es decir, el número de imágenes que se han descargado. 

    # Fase 2: listings de productos
    print("\n[2/4] Cargando listings de productos...")
    productos_filtrados = []
    URL_LISTINGS_BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/"

    for num in range(10):  # archivos 0 al 9
        print(f"  Archivo listings_{num}...")
        try:
            url   = URL_LISTINGS_BASE + f"listings_{num}.json.gz"
            datos = descargar_con_cache(url, f"listings_{num}.json.gz")
            with gzip.open(io.BytesIO(datos), "rt", encoding="utf-8") as f:
                for linea in f:
                    try:
                        prod = json.loads(linea.strip())
                    except Exception:
                        continue
                    product_type = extraer_product_type(prod)
                    if product_type in CATEGORIAS_INTERES:
                        main_image_id = prod.get("main_image_id", "")
                        if not main_image_id:
                            continue
                        productos_filtrados.append({
                            "item_id":      prod.get("item_id", ""),
                            "categoria":    MAPA_CATEGORIAS.get(product_type, product_type),
                            "nombre":       extraer_nombre(prod),
                            "marca":        extraer_marca(prod),
                            "main_image_id": main_image_id,
                        })
        except Exception as e:
            print(f"    Error al cargar listings_{num}: {e}")
            continue

        if len(productos_filtrados) >= MAX_PROD:
            break

    # Deduplicar y limitar
    vistos = set()
    unicos = []
    for p in productos_filtrados:
        if p["item_id"] not in vistos:
            vistos.add(p["item_id"])
            unicos.append(p)
    productos_filtrados = unicos[:MAX_PROD]
    print(f"\n  Productos encontrados: {len(productos_filtrados)}")

    # Fase 3: descargar imágenes y calcular color
    print("\n[3/4] Descargando imágenes y calculando color...")
    filas = []
    for i, prod in enumerate(tqdm(productos_filtrados, desc="Procesando", unit="prod")):
        image_id      = prod["main_image_id"]
        imagen_url    = f"https://m.media-amazon.com/images/I/{image_id}._SX512_.jpg"
        nombre_limpio = re.sub(r'[^a-zA-Z0-9]', '_', prod["nombre"])[:35]
        nombre_archivo = f"abo_{i:04d}_{prod['item_id']}_{nombre_limpio}.jpg"
        ruta = descargar_imagen(imagen_url, os.path.join(IMGS_DIR, nombre_archivo), HEADERS_ABO)
        if ruta is None:
            continue
        try:
            mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = calcular_color(ruta)
        except Exception:
            continue
        filas.append({
            "fuente":     "Amazon Berkeley Objects",
            "categoria":  prod["categoria"],
            "nombre":     prod["nombre"],
            "imagen_url": imagen_url,
            "mean_R":     round(mean_R, 4), "mean_G": round(mean_G, 4), "mean_B": round(mean_B, 4),
            "mean_L":     round(mean_L, 4), "mean_a": round(mean_a, 4), "mean_b": round(mean_b, 4),
            "contrast_L": round(contrast_L, 4),
        })

    df = pd.DataFrame(filas)
    df.to_csv(CSV_ABO, index=False, encoding="utf-8-sig")
    print(f"\n✓ ABO guardado: {len(df)} productos")
    return df


# ============================================================
# PARTE 3 — OPEN FOOD FACTS (API pública)
# ============================================================

def scraper_openfoodfacts():
    print("\n" + "="*55)
    print("SCRAPER OPEN FOOD FACTS")
    print("="*55)

    HEADERS_OFF = {"User-Agent": "TFG-ColorAnalysis/1.0 (universidad; uso-academico)"}
    API_BASE    = "https://world.openfoodfacts.org/api/v2/search"
    DELAY       = 1.0
    PAGE_SIZE   = 50
    MAX_PROD    = 10000
    IMGS_DIR    = CARPETA + r"\openfoodfacts_data\imagenes"

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

    CAMPOS_API = ",".join(["code", "product_name", "brands", "categories_tags", "image_front_url", "image_url", "countries_tags"])

    def buscar_productos(categoria_tag, max_productos):
        productos = []
        pagina = 1
        while len(productos) < max_productos:
            params = {
                "categories_tags": categoria_tag,
                "countries_tags":  "en:spain",
                "fields":          CAMPOS_API,
                "page_size":       PAGE_SIZE,
                "page":            pagina,
                "sort_by":         "unique_scans_n",
                "json":            1,
            }
            try:
                resp = requests.get(API_BASE, params=params, headers=HEADERS_OFF, timeout=15)
                resp.raise_for_status()
                datos = resp.json()
            except Exception as e:
                print(f"    Error en API (página {pagina}): {e}")
                break
            items = datos.get("products", [])
            if not items:
                break
            for item in items:
                imagen_url = (item.get("image_front_url") or item.get("image_url") or "")
                if not imagen_url:
                    continue
                nombre = item.get("product_name", "").strip() or "Sin nombre"
                productos.append({"codigo": item.get("code", ""), "nombre": nombre, "marca": item.get("brands", "").strip(), "imagen_url": imagen_url})
            total = datos.get("count", 0)
            if pagina * PAGE_SIZE >= total:
                break
            pagina += 1
            time.sleep(DELAY)
        return productos[:max_productos]

    max_por_categoria = max(10, MAX_PROD // len(CATEGORIAS))

    print(f"\n[1/3] Descargando datos de la API...")
    todos = []
    for nombre_cat, tag_cat in CATEGORIAS.items():
        print(f"  {nombre_cat}...", end=" ", flush=True)
        productos = buscar_productos(tag_cat, max_por_categoria)
        for p in productos:
            p["categoria"] = nombre_cat
        todos.extend(productos)
        print(f"{len(productos)} productos")
        time.sleep(DELAY)

    # Deduplicar
    vistos = set()
    todos_unicos = []
    for p in todos:
        clave = p["codigo"] or p["imagen_url"]
        if clave not in vistos:
            vistos.add(clave)
            todos_unicos.append(p)
    print(f"\n  Total productos únicos: {len(todos_unicos)}")

    print(f"\n[2/3] Descargando imágenes y calculando color...")
    filas = []
    for i, prod in enumerate(tqdm(todos_unicos, desc="Procesando", unit="img")):
        nombre_limpio  = re.sub(r'[^a-zA-Z0-9]', '_', prod["nombre"])[:40]
        codigo_limpio  = re.sub(r'[^a-zA-Z0-9]', '_', prod["codigo"])[:15]
        nombre_archivo = f"off_{i:04d}_{codigo_limpio}_{nombre_limpio}.jpg"
        ruta = descargar_imagen(prod["imagen_url"], os.path.join(IMGS_DIR, nombre_archivo), HEADERS_OFF)
        if ruta is None:
            continue
        try:
            mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = calcular_color(ruta)
        except Exception:
            continue
        filas.append({
            "fuente":     "Open Food Facts",
            "categoria":  prod["categoria"],
            "nombre":     prod["nombre"],
            "imagen_url": prod["imagen_url"],
            "mean_R":     round(mean_R, 4), "mean_G": round(mean_G, 4), "mean_B": round(mean_B, 4),
            "mean_L":     round(mean_L, 4), "mean_a": round(mean_a, 4), "mean_b": round(mean_b, 4),
            "contrast_L": round(contrast_L, 4),
        })

    df = pd.DataFrame(filas)
    df.to_csv(CSV_OFF, index=False, encoding="utf-8-sig")
    print(f"\n✓ Open Food Facts guardado: {len(df)} productos")
    return df


# ============================================================
# PARTE 4 — UNIFICAR LOS TRES DATASETS
# ============================================================

def unificar_datasets():
    print("\n" + "="*55)
    print("UNIFICANDO LOS TRES DATASETS")
    print("="*55)

    COLUMNAS = ["fuente", "categoria", "nombre", "imagen_url",
                "mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]

    dfs = []
    for path, nombre in [(CSV_ABO, "ABO"), (CSV_MAHOU, "Mahou"), (CSV_OFF, "Open Food Facts")]:
        print(f"  Cargando {nombre}...")
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df[COLUMNAS]
        print(f"    → {len(df)} filas")
        dfs.append(df)

    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.dropna(subset=["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"])
    df_final = df_final.drop_duplicates(subset=["imagen_url"])
    df_final = df_final.reset_index(drop=True)

    df_final.to_csv(CSV_FINAL, index=False, encoding="utf-8-sig")
    print(f"\n✓ Dataset combinado: {len(df_final)} productos")
    return df_final


# ============================================================
# PARTE 5 — INGENIERÍA DEL DATO (limpieza + análisis + gráficos)
# ============================================================

def ingenieria_del_dato():
    print("\n" + "="*55)
    print("INGENIERÍA DEL DATO")
    print("="*55)

    df = pd.read_csv(CSV_FINAL, encoding="utf-8-sig")
    print(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")

    variables_color = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"]

    # --- Nulos ---
    print(f"\nValores nulos por columna:")
    print(df.isnull().sum().to_string())
    df = df.dropna(subset=variables_color)

    # --- Duplicados ---
    df = df.drop_duplicates(subset=["imagen_url"])
    df = df.reset_index(drop=True)
    print(f"\nFilas tras limpiar: {len(df)}")

    # --- Validación de rangos ---
    rangos = {"mean_R": (0, 255), "mean_G": (0, 255), "mean_B": (0, 255),
              "mean_L": (0, 100), "mean_a": (-128, 128), "mean_b": (-128, 128), "contrast_L": (0, 100)}
    print("\nValidación de rangos:")
    for var, (vmin, vmax) in rangos.items():
        fuera = ((df[var] < vmin) | (df[var] > vmax)).sum()
        print(f"  {var:<12}: {'✓ OK' if fuera == 0 else f'⚠ {fuera} fuera de rango'}")

    # --- Estadísticos descriptivos ---
    print("\nEstadísticos descriptivos:")
    descripcion = df[variables_color].describe().round(3)
    print(descripcion.to_string())
    descripcion.to_csv(CARPETA_GRAFICOS + r"\estadisticos_descriptivos.csv", encoding="utf-8-sig")

    # --- Outliers IQR ---
    print("\nOutliers (método IQR):")
    for var in variables_color:
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)).sum()
        print(f"  {var:<12}: {outliers} outliers ({round(outliers/len(df)*100, 2)}%)")

    # --- Gráfico distribución por fuente ---
    plt.figure(figsize=(8, 5))
    df["fuente"].value_counts().plot(kind="bar", color=["#2E75B6", "#1F4E79", "#70AD47"])
    plt.title("Número de productos por fuente de datos", fontsize=14, fontweight="bold")
    plt.xlabel("Fuente")
    plt.ylabel("Número de productos")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS + r"\distribucion_fuente.png", dpi=150)
    plt.close()

    # --- Gráfico distribución por categoría ---
    plt.figure(figsize=(12, 6))
    df["categoria"].value_counts().head(15).plot(kind="barh", color="#2E75B6")
    plt.title("Top 15 categorías con más productos", fontsize=14, fontweight="bold")
    plt.xlabel("Número de productos")
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS + r"\distribucion_categoria.png", dpi=150)
    plt.close()

    # --- Histogramas ---
    nombres_variables = {
        "mean_R": "Canal Rojo — RGB (0-255)", "mean_G": "Canal Verde — RGB (0-255)",
        "mean_B": "Canal Azul — RGB (0-255)", "mean_L": "Luminosidad L* — CIELAB (0-100)",
        "mean_a": "Componente a* — CIELAB (-128 a 128)", "mean_b": "Componente b* — CIELAB (-128 a 128)",
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
    axes[7].set_visible(False)
    axes[8].set_visible(False)
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS + r"\histogramas_color.png", dpi=150)
    plt.close()

    # --- Matriz de correlación ---
    correlacion = df[variables_color].corr().round(2)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(correlacion, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlación de Pearson")
    ax.set_xticks(range(len(variables_color)))
    ax.set_yticks(range(len(variables_color)))
    ax.set_xticklabels(variables_color, rotation=45, ha="right")
    ax.set_yticklabels(variables_color)
    for i in range(len(variables_color)):
        for j in range(len(variables_color)):
            valor = correlacion.iloc[i, j]
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(valor) > 0.6 else "black", fontweight="bold")
    ax.set_title("Matriz de correlación — Variables de color", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS + r"\correlacion.png", dpi=150)
    plt.close()

    # --- Guardar dataset limpio ---
    df.to_csv(CSV_LIMPIO, index=False, encoding="utf-8-sig")
    print(f"\n✓ Dataset guardado: {len(df)} productos, {df['categoria'].nunique()} categorías")
    print(f"  Gráficos guardados en: {CARPETA_GRAFICOS}")

    return df


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    print("="*55)
    print("TFG - ADRIÁN JULVE NAVARRO")
    print("Recolección y análisis de color en packaging")
    print("="*55)

    # Cambia a False las partes que ya hayas ejecutado antes
    EJECUTAR_MAHOU = True
    EJECUTAR_ABO   = True
    EJECUTAR_OFF   = True

    if EJECUTAR_MAHOU:
        scraper_mahou()

    if EJECUTAR_ABO:
        scraper_abo()

    if EJECUTAR_OFF:
        scraper_openfoodfacts()

    unificar_datasets()
    ingenieria_del_dato()

    print("\n" + "="*55)
    print("TODO COMPLETADO")
    print("="*55)