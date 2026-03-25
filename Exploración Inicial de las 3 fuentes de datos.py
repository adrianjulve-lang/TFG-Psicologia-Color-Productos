# ============================================================
# TFG - ADRIÁN JULVE NAVARRO
# EXPLORACIÓN PREVIA DE LAS TRES FUENTES DE DATOS
# ============================================================
# Este script NO descarga imágenes ni genera datasets.
# Solo se conecta a cada fuente, obtiene una muestra mínima
# y muestra qué variables existen, qué tipo son y cómo vienen.
# ============================================================

import requests
import gzip
import json
import io
import time
import pandas as pd

SEPARADOR     = "═" * 60
SEPARADOR_FIN = "─" * 60


# ══════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════

def imprimir_cabecera(titulo):
    print(f"\n{SEPARADOR}")
    print(f"  {titulo}")
    print(SEPARADOR)

def imprimir_seccion(titulo):
    print(f"\n{SEPARADOR_FIN}")
    print(f"  {titulo}")
    print(SEPARADOR_FIN)

def analizar_campo(nombre, valores):
    """Analiza un campo y devuelve tipo, ejemplo y % nulos."""
    no_nulos = [v for v in valores if v is not None and v != "" and v != []]
    pct_nulos = (1 - len(no_nulos) / len(valores)) * 100 if valores else 100

    if not no_nulos:
        return "desconocido", "N/A", pct_nulos

    ejemplo = no_nulos[0]
    if isinstance(ejemplo, list):
        tipo = f"lista ({len(ejemplo)} elementos)"
        ejemplo = str(ejemplo[0])[:60] if ejemplo else "[]"
    elif isinstance(ejemplo, dict):
        tipo = f"diccionario ({list(ejemplo.keys())})"
        ejemplo = str(ejemplo)[:60]
    elif isinstance(ejemplo, (int, float)):
        tipo = "numérico"
        ejemplo = str(ejemplo)
    else:
        tipo = "texto"
        ejemplo = str(ejemplo)[:60]

    return tipo, ejemplo, pct_nulos


# ══════════════════════════════════════════════════════════════
# FUENTE 1 — AMAZON BERKELEY OBJECTS (ABO)
# ══════════════════════════════════════════════════════════════

def explorar_abo():
    imprimir_cabecera("FUENTE 1 — AMAZON BERKELEY OBJECTS (ABO)")

    print("""
  Tipo de acceso:  Descarga directa desde Amazon S3 (archivos públicos)
  Formato:         Archivos JSON comprimidos (.json.gz) + índice CSV
  Sin autenticación requerida
    """)

    HEADERS = {"User-Agent": "TFG-ColorAnalysis/1.0 (uso-academico)"}

    # ── Paso 1: índice de imágenes ────────────────────────────
    imprimir_seccion("1a. Índice de imágenes (images.csv.gz)")
    print("  Descargando muestra del índice de imágenes...")

    URL_IMGS = "https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz"
    try:
        resp = requests.get(URL_IMGS, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as f:
            df_imgs = pd.read_csv(f)

        print(f"\n  Observaciones (imágenes en el índice): {len(df_imgs):,}")
        print(f"  Variables: {len(df_imgs.columns)}")
        print(f"\n  {'Variable':<25} {'Tipo':<15} {'Ejemplo':<50} {'Nulos'}")
        print(f"  {'-'*25} {'-'*15} {'-'*50} {'-'*6}")
        for col in df_imgs.columns:
            tipo = str(df_imgs[col].dtype)
            ejemplo = str(df_imgs[col].dropna().iloc[0])[:50] if df_imgs[col].notna().any() else "N/A"
            pct = df_imgs[col].isnull().mean() * 100
            print(f"  {col:<25} {tipo:<15} {ejemplo:<50} {pct:.1f}%")

    except Exception as e:
        print(f"  ⚠ Error al acceder al índice de imágenes: {e}")

    # ── Paso 2: listings de productos ────────────────────────
    imprimir_seccion("1b. Listings de productos (listings_0.json.gz)")
    print("  Descargando muestra de productos (primeras 20 líneas)...")

    URL_LISTING = "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_0.json.gz"
    try:
        productos = []
        raw = b""
        with requests.get(URL_LISTING, headers=HEADERS, timeout=60, stream=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                raw += chunk
                try:
                    tmp = []
                    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
                        for i, linea in enumerate(f):
                            if i >= 20:
                                break
                            try:
                                tmp.append(json.loads(linea.strip()))
                            except json.JSONDecodeError:
                                continue
                    if len(tmp) >= 20:
                        productos = tmp
                        break
                except Exception:
                    continue  # stream incompleto, seguir descargando

        print(f"\n  Productos en la muestra: {len(productos)}")
        print(f"  (El archivo completo tiene miles de productos por fichero, hay 10 ficheros)")

        if productos:
            # Ver todos los campos únicos
            todos_campos = set()
            for p in productos:
                todos_campos.update(p.keys())

            print(f"\n  Campos disponibles en cada producto: {len(todos_campos)}")
            print(f"\n  {'Campo':<30} {'Tipo':<30} {'Ejemplo':<50} {'% nulos'}")
            print(f"  {'-'*30} {'-'*30} {'-'*50} {'-'*7}")

            for campo in sorted(todos_campos):
                valores = [p.get(campo) for p in productos]
                tipo, ejemplo, pct_nulos = analizar_campo(campo, valores)
                print(f"  {campo:<30} {tipo:<30} {ejemplo:<50} {pct_nulos:.0f}%")

            # Mostrar un producto de ejemplo completo
            imprimir_seccion("1c. Ejemplo de producto completo (primer registro)")
            p = productos[0]
            for campo, valor in p.items():
                print(f"  {campo}: {str(valor)[:100]}")

    except Exception as e:
        print(f"  ⚠ Error al acceder a los listings: {e}")


# ══════════════════════════════════════════════════════════════
# FUENTE 2 — OPEN FOOD FACTS (API REST)
# ══════════════════════════════════════════════════════════════

def explorar_off():
    imprimir_cabecera("FUENTE 2 — OPEN FOOD FACTS (API REST)")

    print("""
  Tipo de acceso:  API REST pública (sin autenticación)
  Formato:         JSON por petición
  Documentación:   https://world.openfoodfacts.org/api/v2
    """)

    HEADERS  = {"User-Agent": "TFG-ColorAnalysis/1.0 (universidad; uso-academico)"}
    API_BASE = "https://world.openfoodfacts.org/api/v2/search"

    # ── Petición de prueba: 5 cervezas ───────────────────────
    imprimir_seccion("2a. Petición de prueba (5 productos de la categoría cervezas)")
    print("  Consultando la API...")

    try:
        params = {
            "categories_tags": "en:beers",
            "page_size":       5,
            "page":            1,
            "sort_by":         "unique_scans_n",
            "json":            1,
        }
        resp = requests.get(API_BASE, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        datos = resp.json()

        total_disponible = datos.get("count", 0)
        productos        = datos.get("products", [])

        print(f"\n  Total productos disponibles en 'cervezas': {total_disponible:,}")
        print(f"  Productos recibidos en esta petición: {len(productos)}")

        if productos:
            # Ver todos los campos del primer producto
            todos_campos = set()
            for p in productos:
                todos_campos.update(p.keys())

            print(f"\n  Campos disponibles por producto: {len(todos_campos)}")
            print(f"\n  {'Campo':<40} {'Tipo':<20} {'Ejemplo':<50} {'% nulos'}")
            print(f"  {'-'*40} {'-'*20} {'-'*50} {'-'*7}")

            for campo in sorted(todos_campos):
                valores = [p.get(campo) for p in productos]
                tipo, ejemplo, pct_nulos = analizar_campo(campo, valores)
                print(f"  {campo:<40} {tipo:<20} {ejemplo:<50} {pct_nulos:.0f}%")

            # Mostrar los campos que usamos nosotros
            imprimir_seccion("2b. Campos que usamos en el proyecto")
            campos_usados = {
                "code":             "Código de barras (identificador único)",
                "product_name":     "Nombre del producto",
                "brands":           "Marca",
                "categories_tags":  "Categorías del producto",
                "image_front_url":  "URL de la imagen frontal (la que usamos)",
                "image_url":        "URL alternativa de imagen",
                "countries_tags":   "Países donde está disponible",
            }
            for campo, descripcion in campos_usados.items():
                valor = productos[0].get(campo, "NO DISPONIBLE")
                disponible = "✓" if campo in productos[0] else "✗ NO EXISTE"
                print(f"  {disponible}  {campo:<30} → {descripcion}")
                if campo in productos[0]:
                    print(f"     Ejemplo: {str(valor)[:80]}")

            # Mostrar un producto de ejemplo completo
            imprimir_seccion("2c. Ejemplo de producto completo (primer registro)")
            for campo, valor in productos[0].items():
                print(f"  {campo}: {str(valor)[:100]}")

    except Exception as e:
        print(f"  ⚠ Error al acceder a la API de Open Food Facts: {e}")

    # ── Cuántos productos hay por categoría ──────────────────
    imprimir_seccion("2d. Volumen disponible por categoría principal")
    print("  Consultando cuántos productos hay en cada categoría...")

    categorias_muestra = {
        "Cervezas":            "en:beers",
        "Cereales y desayuno": "en:breakfast-cereals",
        "Galletas":            "en:biscuits",
        "Chocolates":          "en:chocolates",
        "Lácteos":             "en:dairy-products",
        "Bebidas":             "en:beverages",
        "Vinos":               "en:wines",
        "Snacks":              "en:snacks",
        "Pasta":               "en:pastas",
        "Refrescos":           "en:sodas",
    }

    for nombre, tag in categorias_muestra.items():
        try:
            params = {"categories_tags": tag, "page_size": 1, "json": 1}
            r = requests.get(API_BASE, params=params, headers=HEADERS, timeout=10)
            r.raise_for_status()
            total = r.json().get("count", 0)
            barra = "█" * min(int(total / 5000), 30)
            print(f"  {nombre:<30} {total:>8,} productos  {barra}")
        except Exception as e:
            print(f"  {nombre:<30} error al consultar ({e})")
        time.sleep(0.5)


# ══════════════════════════════════════════════════════════════
# FUENTE 3 — MAHOU SAN MIGUEL (Web scraping)
# ══════════════════════════════════════════════════════════════

def explorar_mahou():
    imprimir_cabecera("FUENTE 3 — MAHOU SAN MIGUEL (Web scraping)")

    print("""
  Tipo de acceso:  Web scraping con Selenium (Chrome automatizado)
  Formato:         HTML dinámico con JavaScript
  URL:             https://www.mahou-sanmiguel.com/tienda
  Requiere:        Chrome + ChromeDriver
    """)

    HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
    BASE_URL = "https://www.mahou-sanmiguel.com"

    # ── Explorar la API interna de la tienda ─────────────────
    imprimir_seccion("3a. Estructura de la tienda y categorías disponibles")

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

    print(f"  Categorías definidas en la tienda: {len(CATEGORIAS)}")
    print(f"\n  {'Categoría':<30} {'Ruta URL'}")
    print(f"  {'-'*30} {'-'*45}")
    for cat, ruta in CATEGORIAS.items():
        print(f"  {cat:<30} {ruta}")

    # ── Intentar acceso básico ────────────────────────────────
    imprimir_seccion("3b. Comprobación de acceso a la web")
    try:
        resp = requests.get(BASE_URL + "/tienda", headers=HEADERS, timeout=15)
        print(f"  Estado HTTP: {resp.status_code}")
        if resp.status_code == 200:
            print(f"  ✓ La web responde correctamente")
            print(f"  Tamaño de la respuesta: {len(resp.content):,} bytes")
            print(f"  Nota: el contenido de productos NO viene en este HTML inicial")
            print(f"        porque la web usa JavaScript dinámico → necesita Selenium")
        else:
            print(f"  ⚠ La web respondió con código {resp.status_code}")
    except Exception as e:
        print(f"  ⚠ Error al acceder a la web: {e}")

    # ── Estructura del dataset que generaremos ────────────────
    imprimir_seccion("3c. Estructura del dataset que generará el scraper")
    print(f"""
  Variables que extraeremos de cada producto de Mahou:

  {'Variable':<15} {'Tipo':<12} {'Origen':<35} {'Ejemplo'}
  ─────────────────────────────────────────────────────────────────────────
  fuente         texto        Fijo                   "Mahou San Miguel"
  categoria      texto        Nombre de categoría    "Cerveza Mahou"
  nombre         texto        Etiqueta h1 del HTML   "Mahou 5 Estrellas 33cl"
  imagen_url     texto        Meta og:image del HTML "https://...jpg"
  mean_R         numérico     calcular_color()        233.77
  mean_G         numérico     calcular_color()        201.45
  mean_B         numérico     calcular_color()        89.12
  mean_L         numérico     calcular_color()        84.31
  mean_a         numérico     calcular_color()        3.21
  mean_b         numérico     calcular_color()        18.44
  contrast_L     numérico     calcular_color()        22.10

  Total variables: 11
  Estimación de productos: ~81 (catálogo completo de la tienda online)
    """)


# ══════════════════════════════════════════════════════════════
# RESUMEN COMPARATIVO FINAL
# ══════════════════════════════════════════════════════════════

def resumen_comparativo():
    imprimir_cabecera("RESUMEN COMPARATIVO DE LAS TRES FUENTES")

    print(f"""
  {'Característica':<30} {'ABO (Amazon)':<30} {'Open Food Facts':<30} {'Mahou San Miguel'}
  {'─'*30} {'─'*30} {'─'*30} {'─'*20}
  {'Tipo de acceso':<30} {'Descarga directa S3':<30} {'API REST pública':<30} {'Web scraping Selenium'}
  {'Autenticación':<30} {'No requiere':<30} {'No requiere':<30} {'No requiere'}
  {'Formato de datos':<30} {'JSON.gz + CSV.gz':<30} {'JSON por petición':<30} {'HTML dinámico'}
  {'Volumen estimado':<30} {'~100.000+ productos':<30} {'Millones de productos':<30} {'~81 productos'}
  {'Categorías':<30} {'Tecnología, hogar...':<30} {'Alimentación y bebidas':<30} {'Cervezas, aguas...'}
  {'Variables por producto':<30} {'~20 campos JSON':<30} {'~100+ campos JSON':<30} {'11 (tras extracción)'}
  {'Variables que usamos':<30} {'5 campos + color':<30} {'7 campos + color':<30} {'4 campos + color'}
  {'Condición fotográfica':<30} {'Fondo blanco estándar':<30} {'Variable (consumidores)':<30} {'Marketing profesional'}
  {'Sesgo cromático':<30} {'L* alto, b* bajo':<30} {'L* medio, b* medio':<30} {'Packaging premium'}
  {'Velocidad extracción':<30} {'Rápida (descarga bulk)':<30} {'Media (1s entre req)':<30} {'Lenta (3s por página)'}
    """)

    print(f"""
  Variables que genera calcular_color() para TODOS los productos:
  ─────────────────────────────────────────────────────────────────
  mean_R      Media canal Rojo (RGB)           Rango: 0–255
  mean_G      Media canal Verde (RGB)           Rango: 0–255
  mean_B      Media canal Azul (RGB)            Rango: 0–255
  mean_L      Luminosidad L* (CIELAB)           Rango: 0–100
  mean_a      Componente a* rojo-verde (CIELAB) Rango: -128 a +128
  mean_b      Componente b* amarillo-azul (CIELAB) Rango: -128 a +128
  contrast_L  Desv. típica de L* (contraste)   Rango: 0–100

  Dataset combinado resultante: ~10.000 productos × 11 columnas
    """)


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("  TFG — ADRIÁN JULVE NAVARRO")
        print("  EXPLORACIÓN PREVIA DE LAS TRES FUENTES DE DATOS")
        print("  (sin descargar imágenes ni generar datasets)")
        print("=" * 60)

        explorar_abo()
        explorar_off()
        explorar_mahou()
        resumen_comparativo()

        print(f"\n{'='*60}")
        print("  EXPLORACIÓN PREVIA COMPLETADA")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"  ERROR INESPERADO: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()

    finally:
        input("\nPulsa Enter para cerrar...")