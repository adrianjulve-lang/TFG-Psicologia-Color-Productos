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
        "PET_BED", "PET_BOWL", "PET_TOY", "CANDLE", "SOAP", "DIFFUSER", "LOTION_PUMP",
        "SUNSCREEN", "FACE_MASK", "SERUM",
        "PROTEIN_POWDER", "SUPPLEMENT",
        "DOG_FOOD", "CAT_FOOD",
        "CLEANING_PRODUCT", "DETERGENT",
    }


    MAPA_CATEGORIAS = { # Generamos un diccionario para que ese valor como "SHIRT" que es util para el código, se convierta en una categoría más utilizada en 
        # nuestro día a día
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
        "CANDLE":           "Hogar - Velas",
        "SOAP":             "Cosmética - Jabones",
        "DIFFUSER":         "Hogar - Difusores",
        "LOTION_PUMP":      "Cosmética - Lociones",
        "SUNSCREEN":        "Cosmética - Protectores solares",
        "FACE_MASK":        "Cosmética - Mascarillas",
        "SERUM":            "Cosmética - Sérums",
        "PROTEIN_POWDER":   "Deporte - Proteínas",
        "SUPPLEMENT":       "Deporte - Suplementos",
        "DOG_FOOD":         "Mascotas - Comida perro",
        "CAT_FOOD":         "Mascotas - Comida gato",
        "CLEANING_PRODUCT": "Hogar - Limpieza",
        "DETERGENT":        "Hogar - Detergentes",
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
    print("\n[2/4] Cargando listings de productos...") # Muestra un mensaje en pantalla para saber en qué paso está el programa
    productos_filtrados = [] # Se crea una lista vacía de productos filtrados en la que se meterán aquellos que nos interesan. 
    URL_LISTINGS_BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/" # Se guarda la URL de donde están los archivos de Amazon


    for num in range(10):  # archivos 0 al 9
        print(f"  Archivo listings_{num}...") # Se muestra en pantalla qué archivo está procesando en este momento
        try:
            url   = URL_LISTINGS_BASE + f"listings_{num}.json.gz" # COnstruye la URL completa sumando la dirección base con el nombre del archivo completo
            datos = descargar_con_cache(url, f"listings_{num}.json.gz") # Llama a la función definida antes con caché. Esta era la que si ya lo descargamos antes, lo 
            # reutiliza del disco en vez de volver a descargarlo para ahorrar tiempo y espacio
            with gzip.open(io.BytesIO(datos), "rt", encoding="utf-8") as f: # Los archivos .json.gc están comprimidos como un ZIP. Esta línea los descomprime
                # El "rt" significa "leer con texto" y "utf-8" es el tipo de codificación de caracteres. 
                for linea in f: # Para cada línea haz lo siguiente
                    try:
                        prod = json.loads(linea.strip()) # COnvierte este archivo JSON en un objeto pyhton (un diccionario). strip() elimina espacios
                    except Exception: # SI falla continua a la siguiente línea sin detenerte. 
                        continue
                    product_type = extraer_product_type(prod) # Llama a la función definida antes para obtener el tipo de producto si está en la lista de 
                    # categorías que nos interesan 
                    if product_type in CATEGORIAS_INTERES: # Si el tipo de producto nos interesa:
                        main_image_id = prod.get("main_image_id", "") # Intenta obtener el identificador de la imagen principal del producto. 
                        if not main_image_id:
                            continue # Si no lo saltamos y vamos a la siguiente
                        productos_filtrados.append({ # A la lista que hemos hecho antes, le metemos lo siguiente: 
                            "item_id":      prod.get("item_id", ""), #Id de producto
                            "categoria":    MAPA_CATEGORIAS.get(product_type, product_type), # la categoría traducida al español
                            "nombre":       extraer_nombre(prod), # el nombre del producto
                            "marca":        extraer_marca(prod), # la marca del producto
                            "main_image_id": main_image_id, # El ID de su imagen
                        })
        except Exception as e:
            print(f"    Error al cargar listings_{num}: {e}") # Aquí se imprime en pantalla qué archivo falló y por qué
            continue


        if len(productos_filtrados) >= MAX_PROD: # Si la cantidad de productos que hemos guardado ya alcanza el máximo que nos hemos fijado
            break # para el bucle con un break y no descarga más archivos. 


    # Deduplicar y limitar
    vistos = set() # Se crea un conjunto vacío, como una lista pero que no admite repetidos por lo que si se ha visto este elemento no se puede volver a meter
    unicos = [] # unicos es una lista vacía donde guardaremos los productos sin repetir. 
    for p in productos_filtrados:
        if p["item_id"] not in vistos: # Si el ID de producto no está en vistos, sigue adelante
            vistos.add(p["item_id"]) # Si ee nuevo, lo añade a vistos para recordarlo
            unicos.append(p) # Además lo mete en la lista de unicos
    productos_filtrados = unicos[:MAX_PROD] # Reemplaza la lista original por la lista sin duplicaods, y además la recorta para quedarse solo con los primeros 
    # MAX_PROD productos. :MAX_PROD significa "dame desde el principio hasta el numero máximo".
    print(f"\n  Productos encontrados: {len(productos_filtrados)}") # Muestra en pantalla cuántos productos hemos encontrado


    # Fase 3: descargar imágenes y calcular color
    print("\n[3/4] Descargando imágenes y calculando color...")
    filas = [] # Se crea uan lista de filas donde iremos guardando los datos de cada producto. 
    for i, prod in enumerate(tqdm(productos_filtrados, desc="Procesando", unit="prod")): # Recorre todos los productos, aqui hay dos cosas a la vez. 
        # enumerate da un número de orden i a cada producto (0,1,2,3...) y tqdm muestra una barra de progreso en pantalla. desc = Procesando es el texto que aperece
        # junto a la barra y unit = "Prod" es la unidad que muestra (productos por segundo)
        image_id      = prod["main_image_id"] # Se obtiene el ID de la imagen
        imagen_url    = f"https://m.media-amazon.com/images/I/{image_id}._SX512_.jpg" # con el ID de imagen se construya la url completa. Pedimos la versión de 512 pixels
        nombre_limpio = re.sub(r'[^a-zA-Z0-9]', '_', prod["nombre"])[:35] # Limpia el nombre del producto para poder usarlo como nombre de archivo. re.sub reemplaza 
        # cualquier carácter que no sea una letra o número por un guion bajo _ — porque los nombres de archivo no pueden tener tildes, espacios ni caracteres raros. 
        # El [:35] lo recorta a 35 caracteres máximo para que el nombre no sea demasiado largo.
        nombre_archivo = f"abo_{i:04d}_{prod['item_id']}_{nombre_limpio}.jpg" # Construye el nombre del archivo que se guardará en disco. i:04d formatea el número
        # con 4 dígitos (0001, 00042...) para que los archivos se ordenen bien en el explorados 
        ruta = descargar_imagen(imagen_url, os.path.join(IMGS_DIR, nombre_archivo), HEADERS_ABO) # Llama a la función que descarga la imagen y la guarda en la carpeta 
        # imágenes 
        if ruta is None:
            continue # Si la descarga falla y devuelve None, saltamos este producto con continue
        try:
            mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, contrast_L = calcular_color(ruta) # Llama a la función calcular_color con la imagen descargada
        except Exception: # de esta manera obtenemos las 7 variables a la vez
            continue
        filas.append({ # Añade cada una de estas filas a la lista con todos los datos del producto. 
            "fuente":     "Amazon Berkeley Objects",
            "categoria":  prod["categoria"],
            "nombre":     prod["nombre"],
            "imagen_url": imagen_url,
            "mean_R":     round(mean_R, 4), "mean_G": round(mean_G, 4), "mean_B": round(mean_B, 4),
            "mean_L":     round(mean_L, 4), "mean_a": round(mean_a, 4), "mean_b": round(mean_b, 4),
            "contrast_L": round(contrast_L, 4),
        })


    df = pd.DataFrame(filas) # COnvierte la lista de filas en una tabla estructurada. 
    df.to_csv(CSV_ABO, index=False, encoding="utf-8-sig") # Guarda esa tabla como archivo CSV. Index = False le dice que no añada una columna extra con números de fila
    # encoding = "utf - 8 - sig" asegura que las tildes y caracteres españoles se guarden correctamente. 
    print(f"\n✓ ABO guardado: {len(df)} productos")
    return df

