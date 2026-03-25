# ==============================================================
# TFG - ADRIÁN JULVE NAVARRO
# Script 2: Ingeniería del Dato
# Parte del CSV generado por el Script 1 y lo enriquece con:
#   - Limpieza completa con antes/después documentado
#   - Variables HSV
#   - Variable objetivo: emoción (Gilbert et al., 2016)
#   - Variables de negocio (coherencia, alineación, temperatura)
#   - Normalización
#   - Gráficos clave para el análisis exploratorio
# ==============================================================

import pandas as pd # Se importa pandas para trabajar con tablas de datos
import numpy as np # Importamos numpy para trabajar con cálculos atemáticos. Lo necesito porque las variables de color son arrays numéricos. 
import matplotlib.pyplot as plt # Librería necesaria para hacer gráficos. 
from sklearn.preprocessing import MinMaxScaler # De la librería sklearn importamos la herramienta necesaria para normalizar variables al rango 0 - 1
import os # Esta es la librería del sistema operativo. SIrve para trabajar con archivos y carpetas. 

# --------------------------------------------------------------
# RUTAS — cambia solo esto si es necesario
# --------------------------------------------------------------
RUTA_ENTRADA  = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\Dataset combinado sin emociones.csv" # Aquí se determinan todas las rutas del csv del que se deben obtener los datos para trabajar
RUTA_SALIDA   = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\Dataset_final_con_emociones.csv" # Es en esta ruta en la que se guardarán todas las transformaciones aplicadas. 
RUTA_GRAFICOS = r"C:\Users\34625\Desktop\4 Carrera\TFG\DATOS SCRAPPING\graficos"
os.makedirs(RUTA_GRAFICOS, exist_ok=True) # Aquí de especifica que se cree la carpeta de gráficos si no existe. 


# ==============================================================
# 1. CARGAR DATOS
# ==============================================================
df = pd.read_csv(RUTA_ENTRADA, encoding="utf-8-sig") # Aquí se lee el archvio CSV y se convierte en un Dataframe. El encoding utf-8-sig se utiliza para que se traten correctamente los datos en el csv. 
print(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas") # Se imprime por pantalla el numero de filas que tiene el dataset cargado. 

VARS_COLOR = ["mean_R", "mean_G", "mean_B", "mean_L", "mean_a", "mean_b", "contrast_L"] # Creo una lista con las 7 variables numéricas de colores ya que se va a utilizar más delante a menudo. 


# ==============================================================
# 2. GRÁFICOS INICIALES — estado del dataset antes de nada
# Mostramos de dónde vienen los datos y cómo se distribuyen
# por categoría antes de aplicar ninguna transformación.
# ==============================================================

# Gráfico A: productos por fuente de datos
plt.figure(figsize=(8, 5)) # Estos dos datos se hacen antes de cualquier transformacion deliberadamente. Se hace así para comparar los resultados antes y después de las transformaciones. 
# El gráfico es de 8 x 5 pulgadas. 
df["fuente"].value_counts().plot(kind="bar", color=["#2E75B6", "#099204", "#FF3838"]) # Seleccionamos solo la variables fuente del dataframe y contamos cuantso valores distintos tienen. Definimos que 
# buscamos un gráfico de barras y con los colores determinados 
plt.title("Número de productos por fuente de datos", fontsize=14, fontweight="bold") # Aquí especifico el título del gráfico y las características que quiero que tenga dicho titulo. 
plt.xlabel("Fuente")
plt.ylabel("Número de productos")
plt.xticks(rotation=15)
plt.tight_layout() # Se especifica que se ajusten automáticamente los márgenes para que no se corte nada. 
plt.savefig(os.path.join(RUTA_GRAFICOS, "0a_distribucion_fuente.png"), dpi=150) # Aquí se especifica donde se va a guardar el gráfico creando la ruta con os.path.join. dpi = 150 es la resolución, 150 puntos por pulgada
plt.close()
print("✓ 0a_distribucion_fuente.png")

# Gráfico B: top 15 categorías con más productos
plt.figure(figsize=(12, 6)) # Quiero que este gráfico sea de 12 pulgadas por 6. 
df["categoria"].value_counts().head(15).plot(kind="barh", color="#2E75B6") # Aquí buscamos que se cogan las 15 categorías con más imagenes y que se represente en un diagrama de barras. 
plt.title("Top 15 categorías con más productos", fontsize=14, fontweight="bold")
plt.xlabel("Número de productos")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "0b_distribucion_categoria.png"), dpi=150)
plt.close()
print("✓ 0b_distribucion_categoria.png")


# ==============================================================
# 3. NULOS
# Si hay menos del 1% los eliminamos y lo documentamos.
# Si hubiera más usaríamos interpolación con la mediana.
# ==============================================================
nulos = df[VARS_COLOR].isnull().sum().sum() # Aqui se suman todos los nulos de las variables cromáticas que hemos guardado antes en VARS_COLOR
pct   = nulos / (len(df) * len(VARS_COLOR)) * 100 # Y además calculamos el porcentaje que representan esos nulos sobre el total
print(f"\nNulos en variables de color: {nulos} ({pct:.2f}%)")

if pct < 1.0: # Si ese porcentaje de nulos es menor que el 1%, no se hace nada porque la cantidad es muy baja como para que afecte a nuestro modelo. 
    df = df.dropna(subset=VARS_COLOR).reset_index(drop=True)
    print(f"→ Menos del 1%: filas con nulos eliminadas. Quedan {len(df)} filas.")
else:
    for v in VARS_COLOR: # EN cambio, eliminarlas sesgaría el dataset por lo que lo que se hace es sustituirlo por la mediana de la variable. 
        df[v] = df[v].fillna(df[v].median())
    print("→ Más del 1%: nulos reemplazados por la mediana de cada variable.")


# ==============================================================
# 4. DUPLICADOS
# Eliminamos filas con la misma URL de imagen (mismo producto).
# ==============================================================
antes = len(df) # drop_duplicates(subset=["imagen_url"]) elimina filas que tengan la misma URL de imagen. Se usa la URL como identificador único del 
df = df.drop_duplicates(subset=["imagen_url"]).reset_index(drop=True) # producto porque si dos filas tienen la misma imagen son el mismo producto aunque tengan nombres distintos.
print(f"\nDuplicados eliminados: {antes - len(df)}. Quedan {len(df)} filas.")


# ==============================================================
# 5. VALIDACIÓN DE RANGOS
# Comprobamos que los valores están dentro de sus rangos
# teóricos antes de aplicar el tratamiento de outliers.
# ==============================================================
RANGOS = {"mean_R":(0,255),"mean_G":(0,255),"mean_B":(0,255), # Se crea el diccionario rangos donde cada clave es el nombre d euna variable y cada valor es una tupla con el min y máx teórico. 
          "mean_L":(0,100),"mean_a":(-128,128),"mean_b":(-128,128),"contrast_L":(0,100)}
print("\nValidación de rangos:")
for var, (vmin, vmax) in RANGOS.items(): # Se hace un bucle for para ir "deseampaquetando" cada tupla en dos variables. 
    fuera = ((df[var] < vmin) | (df[var] > vmax)).sum() # Se utiliza | para que haga la función de un "or". Se crea una serie de Tru / False donde True significa que hay algún valor fuera de rango. 
    print(f"  {var:<12}: {'✓ OK' if fuera == 0 else f'⚠ {fuera} fuera de rango'}") # Si no hay ningún valor fuera del rango de valores cromáticos, imprime un ok, si no un mensaje de warning. 


# ==============================================================
# 6. OUTLIERS — Método IQR con capping
# No eliminamos filas porque colores extremos son información
# válida. Los reemplazamos por el límite del rango IQR.
# ==============================================================
print("\nTratamiento de outliers (IQR con capping):")

# Guardamos L* antes del capping para el gráfico antes/después
l_antes = df["mean_L"].copy() # Se utiliza el .copy() para evitar que se modifique la columna original. 

for v in VARS_COLOR:
    Q1, Q3 = df[v].quantile(0.25), df[v].quantile(0.75) # Se guardan los valores de Q1 y Q3 para cada variable
    IQR    = Q3 - Q1 # Se calcula el IQR para cada una de ellas. 
    lim_inf = max(Q1 - 1.5 * IQR, RANGOS[v][0]) # Todo lo que se quede por fuera de estos límites se considera putlier. 
    lim_sup = min(Q3 + 1.5 * IQR, RANGOS[v][1])
    n_out   = ((df[v] < lim_inf) | (df[v] > lim_sup)).sum() # Se hace una suma para ver cuántos valores hay que estén fuera del rango intercuartílico. 
    df[v]   = df[v].clip(lim_inf, lim_sup) # Cualquier valor por encima del limite superior se remplaza por el limite superiro y lo mismo con el inferior. 
    print(f"  {v:<12}: {n_out} outliers → limitados a [{lim_inf:.1f}, {lim_sup:.1f}]")


# ==============================================================
# 7. VARIABLES HSV
# Convertimos RGB a HSV para añadir tono (H), saturación (S)
# y brillo (V) como variables independientes para el modelo. Estas variables se añaden al modelo porque capturan información sobre el color de una forma más intuitiva y complementaria a CIELAB
# ==============================================================
def rgb_a_hsv(r, g, b): # Esta función recibe tres números, los valores RGB y devuelve los HSV
    r, g, b = r/255, g/255, b/255 # Normaliza todos los valores en una escala de 0 a 1 dividiendo entre 255. 
    cmax, cmin = max(r,g,b), min(r,g,b)
    delta = cmax - cmin # Si el resultado es 0, significa que R,g y B son iguales, es decir, el color es un gris puro sin tono. 
    if delta == 0:    h = 0 # Esta es la fórmula matemática del tono H. Dependiendo de qué canal sea el dominante, el tono se calcula de forma distinta.
    elif cmax == r:   h = 60 * (((g-b)/delta) % 6) # El % 6 en el primer caso es el operador módulo, que evita que el resultado se salga del rango correcto.
    elif cmax == g:   h = 60 * (((b-r)/delta) + 2) # El resultado es un ángulo en grados de 0 a 360, donde 0° es rojo, 120° es verde y 240° es azul
    else:             h = 60 * (((r-g)/delta) + 4) 
    s = 0 if cmax == 0 else (delta/cmax)*100 # La saturación S es el cociente entre delta y cmax, multiplicado por 100 para expresarlo en porcentaje. Si cmax es 0 (pixel negro) la saturación es 0 para evitar división por cero.
    v = cmax * 100 # El brillo V es directamente el canal dominante multiplicado por 100
    return round(h,2), round(s,2), round(v,2)

hsv = df.apply(lambda row: rgb_a_hsv(row.mean_R, row.mean_G, row.mean_B), axis=1) # df.apply() aplica una función a cada fila del DataFrame. azis = 1 significa "recorre fila por fila"
# lambda row: rgb_a_hsv(row.mean_R, row.mean_G, row.mean_B) significa "para cada fila, llama a rgb_a_hsv pasándole los tres valores RGB de esa fila".
df["hsv_h"] = hsv.apply(lambda x: x[0]) # Estas tres líneas extraen cada componente de la tupla usando su índice (0,1,2) y crean tres columnas nuevas en el dataframe. 
df["hsv_s"] = hsv.apply(lambda x: x[1])
df["hsv_v"] = hsv.apply(lambda x: x[2])
print(f"\n✓ Variables HSV añadidas")

# ==============================================================
# 8. VARIABLE OBJETIVO: EMOCIÓN
# Sistema de scoring gaussiano ponderado basado en:
#   - Valdez & Mehrabian (1994): pesos L* para emociones pasivas
#   - Gilbert et al. (2016): pesos C* y H* para emociones activas
#   - Russell (1980): modelo circunflejo → espacio continuo
# ==============================================================

import math

# Centroides emocionales (µ_L, σ_L, µ_C, σ_C, µ_H, σ_H, w_L, w_C, w_H)
CENTROIDES = {
    "Ira":           (22,  8,  10,  6,  0.10, 0.60, 0.50, 0.35, 0.15),
    "Tristeza":      (42, 10,   4,  4,  0.00, 1.50, 0.65, 0.30, 0.05),
    "Romanticismo":  (50, 12,  12,  7,  0.35, 0.55, 0.30, 0.35, 0.35),
    "Energía":       (62, 10,  15,  8,  0.25, 0.60, 0.25, 0.40, 0.35),
    "Alegría":       (80,  8,  16,  7,  0.85, 0.55, 0.30, 0.35, 0.35),
    "Relajación":    (92,  6,   3,  3,  0.00, 1.80, 0.70, 0.25, 0.05),
    "Calma":         (85,  7,   5,  4, -0.40, 1.20, 0.55, 0.30, 0.15),
    "Aburrimiento":  (70, 12,   5,  4,  0.00, 2.00, 0.40, 0.40, 0.20),
}

UMBRAL_CONFIANZA = 0.22  # Si ninguna emoción supera este umbral → "Neutro/Ambiguo"

def _gauss(x, mu, sig):
    # Función gaussiana: devuelve 1.0 en el centro y cae suavemente hacia los lados
    return math.exp(-0.5 * ((x - mu) / sig) ** 2)

def _dist_angular(h1, h2):
    # Distancia circular entre dos ángulos en radianes (H* es circular: va de -π a π)
    d = abs(h1 - h2)
    return min(d, 2 * math.pi - d)

def asignar_emocion_v2(L, a, b):
    """
    Asigna emoción mediante scoring gaussiano ponderado en el espacio
    CIELAB extendido con Croma C* y ángulo de tono H*.
    Devuelve (emocion, confianza, dict_scores).
    """
    C = math.sqrt(a**2 + b**2)   # Croma: intensidad total del color
    H = math.atan2(b, a)          # Ángulo de tono: dirección del color en el plano a*b*

    scores_raw = {}
    for emo, (muL, sL, muC, sC, muH, sH, wL, wC, wH) in CENTROIDES.items():
        sL_ = _gauss(L, muL, sL)                          # Score en luminosidad
        sC_ = _gauss(C, muC, sC)                          # Score en croma
        sH_ = math.exp(-0.5 * (_dist_angular(H, muH) / sH) ** 2)  # Score en tono
        scores_raw[emo] = wL * sL_ + wC * sC_ + wH * sH_ # Score ponderado final

    total = sum(scores_raw.values())
    scores_prob = {e: v / total for e, v in scores_raw.items()}  # Softmax → probabilidades

    ganadora = max(scores_prob, key=scores_prob.get)
    confianza = scores_prob[ganadora]

    if confianza < UMBRAL_CONFIANZA:
        ganadora = "Neutro/Ambiguo"

    return ganadora, round(confianza, 4), scores_prob

# Aplicamos la función fila por fila al DataFrame
resultados = df.apply(
    lambda r: asignar_emocion_v2(r["mean_L"], r["mean_a"], r["mean_b"]), axis=1
)

df["emocion"]             = resultados.apply(lambda x: x[0])  # Emoción ganadora
df["confianza_emocional"] = resultados.apply(lambda x: x[1])  # Probabilidad softmax [0-1]

# Ocho columnas de score individual (una por emoción)
# Capturan la ambigüedad emocional del color de forma continua
nombres_emo = list(CENTROIDES.keys())
for emo in nombres_emo:
    col = "score_" + emo.lower().replace("é", "e").replace("ó", "o").replace("/", "_")
    df[col] = resultados.apply(lambda x: round(x[2][emo], 4))

# Distancia al centroide de la emoción asignada (solo para emociones con centroide definido)
CENTROIDES_ZONA = {
    "Ira": (20, 5, 3), "Tristeza": (40, 1, 2), "Romanticismo": (48, 7, 10),
    "Energía": (62, 8, 13), "Alegría": (82, 5, 18), "Relajación": (93, 0, 1),
    "Calma": (86, 1, 2), "Aburrimiento": (72, 2, 4),
}

def distancia_zona(row):
    # Distancia euclidiana al centroide de la emoción asignada
    emo = row["emocion"]
    if emo not in CENTROIDES_ZONA:
        return 0.0
    cL, ca, cb = CENTROIDES_ZONA[emo]
    return round(((row["mean_L"]-cL)**2 + (row["mean_a"]-ca)**2 + (row["mean_b"]-cb)**2)**0.5, 2)

df["distancia_centroide"] = df.apply(distancia_zona, axis=1)

print("\n✓ Emoción asignada (sistema gaussiano v2). Distribución:")
print(df["emocion"].value_counts().to_string())
print(f"\n  Confianza media:   {df['confianza_emocional'].mean():.3f}")
print(f"  Productos Neutro/Ambiguo: {(df['emocion']=='Neutro/Ambiguo').sum()}")

# ==============================================================
# 9. VARIABLES DE NEGOCIO
# Traducen valores técnicos a conceptos útiles para marketing.
# ==============================================================

# Temperatura de color
def temperatura(a, b): # Aquí convertimos los valores numéricos de a* y b* en una etiqueta cualitativa que sea más interpretable por humanos. 
    if a > 3 or b > 10:    return "Cálido"
    elif a < -3 or b < -5: return "Frío"
    else:                   return "Neutro"
df["temperatura_color"] = df.apply(lambda r: temperatura(r.mean_a, r.mean_b), axis=1)

# Luminosidad y saturación categorizadas
df["luminosidad_cat"] = pd.cut(df["mean_L"], bins=[0,40,70,100], # pd.cut() divide una variable continua en intervalos y asigna una etiqueta a cada uno. bins=[0,40,70,100] define los límites: 
                                labels=["Oscuro","Medio","Luminoso"]) # de 0 a 40 es Oscuro, de 40 a 70 es Medio, de 70 a 100 es Luminoso. Lo mismo hace saturacion_cat con hsv_s
df["saturacion_cat"]  = pd.cut(df["hsv_s"], bins=[0,25,60,100],
                                labels=["Apagado","Moderado","Intenso"])

# Coherencia emocional: qué tan clara es la señal (0=ambiguo, 100=perfecto)
dist_max = df["distancia_centroide"].max() # Convierte la distancia al centroide en una escala de 0 a 100 donde 100 es máxima coherencia (producto justo en el centro de su zona) 
df["coherencia_emocional"] = ((1 - df["distancia_centroide"] / dist_max) * 100).round(1) # y 0 es mínima (producto en el extremo más alejado). La fórmula 1 - distancia/distancia_máxima invierte la escala: a mayor distancia, menor coherencia.

# Emoción óptima por categoría y si el producto está alineado
EMOCION_OPTIMA = { # Este es un diccionario que asigna a cada categoría la emoción que debería transmitir según criterior de marketing. 
    "Bebidas":"Energía","Agua":"Calma","Zumos":"Alegría","Refrescos":"Energía",
    "Cervezas":"Relajación","Cerveza Mahou":"Relajación","Cerveza San Miguel":"Relajación",
    "Cerveza Alhambra":"Relajación","Corona":"Relajación","Founders Brewing":"Relajación",
    "Budweiser":"Energía","Nómada Brewing":"Relajación","Brutus":"Energía",
    "Vinos":"Romanticismo","Lácteos":"Calma","Leche":"Calma","Yogures":"Alegría",
    "Quesos":"Relajación","Chocolates":"Romanticismo","Snacks":"Alegría",
    "Patatas fritas":"Alegría","Galletas":"Alegría","Cereales y desayuno":"Energía",
    "Dulces y caramelos":"Alegría","Conservas":"Calma","Salsas":"Energía",
    "Aceites":"Relajación","Pasta":"Alegría","Arroz":"Calma","Pan y bollería":"Alegría",
    "Congelados":"Calma","Carne":"Energía","Pescado":"Calma","Frutas":"Alegría",
    "Verduras":"Calma","Legumbres":"Calma","Café e infusiones":"Relajación",
    "Condimentos":"Energía","Agua Solán de Cabras":"Calma","Agua Sierra Natura":"Calma",
    "Malta y otras bebidas":"Energía","Cristalería y hogar":"Relajación",
    "Moda y accesorios":"Alegría",
    "Moda - Camisetas":"Alegría","Moda - Pantalones":"Calma","Moda - Vestidos":"Romanticismo",
    "Moda - Chaquetas":"Calma","Moda - Abrigos":"Calma","Moda - Jerseys":"Relajación",
    "Moda - Faldas":"Romanticismo","Moda - Zapatos":"Energía","Moda - Botas":"Calma",
    "Moda - Zapatillas":"Energía","Moda - Sandalias":"Alegría","Moda - Bolsos":"Romanticismo",
    "Moda - Mochilas":"Energía","Moda - Carteras":"Calma","Moda - Cinturones":"Calma",
    "Moda - Relojes":"Calma","Moda - Gafas de sol":"Alegría","Moda - Sombreros":"Alegría",
    "Moda - Bufandas":"Relajación","Moda - Guantes":"Calma","Moda - Joyería":"Romanticismo",
    "Moda - Collares":"Romanticismo","Moda - Pulseras":"Romanticismo",
    "Moda - Pendientes":"Romanticismo","Moda - Anillos":"Romanticismo",
    "Hogar - Sofás":"Relajación","Hogar - Sillas":"Calma","Hogar - Mesas":"Calma",
    "Hogar - Camas":"Relajación","Hogar - Lámparas":"Relajación","Hogar - Escritorios":"Calma",
    "Hogar - Estantes":"Calma","Hogar - Armarios":"Calma","Hogar - Espejos":"Calma",
    "Hogar - Alfombras":"Relajación","Hogar - Cortinas":"Calma","Hogar - Cojines":"Relajación",
    "Hogar - Mantas":"Relajación","Hogar - Jarrones":"Calma","Hogar - Marcos":"Calma",
    "Hogar - Relojes pared":"Calma",
    "Cocina - Tazas":"Alegría","Cocina - Vasos":"Calma","Cocina - Platos":"Calma",
    "Cocina - Boles":"Calma","Cocina - Ollas":"Calma","Cocina - Sartenes":"Calma",
    "Cocina - Cuchillos":"Calma","Cocina - Tablas":"Calma","Cocina - Teteras":"Relajación",
    "Cocina - Tostadoras":"Alegría","Cocina - Batidoras":"Energía","Cocina - Cafeteras":"Relajación",
    "Cocina - Copas":"Romanticismo","Cocina - Botellas":"Calma","Cocina - Recipientes":"Calma",
    "Tecnología - Portátiles":"Calma","Tecnología - Tablets":"Calma",
    "Tecnología - Auriculares":"Energía","Tecnología - Altavoces":"Energía",
    "Tecnología - Cámaras":"Calma","Tecnología - Teclados":"Calma",
    "Tecnología - Ratones":"Calma","Tecnología - Monitores":"Calma",
    "Tecnología - Fundas móvil":"Alegría",
    "Cosmética - Perfumes":"Romanticismo","Cosmética - Cremas":"Calma",
    "Cosmética - Champús":"Relajación","Cosmética - Cepillos":"Calma",
    "Cosmética - Secadores":"Calma","Cosmética - Maquillaje":"Romanticismo",
    "Cosmética - Labiales":"Romanticismo","Cosmética - Esmaltes":"Romanticismo",
    "Juguetes - General":"Alegría","Juguetes - Puzzles":"Calma",
    "Juguetes - Juegos de mesa":"Alegría","Juguetes - Muñecas":"Alegría",
    "Juguetes - Figuras":"Energía","Ocio - Libros":"Calma","Ocio - Cuadernos":"Calma",
    "Ocio - Bolígrafos":"Calma","Deporte - Esterillas":"Energía","Deporte - Botellas":"Energía",
    "Deporte - Bolsas":"Energía","Deporte - Bicis":"Energía","Deporte - Cascos":"Energía",
    "Mascotas - Camas":"Relajación","Mascotas - Comederos":"Alegría","Mascotas - Juguetes":"Alegría",
}
df["emocion_optima"]       = df["categoria"].map(EMOCION_OPTIMA).fillna("No definida") # .map() sustitutye cada valor de la columna categoria por su valor correspondiente en el diccionario. 
df["alineacion_emocional"] = (df["emocion"] == df["emocion_optima"]).astype(int) # Aquí se comparan las columnas de emocion y emocion optima fila a fila y devuelve True o False y convierto los valores en integers. 
pct_alineados = df["alineacion_emocional"].mean() * 100 # Se calculan cuánt oporcentaje de variables estaban alineadas. 
print(f"\n✓ Variables de negocio añadidas. Alineación media: {pct_alineados:.1f}%")


# ==============================================================
# 10. NORMALIZACIÓN Min-Max
# Escala todas las variables numéricas al rango [0,1] para que
# el modelo no se vea afectado por diferencias de escala.
# ==============================================================
VARS_NORM = ["mean_R","mean_G","mean_B","mean_L","mean_a","mean_b",
             "contrast_L","hsv_h","hsv_s","hsv_v"] # Aquí determino en una lista las variables que hay que normalizar. 
scaler    = MinMaxScaler() # Aquí se llama al objeto normalizador. (valor - minimo) / (máximo - minimo). 
cols_norm = [v + "_norm" for v in VARS_NORM] #[v + "_norm" for v in VARS_NORM] es una list comprehension, una forma compacta de crear una lista aplicando una operación a cada elemento. 
df[cols_norm] = scaler.fit_transform(df[VARS_NORM]).round(4) # Genera ["mean_R_norm", "mean_G_norm", ...] añadiendo el sufijo _norm a cada nombre.
print(f"\n✓ Normalización aplicada a {len(VARS_NORM)} variables") # fit calcula el mínimo y máximo de cada variable en el dataset, y transform aplica la normalización usando esos valores. El resultado es una matriz 
# numérica que se asigna directamente a las columnas nuevas del DataFrame.

# ==============================================================
# 11. GRÁFICOS DE ANÁLISIS
# ==============================================================
print("\nGenerando gráficos...")

COLORES = {"Alegría":"#F4D03F","Energía":"#E67E22","Calma":"#5DADE2", # Aquí se determinan los colores que se van a utilizar para representar cada una de las emociones. 
           "Romanticismo":"#F1948A","Tristeza":"#85929E","Ira":"#C0392B",
           "Aburrimiento":"#A9A9A9","Relajación":"#82E0AA"}
ORDEN = list(COLORES.keys()) # Aquí se extrae solo las claves del diccionario y las convierte en lista. Esto fija el orden en que aparecen las emociones en los gráficos. 

# Gráfico 1: Antes vs Después de outliers
fig, axes = plt.subplots(1, 2, figsize=(11, 4)) # Aquí se determina que se crea una figura con 1 fila y 2 columnas. axes[0] es el izquierdo y axes[1] es el derecho. 
fig.suptitle("Luminosidad L* — Antes y después del tratamiento de outliers",
             fontsize=12, fontweight="bold")
axes[0].hist(l_antes, bins=40, color="#E74C3C", edgecolor="white", alpha=0.85)
axes[0].set_title("Antes del capping")
axes[0].set_xlabel("L*")
axes[0].set_ylabel("Frecuencia")
axes[1].hist(df["mean_L"], bins=40, color="#27AE60", edgecolor="white", alpha=0.85) # Dibuja un histograma. bins = 40 significa que divide el rango de valores en 40 intervalos y alpha = =.85 es la transparencia
# 0 es invisible y 1 es opaco. 
axes[1].set_title("Después del capping")
axes[1].set_xlabel("L*")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "1_antes_despues_outliers.png"), dpi=150)
plt.close()
print("✓ 1_antes_despues_outliers.png")

# Gráfico 2: Distribución de emociones
fig, ax = plt.subplots(figsize=(9, 5))
conteo = df["emocion"].value_counts().reindex(ORDEN) #.reindex(ORDEN) reordena el resultado para que las barras aparezcan exactamente en el orden definido en ORDEN, no ordenadas por frecuencia como haría 
ax.bar(conteo.index, conteo.values, # .value_counts() por defecto.
       color=[COLORES[e] for e in ORDEN], edgecolor="white")
ax.set_title("Distribución de emociones en el dataset", fontsize=12, fontweight="bold")
ax.set_xlabel("Emoción")
ax.set_ylabel("Productos")
for i, v in enumerate(conteo.values): # Este bucle añade el número exacto encima de cada bara. "enumerate" da el indica i y el valor v a la vez. 
    ax.text(i, v + 30, str(v), ha="center", fontsize=9, fontweight="bold") # ax.test escribe texto en las coordenas x e y del gráfico. v+30 coloa el texto 30 uds por encima de la barra para no solapar. 
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "2_distribucion_emociones.png"), dpi=150)
plt.close()
print("✓ 2_distribucion_emociones.png")

# Gráficos 3-5: Boxplots por emoción (análisis bivariante clave)
for var, titulo in [("mean_L","Luminosidad L*"), # en lugar de scribir 3 veces el mismo código para tres gráficos identicos, se ha realizado un bucle. 
                    ("mean_a","Componente a* (rojo-verde)"),
                    ("mean_b","Componente b* (amarillo-azul)")]:
    fig, ax = plt.subplots(figsize=(11, 5))
    datos = [df[df["emocion"] == e][var].values for e in ORDEN] # Aquí se crean 8 arrays, 1 por emoción. Cada array contiene todos los valores de esa variable para los productos de esa emoción. 
    bp = ax.boxplot(datos, patch_artist=True, tick_labels=ORDEN) # patch_artist=True permite colorear el interior de las cajas, sin eso solo se dibujan los contornos. bp["boxes"] es la lista de objetos caja del gráfico. 
    for patch, emo in zip(bp["boxes"], ORDEN): # zip() empareja cada caja con su emoción para asignarle el color correcto.
        patch.set_facecolor(COLORES[emo])
        patch.set_alpha(0.8)
    ax.set_title(f"{titulo} por emoción", fontsize=12, fontweight="bold")
    ax.set_xlabel("Emoción")
    ax.set_ylabel(var)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, f"boxplot_{var}.png"), dpi=150)
    plt.close()
print("✓ boxplots L*, a*, b* por emoción")

# Gráfico 6: Matriz de correlación
vars_corr = ["mean_L","mean_a","mean_b","contrast_L","hsv_h","hsv_s","hsv_v"]
corr = df[vars_corr].corr().round(2)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1) # ax.imshow() convierte esa tabla numérica en una imagen donde cada número se representa con un color.
# cmap="coolwarm" es la paleta de colores: azul para valores negativos, rojo para positivos, blanco para cero.
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(vars_corr)))
ax.set_yticks(range(len(vars_corr)))
ax.set_xticklabels(vars_corr, rotation=45, ha="right")
ax.set_yticklabels(vars_corr)
for i in range(len(vars_corr)): # Estos son dos bucles anidados que recorren todas las celdas de la matriz y escriben el número encima de cada color. El texto es blanco si la correlación
    for j in range(len(vars_corr)): # es muy alta y negro u oscuro si es muy baja. 
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                fontweight="bold", color="white" if abs(v) > 0.6 else "black")
ax.set_title("Matriz de correlación", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "6_correlacion.png"), dpi=150)
plt.close()
print("✓ 6_correlacion.png")

# Gráfico 7: Paleta de colores real del dataset
# Cada rectángulo es el color medio real de un producto,
# ordenados por tono (hsv_h) para que se vea el arcoíris.
fig, ax = plt.subplots(figsize=(14, 3))
df_sorted = df.sort_values("hsv_h").reset_index(drop=True) # ordena los productos por tono HSV para que los colores aparezcan como un arco´iris de izquierda a derrehca. 
n = len(df_sorted)
for i, row in df_sorted.iterrows(): # iterrows() recorre el DataFrame fila a fila devolviendo el índice y la fila. Para cada producto dibuja un rectángulo de 1×1 en la posición i con el color real del producto.
    color = (row["mean_R"]/255, row["mean_G"]/255, row["mean_B"]/255)
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
ax.set_xlim(0, n)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title(f"Paleta de colores real del dataset — {n} productos ordenados por tono",
             fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "7_paleta_colores_real.png"), dpi=150)
plt.close()
print("✓ 7_paleta_colores_real.png")

# Gráfico 8: Mapa de calor emoción × categoría
# Muestra qué emoción predomina en cada categoría de producto.
# Valor = % de productos de esa categoría con esa emoción.
top_cats = df["categoria"].value_counts().head(12).index.tolist()
df_top   = df[df["categoria"].isin(top_cats)]
tabla    = pd.crosstab(df_top["categoria"], df_top["emocion"], normalize="index") * 100 # pd.crosstab() crea una tabla de contingencia: filas son categorías, columnas son emociones, y cada celda cuenta cuántos productos hay en esa combinación
tabla    = tabla.reindex(columns=ORDEN, fill_value=0).round(1) # normalize="index" divide cada fila entre su total, convirtiendo los conteos en porcentajes por categoría. Multiplicar por 100 lo pasa a escala 0-100.
fig, ax = plt.subplots(figsize=(13, 6)) # Se reordenan las columnas para que las emociones aparezcan siempre en el mismo orden. 
im = ax.imshow(tabla.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label="% de productos")
ax.set_xticks(range(len(tabla.columns)))
ax.set_yticks(range(len(tabla.index)))
ax.set_xticklabels(tabla.columns, rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(tabla.index, fontsize=9)
for i in range(len(tabla.index)):
    for j in range(len(tabla.columns)):
        v = tabla.values[i, j]
        if v > 5:
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if v > 60 else "black")
ax.set_title("Emoción predominante por categoría de producto (%)",
             fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "8_mapa_calor_emocion_categoria.png"), dpi=150)
plt.close()
print("✓ 8_mapa_calor_emocion_categoria.png")

# Gráfico 9: Scatter L* vs a* coloreado por emoción
# Cada punto es un producto. Demuestra visualmente que las
# emociones son separables en el espacio de color CIELAB,
# lo que justifica el enfoque del modelo.
fig, ax = plt.subplots(figsize=(11, 7))
for emo in ORDEN:
    sub = df[df["emocion"] == emo]
    sub = sub.sample(min(400, len(sub)), random_state=42) # SI una emoción tiene más de 400 productos, coge una muestra aleatoria de 400. Dibujar más sería contraproducente. 
    ax.scatter(sub["mean_L"], sub["mean_a"], # con random_state, se fija una semilla aleatoria para que la muestra sea siempre la misma. 
               c=COLORES[emo], label=f"{emo} (n={len(df[df['emocion']==emo])})",
               alpha=0.45, s=18, edgecolors="none")
ax.set_xlabel("L* — Luminosidad (oscuro → luminoso)", fontsize=11)
ax.set_ylabel("a* — Componente rojo-verde", fontsize=11)
ax.set_title("Separabilidad emocional en el espacio CIELAB (L* vs a*)",
             fontsize=12, fontweight="bold")
ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
ax.axvline(50, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "9_scatter_separabilidad_emocional.png"), dpi=150)
plt.close()
print("✓ 9_scatter_separabilidad_emocional.png")


# Gráfico 10: Alineación emocional por categoría
# Porcentaje de productos que transmiten la emoción óptima
# para su categoría. Es el gráfico de negocio más potente:
# identifica qué categorías están bien alineadas y cuáles
# son una oportunidad de mejora para las marcas.
cats_con_optima = [c for c in df["categoria"].unique() if c in EMOCION_OPTIMA] # Aquí se recogen las categorías que tienen emoción óptima definida. 
if cats_con_optima: # SI la lista no está vacía siga adelante. 
    df_alin = df[df["categoria"].isin(cats_con_optima)].copy()
    df_alin["alineada"] = df_alin["emocion"] == df_alin["categoria"].map(EMOCION_OPTIMA)
    alineacion_cat = (df_alin.groupby("categoria")["alineada"] # groupby("categoria") agrupa el DataFrame por categoría. ["alineada"].mean() calcula la media de la variable binaria 0/1 dentro de cada grupo,
                      .mean() #  que es equivalente al porcentaje de productos alineados. 
                      .sort_values() #.sort_values() ordena de menor a mayor para que el gráfico de barras horizontales quede con la peor categoría arriba y la mejor abajo.
                      * 100)
    media_alin = alineacion_cat.mean()

    fig, ax = plt.subplots(figsize=(10, max(5, len(alineacion_cat) * 0.5)))
    colores_barras = ["#E74C3C" if v < 40 else "#F39C12" if v < 60 else "#27AE60"
                      for v in alineacion_cat.values]
    barras = ax.barh(alineacion_cat.index, alineacion_cat.values,
                     color=colores_barras, edgecolor="white", height=0.6)
    ax.axvline(media_alin, color="#2C3E50", linestyle="--", linewidth=1.5,
               label=f"Media: {media_alin:.1f}%")
    for barra, val in zip(barras, alineacion_cat.values): #barra.get_y() devuelve la posición vertical de la barra y barra.get_height()/2 añade la mitad de su altura, colocando el texto exactamente centrado 
        ax.text(val + 0.8, barra.get_y() + barra.get_height() / 2, # verticalmente respecto a cada barra. val + 0.8 lo desplaza ligeramente a la derecha del extremo de la barra.
                f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("% de productos alineados con su emoción óptima", fontsize=11)
    ax.set_title("Alineación emocional por categoría\n"
                 "(verde ≥ 60% · naranja 40-60% · rojo < 40%)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, "10_alineacion_por_categoria.png"), dpi=150)
    plt.close()
    print("✓ 10_alineacion_por_categoria.png")

# Gráfico 11: Posicionamiento cromático por fuente de datos
# Muestra las diferencias estructurales de color entre las tres
# fuentes en el espacio L* vs b*. Demuestra que el analista ha
# identificado y documentado los sesgos cromáticos del dataset.
fig, ax = plt.subplots(figsize=(9, 6))
COLORES_FUENTE = {
    "Amazon Berkeley Objects": "#2E75B6",
    "Open Food Facts":         "#70AD47",
    "Mahou San Miguel":        "#C00000",
}
for fuente, sub in df.groupby("fuente"): # df.groupby("fuente") divide el DataFrame en tres grupos, uno por fuente, y el bucle los recorre uno a uno. 
    muestra = sub.sample(min(500, len(sub)), random_state=42) # sub es el subDataFrame de esa fuente concreta. Primero dibuja hasta 500 puntos pequeños y transparentes para 
    ax.scatter(muestra["mean_L"], muestra["mean_b"], # mostrar la nube de productos, y luego encima dibuja el centroide.
               c=COLORES_FUENTE.get(fuente, "#999999"),
               label=fuente, alpha=0.35, s=14, edgecolors="none")
    # Centroide de cada fuente
    cx, cy = sub["mean_L"].mean(), sub["mean_b"].mean() # cx y cy son las coordenadas del centroide, simplemente la media de L* y b* de todos los productos de esa fuente
    ax.scatter(cx, cy, c=COLORES_FUENTE.get(fuente, "#999999"),
               s=180, edgecolors="black", linewidths=1.5, zorder=5) # zorder controla el orden de capas: 
    ax.annotate(f"{fuente}\nL*={cx:.1f}  b*={cy:.2f}", # ax.annotate() dibuja una etiqueta con flecha. xy es el punto al que apunta la flecha (el centroide), xytext es donde se escribe el texto 
                xy=(cx, cy), xytext=(cx + 2, cy + 1.2), # (desplazado para no tapar el punto). arrowprops
                fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
ax.set_xlabel("L* — Luminosidad", fontsize=11)
ax.set_ylabel("b* — Componente amarillo-azul", fontsize=11)
ax.set_title("Posicionamiento cromático por fuente de datos\n"
             "(puntos grandes = centroide de cada fuente)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.9)
ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
ax.axvline(50, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "11_posicionamiento_cromatico_fuente.png"), dpi=150)
plt.close()
print("✓ 11_posicionamiento_cromatico_fuente.png")

# Gráfico 12: Contraste interno por emoción (boxplot)
# El contraste (contrast_L = desv. típica de L*) mide la
# homogeneidad visual del producto. Relajación → productos
# uniformes; Tristeza y Aburrimiento → más heterogéneos.
# Hallazgo original no presente en la literatura de referencia.
fig, ax = plt.subplots(figsize=(11, 6))
datos_contraste = [df[df["emocion"] == emo]["contrast_L"].dropna().values
                   for emo in ORDEN]
bp = ax.boxplot(datos_contraste, labels=ORDEN, patch_artist=True, # Este boxplot tiene más parámetros de ersonalización que los anteriores. Cada propos configura una parte distinta del boxplot. 
                medianprops=dict(color="black", linewidth=2), # Esto es la línea central de la caja. 
                whiskerprops=dict(linewidth=1.2), # son los bigotes, las lineas que salen de la caja hacia los extremos
                capprops=dict(linewidth=1.2), # son los extremos de los bigotes, las lineas horizontales de los extremos
                flierprops=dict(marker="o", markersize=2, alpha=0.3)) # Estos son los puntos outliers que quedan fuera de los bigotes. 
for patch, emo in zip(bp["boxes"], ORDEN):
    patch.set_facecolor(COLORES[emo])
    patch.set_alpha(0.75)
# Anotar la media encima de cada caja
for i, emo in enumerate(ORDEN): # Aquí se añade el valor de la media encima de cada caja para que sea facil comparar numeros exactos sin tener que estimarlos visualmente. 
    media = df[df["emocion"] == emo]["contrast_L"].mean()
    ax.text(i + 1, media + 1.5, f"{media:.1f}", ha="center",
            fontsize=8, fontweight="bold", color="#2C3E50")
ax.set_xlabel("Emoción asignada", fontsize=11)
ax.set_ylabel("Contraste interno (desv. típica de L*)", fontsize=11)
ax.set_title("Contraste visual interno por emoción\n"
             "(mayor valor = producto visualmente más heterogéneo)",
             fontsize=12, fontweight="bold")
ax.set_xticklabels(ORDEN, rotation=20, ha="right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "12_contraste_por_emocion.png"), dpi=150)
plt.close()
print("✓ 12_contraste_por_emocion.png")


# ==============================================================
# 12. GUARDAR CSV FINAL Y RESUMEN
# ==============================================================
df.to_csv(RUTA_SALIDA, index=False, encoding="utf-8-sig") # Esta es la linea que guarda el csv final con todas las columnas nuevas que se han ido añadiendo a lo largo del script. 

print(f"""
{'='*55}
✓ COMPLETADO 
{'='*55}
  Filas:            {len(df)}
  Columnas totales: {len(df.columns)}
  Emociones:        {df['emocion'].nunique()} clases
  Coherencia media: {df['coherencia_emocional'].mean():.1f}/100
  Alineación media: {pct_alineados:.1f}%
  CSV guardado en:  {RUTA_SALIDA}
{'='*55}
""") # Aquí se imprimen unas mini estadisticas basicas del dataframe final.