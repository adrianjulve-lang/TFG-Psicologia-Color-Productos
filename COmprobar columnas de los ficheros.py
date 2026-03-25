import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

files = {
    "oidv7-class-descriptions-boxable.csv": os.path.join(BASE_DIR, "oidv7-class-descriptions-boxable.csv"),
    "train-annotations-human-imagelabels-boxable.csv": os.path.join(BASE_DIR, "train-annotations-human-imagelabels-boxable.csv"),
    "open-images-dataset-train0.tsv": os.path.join(BASE_DIR, "open-images-dataset-train0.tsv"),
}

for name, path in files.items():
    print("\n", name)
    if not os.path.exists(path):
        print("  ❌ No existe en:", path)
        continue

    try:
        if name.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", nrows=5)
        else:
            df = pd.read_csv(path, nrows=5)

        print("  ✅ Columnas:", df.columns.tolist())

    except Exception as e:
        print("  ❌ Error al leer:", e)
