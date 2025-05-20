from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import hdbscan
import json

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

def common_prefix(strings):
    from os.path import commonprefix
    return commonprefix(strings).rstrip("_- .")

def smart_group_name(files):
    if len(files) == 1:
        # Nur eine Datei → verwende den Namen ohne Erweiterung
        return files[0].rsplit('.', 1)[0].replace("_", " ").replace("-", " ")
    else:
        # Mehrere Dateien → gemeinsamen Teil suchen
        prefix = common_prefix(files)
        # Falls sinnvoller Prefix gefunden
        if prefix and len(prefix.strip()) >= 3:
            return prefix.strip().replace("_", " ").replace("-", " ")
        else:
            return "Unsortiert"

@app.post("/cluster")
async def cluster_files(request: Request):
    file_list = await request.json()

    if not isinstance(file_list, list):
        return {"error": "Expected a list of filenames"}

    # 🔐 Schutz bei nur 1 Datei → keine Clustering-Versuche
    if len(file_list) < 2:
        name = smart_group_name(file_list)
        return {name: file_list}

    # Normaler Clustering-Ablauf
    embeddings = model.encode(file_list)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(file_list[i])

    # 🔁 Gruppennamen pro Cluster intelligent bestimmen
    result = {smart_group_name(v): v for v in clusters.values()}
    return result
