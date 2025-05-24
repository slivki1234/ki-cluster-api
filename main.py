from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import hdbscan
import re
from os.path import commonprefix

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

def cleaned_filename(name):
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\d+', '', name)
    name = name.replace('_', ' ').replace('-', ' ')
    return name.strip()

def smart_group_name(files):
    base_names = [cleaned_filename(f.rsplit('.', 1)[0]) for f in files]
    if len(base_names) == 1:
        return base_names[0]
    prefix = commonprefix(base_names).strip()
    return prefix if len(prefix) >= 3 else "Unsortiert"

@app.post("/cluster")
async def cluster_files(request: Request):
    file_list = await request.json()
    if not isinstance(file_list, list) or not all(isinstance(x, str) for x in file_list):
        return {"error": "Ungültige Eingabe"}
    if len(file_list) < 2:
        return {smart_group_name(file_list): file_list}
    embeddings = model.encode(file_list)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(embeddings)
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(file_list[i])
    if not clusters:
        return {"Unsortiert": file_list}
    return {smart_group_name(v): v for v in clusters.values()}
