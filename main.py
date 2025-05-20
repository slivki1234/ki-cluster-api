from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import hdbscan
import json

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

def common_prefix(strings):
    from os.path import commonprefix
    return commonprefix(strings).rstrip("_- ")

@app.post("/cluster")
async def cluster_files(request: Request):
    file_list = await request.json()
    if not isinstance(file_list, list):
        return {"error": "Expected a list of filenames"}

    embeddings = model.encode(file_list)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(file_list[i])

    result = {common_prefix(v): v for v in clusters.values()}
    return result
