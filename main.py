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

    def smart_group_name(file_list):
    prefix = common_prefix(file_list)

    # Wenn Prefix zu kurz, nimm GPT oder einfach Ordner „Unsortiert“
    if len(file_list) >= 2 and len(prefix) >= 4:
        return prefix.strip("_- ")
    elif len(file_list) == 1:
        return file_list[0].split(".")[0].strip("_- ")  # z. B. "190 Hausaufgaben"
    else:
        return "Unsortiert"

    result = {smart_group_name(v): v for v in clusters.values()}
