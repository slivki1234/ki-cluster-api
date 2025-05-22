from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import hdbscan
import re
from os.path import commonprefix
import os
import uvicorn

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

def cleaned_filename(name):
    name = re.sub(r'\(.*?\)', '', name)  # Entferne (1), (2), etc.
    name = re.sub(r'\d+', '', name)      # Entferne Zahlen
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
    try:
        file_list = await request.json()

        if not isinstance(file_list, list):
            return {"error": "Expected a list of filenames"}

        if not file_list or not all(isinstance(x, str) for x in file_list):
            return {"error": "Invalid input. Expected list of strings."}

        if len(file_list) < 2:
            name = smart_group_name(file_list)
            return {name: file_list}

        embeddings = model.encode(file_list)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(embeddings)

        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue  # Einzeldateien werden nicht gruppiert
            clusters.setdefault(label, []).append(file_list[i])

        if not clusters:
            return {"Unsortiert": file_list}

        result = {smart_group_name(v): v for v in clusters.values()}
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": "Internal server error", "details": str(e)}

# ✅ Für Render: automatischer Port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"💡 Starte auf PORT {port} ...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)

