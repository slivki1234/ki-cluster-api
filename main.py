from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import hdbscan
import re
from os.path import commonprefix
import os
import uvicorn

app = FastAPI()

# 🧠 Modell laden
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🌐 CORS für lifeos.live aktivieren (Frontend-Domain erlauben)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lifeos.live"],  # Für Render-API Zugriff aus Netcup-Frontend
    allow_methods=["*"],
    allow_headers=["*"]
)

# 🔤 Hilfsfunktion: Dateinamen bereinigen
def cleaned_filename(name):
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\d+', '', name)
    name = name.replace('_', ' ').replace('-', ' ')
    return name.strip()

# 📁 Gruppennamen bestimmen
def smart_group_name(files):
    base_names = [cleaned_filename(f.rsplit('.', 1)[0]) for f in files]
    if len(base_names) == 1:
        return base_names[0]
    prefix = commonprefix(base_names).strip()
    return prefix if len(prefix) >= 3 else "Unsortiert"

# 🚀 API-Endpunkt für das Clustering
@app.post("/cluster")
async def cluster_files(request: Request):
    try:
        file_list = await request.json()

        # Validierung
        if not isinstance(file_list, list):
            return {"error": "Expected a list of filenames"}
        if not file_list or not all(isinstance(x, str) for x in file_list):
            return {"error": "Invalid input. Expected list of strings."}

        # Nur 1 Datei → keine Clustering nötig
        if len(file_list) < 2:
            name = smart_group_name(file_list)
            return {name: file_list}

        # Vektorisierung und Clustering
        embeddings = model.encode(file_list)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(embeddings)

        # Clustern
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue  # Einzeldateien werden ignoriert
            clusters.setdefault(label, []).append(file_list[i])

        if not clusters:
            return {"Unsortiert": file_list}

        # Gruppennamen berechnen
        result = {smart_group_name(v): v for v in clusters.values()}
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": "Internal server error", "details": str(e)}

# 🟢 Startbefehl für Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"💡 Starte auf PORT {port} ...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
