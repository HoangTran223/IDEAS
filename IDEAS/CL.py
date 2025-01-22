import numpy as np
import os

DATASET = "data/IMDB"
OUTPUT_FOLDER = os.path.join(DATASET, "LLM")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

dataset = []
with open(os.path.join(DATASET, "train_texts.txt")) as fIn:
    for line in fIn:
        dataset.append(line.strip())

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
query_prompt_name = "s2s_query"
embeddings = embedding_model.encode(dataset, prompt_name=query_prompt_name)

def embed_texts(texts):
    return embedding_model.encode(
        texts,
        show_progress_bar=True,
        prompt_name="s2s_query"
    )


from umap import UMAP

# Default
dim_rec_model = UMAP(n_neighbors= 30, n_components=5, min_dist=1e-3, metric='euclidean', random_state=42)

from sklearn.cluster import AgglomerativeClustering
cluster_model = AgglomerativeClustering(
    n_clusters=None,
    linkage='ward'
)