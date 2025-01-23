# import numpy as np
# import os

# DATASET = "data/IMDB"
# OUTPUT_FOLDER = os.path.join(DATASET, "LLM")
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# dataset = []
# with open(os.path.join(DATASET, "train_texts.txt")) as fIn:
#     for line in fIn:
#         dataset.append(line.strip())

# from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
# query_prompt_name = "s2s_query"
# embeddings = embedding_model.encode(dataset, prompt_name=query_prompt_name)

# def embed_texts(texts):
#     return embedding_model.encode(
#         texts,
#         show_progress_bar=True,
#         prompt_name="s2s_query"
#     )


# from umap import UMAP

# # Default
# dim_rec_model = UMAP(n_neighbors= 30, n_components=5, min_dist=1e-3, metric='euclidean', random_state=42)

# from sklearn.cluster import AgglomerativeClustering
# cluster_model = AgglomerativeClustering(
#     n_clusters=None,
#     linkage='ward'
# )



import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

class CL(nn.Module):
    def __init__(self, topic_embeddings, num_groups, num_topics, threshold_cl=0.5):
        super().__init__()
        self.topic_embeddings = topic_embeddings  # Embeddings của các topic
        self.num_groups = num_groups  # Số lượng nhóm lớn (clusters)
        self.num_topics = num_topics  # Số lượng topic
        self.threshold_cl = threshold_cl  # Ngưỡng để phân biệt positive và negative pairs
        self.sub_cluster = {}  # Lưu trữ các nhóm con của các nhóm lớn

    def create_group_connection_regularizer(self):
        # Step 1: Hierarchical Agglomerative Clustering (HAC) to find large clusters
        distances = torch.cdist(self.topic_embeddings, self.topic_embeddings, p=2)  # Tính khoảng cách Euclidean
        distances = distances.numpy()

        # Sử dụng linkage để thực hiện HAC
        Z = linkage(distances, method='ward')  # Phương pháp 'ward' cho HAC

        # Chia thành số cụm lớn (num_groups)
        group_id = fcluster(Z, t=self.num_groups, criterion='maxclust')

        # Step 2: Tạo sub-clusters trong mỗi nhóm lớn sử dụng K-means
        self.group_topic = [[] for _ in range(self.num_groups)]
        for i in range(self.num_topics):
            self.group_topic[group_id[i] - 1].append(i)  # Lưu topic vào mỗi nhóm lớn

        # Step 3: Tạo sub-clusters trong mỗi nhóm lớn
        for group_idx, topics in enumerate(self.group_topic):
            sub_embeddings = self.topic_embeddings[topics]  # Lấy embedding của các topic trong nhóm lớn
            kmean_model = KMeans(n_clusters=min(3, len(topics)), max_iter=1000, seed=0, verbose=False)
            sub_group_id = kmean_model.fit_predict(sub_embeddings.cpu().detach().numpy())  # Phân nhóm con trong mỗi cụm lớn

            self.sub_cluster[group_idx] = {}
            for sub_idx, topic_idx in enumerate(topics):
                self.sub_cluster[group_idx].setdefault(sub_group_id[sub_idx], []).append(topic_idx)

        # In ra sub-clusters để kiểm tra
        print(self.sub_cluster)

    def contrastive_loss_sub(self):
        # Hàm này tính Contrastive loss_cl giữa các topic con trong cùng một nhóm lớn
        loss_cl = 0.0
        for group_idx, sub_clusters in self.sub_cluster.items():
            for sub_group_id, topics in sub_clusters.items():
                # Lấy các embedding của topic con
                embeddings = self.topic_embeddings[topics]

                # Tính cosine similarity giữa các embedding trong cùng một sub-cluster
                similarity_matrix = torch.mm(embeddings, embeddings.T)
                norm_embeddings = torch.norm(embeddings, p=2, dim=1, keepdim=True)
                similarity_matrix = similarity_matrix / (norm_embeddings * norm_embeddings.T)

                # Tính loss: Nếu sim > threshold_cl, coi là positive pair, ngược lại là negative pair
                for i in range(similarity_matrix.shape[0]):
                    for j in range(i + 1, similarity_matrix.shape[1]):
                        sim = similarity_matrix[i, j]
                        if sim > self.threshold_cl:  # Positive pair
                            loss_cl += F.relu(1 - sim)  # loss_cl cho positive pair
                        else:  # Negative pair
                            loss_cl += F.relu(sim)  # loss_cl cho negative pair

        return loss_cl
