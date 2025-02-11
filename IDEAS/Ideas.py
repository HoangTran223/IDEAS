import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
#from .GR import GR
from .DT_ETP import DT_ETP
from .TP import TP
import torch_kmeans
import logging
import sentence_transformers
from heapq import nlargest
from sklearn.metrics import silhouette_score
import hdbscan

##
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from utils import static_utils

class IDEAS(nn.Module):
    def __init__(self, vocab_size, data_name = '20NG', num_topics=50, num_groups=50, en_units=200, dropout=0.,
                 cluster_distribution=None, cluster_mean=None, cluster_label=None, threshold_epochs = 10, doc2vec_size=384,
                 pretrained_WE=None, embed_size=200, beta_temp=0.2, weight_loss_cl_words=1.0, threshold_cluster=10,
                 weight_loss_ECR=250.0, weight_loss_TP = 250.0, alpha_TP = 20.0, threshold_cl_large = 0.5,
                 DT_alpha: float=3.0, weight_loss_DT_ETP = 10.0, threshold_cl = 0.5, vocab = None, doc_embeddings=None,
                 weight_loss_cl_large = 1.0, num_large_clusters = 5, method_cl = 'HAC', metric_cl = 'euclidean',
                 alpha_GR=20.0, alpha_ECR=20.0, sinkhorn_alpha = 20.0, sinkhorn_max_iter=5000):
        super().__init__()

        self.method_cl = method_cl
        self.metric_cl = metric_cl
        self.threshold_epochs = threshold_epochs
        self.threshold_cluster = threshold_cluster
        self.num_large_clusters = num_large_clusters
        self.num_topics = num_topics
        self.num_groups = num_groups
        self.beta_temp = beta_temp
        self.data_name = data_name
        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor(
            (np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor(
            (((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(
                torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, alpha_ECR, sinkhorn_max_iter)
        

        ##
        self.weight_loss_cl_words= weight_loss_cl_words
        self.weight_loss_cl_large = weight_loss_cl_large


        self.vocab = vocab
        self.matrixP = None
        self.DT_ETP = DT_ETP(weight_loss_DT_ETP, DT_alpha)

        self.doc_embeddings = doc_embeddings.to(self.topic_embeddings.device)
        self.group_topic = None

        self.TP = TP(weight_loss_TP, alpha_TP)

        print(f"chieuX cua doc_embeddings {len(self.doc_embeddings)}")
        print(f"chieuY cua doc_embeddings : {len(self.doc_embeddings[0])}")

        self.document_emb_prj = nn.Sequential(
            nn.Linear(doc2vec_size, embed_size), 
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.topic_embeddings.device)

        self.topics = []
        self.topic_index_mapping = {}


    def create_group_topic(self):
        with torch.no_grad():  
            distances = torch.cdist(self.topic_embeddings, self.topic_embeddings, p=2)
            distances = distances.detach().cpu().numpy()

        if self.method_cl == 'HAC':
            Z = linkage(distances, method='average', optimal_ordering=True) 
            group_id = fcluster(Z, t= self.num_large_clusters, criterion='maxclust') - 1
        
        elif self.method_cl == 'HDBSCAN':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
            group_id = clusterer.fit_predict(distances)
        
        else:
            raise ValueError("method_cl must be either 'HAC' or 'HDBSCAN'")
        
        self.group_topic = [[] for _ in range(self.num_large_clusters)]
        for i in range(self.num_topics):
            self.group_topic[group_id[i]].append(i) 
        
        topic_idx_counter = 0
        # Lấy danh sách từ trong từng topic
        word_topic_assignments = self.get_word_topic_assignments()
        for topic_idx in range(self.num_topics):
            self.topics.append(word_topic_assignments[topic_idx])
            self.topic_index_mapping[topic_idx] = topic_idx_counter
            topic_idx_counter += 1
    

    """Gán các từ vào chủ đề tương ứng dựa trên độ tương đồng giữa từ và chủ đề:
    - Tính cosine similarity giữa 1 word embedding và tất cả các topic embeddings
    - Tìm topic có similarity lớn nhất với từ đó
    """
    
    def get_word_topic_assignments(self):
        word_topic_assignments = [[] for _ in range(self.num_topics)]

        for word_idx, word in enumerate(self.vocab):
            topic_idx = self.word_to_topic_by_similarity(word)
            word_topic_assignments[topic_idx].append(word_idx)
        return word_topic_assignments


    def word_to_topic_by_similarity(self, word):
        word_idx = self.vocab.index(word)
        word_embedding = self.word_embeddings[word_idx].unsqueeze(0)

        similarity_scores = F.cosine_similarity(word_embedding, self.topic_embeddings)
        topic_idx = torch.argmax(similarity_scores).item()

        return topic_idx


    def get_contrastive_loss_large_clusters(self, margin=0.2, num_negatives=10):
        loss_cl_large = 0.0

        # Duyệt qua từng cụm lớn
        for group_idx, group_topics in enumerate(self.group_topic):
            if len(group_topics) < 1:
                continue

            anchor = torch.mean(self.topic_embeddings[group_topics], dim=0, keepdim=True)

            positive_topic_idx = np.random.choice(group_topics)
            positive = self.topic_embeddings[positive_topic_idx].unsqueeze(0)

            negative_candidates = []
            for neg_group_idx, neg_group_topics in enumerate(self.group_topic):
                if neg_group_idx != group_idx:  
                    negative_candidates.extend(neg_group_topics)

            if len(negative_candidates) < num_negatives:
                continue

            negative_topic_idxes = np.random.choice(negative_candidates, size=num_negatives, replace=False)
            negatives = self.topic_embeddings[negative_topic_idxes]


            if self.metric_cl == 'euclidean':
                pos_distance = F.pairwise_distance(anchor, positive)
                neg_distances = F.pairwise_distance(anchor.repeat(num_negatives, 1), negatives)
                
            elif self.metric_cl == 'cosine':
                pos_similarity = F.cosine_similarity(anchor, positive)
                neg_similarities = F.cosine_similarity(anchor.repeat(num_negatives, 1), negatives)
                pos_distance = 1 - pos_similarity
                neg_distances = 1 - neg_similarities
            else:
                raise ValueError(f"Invalid metric_cl: {self.metric_cl}")

            loss = torch.clamp(pos_distance - neg_distances + margin, min=0.0)
            loss_cl_large += loss.mean()

        loss_cl_large *= self.weight_loss_cl_large
        return loss_cl_large

    
    def get_contrastive_loss_words(self):
        loss_cl_words = 0.0
        margin = 0.2 
        num_negatives = 10

        # Duyệt qua từng cụm lớn
        for group_idx, group_topics in enumerate(self.group_topic):
            # Duyệt qua từng topic trong cụm lớn 
            for anchor_topic_idx in group_topics:
                anchor_words_idxes = self.topics[self.topic_index_mapping[anchor_topic_idx]]

                if len(anchor_words_idxes) < 1:
                    continue

                anchor = torch.mean(self.word_embeddings[anchor_words_idxes], dim=0, keepdim=True)
                
                positive_word_idx = np.random.choice(anchor_words_idxes)
                positive = self.word_embeddings[positive_word_idx].unsqueeze(0)

                negative_candidates = []
                # Lấy các từ từ các topic khác
                for neg_topic_idx in range(self.num_topics):
                    if neg_topic_idx not in group_topics:
                        negative_candidates.extend(self.topics[self.topic_index_mapping[neg_topic_idx]])
                
                if len(negative_candidates) < num_negatives:
                    continue
                
                negative_word_idxes = np.random.choice(negative_candidates, size=num_negatives, replace=False)
                negatives = self.word_embeddings[negative_word_idxes]

                if self.metric_cl == 'euclidean':
                    pos_distance = F.pairwise_distance(anchor, positive)
                    neg_distances = F.pairwise_distance(anchor.repeat(num_negatives, 1), negatives)
                
                elif self.metric_cl == 'cosine':
                    pos_similarity = F.cosine_similarity(anchor, positive)
                    neg_similarities = F.cosine_similarity(anchor.repeat(num_negatives, 1), negatives)
                    pos_distance = 1 - pos_similarity
                    neg_distances = 1 - neg_similarities
                else:
                    raise ValueError(f"Invalid metric_cl: {self.metric_cl}")
                
                loss = torch.clamp(pos_distance - neg_distances + margin, min=0.0)
                loss_cl_words += loss.mean()

        loss_cl_words *= self.weight_loss_cl_words
        return loss_cl_words


    def get_beta(self):
        dist = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def get_representation(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        return theta, mu, logvar

    def encode(self, input):
        theta, mu, logvar = self.get_representation(input)
        loss_KL = self.compute_loss_KL(mu, logvar)
        return theta, loss_KL

    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term +
                     logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR
    

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + \
            torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def create_matrixP(self, minibatch_embeddings, indices):
        num_minibatch = len(indices)
        self.matrixP = torch.ones(
            (num_minibatch, num_minibatch), device=self.topic_embeddings.device) / num_minibatch

        norm_embeddings = F.normalize(minibatch_embeddings, p=2, dim=1).clamp(min=1e-6)
        self.matrixP = torch.matmul(norm_embeddings, norm_embeddings.T)
        self.matrixP.fill_diagonal_(0)
        self.matrixP = self.matrixP.clamp(min=1e-4)
        return self.matrixP

    def get_loss_TP(self, doc_embeddings, indices):
        indices = indices.to(self.doc_embeddings.device)
        minibatch_embeddings = self.doc_embeddings[indices]
        # minibatch_indices = minibatch_indices.to(self.topic_embeddings.device)
        # minibatch_embeddings = doc_embeddings[minibatch_indices]
        cost = self.pairwise_euclidean_distance(minibatch_embeddings, minibatch_embeddings) \
           + 1e1 * torch.ones(minibatch_embeddings.size(0), minibatch_embeddings.size(0)).to(minibatch_embeddings.device)

        self.matrixP = self.create_matrixP(minibatch_embeddings, indices)
        loss_TP = self.TP(cost, self.matrixP)
        return loss_TP
    
    
    def get_loss_DT_ETP(self, doc_embeddings):
        document_prj = self.document_emb_prj(doc_embeddings)

        loss_DT_ETP, transp_DT = self.DT_ETP(document_prj, self.topic_embeddings)
        return loss_DT_ETP


    def forward(self, indices, input, epoch_id=None, doc_embeddings=None):

        bow = input[0]
        # contextual_emb = input[1]
        doc_embeddings = doc_embeddings.to(self.topic_embeddings.device)

        rep, mu, logvar = self.get_representation(bow)
        loss_KL = self.compute_loss_KL(mu, logvar)
        theta = rep

        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()
        loss_TP = self.get_loss_TP(doc_embeddings, indices)
        loss_DT_ETP = self.get_loss_DT_ETP(doc_embeddings)

        loss_cl_large = 0.0
        loss_cl_words = 0.0

        if epoch_id >= self.threshold_epochs and (epoch_id == self.threshold_epochs or (epoch_id > self.threshold_epochs and epoch_id % self.threshold_cluster == 0)):
            self.create_group_topic()

        if epoch_id >= self.threshold_epochs and self.weight_loss_cl_large != 0:
            loss_cl_large = self.get_contrastive_loss_large_clusters()
        
        if epoch_id >= self.threshold_epochs and self.weight_loss_cl_words != 0:
            loss_cl_words = self.get_contrastive_loss_words()
            
        loss = loss_TM + loss_ECR + loss_TP + loss_DT_ETP + loss_cl_large + loss_cl_words
        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR,
            'loss_DT_ETP': loss_DT_ETP,
            'loss_TP': loss_TP,
            'loss_cl_large': loss_cl_large,
            'loss_cl_words': loss_cl_words,
        }

        return rst_dict





    """ InfoNCE Loss: similarity_matrix là ma trận cosine similarity giữa các cụm (xđịnh = tâm cụm). Nó:
        - similarity_matrix[i, i] (similarity của một cụm lớn với chính nó) cao.
        - similarity_matrix[i, j] (similarity của một cụm lớn với các cụm lớn khác) thấp, với i != j
        - positive pair: cặp embedding của 1 cụm với chính nó -> ptu nằm trên đg chéo
        - negative pair: cặp embedding của 1 cụm với các cụm khác
        => Mong similarity_matrix gần giống với ma trận đơn vị
    """


