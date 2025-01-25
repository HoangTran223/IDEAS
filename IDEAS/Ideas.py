import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
#from .GR import GR
from .CL import CL
from .DT_ETP import DT_ETP
from .TP import TP
import torch_kmeans
import logging
import sentence_transformers

##
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from utils import static_utils

class IDEAS(nn.Module):
    def __init__(self, vocab_size, data_name = '20NG', num_topics=50, num_groups=50, en_units=200, dropout=0.,
                 cluster_distribution=None, cluster_mean=None, cluster_label=None, 
                 pretrained_WE=None, embed_size=200, beta_temp=0.2, num_documents=None,
                 weight_loss_ECR=250.0, weight_loss_TP = 250.0, alpha_TP = 20.0, threshold_cl_large = 0.5,
                 DT_alpha: float=3.0, weight_loss_DT_ETP = 10.0, threshold_cl = 0.5, vocab = None,
                 weight_loss_cl = 1.0, weight_loss_cl_large = 1.0,
                 alpha_GR=20.0, alpha_ECR=20.0, sinkhorn_alpha = 20.0, sinkhorn_max_iter=1000):
        super().__init__()

        
        self.num_documents = num_documents
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
        self.weight_loss_cl = weight_loss_cl
        self.weight_loss_cl_large = weight_loss_cl_large
        self.threshold_cl = threshold_cl
        self.threshold_cl_large = threshold_cl_large

        self.top_words_dict = {}
        self.vocab = vocab
        self.matrixP = None
        self.DT_ETP = DT_ETP(weight_loss_DT_ETP, DT_alpha)

        self.doc_embeddings = torch.empty((self.num_documents, self.num_documents))
        nn.init.trunc_normal_(self.doc_embeddings, std=0.1)
        self.doc_embeddings = nn.Parameter(F.normalize(self.doc_embeddings))
        # self.doc_embeddings = nn.Parameter(
        #     torch.randn((self.num_documents, self.num_documents))
        # )
        self.group_topic = None
        self.sub_cluster = None

        print(f"chieuX cua doc_embeddings {len(self.doc_embeddings)}")
        print(f"chieuY cua doc_embeddings : {len(self.doc_embeddings[0])}")
        self.TP = TP(weight_loss_TP, alpha_TP)

        self.document_emb_prj = nn.Sequential(nn.Linear(self.num_documents, self.word_embeddings.shape[1] ),
                                       nn.Dropout(dropout))
        self.topic_top_words = {}
        
        self.word_embeddings_dict = {
            word: self.word_embeddings[idx].detach().cpu().numpy()
            for idx, word in enumerate(self.vocab)
        }


    # def create_group_topic(self):
    #     if self.num_topics == 0 or self.num_groups == 0:
    #         print("Warning: num_topics or num_groups is zero!")
    #         return

    #     distances = torch.cdist(self.topic_embeddings, self.topic_embeddings, p=2)  
    #     distances = distances.detach().cpu().numpy()

    #     # Dùng linkage để thực hiện HAC
    #     Z = linkage(distances, method='ward') 

    #     # Chia thành số cụm lớn (num_groups)
    #     group_id = fcluster(Z, t=self.num_groups, criterion='maxclust')
        
    #     self.group_topic = [[] for _ in range(self.num_groups)]
    #     for i in range(self.num_topics):
    #         self.group_topic[group_id[i] - 1].append(i)  # Lưu topic vào mỗi nhóm lớn

    #     # Tạo sub-clusters trong mỗi nhóm lớn
    #     self.sub_cluster = {}
    #     for group_idx, topics in enumerate(self.group_topic):
    #         sub_embeddings = self.topic_embeddings[topics]  
    #         kmean_model = KMeans(n_clusters=min(3, len(topics)), max_iter=1000, verbose=False, n_init='auto')
    #         sub_group_id = kmean_model.fit_predict(sub_embeddings.cpu().detach().numpy()) 

    #         # if len(sub_embeddings) < 2:
    #         #     self.sub_cluster[group_idx] = {0: topics} 
    #         #     continue

    #         # sub_distances = torch.cdist(sub_embeddings, sub_embeddings, p=2).detach().cpu().numpy() 
    #         # sub_Z = linkage(sub_distances, method='ward')
    #         # sub_group_id = fcluster(sub_Z, t = 0.5, criterion='inconsistent')

    #         self.sub_cluster[group_idx] = {}
    #         for sub_idx, topic_idx in enumerate(topics):
    #             self.sub_cluster[group_idx].setdefault(sub_group_id[sub_idx], []).append(topic_idx)
    

    def create_group_topic_op2(self):
        # Bước 1: Sử dụng KMeans để tạo các cụm lớn (num_groups)
        kmeans = KMeans(n_clusters=self.num_groups, max_iter=2000, verbose=False, n_init=10)
        group_id = kmeans.fit_predict(self.topic_embeddings.cpu().detach().numpy())
        print("Unique cluster IDs:", len(set(group_id)))


        self.group_topic = [[] for _ in range(self.num_groups)]
        for i in range(self.num_topics):
            self.group_topic[group_id[i]].append(i)  # Lưu topic vào mỗi cụm lớn

        # Bước 2: Tạo các cụm con trong từng cụm lớn bằng HAC
        self.sub_cluster = {}
        for group_idx, topics in enumerate(self.group_topic):
            if len(topics) < 2:
                # Nếu cụm lớn có ít hơn 2 topic, gán cụm con là chính nó
                self.sub_cluster[group_idx] = {0: topics}
                continue

            sub_embeddings = self.topic_embeddings[topics]
            sub_distances = torch.cdist(sub_embeddings, sub_embeddings, p=2).detach().cpu().numpy()

            # Dùng linkage để thực hiện HAC
            sub_Z = linkage(sub_distances, method='ward')

            sub_group_id = fcluster(sub_Z, t=2, criterion='distance')

            self.sub_cluster[group_idx] = {}
            for sub_idx, topic_idx in enumerate(topics):
                self.sub_cluster[group_idx].setdefault(sub_group_id[sub_idx] - 1, []).append(topic_idx)


    # Options 2: num_groups = num_topics
    def create_group_topic(self):
        # Bước 1: Sử dụng KMeans để tạo các cụm lớn (num_groups)
        kmeans = KMeans(n_clusters=self.num_groups, max_iter=2000, verbose=False, n_init=10)
        group_id = kmeans.fit_predict(self.topic_embeddings.cpu().detach().numpy())
        print("Unique cluster IDs:", len(set(group_id)))


        self.group_topic = [[] for _ in range(self.num_groups)]
        for i in range(self.num_topics):
            self.group_topic[group_id[i]].append(i) 

        # Bước 2: Tạo các cụm con trong từng cụm lớn bằng HAC
        self.sub_cluster = {}
        for group_idx, topics in enumerate(self.group_topic):

            sub_embeddings = self.topic_embeddings[topics]
            sub_distances = torch.cdist(sub_embeddings, sub_embeddings, p=2).detach().cpu().numpy()

            # Dùng linkage để thực hiện HAC
            sub_Z = linkage(sub_distances, method='ward')

            sub_group_id = fcluster(sub_Z, t=2, criterion='distance')

            self.sub_cluster[group_idx] = {}
            for sub_idx, topic_idx in enumerate(topics):
                self.sub_cluster[group_idx].setdefault(sub_group_id[sub_idx] - 1, []).append(topic_idx)
            
            print(f"Group {group_idx} has {len(self.sub_cluster[group_idx])} sub-clusters")



    def get_contrastive_loss(self):
        loss_cl = 0.0
        for group_idx, sub_clusters in self.sub_cluster.items():
            for sub_group_id, topics in sub_clusters.items():
                if len(topics) <= 1:
                    continue
                embeddings = self.topic_embeddings[topics]
                
                # Tính cosine similarity giữa các embedding trong cùng một sub-cluster
                norm_embeddings = F.normalize(embeddings, p=2, dim=1)
                similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.T)
                
                positive_pairs = similarity_matrix[similarity_matrix >= self.threshold_cl]
                negative_pairs = similarity_matrix[similarity_matrix < self.threshold_cl]

                # Tính loss
                loss_pos = torch.sum(F.relu(1 - positive_pairs)) 
                loss_neg = torch.sum(F.relu(negative_pairs - self.threshold_cl))  
                loss_cl += loss_pos + loss_neg

        loss_cl *= self.weight_loss_cl
        return loss_cl

    
    def get_top_words(self, vocab, group_index, num_top_words=15):
        beta = self.get_beta().detach().cpu().numpy()
        group_beta = beta[group_index]
        top_words = []

        for topic_dist in group_beta:
            top_indices = np.argsort(topic_dist)[-num_top_words:][::-1]
            #topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-(num_top_words + 1):-1]
            topic_words = [self.vocab[idx] for idx in top_indices if self.vocab[idx] in self.vocab]
            top_words.extend(topic_words)

        return top_words[:num_top_words]
    

    def get_contrastive_loss_large_clusters(self):

        loss_cl_large = 0.0
        for i, group_i in enumerate(self.group_topic):
            for j in range(i + 1, len(self.group_topic)):
                group_j = self.group_topic[j]

                top15_i = self.get_top_words(self.vocab, group_i)  
                top15_j = self.get_top_words(self.vocab, group_j) 

                similarity_matrix = self.compute_similarity(top15_i, top15_j)

                # Tính độ tương đồng trung bình
                sim_avg = similarity_matrix.mean()
                if sim_avg > self.threshold_cl_large:
                    loss_cl_large += F.relu(1 - sim_avg)
                else:
                    loss_cl_large += F.relu(sim_avg - self.threshold_cl_large)

        loss_cl_large *= self.weight_loss_cl_large
        return loss_cl_large

    def compute_similarity(self, top_words_i, top_words_j):

        # vec_i = torch.stack([torch.tensor(self.word_embeddings_dict[word]) for word in top_words_i 
        #                     if word in self.word_embeddings_dict])
        # vec_j = torch.stack([torch.tensor(self.word_embeddings_dict[word]) for word in top_words_j 
        #                     if word in self.word_embeddings_dict])
        vec_i = torch.stack([self.word_embeddings[self.vocab.index(word)] for word in top_words_i 
                    if word in self.vocab])
        vec_j = torch.stack([self.word_embeddings[self.vocab.index(word)] for word in top_words_j 
                    if word in self.vocab])

        #print(f"vec_i shape: {vec_i.shape}, vec_j shape: {vec_j.shape}")

        similarity_matrix = F.cosine_similarity(vec_i, vec_j, dim=-1)
        return similarity_matrix


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


    def create_matrixP(self, minibatch_indices):
        num_minibatch = len(minibatch_indices)
        minibatch_embeddings = self.doc_embeddings[minibatch_indices]
        self.matrixP = torch.ones(
            (num_minibatch, num_minibatch), device=self.topic_embeddings.device) / num_minibatch
        
        norm_embeddings = minibatch_embeddings / (torch.norm(minibatch_embeddings, dim=1, keepdim=True).clamp(min=1e-6))
        self.matrixP = torch.matmul(norm_embeddings, norm_embeddings.T)
        self.matrixP.fill_diagonal_(0)
        self.matrixP = self.matrixP.clamp(min=1e-4)
        return self.matrixP

    def get_loss_TP(self, minibatch_indices):
        minibatch_embeddings = self.doc_embeddings[minibatch_indices]
        cost = self.pairwise_euclidean_distance(minibatch_embeddings, minibatch_embeddings) \
           + 1e1 * torch.ones(minibatch_embeddings.size(0), minibatch_embeddings.size(0)).to(minibatch_embeddings.device)

        self.matrixP = self.create_matrixP(minibatch_indices)
        loss_TP = self.TP(cost, self.matrixP)
        return loss_TP
    
    
    def get_loss_DT_ETP(self):
        document_prj = self.document_emb_prj(self.doc_embeddings)

        loss_DT_ETP, transp_DT = self.DT_ETP(document_prj, self.topic_embeddings)
        return loss_DT_ETP



    def forward(self, indices, input, epoch_id=None):

        if self.sub_cluster is None or (epoch_id is not None and epoch_id % 500 == 0):
            self.create_group_topic()

        bow = input[0]
        contextual_emb = input[1]

        rep, mu, logvar = self.get_representation(bow)
        loss_KL = self.compute_loss_KL(mu, logvar)
        theta = rep

        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()
        loss_TP = self.get_loss_TP(indices)
        loss_DT_ETP = self.get_loss_DT_ETP()
        loss_cl = self.get_contrastive_loss()

        loss_cl_large = 0.0
        if epoch_id is not None and epoch_id > 1 and self.weight_loss_cl_large != 0:
            if not self.group_topic:
                print("Warning: group_topic is empty!")
            else:
                loss_cl_large = self.get_contrastive_loss_large_clusters()
            
        loss = loss_TM + loss_ECR + loss_TP + loss_DT_ETP + loss_cl + loss_cl_large
        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR,
            'loss_DT_ETP': loss_DT_ETP,
            'loss_TP': loss_TP,
            'loss_cl': loss_cl,
            'loss_cl_large': loss_cl_large
        }

        return rst_dict






    
    # Tính loss: Nếu sim > threshold, coi là positive pair, ngược lại là negative pair
    # for i in range(similarity_matrix.shape[0]):
    #     for j in range(i + 1, similarity_matrix.shape[1]):
    #         sim = similarity_matrix[i, j]
    #         if sim > self.threshold_cl:  # Positive pair
    #             loss_cl += F.relu(1 - sim)  # Loss cho positive pair
    #         else:  # Negative pair
    #             loss_cl += F.relu(sim)  # Loss cho negative pair
    # sim = similarity_matrix[torch.triu_indices(similarity_matrix.shape[0], similarity_matrix.shape[1], offset=1)]
    # loss_cl += torch.sum(F.relu(self.threshold_cl - sim))

    # def init_topic_top_words(self, vocab, top_words_dict, num_top_words = 15):
    #     beta = self.get_beta().detach().cpu().numpy()  # Xuất beta từ mô hình
    #     self.topic_top_words = static_utils.print_topic_words(beta, vocab, num_top_words)

    # def compute_similarity(self, top_words_i, top_words_j):
    #     """
    #     Tính ma trận similarity giữa hai tập từ.
    #     """
    #     similarity_matrix = []
    #     for word_i in top_words_i:
    #         row = []
    #         for word_j in top_words_j:
    #             sim = self.word_similarity(word_i, word_j) 
    #             row.append(sim)
    #         similarity_matrix.append(row)
    #     return torch.tensor(similarity_matrix)


    # def get_top_words(self, topic_id, top_k=15):
    #     """
    #     Lấy top-k từ từ topic embeddings, dựa trên độ tương đồng giữa các embeddings.
    #     """
    #     topic_embedding = self.topic_embeddings[topic_id] 

    #     # Tính cosine similarity giữa topic_embedding và tất cả các topic khác
    #     similarities = []
    #     for other_topic_id, other_embedding in self.topic_embeddings.items():
    #         sim = F.cosine_similarity(topic_embedding, other_embedding, dim=0)
    #         similarities.append((other_topic_id, sim.item()))

    #     # Sắp xếp và lấy top-k topic tương đồng
    #     similarities.sort(key=lambda x: x[1], reverse=True)
    #     top_topics = [topic for topic, _ in similarities[:top_k]]

    #     # Lấy từ top-k của các topic tương ứng
    #     top_words = []
    #     for topic in top_topics:
    #         top_words.extend(self.topic_top_words[topic][:top_k])

    #     return top_words


    # for i in range(num_minibatch):
    #     for j in range(num_minibatch):
    #         e_i = minibatch_embeddings[i]
    #         e_j = minibatch_embeddings[j]
            
    #         norm_i = torch.norm(e_i).clamp(min=1e-6)  
    #         norm_j = torch.norm(e_j).clamp(min=1e-6)  
            
    #         p_ij = torch.dot(e_i, e_j) / (norm_i * norm_j)
    #         self.matrixP[i, j] = p_ij
    # self.matrixP.fill_diagonal_(0)
    # self.matrixP = self.matrixP.clamp(min = 1e-4)

    # norms = torch.norm(minibatch_embeddings, dim=1, keepdim=True).clamp(min=1e-6)
    # P = torch.mm(minibatch_embeddings, minibatch_embeddings.t()) / (norms * norms.t() + 1e-6)
    # P = (P + P.T) / 2  # Symmetric matrix
    # P = P / P.sum(dim=1, keepdim=True) 

    # def create_group_connection_regularizer(self):
    #     kmean_model = torch_kmeans.KMeans(
    #         n_clusters=self.num_groups, max_iter=1000, seed=0, verbose=False,
    #         normalize='unit')
    #     group_id = kmean_model.fit_predict(self.topic_embeddings.reshape(
    #         1, self.topic_embeddings.shape[0], self.topic_embeddings.shape[1]))
    #     group_id = group_id.reshape(-1)
    #     self.group_topic = [[] for _ in range(self.num_groups)]
    #     for i in range(self.num_topics):
    #         self.group_topic[group_id[i]].append(i)

    #     self.group_connection_regularizer = torch.ones(
    #         (self.num_topics, self.num_topics), device=self.topic_embeddings.device) / 5.
    #     for i in range(self.num_topics):
    #         for j in range(self.num_topics):
    #             if group_id[i] == group_id[j]:
    #                 self.group_connection_regularizer[i][j] = 1
    #     self.group_connection_regularizer.fill_diagonal_(0)
    #     self.group_connection_regularizer = self.group_connection_regularizer.clamp(min=1e-4)
    #     for _ in range(50):
    #         self.group_connection_regularizer = self.group_connection_regularizer / \
    #             self.group_connection_regularizer.sum(axis=1, keepdim=True) / self.num_topics
    #         self.group_connection_regularizer = (self.group_connection_regularizer \
    #             + self.group_connection_regularizer.T) / 2.


    # def sim(self, rep, bert):
    #     prep = self.prj_rep(rep)
    #     pbert = self.prj_bert(bert)
    #     return torch.exp(F.cosine_similarity(prep, pbert))

    # def csim(self, bow, bert):
    #     pbow = self.prj_rep(bow)
    #     pbert = self.prj_bert(bert)
    #     csim_matrix = (pbow@pbert.T) / (pbow.norm(keepdim=True,
    #                                               dim=-1)@pbert.norm(keepdim=True, dim=-1).T)
    #     csim_matrix = torch.exp(csim_matrix)
    #     csim_matrix = csim_matrix / csim_matrix.sum(dim=1, keepdim=True)
    #     return -csim_matrix.log()