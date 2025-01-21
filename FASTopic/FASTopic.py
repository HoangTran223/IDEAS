import torch
from torch import nn
import torch.nn.functional as F

from ._ETP import ETP
from ._model_utils import pairwise_euclidean_distance
from .OT import OT


class FASTopic(nn.Module):
    def __init__(self,
                 vocab_size: int, embed_size: int, num_topics: int,
                 cluster_distribution=None,
                 cluster_mean=None,
                 cluster_label=None,
                 is_OT=False,
                 theta_temp: float=1.0,
                 DT_alpha: float=3.0,
                 TW_alpha: float=2.0,
                 weight_loss_OT=100.0, sinkhorn_alpha = 20.0, sinkhorn_max_iter=1000,
                ):
        super().__init__()

        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp
        self.num_topics = num_topics
        self.is_OT = is_OT
        self.epsilon = 1e-12
        
        self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((self.num_topics, embed_size))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.word_weights = nn.Parameter((torch.ones(vocab_size) / vocab_size).unsqueeze(1))
        self.topic_weights = nn.Parameter((torch.ones(self.num_topics) / self.num_topics).unsqueeze(1))

        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

        #OT Distance between topic proportion and cluster proportion
        self.weight_loss_OT = weight_loss_OT
        self.cluster_mean = nn.Parameter(torch.from_numpy(cluster_mean).float(), requires_grad=False)
        self.cluster_distribution = nn.Parameter(torch.from_numpy(cluster_distribution).float(), requires_grad=False)
        self.cluster_label = cluster_label
        if not isinstance(self.cluster_label, torch.Tensor):
            self.cluster_label = torch.tensor(self.cluster_label, dtype=torch.long, device='cuda')
        else:
            self.cluster_label = self.cluster_label.to(device='cuda', dtype=torch.long)
        
        self.map_t2c = nn.Linear(self.word_embeddings.shape[1], self.cluster_mean.shape[1], bias=False)
        self.OT = OT(weight_loss_OT, sinkhorn_alpha, sinkhorn_max_iter)

    def get_transp_DT(self,
                      doc_embeddings,
                    ):

        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        _, transp = self.DT_ETP(doc_embeddings, topic_embeddings)

        return transp.detach().cpu().numpy()

    # only for testing
    def get_beta(self):
        _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
        # use transport plan as beta
        beta = transp_TW * transp_TW.shape[0]

        return beta

    # only for testing
    def get_theta(self,
                  doc_embeddings,
                  train_doc_embeddings
                ):
        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
        train_dist = pairwise_euclidean_distance(train_doc_embeddings, topic_embeddings)

        exp_dist = torch.exp(-dist / self.theta_temp)
        exp_train_dist = torch.exp(-train_dist / self.theta_temp)

        theta = exp_dist / (exp_train_dist.sum(0))
        theta = theta / theta.sum(1, keepdim=True)

        return theta
    
    def get_loss_OT(self, input, indices):
        doc_embeddings = input[1]
        _, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)
        theta = transp_DT * transp_DT.shape[0]
        cd_batch = self.cluster_distribution[indices]  
        cost = pairwise_euclidean_distance(self.cluster_mean, self.map_t2c(self.topic_embeddings))  
        loss_OT = self.weight_loss_OT * self.OT(theta, cd_batch, cost)  
        return loss_OT

    def forward(self, indices, input, epoch_id=None):
        train_bow = input[0]
        doc_embeddings = input[1]

        loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)
        loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

        loss_ETP = loss_DT + loss_TW

        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]

        # Dual Semantic-relation Reconstruction
        recon = torch.matmul(theta, beta)

        loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()

        if self.is_OT:
            loss_OT = self.get_loss_OT(input, indices)
        else:
            loss_OT = 0.0

        loss = loss_DSR + loss_ETP + loss_OT

        rst_dict = {
            'loss': loss,
            'loss_OT': loss_OT
        }

        return rst_dict



