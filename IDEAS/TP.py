import torch
from torch import nn

class TP(nn.Module):
    def __init__(self, weight_loss_TP, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_TP = weight_loss_TP
        self.stopThr = stopThr
        self.epsilon = 1e-16
        self.transp = None

    def forward(self, theta, embeddings):
        if self.weight_loss_TP <= 1e-6:
            return 0.
        else:
            # Device setup
            device = theta.device
            
            # Step 1: Compute cost matrix C (D x D)
            C = torch.cdist(theta, theta, p=2) ** 2  # ||theta_i - theta_j||_2^2

            # Step 2: Compute similarity matrix P (D x D)
            norms = torch.norm(embeddings, dim=1, keepdim=True)  # ||e_i||
            P = torch.mm(embeddings, embeddings.t()) / (norms * norms.t() + self.epsilon)  # cosine similarity
            P = P / P.sum(dim=1, keepdim=True)  # Row-normalize P
            P = (P + P.T) / 2

            # Step 3: Sinkhorn's algorithm to solve for Q
            D = theta.size(0)
            a = torch.ones(D, 1, device=device) / D
            b = torch.ones(D, 1, device=device) / D

            u = torch.ones_like(a)  # D x 1
            K = torch.exp(-C * self.sinkhorn_alpha)  # Exponential kernel for Sinkhorn

            err = 1
            cpt = 0
            while err > self.stopThr and cpt < self.OT_max_iter:
                v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
                u = torch.div(a, torch.matmul(K, v) + self.epsilon)
                cpt += 1
                if cpt % 50 == 1:
                    bb = torch.mul(v, torch.matmul(K.t(), u))
                    err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

            Q = u * (K * v.t())
            Q = Q.clamp(min=1e-6)  # Avoid numerical issues
            Q = Q / Q.sum()
            self.transp = Q

            # Step 4: Compute KL divergence KL(P || Q)
            kl_div = (P * (P.log() - Q.log())).sum()

            # Step 5: Scale loss by weight
            loss_TP = self.weight_loss_TP * kl_div

            return loss_TP