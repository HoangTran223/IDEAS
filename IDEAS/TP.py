import torch
from torch import nn

class TP(nn.Module):
    def __init__(self, weight_loss_TP, sinkhorn_alpha, OT_max_iter=500, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_TP = weight_loss_TP
        self.stopThr = stopThr
        self.epsilon = 1e-6
        self.transp = None


    def forward(self, M, group):
        if self.weight_loss_TP <= 1e-6:
            return 0.
        else:
            # M: KxV cost matrix
            # sinkhorn alpha = 1/ entropic reg weight
            # a: Kx1 source distribution
            # b: Vx1 target distribution
            device = M.device
            group = group.to(device)

            # Sinkhorn's algorithm
            a = (group.sum(axis=1)).unsqueeze(1).to(device)
            b = (group.sum(axis=0)).unsqueeze(1).to(device)

            u = (torch.ones_like(a) / a.size()[0]).to(device) # Kx1

            K = torch.exp(-M * self.sinkhorn_alpha)
            err = 1
            cpt = 0
            while err > self.stopThr and cpt < self.OT_max_iter:
                v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
                u = torch.div(a, torch.matmul(K, v) + self.epsilon)
                cpt += 1
                if cpt % 50 == 1:
                    bb = torch.mul(v, torch.matmul(K.t(), u))
                    err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

            transp = u * (K * v.T)
            transp = transp.clamp(min=1e-4)

            print(f"transp: {transp}")  # ma trận vận chuyển
            print(f"group: {group}")

            #self.transp = transp

            loss_TP = (group * (group.log() - transp.log() - 1) \
                + transp).sum()
            # loss_TP = ((group + self.epsilon) * (torch.log(group + self.epsilon) \
            #              - torch.log(transp) - 1) + transp).sum()
            loss_TP *= self.weight_loss_TP

            return loss_TP





            # def pairwise_euclidean_distance(self, x, y):
            #     cost = torch.sum(x ** 2, axis=1, keepdim=True) + \
            #         torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
            #     return cost

            #             # Device setup
            # device = theta.device
            
            # # C = torch.cdist(theta, theta, p=2) ** 2  # ||theta_i - theta_j||_2^2
            # C = self.pairwise_euclidean_distance(
            #         theta_i, theta_j
            # )
            # C.fill_diagonal_(float('inf'))

            # # Step 2: Compute similarity matrix P (D x D)
            # norms = torch.norm(embeddings, dim=1, keepdim=True)  # ||e_i||
            # P = torch.mm(embeddings, embeddings.t()) / (norms * norms.t() + self.epsilon)  # cosine similarity
            # P = P / (norms * norms.t())  # Adjusted similarity (based on your formula)
            # P = (P + P.T) / 2  # Symmetric matrix

            # # Step 3: Sinkhorn's algorithm to solve for Q
            # D = theta.size(0)
            # a = torch.ones(D, 1, device=device) / D
            # b = torch.ones(D, 1, device=device) / D

            # u = torch.ones_like(a)  # D x 1
            # K = torch.exp(-C * self.sinkhorn_alpha)  # Exponential kernel for Sinkhorn
            # K.fill_diagonal_(0)

            # err = 1
            # cpt = 0
            # while err > self.stopThr and cpt < self.OT_max_iter:
            #     v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            #     u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            #     cpt += 1
            #     if cpt % 50 == 1:
            #         bb = torch.mul(v, torch.matmul(K.t(), u))
            #         err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

            # Q = u * (K * v.t())
            # Q = Q.clamp(min=1e-6)  # Avoid numerical issues
            # Q = Q / Q.sum()
            # Q.fill_diagonal_(0)
            # self.transp = Q

            # # Step 4: Compute KL divergence KL(P || Q)
            # kl_div = (P * (P.log() - Q.log())).sum()

            # # Step 5: Scale loss by weight
            # loss_TP = self.weight_loss_TP * kl_div

            # return loss_TP