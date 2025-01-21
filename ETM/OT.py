import torch
from torch import nn


class OT(nn.Module):
    def __init__(self, weight_loss_OT, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_OT = weight_loss_OT
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, a, b, M):
        # a: B x K
        # b: B x V
        # M: K x V

        if self.weight_loss_OT <= 1e-6:
            return 0.0

        B, K = a.size()
        _, V = b.size()
        M = M.unsqueeze(0).expand(B, -1, -1)  
        M = M.transpose(1, 2) # Shape: (B, K, V)
        device = M.device

        # Initialize u and v
        u = torch.ones(B, K, device=device) / K  # Shape: (B, K)
        v = torch.ones(B, V, device=device) / V  # Shape: (B, V)

        # Compute the kernel matrix
        K_mat = torch.exp(-M * self.sinkhorn_alpha)  # Shape: (B, K, V)
        err = float('inf')
        cpt = 0

        while err > self.stopThr and cpt < self.OT_max_iter:
            # Update v: v = b / (K^T u)
            KTu = torch.bmm(K_mat.transpose(1, 2), u.unsqueeze(2)).squeeze(2)  # Shape: (B, V)
            v = b / (KTu + self.epsilon)  # Shape: (B, V)

            # Update u: u = a / (K v)
            Kv = torch.bmm(K_mat, v.unsqueeze(2)).squeeze(2)  # Shape: (B, K)
            u = a / (Kv + self.epsilon)  # Shape: (B, K)

            cpt += 1
            if cpt % 50 == 1:
                # Compute the marginal constraint error
                err_u = torch.max(torch.abs(torch.sum(u * Kv, dim=1) - a.sum(dim=1)))
                err_v = torch.max(torch.abs(torch.sum(v * KTu, dim=1) - b.sum(dim=1)))
                err = max(err_u.item(), err_v.item())

        # Transport matrix for the batch
        transp = u.unsqueeze(2) * K_mat * v.unsqueeze(1)  # Shape: (B, K, V)

        # Compute the loss
        loss_OT = torch.mean(torch.sum(transp * M, dim=(1, 2)))  # Scalar
        return loss_OT
