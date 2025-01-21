import torch
import torch.nn.functional as F

class KLDivergence:
    # Choose "forward"
    def __init__(self, kl_type: str = "forward"):
        self.kl_type = kl_type

        if self.kl_type == "forward":
            self.klfn = lambda x, y: self._kl(x, y)

        elif self.kl_type == "reverse":
            self.klfn = lambda x, y: self._kl(y, x)

        elif self.kl_type == "symmetric":
            self.klfn = lambda x, y: self._kl(x, y) + self._kl(y, x)
             
    def _kl(self, x: torch.Tensor, y: torch.Tensor):
        return F.kl_div(
            input=F.log_softmax(y, dim=-1, dtype=torch.float32),
            target=F.log_softmax(x, dim=-1, dtype=torch.float32),
            log_target=True,
            reduction="mean"
        )

    def get_divergence(self, x: torch.Tensor, y: torch.Tensor):
        return self.klfn(x, y) / x.size(0)


class DREAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, device, adaptive=True, lr=0.002, sigma=1, lmbda=0.9):
        defaults = dict(adaptive=adaptive, lr=lr)
        super(DREAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.device = device
        self.sigma = sigma
        self.lmbda = lmbda


    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(self.device)
                        for group in self.param_groups for p in group["params"] if p.grad is not None ]),
                    p=2)
        return norm

    @torch.no_grad()
    def first_step(self, loss_CTR, zero_grad=False, device='cuda'):

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue

                grad = p.grad.clone()
                state = self.state[p]

                if "momentum" not in state:
                    state["momentum"] = grad 
                else:
                    p.grad.add_(state["momentum"], alpha=-self.sigma)
                    state["momentum"].mul_(self.lmbda).add_(grad, alpha=1 - self.lmbda)  # In-place update momentum


        grad_norm = self._grad_norm()
        scale = loss_CTR / (grad_norm + 1e-12)

        for group in self.param_groups: 
            for p in group["params"]:
                if p.grad is None: 
                    continue

                state = self.state[p]
                state["old_p"] = p.data.clone() 

                e_w = (torch.pow(p, 2) if  group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w) 

        if zero_grad: self.zero_grad()
    

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # Khôi phục 
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue

                p.data.copy_(self.state[p]["old_p"])  # Khôi phục trạng thái cũ

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        closure = torch.enable_grad()(closure)  
        self.first_step(zero_grad=True)        
        closure()
        self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



