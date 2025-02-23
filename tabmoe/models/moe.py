import torch
from torch import nn, Tensor
from torch.nn import functional as F

from tabmoe.models.utils import init_rsqrt_uniform_ # # TODO: could be a local import?

class GumbelGatingNetwork(nn.Module):
    """
    MLP gating with BayesianLinear for each layer.
    We'll provide a method to sum the KL from both layers.
    """

    def __init__(self, in_features=784, num_experts=3, tau=1.0, device='cuda'):
        super().__init__()
        self.lin = nn.Linear(in_features, num_experts)
        self.tau = tau
        self.device = device
        print(f'tau={self.tau}')

    def forward(self, x, num_samples: int, hard: bool = False):
        logits = self.lin(x)  # shape: (batch, num_experts) or (num_samples, batch, num_experts)
        if num_samples < 2:
            # alpha = torch.softmax(logits, dim=-1)  # gating coefficients
            alpha = F.gumbel_softmax(logits, dim=-1, tau=self.tau, hard=hard)  # gating coefficients
        else:
            # Expand logits along the batch dimension for num_samples
            logits_expanded = logits.unsqueeze(0).expand(num_samples, -1, -1)

            # Sample using PyTorch's gumbel_softmax function
            alpha = F.gumbel_softmax(logits_expanded, tau=self.tau, hard=hard, dim=-1)
        return alpha


class MoE(nn.Module):
    def __init__(
            self,
            *,
            d_in: None | int = None,
            d_out: None | int = None,
            n_blocks: int,
            d_block: int,
            dropout: float,
            activation: str = 'ReLU',
            num_experts: None | int = None,
            gating_type: str,  # ['standard' or 'bayesian'] #TODO: add Literal and enum
            kl_factor: float = 1e-2,
            d_block_per_expert: None | int = None,
            default_num_samples: int = 5,
            tau: float = 1.0,
    ) -> None:
        assert d_out is not None, "the output layer must be added to the MoE"
        assert gating_type in ['standard', 'bayesian']
        super().__init__()
        if d_block_per_expert is not None:
            num_experts = d_block // d_block_per_expert
            print(f'num experts is set to :{num_experts}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        self.device = device
        self.n_blocks = n_blocks
        self.kl_factor = kl_factor
        self.num_experts = num_experts
        self.gating_type = gating_type
        self.default_num_samples = default_num_samples
        print(f'gating type:{self.gating_type}')
        print(f'default num samples:{self.default_num_samples}')
        print(f'kl_factor: {kl_factor}')
        d_first = d_block // num_experts if d_in is None else d_in

        self.stat_alpha_sum = None
        # Gating network
        self.gating_type = gating_type

        self.Weights = nn.ParameterList()
        for i in range(n_blocks + 1):  # one more for the output layer!
            w = torch.zeros(num_experts, d_first if i == 0 else d_block // num_experts,
                            d_out if i == n_blocks else d_block // num_experts)
            w = init_rsqrt_uniform_(w, w.shape[-1])
            self.Weights.append(nn.Parameter(w))

        self.activation = getattr(nn, activation)()

        self.dropout = nn.Dropout(dropout)  # if self.gating_type == 'standard' else None

        if self.gating_type == 'standard':
            self.gate = nn.Sequential(
                nn.Linear(d_first, num_experts),
                nn.Softmax(dim=-1)
            )

        elif self.gating_type == 'bayesian':
            # self.gate = BayesianGatingNetwork(
            #     in_features=d_first,
            #     num_experts=num_experts,
            #     prior_std=gating_prior_std,
            #     device=self.device,
            # )
            self.gate = GumbelGatingNetwork(d_first, num_experts, tau=tau, device=device)
        else:
            raise ValueError(f'The gating type "{self.gating_type}" is not supported.')

    def forward(self, x: Tensor, num_samples: None | int = None, return_average: bool = True) -> Tensor:
        """
        If self.training is True:
           - Sample one alpha from gate (as usual),
           - Optionally store statistics,
           - Compute and return the weighted sum of expert outputs.
        If self.training is False (eval mode):
           - Sample 10 alphas from gate,
           - Compute expert outputs once (they're standard),
           - Average the weighted sums over those 10 alpha samples.
        """
        # print(f'num samples:{num_samples}')
        # TODO: improve code clarity
        if self.training or self.gating_type == 'standard':
            num_samples = 1
        elif num_samples is None:
            num_samples = self.default_num_samples

        if self.training or num_samples < 2 or self.gating_type == 'standard':
            # [batch_size, num_experts] -> [num_experts, batch_size]
            alpha = self.gate(x, num_samples=num_samples) if self.gating_type == 'bayesian' \
                else self.gate(x)
            alpha = alpha.transpose(-1, -2)
        else:
            # [num_samples, batch_size, num_experts] -> [num_samples, num_experts, batch_size]
            alpha = self.gate(x, num_samples=num_samples).permute(0, 2, 1)

        for i in range(self.n_blocks + 1):
            x = torch.einsum('...nd,...dh->...nh', x, self.Weights[i])
            if i < self.n_blocks:
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)

        if self.training or num_samples < 2 or self.gating_type == 'standard':
            output = torch.sum(alpha.unsqueeze(-1) * x, dim=0)
        else:
            # EVAL MODE (Bayesian ensemble)
            weighted_expert_outputs = alpha.unsqueeze(-1) * x.unsqueeze(0)

            # 4) Sum over experts => [10, batch_size, output_dim]
            weighted_sums = torch.sum(weighted_expert_outputs, dim=1)

            # [ num_samples, batch_size, output_dim]
            if return_average:
                output = weighted_sums.mean(dim=0)
            else:
                output = weighted_sums
        return output