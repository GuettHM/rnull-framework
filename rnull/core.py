import torch
import torch.nn as nn


class RNullTensor:
    """
    Vectorized Dual-Tensor (v, m) PyTorch Emulator for Causal Transparency Algebra (R_null).
    - self.v (torch.float32): dense numerical payload.
    - self.m (torch.bool): causal abstention bitmask (True <=> x = null).
    """

    def __init__(self, data, is_null=None):
        if isinstance(data, RNullTensor):
            self.v = data.v.clone()
            self.m = (
                data.m.clone()
                if is_null is None
                else torch.as_tensor(is_null, dtype=torch.bool, device=self.v.device).expand_as(self.v)
            )
        else:
            self.v = torch.as_tensor(data, dtype=torch.float32)
            if is_null is None:
                self.m = torch.zeros_like(self.v, dtype=torch.bool)
            else:
                self.m = torch.as_tensor(is_null, dtype=torch.bool, device=self.v.device).expand_as(self.v)

    @classmethod
    def null_like(cls, tensor):
        """Creates a tensor completely in epistemic abstention state (X = null)."""
        v = tensor.v if isinstance(tensor, RNullTensor) else torch.as_tensor(tensor, dtype=torch.float32)
        return cls(torch.zeros_like(v), is_null=torch.ones_like(v, dtype=torch.bool))

    def __add__(self, other):
        """
        Extended Addition (Bi-Identity (+)): x (+) null = x, null (+) null = null.
        Replaces masked elements with additive identity 0.0 and applies logical AND (self.m & other.m).
        """
        other = other if isinstance(other, RNullTensor) else RNullTensor(other)
        v_out = torch.where(self.m, 0.0, self.v) + torch.where(other.m, 0.0, other.v)
        return RNullTensor(v_out, is_null=self.m & other.m)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Extended Subtraction ((-)): x (-) null = x, null (-) x = -x, null (-) null = null."""
        other = other if isinstance(other, RNullTensor) else RNullTensor(other)
        v_out = torch.where(self.m, 0.0, self.v) - torch.where(other.m, 0.0, other.v)
        return RNullTensor(v_out, is_null=self.m & other.m)

    def __mul__(self, other):
        """
        Symmetric Multiplicative Fusion (Bi-Identity (*)): x (*) null = x, null (*) null = null.
        Replaces masked elements with multiplicative identity 1.0 and applies logical AND (self.m & other.m).
        Enables fault-tolerant multimodal fusion: S_cam (*) null (*) S_radar = S_cam (*) S_radar.
        """
        other = other if isinstance(other, RNullTensor) else RNullTensor(other)
        v_out = torch.where(self.m, 1.0, self.v) * torch.where(other.m, 1.0, other.v)
        return RNullTensor(v_out, is_null=self.m & other.m)

    def __rmul__(self, other):
        return self.__mul__(other)

    def synaptic_mul(self, x):
        """
        Absorbing Synaptic MAC Coupling ((.)):
        Propagates null via logical OR (self.m | x.m) when either weight or input is null,
        allowing the subsequent (+)-accumulator to bypass the inactive synapse.
        """
        x = x if isinstance(x, RNullTensor) else RNullTensor(x)
        m_out = self.m | x.m
        v_out = torch.where(m_out, 0.0, self.v * x.v)
        return RNullTensor(v_out, is_null=m_out)

    def synaptic_skip(self, x):
        """
        Directed Synaptic Skip Coupling ((|>)):
        - If weight self.m is True and input x.m is False: acts as passive skip-wire (null |> x = x).
        - If input x.m is True: synapse abstains (W |> null = null).
        """
        x = x if isinstance(x, RNullTensor) else RNullTensor(x)
        v_out = torch.where(self.m, 1.0, self.v) * torch.where(x.m, 0.0, x.v)
        return RNullTensor(v_out, is_null=x.m)

    def chi_null(self, evaluate_polynomial: bool = False) -> torch.Tensor:
        """
        Causal State Detector chi_null(x) in {0.0, 1.0}.
        If evaluate_polynomial=True, explicitly evaluates the non-distributivity polynomial:
            ((1 (*) 0) (+) (1 (*) x)) (-) (1 (*) (0 (+) x))
        proving exact equivalence between the algebraic equation and the boolean mask.
        """
        if not evaluate_polynomial:
            return self.m.to(torch.float32)
        one = RNullTensor(torch.ones_like(self.v))
        zero = RNullTensor(torch.zeros_like(self.v))
        return (((one * zero) + (one * self)) - (one * (zero + self))).v

    def __repr__(self):
        return f"RNullTensor(v={self.v}, is_null={self.m})"


class RNullLinear(nn.Module):
    """
    Linear ∅-Net Layer with absorbing synaptic MAC and Native Epistemic Abstention (OOD Safety).
    If all inputs x_j for a sample are null, outputs Y = null deterministically without bias hallucination.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / (in_features ** 0.5)))
        self.register_buffer("weight_null_mask", torch.zeros(out_features, in_features, dtype=torch.bool))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: RNullTensor) -> RNullTensor:
        x_v = x.v.unsqueeze(1)                      # [B, 1, in]
        x_m = x.m.unsqueeze(1)                      # [B, 1, in]
        w_v = self.weight.unsqueeze(0)              # [1, out, in]
        w_m = self.weight_null_mask.unsqueeze(0)    # [1, out, in]

        # 1. Absorbing synaptic product (logical OR: w_m | x_m)
        syn_m = w_m | x_m
        syn_v = torch.where(syn_m, 0.0, w_v * x_v)

        # 2. Transparent additive accumulation (+) across in_features (logical AND on masks)
        out_v = syn_v.sum(dim=-1)
        out_m = syn_m.all(dim=-1)                   # True (null) iff ALL synapses abstained

        # 3. Suppress static bias if the neuron is in epistemic abstention (out_m == True)
        if self.bias is not None:
            out_v = torch.where(out_m, 0.0, out_v + self.bias.unsqueeze(0))

        return RNullTensor(out_v, is_null=out_m)


@torch.no_grad()
def continual_learning_step(param: nn.Parameter, raw_delta: torch.Tensor, frozen_mask: torch.Tensor) -> RNullTensor:
    """
    Drift-Free Continual Learning Update: W(t+1) = W(t) (+) Delta_W.
    Protected weights (frozen_mask == True) receive Delta_W = null, guaranteeing exact zero drift (MSE = 0.0)
    even in the presence of weight decay (-lambda * W) and momentum tails.
    """
    w_t = RNullTensor(param.data, is_null=torch.zeros_like(param.data, dtype=torch.bool))
    delta_w = RNullTensor(raw_delta, is_null=frozen_mask)
    w_next = w_t + delta_w
    param.data.copy_(w_next.v)
    return w_next
