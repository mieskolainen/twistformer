# Flow configuration
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass, field

@dataclass
class FlowConfig:
    type: str = "continuous"
    mode: str = "unconditional"
    shape: list[int] = field(default_factory=list)
    space: str = "direct"
    condition: str = "parameters"
    flip_parameter: str = ""
    network: str = "mlp"
    width: int = 64
    depth: int = 3
    activation: str = "silu"
    layer_norm: bool = False
    skip_connections: bool = True
    frequencies: int = 3
    dct: bool = False
    path: str = "CFM"
    sigma: float = 0.0
    warp: float = 1.0
    steps: int = 32
    integrator: str = "heun"
    base: str = "gaussian"
    trace: str = "hutchinson"
    trace_samples: int = 1
    finite_difference: float = 1e-3
    ot_method: str = "exact"
    ot_regularization: float = 0.05
    ot_mass_regularization: float = 1.0
    coupling: str = "rqs"
    layers: int = 4
    bins: int = 8
    bound: float = 3.0
    scale_limit: float = 3.0
    permutation: str = "reverse"
    linear: bool = True
    normalization: str = "actnorm"
    latent_dim: int = 16
    codec: str = "ae"
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.001
    lipschitz_weight: float = 0.0
    noise_regularization: float = 0.0
    matching_loss: str = "mse"
    evaluation_samples: int = 16
    temperature: float = 1.0
    max_attempts: int = 10
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)

    def __post_init__(self):
        """
        Validate flow algorithms and their physical operation modes
        """
        choices = {
            "type": {"continuous", "discrete"},
            "mode": {"unconditional", "conditional", "conditional-flip", "morph", "morph-shuffle",
                     "morph-conditional", "morph-conditional-shuffle"},
            "space": {"direct", "latent"},
            "condition": {"parameters", "encoder", "image", "graph"},
            "network": {"mlp", "lzmlp", "fastkan", "unet"},
            "activation": {"relu", "tanh", "silu", "elu", "gelu", "softplus"},
            "path": {"FM", "CFM", "COS-CFM", "OT", "S-BRIDGE"},
            "integrator": {"euler", "heun"},
            "base": {"gaussian", "uniform", "triangle"},
            "trace": {"exact", "vmap", "hutchinson", "finite_difference", "finite_hutchinson"},
            "ot_method": {"exact", "sinkhorn", "unbalanced", "partial"},
            "coupling": {"affine", "kan", "rqs"},
            "permutation": {"none", "reverse", "random"},
            "normalization": {"none", "actnorm", "batchnorm"},
            "codec": {"ae", "vae"},
            "matching_loss": {"mse", "momentum"},
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"flow.{name} must be one of {sorted(allowed)}")
        if min(self.width, self.depth, self.steps, self.trace_samples, self.layers, self.latent_dim,
               self.evaluation_samples, self.max_attempts) < 1 or self.bins < 2 or self.frequencies < 0:
            raise ValueError("Flow counts must be positive and bins must be at least two")
        if any(value < 1 for value in self.shape):
            raise ValueError("Flow shape entries must be positive")
        if min(self.sigma, self.reconstruction_weight, self.kl_weight, self.lipschitz_weight) < 0:
            raise ValueError("Flow regularization weights must be nonnegative")
        if not 0 <= self.warp <= 1.5 or not 0 <= self.noise_regularization <= 1:
            raise ValueError("Invalid flow time warp or noise regularization")
        if min(self.finite_difference, self.ot_regularization, self.ot_mass_regularization,
               self.bound, self.scale_limit, self.temperature) <= 0:
            raise ValueError("Flow numerical scales must be positive")
        if bool(self.lower) != bool(self.upper) or len(self.lower) != len(self.upper):
            raise ValueError("Flow constraints need matching lower and upper bounds")
        if any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError("Flow lower bounds must be below upper bounds")
        if self.type == "discrete" and "morph" in self.mode:
            raise ValueError("Paired transport uses continuous flow matching")
        if "morph" in self.mode and self.path == "FM":
            raise ValueError("FM has no arbitrary source distribution. Use CFM for morphing")
        if self.path == "S-BRIDGE" and self.sigma <= 0:
            raise ValueError("S-BRIDGE requires positive sigma")
        if self.space == "latent" and self.network == "unet":
            raise ValueError("Latent vectors require a vector network")
        if self.type == "discrete" and self.space == "latent":
            raise ValueError("Latent codecs require continuous flow matching")
        if self.condition == "image" and self.mode in {"conditional", "conditional-flip", "morph-conditional", "morph-conditional-shuffle"} and self.network != "unet":
            raise ValueError("Image conditions require a U-Net vector field")
        if self.network == "unet" and (self.type != "continuous" or self.width % 32):
            raise ValueError("U-Net requires continuous flow and a width divisible by 32")
        if self.lipschitz_weight and (self.network != "lzmlp" or self.type != "continuous"):
            raise ValueError("Lipschitz regularization requires a continuous lzmlp field")
        if self.matching_loss == "momentum" and self.space != "direct":
            raise ValueError("Momentum matching requires direct momentum coordinates")
        if self.path == "FM" and self.base != "gaussian":
            raise ValueError("FM requires a Gaussian base")
