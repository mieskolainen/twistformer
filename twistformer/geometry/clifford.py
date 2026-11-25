# Clifford algebras for spatial and Lorentz geometry
#
# GATr:           https://arxiv.org/abs/2305.18415
# E/P/C-GATr:     https://arxiv.org/abs/2311.04744
# L-GATr:         https://arxiv.org/abs/2405.14806
# L-GATr physics: https://arxiv.org/abs/2411.00446
#
# mikael.mieskolainen@cern.ch (2026)

from importlib.util import find_spec
from itertools import combinations
from functools import lru_cache
import math

import torch
from torch import nn
from torch.nn import functional as F

SIGNATURES = {"euclidean": (1, 1, 1), "projective": (0, 1, 1, 1), "conformal": (1, 1, 1, 1, -1), "lorentz": (1, -1, -1, -1)}

_HAS_TRITON = find_spec("triton") is not None


def _launch_blades(left: torch.Tensor, right: torch.Tensor, form: str, adjoint: int) -> torch.Tensor:
    """
    Launch one fixed Clifford contraction inside eager or compiled training

    Args:
        left, right:  Equal-shaped FP32 multivectors
        form:  Algebra name for a product, or projective_join
        adjoint:  Forward, left derivative or right derivative index
    Returns:
        output: Contiguous multivectors
    """
    from .projective_kernels import launch

    return launch(left, right, form, adjoint)


_compiled_blades = torch.library.custom_op("twistformer::blade_contraction", _launch_blades, mutates_args=())


@_compiled_blades.register_fake
def _blade_shape(left, right, form, adjoint):
    """Propagate the output layout through dynamic graph capture"""
    return left.new_empty(left.shape)


def _blade_contraction(left, right, form, adjoint):
    """Expose the fused kernel as one operator during graph capture"""
    if torch.compiler.is_compiling():
        return _compiled_blades(left, right, form, adjoint)
    return _launch_blades(left, right, form, adjoint)


def sparse_product(left, right, indices, signs):
    """
    Contract one right blade for each output and left blade

    Args:
        left:  Left multivectors (...,D)
        right:  Right multivectors (...,D)
        indices:  Right blade indices (D,D)
        signs:  Signed algebra coefficients (D,D)
    Returns:
        product: Contracted multivectors (...,D)
    """
    return (left.unsqueeze(-2) * right[..., indices] * signs).sum(-1)

class BladeProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, indices, signs, right_indices, right_signs, form=None):
        """
        Save bilinear operands without particle-sized blade intermediates

        Args:
            ctx:  Autograd state
            left:  Left multivectors (...,D)
            right:  Right multivectors (...,D)
            indices:  Forward right-blade indices
            signs:  Forward algebra coefficients
            right_indices:  Output-blade indices for the right derivative
            right_signs:  Algebra coefficients for the right derivative
            form:  Optional algebra product or join name
        Returns:
            product: Broadcast multivector product
        """
        ctx.save_for_backward(left, right, indices, signs, right_indices, right_signs)
        ctx.form = form
        if form is not None:
            return _blade_contraction(left, right, form, 0)
        return sparse_product(left, right, indices, signs)

    @staticmethod
    def backward(ctx, gradient):
        """
        Contract both bilinear derivatives without scatter accumulation

        Args:
            ctx:  Saved operands and algebra coefficients
            gradient:  Output derivative (...,D)
        Returns:
            gradients: Operand derivatives and no algebra derivatives
        """
        left, right, indices, signs, right_indices, right_signs = ctx.saved_tensors
        if ctx.form is not None and not torch.is_grad_enabled():
            dl = _blade_contraction(gradient, right, ctx.form, 1) if ctx.needs_input_grad[0] else None
            dr = _blade_contraction(left, gradient, ctx.form, 2) if ctx.needs_input_grad[1] else None
            return dl, dr, None, None, None, None, None
        left_gradient = right_gradient = None
        if ctx.needs_input_grad[0]:
            left_gradient = sparse_product(gradient, right, indices.T, signs.T).sum_to_size(left.shape)
        if ctx.needs_input_grad[1]:
            right_gradient = sparse_product(left, gradient, right_indices, right_signs).sum_to_size(right.shape)
        gradients = (left_gradient, right_gradient, None, None, None, None, None)
        return gradients[:len(ctx.needs_input_grad)]

@lru_cache(maxsize=1)
def sparse_product_kernel():
    """
    Compile CUDA blade contractions without storing particle-sized products

    Returns:
        kernel: Fused geometric product and backward contractions
    """
    return torch.compile(BladeProduct.apply, dynamic=True)

def contract_blades(left, right, indices, signs, right_indices, right_signs, kernel=None, form=None):
    """
    Select fused CUDA contractions or direct graph capture

    Args:
        left:  Left multivectors (...,D)
        right:  Right multivectors (...,D)
        indices:  Right blade indices (D,D)
        signs:  Signed algebra coefficients (D,D)
        right_indices:  Output-blade indices for the right derivative
        right_signs:  Algebra coefficients for the right derivative
        kernel:  Blade coefficient tensor for CPU contractions
        form:  Algebra name for a product, or projective_join
    Returns:
        product: Contracted multivectors (...,D)
    """
    if kernel is not None and left.dtype == right.dtype == kernel.dtype:
        if left.device.type == "cpu" and not torch.is_autocast_enabled("cpu"):
            # BLAS contracts the blade outer product without an indexed scatter in backward
            pairs = (left.unsqueeze(-1) * right.unsqueeze(-2)).flatten(-2)
            return F.linear(pairs, kernel.flatten(1))
        if (left.is_cuda and _HAS_TRITON and torch.version.hip is None
                and left.dtype == torch.float32 and left.ndim >= 2
                and left.shape == right.shape and left.numel() > 0):
            return BladeProduct.apply(left, right, indices, signs, right_indices, right_signs, form)
    if not left.is_cuda:
        return sparse_product(left, right, indices, signs)
    if not torch.compiler.is_compiling():
        return sparse_product_kernel()(left, right, indices, signs, right_indices, right_signs)
    return BladeProduct.apply(left, right, indices, signs, right_indices, right_signs)

def blade_product(left, right, signature):
    """
    Multiply ordered basis blades using the Clifford anticommutator

    Args:
        left:  Bit mask of the left blade
        right:  Bit mask of the right blade
        signature:  Diagonal vector metric
    Returns:
        result: Output blade mask and signed coefficient
    """
    swaps = sum((right & ((1 << i) - 1)).bit_count() for i in range(len(signature)) if left & (1 << i))
    factor = (-1) ** swaps
    for i, metric in enumerate(signature):
        if left & right & (1 << i):
            factor *= metric
    return left ^ right, factor

@lru_cache(maxsize=None)
def blade_tables(signature):
    """
    Enumerate oriented blade products and dual exterior products

    Args:
        signature:  Diagonal Clifford metric
    Returns:
        tables: Blade masks and (output, left, right, sign) entries for product and join
    """
    n = len(signature)
    blades = tuple(sum(1 << i for i in indices) for grade in range(n + 1) for indices in combinations(range(n), grade))
    index = {blade: i for i, blade in enumerate(blades)}
    full = (1 << n) - 1
    # Complementary blades are disjoint, so their orientation is independent of the metric
    dual_sign = {a: blade_product(a, full ^ a, signature)[1] for a in blades}
    products, joins = [], []
    for i, a in enumerate(blades):
        for j, b in enumerate(blades):
            out, sign = blade_product(a, b, signature)
            products.append((index[out], i, j, sign))
            if (a | b) == full:
                # Dual of the wedge of the two complementary blades
                _, wedge_sign = blade_product(full ^ a, full ^ b, signature)
                joins.append((index[full ^ out], i, j, dual_sign[a] * dual_sign[b] * wedge_sign * dual_sign[out]))
    return blades, tuple(products), tuple(joins)


class CliffordAlgebra(nn.Module):
    def __init__(self, kind, proper=True):
        """
        Construct products and complete equivariant linear bases

        Args:
            kind:  Euclidean, projective, conformal or Lorentz algebra
            proper:  Include Lorentz Hodge-dual maps for SO+(1,3)
        """
        super().__init__()
        self.kind, self.proper = kind, proper
        self.signature = SIGNATURES[kind]
        n = len(self.signature)
        self.blades, products, joins = blade_tables(self.signature)
        self.index = {blade: i for i, blade in enumerate(self.blades)}
        self.dim = len(self.blades)
        grades = torch.tensor([blade.bit_count() for blade in self.blades])
        grade_mask = torch.stack([grades == k for k in range(n + 1)]).float()
        metric = torch.tensor([math.prod(self.signature[i] for i in range(n) if blade & (1 << i)) for blade in self.blades]).float()
        kernel = torch.zeros(self.dim, self.dim, self.dim)
        output, left, right, coefficient = torch.tensor(products, dtype=torch.long).T
        kernel[output, left, right] = coefficient.to(kernel.dtype)
        right_index = torch.empty(self.dim, self.dim, dtype=torch.long)
        signs = torch.empty_like(kernel[0])
        right_index[output, left] = right
        signs[output, left] = coefficient.to(kernel.dtype)
        projectors = torch.diag_embed(grade_mask)
        basis = list(projectors)
        invariant = torch.zeros(self.dim)
        if kind == "projective":
            invariant[self.index[1]] = 1
            left = torch.einsum("oij,i->oj", kernel, invariant)
            basis.extend(projectors[k] @ left for k in range(1, n + 1))
        elif kind == "conformal":
            # Infinity = e_minus - e_plus, origin = (e_minus + e_plus) / 2
            invariant[self.index[1 << 4]] = 1
            invariant[self.index[1 << 3]] = -1
            left = torch.einsum("oij,i->oj", kernel, invariant)
            basis.extend(projectors[k - 1] @ left @ projectors[k] for k in range(1, n + 1))
            basis.extend(projectors[k + 1] @ left @ projectors[k] for k in range(n))
            basis.extend(left @ projectors[k - 1] @ left @ projectors[k] for k in range(1, n))
        elif kind == "lorentz" and proper:
            dual = kernel[:, -1, :]
            basis.extend(dual @ projector for projector in projectors)
        basis = torch.stack(basis)
        if kind != "lorentz":
            basis = basis / basis.square().sum((-2, -1), keepdim=True).sqrt()
        scalar = torch.zeros(self.dim)
        scalar[0] = 1
        constants = [scalar]
        if invariant.any():
            constants.append(invariant)
        if kind == "lorentz" and proper:
            pseudo = torch.zeros_like(scalar)
            pseudo[-1] = 1
            constants.append(pseudo)
        constants = torch.stack(constants)
        functionals = constants * metric
        functionals = functionals[functionals.abs().sum(-1) > 0]
        join = torch.zeros_like(kernel)
        output, left, right, coefficient = torch.tensor(joins, dtype=torch.long).T
        join[output, left, right] = coefficient.to(join.dtype)
        self.register_buffer("gp_kernel", kernel, persistent=False)
        join_right = join.abs().argmax(-1)
        join_sign = join.gather(-1, join_right[..., None]).squeeze(-1)
        gp_derivative = kernel.permute(2, 1, 0)
        join_derivative = join.permute(2, 1, 0)
        gp_gradient_right = gp_derivative.abs().argmax(-1)
        join_gradient_right = join_derivative.abs().argmax(-1)
        for name, value in {
            "metric": metric,
            "grades": grades,
            "grade_mask": grade_mask,
            "linear_basis": basis,
            "constants": constants,
            "functionals": functionals,
            "gp_right": right_index,
            "gp_sign": signs,
            "gp_gradient_right": gp_gradient_right,
            "gp_gradient_sign": gp_derivative.gather(-1, gp_gradient_right[..., None]).squeeze(-1),
            "join_kernel": join,
            "join_right": join_right,
            "join_sign": join_sign,
            "join_gradient_right": join_gradient_right,
            "join_gradient_sign": join_derivative.gather(-1, join_gradient_right[..., None]).squeeze(-1),
        }.items():
            self.register_buffer(name, value, persistent=False)

    def product(self, left, right):
        """
        Contract multivectors with the signed Clifford product

        Args:
            left:  Left multivectors (...,D)
            right:  Right multivectors (...,D)
        Returns:
            product: Broadcast multivectors (...,D)
        """
        return contract_blades(
            left, right, self.gp_right, self.gp_sign, self.gp_gradient_right, self.gp_gradient_sign, self.gp_kernel, self.kind
        )

    def join(self, left, right, reference):
        """
        Form the projective join with a data-dependent orientation reference

        Args:
            left:  Left projective multivectors (...,16)
            right:  Right projective multivectors (...,16)
            reference:  Reference e123 coefficient broadcast to (...,1)
        Returns:
            joined: Equivariant projective join
        """
        return reference * contract_blades(
            left, right, self.join_right, self.join_sign, self.join_gradient_right, self.join_gradient_sign,
            self.join_kernel if self.kind == "projective" else None, "projective_join",
        )

    def invariants(self, multivectors):
        """
        Extract invariant linear functionals

        Args:
            multivectors:  Algebra features (...,C,D)
        Returns:
            scalars: Invariant features (...,C,K)
        """
        return torch.einsum("...cd,kd->...ck", multivectors, self.functionals)

    def norm_squared(self, multivectors):
        """
        Sum absolute grade norms without cancelling distinct grades

        Args:
            multivectors:  Algebra features (...,C,D)
        Returns:
            norm: Invariant nonnegative norm (...,C,1)
        """
        if self.kind in {"euclidean", "projective"}:
            return (multivectors.square() * self.metric).sum(-1, keepdim=True)
        grade_norm = torch.einsum("...d,gd->...g", multivectors.square() * self.metric, self.grade_mask)
        return grade_norm.abs().sum(-1, keepdim=True)

    def embed(self, coordinates):
        """
        Embed spatial points or Lorentz four-vectors in grade-major coordinates

        Args:
            coordinates:  Points (...,3) or Lorentz vectors (...,4), E first
        Returns:
            multivectors: Geometric embeddings (...,D)
        """
        mv = coordinates.new_zeros(*coordinates.shape[:-1], self.dim)
        if self.kind == "projective":
            mv[..., self.index[14]] = 1
            mv[..., self.index[13]] = -coordinates[..., 0]
            mv[..., self.index[11]] = coordinates[..., 1]
            mv[..., self.index[7]] = -coordinates[..., 2]
        elif self.kind == "conformal":
            mv[..., 1:4] = coordinates
            square = coordinates.square().sum(-1)
            mv[..., 4] = (1 - square) / 2
            mv[..., 5] = (1 + square) / 2
        else:
            mv[..., 1 : 1 + coordinates.shape[-1]] = coordinates
        return mv
