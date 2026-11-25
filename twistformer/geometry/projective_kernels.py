# Fused Clifford products and projective join on CUDA
#
# mikael.mieskolainen@cern.ch (2026)

import torch
import triton
import triton.language as tl


def _forms():
    """
    Build compile-time blade indices and signs with integer arithmetic

    Returns:
        forms: Storage indices and signed lane masks for each forward and adjoint map
    """
    from .clifford import SIGNATURES, blade_tables
    
    forms = {}
    for kind, signature in SIGNATURES.items():
        blades, products, joins = blade_tables(signature)
        indices = tuple(blades.index(mask) for mask in range(len(blades)))
        operations = [(kind, products)]
        if kind == "projective":
            operations.append(("projective_join", joins))
        for name, entries in operations:
            maps = []
            for output, left in ((0, 1), (1, 0), (2, 1)):
                positive = [0] * len(blades)
                negative = [0] * len(blades)
                for entry in entries:
                    if entry[3]:
                        lanes = positive if entry[3] > 0 else negative
                        lanes[blades[entry[left]]] |= 1 << blades[entry[output]]
                maps.append(tuple(zip(positive, negative)))
            forms[name] = indices, tuple(maps)
    return forms


@triton.constexpr_function
def _index(form, blade):
    """Map the basis bit mask to the grade-ordered storage index"""
    return _FORMS[form][0][blade]


@triton.constexpr_function
def _sign_mask(form, adjoint, blade, negative):
    """Read an output lane mask for a signed left blade"""
    return _FORMS[form][1][adjoint][blade][negative]


@triton.jit
def _contract(left, right, output, rows, channels, left_token, left_channel, left_blade,
              right_token, right_channel, right_blade, form: tl.constexpr, adjoint: tl.constexpr,
              dim: tl.constexpr, block: tl.constexpr):
    """Contract one multivector per group of dim contiguous warp lanes"""
    offset = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    row = offset // dim
    blade = (tl.arange(0, block) % dim).to(tl.uint32)
    index = tl.full((block,), 0, tl.int32)
    for i in tl.static_range(1, dim):
        index = tl.where(blade == i, _index(form, i), index)
    left_start = row // channels * left_token + row % channels * left_channel
    right_start = row // channels * right_token + row % channels * right_channel
    a = tl.load(left + left_start + index * left_blade, mask=row < rows, other=0)
    b = tl.load(right + right_start + index * right_blade, mask=row < rows, other=0)
    total = tl.full((block,), 0, tl.float32)
    for i in tl.static_range(dim):
        # Broadcast left blade i inside its group of 8, 16 or 32 lanes
        x = tl.inline_asm_elementwise(
            "shfl.sync.idx.b32 $0, $1, $2, $3, -1;", constraints="=f,f,r,r",
            args=[a, i, ((32 - dim) << 8) | 31], dtype=tl.float32, is_pure=True, pack=1,
        )
        # Product blade masks combine by XOR, the join also takes the complement
        shift = dim - 1 - i if form == "projective_join" else i
        y = tl.inline_asm_elementwise(
            "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;", constraints="=f,f,r",
            args=[b, shift], dtype=tl.float32, is_pure=True, pack=1,
        )
        positive = _sign_mask(form, adjoint, i, False)
        negative = _sign_mask(form, adjoint, i, True)
        if negative:
            x = tl.where(((negative >> blade) & 1) != 0, -x, x)
        updated = tl.fma(x, y, total)
        active = positive | negative
        if active != (1 << dim) - 1:
            updated = tl.where(((active >> blade) & 1) != 0, updated, total)
        total = updated
    tl.store(output + row * dim + index, total, mask=row < rows)


def launch(left, right, form, adjoint):
    """
    Preserve channel strides from the joint operand projection

    Args:
        left, right:  Equal-shaped FP32 multivectors
        form:  Algebra name for a product, or projective_join
        adjoint:  Forward, left derivative or right derivative index
    Returns:
        output: Contiguous multivectors with the input shape
    """
    channels, dim = left.shape[-2:]
    output = torch.empty(left.shape, device=left.device, dtype=left.dtype)
    a = left.reshape(-1, channels, dim)
    b = right.reshape(-1, channels, dim)
    rows = a.shape[0] * channels
    # One FP32 coefficient per CUDA lane
    block = 128
    grid = (triton.cdiv(rows * dim, block),)
    with torch.cuda.device_of(left):
        _contract[grid](a, b, output, rows, channels, *a.stride(), *b.stride(), form, adjoint, dim, block, num_warps=block // 32)
    return output


_FORMS = _forms()
