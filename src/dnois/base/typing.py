# The philosophy of this type system is, when multiple types for a variable
# (arguments, etc.) are allowed to improve flexibility and usability and
# there is a standard type for it so that all the allowed types can be
# converted to the standard type, a new type alias (typically constructed
# through Union) should be defined to cover allowed types and a converter
# function should be provided to convert them to the standard type.
# Converter functions must conduct rigorous type checking and should be named
# as the lower case of corresponding type alias.

from pathlib import Path
import numbers
from numbers import *
import typing
from typing import *

# Support for Self type (Python 3.11+)
try:
    from typing import Self
except ImportError:
    try:
        from typing_extensions import Self
    except ImportError:
        # Fallback for Python < 3.11
        Self = typing.TypeVar('Self', bound='object')

import torch
from torch import is_tensor, Tensor

from .exception import ShapeError

__all__ = [
    'is_scalar',
    'is_tensor',
    'pair',
    'scalar',
    'size2d',
    'sizend',
    'tensor',
    'vector',

    'ConvOut',
    'Double',
    'Numeric',
    'NumInv',
    'Pair',
    'Scalar',
    'Self',
    'Size2d',
    'Sizend',
    'Spacing',
    'Tensor',
    'TextFile',
    'Triple',
    'Ts',
    'Vector',
]
__all__ += typing.__all__
__all__ += numbers.__all__

_T = TypeVar('_T')

_dty = torch.dtype
_dev = torch.device
Device = Union[str, int, _dev]  # as same as torch.DeviceLikeType

# tensor-like
Ts = Tensor
Spacing = Union[float, Ts]  # delta (grid spacing) type
Numeric = Union[Real, Ts]  # support numeric operation
NumInv = TypeVar('NumInv', bound=Numeric)  # invariant numeric type
Scalar = Union[Real, Ts]  # can be converted to 0d tensor
Vector = Union[Real, Sequence[Real], Ts]  # can be converted to 1d tensor

Pair = Union[_T, tuple[_T, _T]]
Size2d = Pair[int]
Sizend = Union[int, Sequence[int]]

Double = tuple[_T, _T]
Triple = tuple[_T, _T, _T]

# options
ConvOut = Literal['full', 'same', 'valid']

TextFile = str | Path | TextIO


def pair(arg: Pair[_T], type_: type[_T] = None) -> tuple[_T, _T]:
    if type_ is None:
        return arg if isinstance(arg, tuple) else (arg, arg)

    cn = type_.__name__
    if isinstance(arg, type_):
        return arg, arg
    elif isinstance(arg, tuple):
        if not isinstance(arg[0], type_) or not isinstance(arg[1], type_):
            raise ValueError(f'A pair of {cn} expected, got {arg}')
        # allow negative
        return cast(tuple[type_, type_], arg)
    else:
        raise ValueError(f'An {cn} or a pair of {cn} expected, got {type(arg)}')


def size2d(size: Size2d) -> tuple[int, int]:
    return pair(size, int)


def sizend(size: Sizend, ndim: int = None) -> list[int]:
    if isinstance(size, int):
        return [size for _ in range(ndim)]
    elif isinstance(size, Sequence):
        if not all(isinstance(s, int) for s in size):
            raise ValueError(f'A sequence of int expected, got {size}')
        if ndim is not None and len(size) != ndim:
            raise ValueError(f'{ndim} integers expected, got {len(size)}')
        return list(size)
    else:
        raise ValueError(f'An int or a sequence of int expected, got {type(size)}')


def vector(arg: Vector, dtype: _dty = None, device: Device = None, **kwargs) -> Ts:
    cfg = {}
    if dtype is not None:
        cfg['dtype'] = dtype
    if device is not None:
        cfg['device'] = device
    if isinstance(arg, Real):
        return torch.tensor([arg], **cfg)
    elif isinstance(arg, Sequence) and all(isinstance(item, Real) for item in arg):
        return torch.tensor(arg, **cfg)
    elif is_tensor(arg):
        if arg.ndim == 0:
            arg = arg.unsqueeze(0)
        if arg.ndim != 1:
            raise ShapeError(f'Trying to convert a tensor with shape {arg.shape} to a vector')
        return arg.to(**cfg, **kwargs)
    else:
        raise TypeError(f'A float, int, a sequence of them or a 1d tensor expected, got {type(arg)}')


def scalar(arg: Scalar, dtype: _dty = None, device: Device = None, **kwargs) -> Ts:
    cfg = {}
    if dtype is not None:
        cfg['dtype'] = dtype
    if device is not None:
        cfg['device'] = device
    if isinstance(arg, Real):
        return torch.tensor(arg, **cfg)
    elif is_tensor(arg):
        if arg.ndim != 0:
            raise ShapeError(f'Trying to convert a tensor with shape {arg.shape} to a scalar')
        return arg.to(**cfg, **kwargs)
    else:
        raise TypeError(f'A float, int or a 0d tensor is expected, got {type(arg)}')


def is_scalar(arg: Any) -> bool:
    return isinstance(arg, Real) or (is_tensor(arg) and arg.ndim == 0)


def tensor(x) -> Ts:
    if is_tensor(x):
        return x
    return torch.tensor(x)
