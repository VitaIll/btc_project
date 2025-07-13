"""
Utilities Module for Time Series Analysis
=========================================

This module provides utilities for time series analysis, with a focus on
fractional differentiation methods used in financial machine learning.

Main Features:
--------------
- Fractional Differentiation (FD)
- Fixed-width Fractional Differentiation (FFD)
- Fast Fractional Differentiation using FFT

Example Usage:
--------------
    >>> import numpy as np
    >>> from utils import frac_diff_ffd
    >>> 
    >>> # Generate sample data
    >>> data = np.random.randn(1000).cumsum()
    >>> 
    >>> # Apply fractional differentiation
    >>> fracdiff_data = frac_diff_ffd(data, d=0.5)

Available Functions:
-------------------
- fast_fracdiff: Fast fractional differentiation using FFT
- frac_diff: Standard fractional differentiation for pandas Series/DataFrame
- frac_diff_ffd: Fixed-width fractional differentiation
- get_weights: Compute fractional differentiation weights
- get_weights_ffd: Compute fixed-width fractional differentiation weights
- validate_fracdiff_params: Validate input parameters

For more information on each function, use help(function_name).
"""

__version__ = "0.1.0"
__author__ = "Vit Illichmann, Jan Jouda"
__email__ = "vit.illichmann@gmail.com"


from .fracdiff import (
    fast_fracdiff,
    frac_diff,
    frac_diff_ffd,
    get_weights,
    get_weights_ffd,
    validate_fracdiff_params
)

__all__ = [
  
    'fast_fracdiff',
    'frac_diff',
    'frac_diff_ffd',
    'get_weights',
    'get_weights_ffd',
    'validate_fracdiff_params',
    
]


