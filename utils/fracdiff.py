import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings

from statsmodels.tsa.stattools import adfuller


def validate_fracdiff_params(d: float, x: Union[np.ndarray, pd.Series, pd.DataFrame]) -> None:
    """Validate input parameters for fractional differentiation."""
    if not isinstance(d, (int, float)):
        raise TypeError(f"Parameter d must be numeric, got {type(d)}")
    
    if d < 0:
        raise ValueError(f"Parameter d must be non-negative, got {d}")
    
    if d > 1:
        warnings.warn(f"Parameter d={d} > 1 may lead to non-stationary series")
    
    if isinstance(x, pd.DataFrame) and x.empty:
        raise ValueError("Input DataFrame is empty")
    elif isinstance(x, (np.ndarray, pd.Series)) and len(x) == 0:
        raise ValueError("Input array/series is empty")


def fast_fracdiff(x: np.ndarray, d: float) -> np.ndarray:
    """
    Fast fractional differentiation using FFT.
    
    Parameters:
    -----------
    x : np.ndarray
        Input time series
    d : float
        Fractional differentiation parameter
    
    Returns:
    --------
    np.ndarray
        Fractionally differentiated series
    """
    validate_fracdiff_params(d, x)
    
    T = len(x)
    if T < 2:
        return x.copy()
    
    # Use next power of 2 for FFT efficiency
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    
    # Compute binomial coefficients
    k = np.arange(1, T)
    b = [1.0]
    for i in range(1, T):
        b.append(b[-1] * (i - d - 1) / i)
    b = np.array(b)
    
    # Zero-pad for FFT
    z1 = np.zeros(np2)
    z1[:T] = b
    z2 = np.zeros(np2)
    z2[:T] = x
    
    # Convolution via FFT
    dx = np.fft.ifft(np.fft.fft(z1) * np.fft.fft(z2))
    
    return np.real(dx[:T])


def get_weights(d: float, size: int) -> np.ndarray:
    """
    Compute weights for fractional differentiation.
    
    Parameters:
    -----------
    d : float
        Fractional differentiation parameter
    size : int
        Number of weights to compute
    
    Returns:
    --------
    np.ndarray
        Weight vector (reversed order)
    """
    if size < 1:
        raise ValueError(f"Size must be positive, got {size}")
    
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    
    return np.array(w[::-1]).reshape(-1, 1)


def get_weights_ffd(d: float, thres: float = 1e-5, max_size: Optional[int] = None) -> np.ndarray:
    """
    Compute weights for Fixed-width Fractional Differentiation (FFD).
    
    Parameters:
    -----------
    d : float
        Fractional differentiation parameter
    thres : float
        Threshold for weight truncation
    max_size : int, optional
        Maximum number of weights
    
    Returns:
    --------
    np.ndarray
        Weight vector (reversed order)
    """
    if thres <= 0:
        raise ValueError(f"Threshold must be positive, got {thres}")
    
    w = [1.0]
    k = 1
    
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        
        if abs(w_k) < thres:
            break
            
        w.append(w_k)
        k += 1
        
        if max_size is not None and k >= max_size:
            break
    
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff(series: Union[pd.Series, pd.DataFrame, np.ndarray], 
              d: float, 
              thres: float = None,
              max_weight_len: int = None) -> Union[pd.Series, pd.DataFrame]:
    """
    Apply fractional differentiation to time series.
    
    Parameters:
    -----------
    series : pd.Series, pd.DataFrame, or np.ndarray
        Input time series
    d : float
        Fractional differentiation parameter
    thres : float, optional
        Weight threshold for determining burn-in period based on cumulative weight.
        Cannot be used together with max_weight_len.
    max_weight_len : int, optional
        Maximum number of weights to use (caps the lookback window).
        Cannot be used together with thres.
    
    Returns:
    --------
    pd.Series or pd.DataFrame
        Fractionally differentiated series
        
    Raises:
    -------
    ValueError
        If both thres and max_weight_len are specified
    ValueError
        If neither thres nor max_weight_len are specified
    """
    # Validate capping options
    if thres is not None and max_weight_len is not None:
        raise ValueError("Only one of 'thres' or 'max_weight_len' can be specified, not both")
    if thres is None and max_weight_len is None:
        raise ValueError("Either 'thres' or 'max_weight_len' must be specified")
    
    # Convert to DataFrame for consistent handling
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    if isinstance(series, pd.Series):
        series = series.to_frame()
    
    validate_fracdiff_params(d, series)
    
    # Compute weights
    w = get_weights(d, len(series))
    
    # Determine burn-in period based on chosen method
    if thres is not None:
        # Original threshold-based method
        w_cumsum = np.cumsum(np.abs(w))
        w_cumsum /= w_cumsum[-1]
        skip = len(w_cumsum[w_cumsum > thres])
    else:
        # New max_weight_len method
        if max_weight_len <= 0:
            raise ValueError("max_weight_len must be a positive integer")
        # Only need to skip enough observations to have max_weight_len history
        skip = min(max_weight_len - 1, len(series) - 1)
    
    # Apply fractional differentiation
    output = {}
    
    for col in series.columns:
        series_col = series[col].ffill().dropna()
        fracdiff_values = pd.Series(index=series_col.index[skip:], dtype=float)
        
        for i in range(skip, len(series_col)):
            if np.isfinite(series_col.iloc[i]):
                if max_weight_len is not None:
                    # When using max_weight_len, limit the window size
                    window_start = max(0, i + 1 - max_weight_len)
                    window = series_col.iloc[window_start:i+1].values
                    # Use only the weights that correspond to the window size
                    weights = w[-(len(window)):]
                else:
                    # Original behavior with threshold
                    window = series_col.iloc[:i+1].values
                    weights = w[-(i+1):]
                
                fracdiff_values.iloc[i-skip] = np.dot(weights.T, window)[0]
        
        output[col] = fracdiff_values
    
    result = pd.DataFrame(output)
    return result[result.columns[0]] if len(result.columns) == 1 else result


def frac_diff_ffd(x: Union[np.ndarray, pd.Series], 
                  d: float, 
                  thres: float = 1e-5,
                  disable_warning: bool = False) -> np.ndarray:
    """
    Fixed-width Fractional Differentiation (FFD).
    
    Parameters:
    -----------
    x : np.ndarray or pd.Series
        Input time series
    d : float
        Fractional differentiation parameter
    thres : float
        Weight threshold
    disable_warning : bool
        Disable warning for non-log transformed data
    
    Returns:
    --------
    np.ndarray
        Fractionally differentiated series with fixed window
    """
    if isinstance(x, pd.Series):
        x = x.values
    
    validate_fracdiff_params(d, x)
    
    # Warning for potentially non-log transformed data
    if not disable_warning and np.max(np.abs(x)) > 10.0:
        warnings.warn(
            "Input values are large (>10). Consider applying log transformation first. "
            "Set disable_warning=True to suppress this message."
        )
    
    # Get FFD weights
    w = get_weights_ffd(d, thres, len(x))
    width = len(w) - 1
    
    # Apply FFD
    output = np.zeros(len(x))
    
    for i in range(width, len(x)):
        window = x[i - width:i + 1]
        output[i] = np.dot(w.T, window)[0]
    
    return output


def get_optimal_fraction_diff(
        y: pd.Series, 
        pvalue: float, 
        regression: str, 
        threshold: Optional[float] = 0.001,
        d_initial: Optional[float] = 0.5,
        step: Optional[float] = 0.01
        ) -> float:
    """
    Find the optimal fractional differencing parameter d that makes the series stationary.
    
    Parameters:
    -----------
    y : pd.Series
        The time series to difference
    pvalue : float
        Target p-value threshold for stationarity test
    regression : str
        Type of regression for ADF test ('c', 'ct', 'ctt', 'n')
    threshold : float
        Threshold for fractional differencing (default 0.01)
    
    Returns:
    --------
    float : Optimal d parameter
    """
    d = d_initial
    max_d = 1.0

    while d <= max_d:

        diffs = frac_diff(y.values, d=d, thres=threshold)
        
        if len(diffs) < 2000:
            print(f"Warning: Length of differenced series is {len(diffs)}, which is less than 2000. "
                  f"Results may be less reliable.")
        
        adf_result = adfuller(diffs, regression=regression)
        current_pvalue = adf_result[1]
        
        if current_pvalue < pvalue:
            return d
        d += step
    
    print(f"Warning: Could not find d that achieves p-value < {pvalue}. Returning d = {max_d}")
    
    return max_d
