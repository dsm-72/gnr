# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_fn.ipynb.

# %% auto 0
__all__ = ['shift_trajectories', 'get_pval_from_granger_causality_tests', 'grangers_causation_matrix',
           'calculate_granger_causation']

# %% ../nbs/02_fn.ipynb 3
import warnings
import pandas as pd, numpy as np

from joblib import Parallel, delayed
from typing import Optional

from statsmodels.tsa.stattools import grangercausalitytests

# %% ../nbs/02_fn.ipynb 4
from iza.types import SeriesLike, DataFrame

# %% ../nbs/02_fn.ipynb 5
from gnr.static import (
    LOG_SAFETY, SSR_CHI2TEST, LAG_ORDER,
    DEFAULT_TEST, DEFAULT_GRANGER_CAUSALITY_TESTS
)

from gnr.utils import (
    _add_suffix, make_stable, add_stability, add_non_const, 
    apply_log2_fold, apply_signed_correlation, apply_standard_scaler,
    _prep_args_for_granger_causality_tests,
    _prep_vars_for_granger_causality_tests
)

# %% ../nbs/02_fn.ipynb 7
def shift_trajectories(
    df: DataFrame, 
    shift: Optional[int] = 10, 
    copy: Optional[bool] = True
) -> DataFrame:
    '''
    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame where rows are the response variable (genes), and 
        columns are predictors (expression).
    
    shift : int, default=10
        number to shift `df`'s values by

    copy : bool, default=True
        Whether or not to copy input `df`

    Returns
    -------
    df_trj : pd.DataFrame
        Pandas DataFrame of `df - df_shift`
    '''
    df_trj = (df.copy() if copy else df)
    df_trj = df_trj.T[::shift]
    df_trj = df_trj - df_trj.shift(1)
    df_trj = df_trj.dropna()
    return df_trj


# %% ../nbs/02_fn.ipynb 8
def get_pval_from_granger_causality_tests(
    df: DataFrame,
    test: Optional[str] = DEFAULT_TEST, 
    lag_order: Optional[int] = LAG_ORDER,
    max_lag: Optional[tuple] = (LAG_ORDER, ), 
) -> float:
    '''
    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame where rows are the response variable (genes), and 
        columns are predictors (expression).

    test : str, default='ssr_chi2test'
        the kind of statistical test to use
    
    lag_order : int, default=1
        how long to lag
    
    max_lag : tuple, optional
        if `None` coerced to `(1, )`

    Returns
    -------
    min_p_value : float
        minimum p-value of Granger Causality Tests

    '''
    test, lag_order, max_lag = _prep_args_for_granger_causality_tests(test, lag_order, max_lag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_result = grangercausalitytests(df, maxlag=max_lag, verbose=False)
    p_values = [test_result[i][0][test][1] for i in max_lag]
    min_p_value = np.min(p_values)
    return min_p_value

# %% ../nbs/02_fn.ipynb 10
def grangers_causation_matrix(
    df: DataFrame, 
    x_vars: Optional[SeriesLike] = None, 
    y_vars: Optional[SeriesLike] = None, 
    test: Optional[str] = DEFAULT_TEST, 
    lag_order: Optional[int] = LAG_ORDER,
    max_lag: Optional[tuple] = (LAG_ORDER, ),    
    n_jobs: Optional[int] = -1
) -> DataFrame:
    '''
    Computes Granger Causality

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame where rows are the response variable (genes), and 
        columns are predictors (expression).
    
    x_vars : SeriesLike, optional
        A subset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.

    y_vars : SeriesLike, optional
        A subset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.

    test : str, default='ssr_chi2test'
        the kind of statistical test to use
    
    lag_order : int, default=1
        how long to lag
    
    max_lag : tuple, optional
        if `None` coerced to `(1, )`

    n_jobs : int, default=-1
        number of cpu threads to use during calculation
        
    Returns
    -------
    df_res : pd.DataFrame    
        Pandas DataFrame with shape `(len(x_vars), len(y_vars))` containing the
        minimum p-value from Granger's Causation Tests

    Notes
    -----
    If `x_var = ['A', B']` and `y_var = ['C', 'D]', the Granger's Causality matrix we
    return has shape:
    ```
           | C_y | D_y |
           -------------
    | A_x |      |     |
    --------------------
    | B_x |      |     |

    ```
    '''
    n_jobs = -1 if n_jobs is None else n_jobs
    
    df, x_vars, y_vars       = _prep_vars_for_granger_causality_tests(df, x_vars, y_vars)
    test, lag_order, max_lag = _prep_args_for_granger_causality_tests(test, lag_order, max_lag)

    delayed_funcs = [
        delayed(get_pval_from_granger_causality_tests)(
            df[[x_var, y_var]], test, lag_order, max_lag
        ) for y_var in y_vars for x_var in x_vars
    ]

    parallel_out = Parallel(n_jobs=n_jobs)(delayed_funcs)
    
    df_res = pd.DataFrame(
        np.array(parallel_out).reshape((len(x_vars), len(y_vars))),
        index=x_vars, columns=y_vars,        
    )
    df_res.index = _add_suffix(x_vars, 'x')
    df_res.columns = _add_suffix(y_vars, 'y')
    return df_res

# %% ../nbs/02_fn.ipynb 11
def calculate_granger_causation(
    df: DataFrame, 
    x_vars: Optional[SeriesLike] = None, 
    y_vars: Optional[SeriesLike] = None, 
    shift: Optional[int] = 10,
    test: Optional[str] = DEFAULT_TEST, 
    lag_order: Optional[int] = LAG_ORDER,
    max_lag: Optional[tuple] = (LAG_ORDER, ),    
    n_jobs: Optional[int] = -1
) -> DataFrame:     
    '''
    Computes Granger Causality

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame where rows are the response variable (genes), and 
        columns are predictors (expression).
    
    x_vars : SeriesLike, optional
        A subset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.


    y_vars : SeriesLike, optional
        A subset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.

    shift : int, default=10
        number to shift `df`'s values by

    test : str, default='ssr_chi2test'
        the kind of statistical test to use
    
    lag_order : int, default=1
        how long to lag
    
    max_lag : tuple, optional
        if `None` coerced to `(1, )`

    n_jobs : int, default=-1
        number of cpu threads to use during calculation
        
    Returns
    -------
    df_res : pd.DataFrame    
        Pandas DataFrame with shape `(len(x_vars), len(y_vars))` containing the
        minimum p-value from Granger's Causation Tests

    Notes
    -----
    If `x_var = ['A', B']` and `y_var = ['C', 'D]', the Granger's Causality matrix we
    return has shape:
    ```
           | C_y | D_y |
           -------------
    | A_x |      |     |
    --------------------
    | B_x |      |     |

    ```
    '''
    n_jobs = -1 if n_jobs is None else n_jobs
    
    df, x_vars, y_vars       = _prep_vars_for_granger_causality_tests(df, x_vars, y_vars)    
    test, lag_order, max_lag = _prep_args_for_granger_causality_tests(test, lag_order, max_lag)
    
    df_trj = shift_trajectories(df, shift)
        
    df_res = grangers_causation_matrix(
        df_trj, x_vars=x_vars, y_vars=y_vars,
        test=test, lag_order=lag_order, max_lag=max_lag,
        n_jobs=n_jobs
    )

    return df_res
