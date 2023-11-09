#| export
import warnings
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

import seaborn as sns
from dataclasses import dataclass, field

from joblib import Parallel, delayed
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import kpss


LOG_SAFETY = 1e-8

SSR_CHI2TEST = 'ssr_chi2test'
LAG_ORDER = 1

DEFAULT_TEST = SSR_CHI2TEST

DEFAULT_GRANGER_CAUSALITY_TESTS = dict(
    test      =  DEFAULT_TEST,
    lag_order =  LAG_ORDER,
    max_lag   = (LAG_ORDER, ) 
)


LOG2_FOLD = 'log2_fold'
SIGNED_CORRELATION= 'signed_correlation'
STANDARD_SCALER = 'standard_scaler'

KNOWN_TRANSFORMS = [LOG2_FOLD, SIGNED_CORRELATION, STANDARD_SCALER]

MOCK_GENES = 'wasf colq gpr1 chrm3 lmod2 tek kank3 oca2 taz map4k1'.split()


def kpss_test(series, print: bool = False, **kwargs) -> None:
    # KPSS Null hypothesis: there is a no unit root, meaning series is stationary
    statistic, p_value, n_lags, critical_values = kpss(series, **kwargs)

    # Format Output    
    result_str = ''
    result_str += f'KPSS Statistic: {statistic}\n'
    result_str += f'p-value: {p_value}\n'
    result_str += f'num lags: {n_lags}\n'
    result_str += f'Critial Values:\n'    
    for key, value in critical_values.items():
        result_str += f'\t{key}:{value}\n'
    result_str += f'Result: The series is {"not " if p_value < 0.05 else ""}stationary'

    if print:
        print(result_str)
    return

#| exporti
def _prep_args_for_granger_causality_tests(
    test: str = DEFAULT_TEST, 
    lag_order: int = LAG_ORDER,
    max_lag: tuple = (LAG_ORDER, ),    
):
    '''
    Parameters
    ----------
    test : str, default='ssr_chi2test'
        the kind of statistical test to use
    
    lag_order : int, default=1
        how long to lag
    
    max_lag : tuple,         if`None` coerced to `(1, )`
    
    Returns
    -------
    test : str
        `'ssr_chi2test'` if `test` was `None`
        
    lag_order : int
        `1` if `lag_order` was `None`

    max_lag : tuple
        `(1, )` if `max_lag` was `None`

    Notes
    -----
    lag_order
        `lag_order = 1` # becuase we got this value before. We are not suppose to add 1 to it
    
    max_lag
        `max_lag = (lag_order, )`
    '''
    if test is None:
        test = SSR_CHI2TEST

    if lag_order is None:
        lag_order = LAG_ORDER

    if max_lag is None:
        max_lag = (lag_order,)
    
    return test, lag_order, max_lag

def _prep_vars_for_granger_causality_tests(
    df: pd.DataFrame, 
    x_vars: list = None, 
    y_vars: list = None,
    do_safety_check: bool = True     
):
    '''
    Ensures `df` is safe for Granger Causality tests and that 
    `x_vars` and `y_vars` are not `None`.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame where rows are the response variable (genes), and 
        columns are predictors (expression).
    
    x_vars : list,         Asubset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.


    y_vars : list,         Asubset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.

    do_safety_check : bool, default=True
        Whether or not to check if there is a constant column in `df`.

        
    Returns
    -------
    df : pd.DataFrame
        input `df` possibly with small random noise added to prevent constant columns

    x_vars : list
        input `x_vars` or `df.index.values` if `x_vars` was `None`

    y_vars : list
        input `y_vars` or `df.index.values` if `y_vars` was `None`

    Notes
    -----
    If `x_var = ['A', B']` and `y_var = ['C', 'D]', then when calling Granger's 
    Causality the matrix returned has shape:
    ```
        | C | D |
        ---------
    | A |   |   |
    -------------
    | B |   |   |

    ```
    '''
    if do_safety_check:
        df = add_non_const(df)

    if x_vars is None:
        x_vars = df.index.values

    if y_vars is None:
        y_vars = df.index.values
    
    return df, x_vars, y_vars

def grangers_causation_matrix(
    df: pd.DataFrame, 
    x_vars: list = None, 
    y_vars: list = None, 
    test: str = DEFAULT_TEST, 
    lag_order: int = LAG_ORDER,
    max_lag: tuple = (LAG_ORDER, ),    
    n_jobs: int = -1
) -> pd.DataFrame:
    '''
    Computes Granger Causality

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame where rows are the response variable (genes), and 
        columns are predictors (expression).
    
    x_vars : list,         Asubset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.

    y_vars : list,         Asubset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.

    test : str, default='ssr_chi2test'
        the kind of statistical test to use
    
    lag_order : int, default=1
        how long to lag
    
    max_lag : tuple,         if`None` coerced to `(1, )`

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

def get_pval_from_granger_causality_tests(
    df: pd.DataFrame,
    test: str = DEFAULT_TEST, 
    lag_order: int = LAG_ORDER,
    max_lag: tuple = (LAG_ORDER, ), 
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
    
    max_lag : tuple,         if`None` coerced to `(1, )`

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


def shift_trajectories(
    df: pd.DataFrame, 
    shift: int = 10, 
    copy: bool = True
) -> pd.DataFrame:
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


def calculate_granger_causation(
    df: pd.DataFrame, 
    x_vars: list = None, 
    y_vars: list = None, 
    shift: int = 10,
    test: str = DEFAULT_TEST, 
    lag_order: int = LAG_ORDER,
    max_lag: tuple = (LAG_ORDER, ),    
    n_jobs: int = -1
) -> pd.DataFrame:     
    '''
    Computes Granger Causality

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame where rows are the response variable (genes), and 
        columns are predictors (expression).
    
    x_vars : list,         Asubset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.


    y_vars : list,         Asubset of response variable (genes) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.

    shift : int, default=10
        number to shift `df`'s values by

    test : str, default='ssr_chi2test'
        the kind of statistical test to use
    
    lag_order : int, default=1
        how long to lag
    
    max_lag : tuple,         if`None` coerced to `(1, )`

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

class GrangerCausality(BaseEstimator):
    '''
    Computes Granger Causality
    Check Granger Causality of all possible combinations of the Time series.

    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame where rows are the response variable (features), and 
        columns are predictors (expression).
    
    x_vars : list,         Asubset of response variable (features) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.


    y_vars : list,         Asubset of response variable (features) to compute granger's causality test with. 
        If not provided, defaults to `df.index.values` i.e. all rows in `df`.

    shift : int, default=10
        number to shift `df`'s values by

    test : str, default='ssr_chi2test'
        the kind of statistical test to use
    
    lag_order : int, default=1
        how long to lag
    
    max_lag : tuple,         if`None` coerced to `(1, )`

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

    See Also
    --------
    GrangerCausality._prep_args_for_granger_causality_tests :
        validates input arguments

    GrangerCausality._prep_vars_for_granger_causality_tests :
        validates input variables
    '''
    _LOG2: str = LOG2_FOLD
    _SCOR: str = SIGNED_CORRELATION
    _STDN: str = STANDARD_SCALER
    KNOWN_TRANSFORMS: str = KNOWN_TRANSFORMS


    def __init__(
        self,
        x_vars: list = None, 
        y_vars: list = None,
        shift: int = 10,
        test: str = DEFAULT_TEST,
        lag_order: int = LAG_ORDER,
        max_lag: tuple = (LAG_ORDER, ),
        n_jobs: int = -1
    ):        
        
    

        test, lag_order, max_lag = _prep_args_for_granger_causality_tests(test, lag_order, max_lag)

        self.x_vars = x_vars
        self.y_vars = y_vars
        self.shift = shift
        self.test = test
        self.max_lag = max_lag
        self.lag_order = lag_order
        self.n_jobs = n_jobs

        # original DataFrame
        self.df_org = None

        # results DataFrame
        self.df_res = None
        
        # scalar if transforming
        self.scaler = None

    
    def process_fit_params(self, **fit_params) -> dict:
        '''
        fit_params : dict, 
        NOTE you can use either of the following. The first is more explicit, the second is more concise.
        ```
            {
                "apply" : str, default=None | "log2_fold" | "signed_correlation" | "standard_scaler"
            }

            {
                "use_cached" : bool, default=False
                "log2_fold" : bool, default=False
                "signed_correlation" : bool, default=False
                "standard_scaler" : bool, default=False
            }
        ```
        if `apply` is not `None`, then the following apply:
        - `log2_fold` : will apply log2 fold change to `df_res`
        - `signed_correlation` : will apply signed correlation to `df_res`
        - `standard_scaler` : will apply standard scaler to `df_org` and `df_res`
        '''
        apply = fit_params.get('apply', None)
        if apply is not None and apply not in self.KNOWN_TRANSFORMS:
            if isinstance(apply, str):
                for app in apply.split(' '):
                    if app not in self.KNOWN_TRANSFORMS:
                        continue
                    fit_params[app] = True
        return fit_params



    def fit(self, X):
        '''
        Update internal parameters prior to computation in `transform`
        '''
        test, lag_order, max_lag = _prep_args_for_granger_causality_tests(self.test, self.lag_order, self.max_lag)
        X, x_vars, y_vars = _prep_vars_for_granger_causality_tests(X, self.x_vars, self.y_vars, do_safety_check=True)
        self.test = test
        self.lag_order = lag_order
        self.max_lag = max_lag
        self.x_vars = x_vars
        self.y_vars = y_vars
        return self
    
    def transform(self, X: pd.DataFrame, y = None, **fit_params):   
        # Save input DataFrame
        self.df_org = X.copy()
        fit_params = self.process_fit_params(**fit_params)

        # Apply standard scaler
        stdn = fit_params.get(self._STDN, False)
        if stdn:
            self.apply_standard_scaler(X)

        # Use saved results
        use_cached = fit_params.get('use_cached', False)
        if use_cached and self.df_res is not None:
            df_res = self.df_res.copy()

        else:
            df_res = calculate_granger_causation(
                X, x_vars=self.x_vars, y_vars=self.y_vars,
                shift=self.shift, test=self.test, 
                lag_order=self.lag_order, max_lag=self.max_lag,    
                n_jobs=self.n_jobs
            )

        # Store results
        self.df_res = df_res.copy()

        log2 = fit_params.get(self._LOG2, False)
        if log2:
            df_res = apply_log2_fold(df_res)

        scor = fit_params.get(self._SCOR, False)
        if scor:
            df_res = apply_signed_correlation(X, df_res)        

        return df_res
    
    def fit_transform(self, X: pd.DataFrame, y = None, **fit_params):
        df_res = self.fit(X).transform(X, y = y, **fit_params)
        return df_res
    
    def apply_log2_fold(self, df: pd.DataFrame) -> pd.DataFrame:
        return apply_log2_fold(df)

    def apply_signed_correlation(self, df_tseries: pd.DataFrame, df_granger: pd.DataFrame) -> pd.DataFrame:
        return apply_signed_correlation(df_tseries, df_granger)
    
    def apply_standard_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        df_res, scaler = apply_standard_scaler(df, return_scaler=True)
        self.scaler = scaler
        return df_res
    
    def invert_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise ValueError
        return self.scaler.inverse_transform(df)
    
    def plot_df_org(self, show_all_yticks: bool = True, **kwargs):
        '''
        Returns
        -------
        fig : matplotlib.pyplot.Figure
        ax : matplotlib.pyplot.Axis
        ClusterMap : seaborn.ClusterMap
        '''
        options = dict(cmap='inferno', robust=True, col_cluster=False, yticklabels=show_all_yticks)
        options.update(kwargs)
        cstrmp = sns.clustermap(self.df_org, **options)
        return cstrmp
    
    def plot_df_res(self, show_all_yticks: bool = True, **kwargs):
        '''
        Returns
        -------
        fig : matplotlib.pyplot.Figure
        ax : matplotlib.pyplot.Axis
        ClusterMap : seaborn.ClusterMap
        '''           
        options = dict(yticklabels=show_all_yticks)
        options.update(kwargs)
        cstrmp = sns.clustermap(self.df_res, **options)
        return cstrmp


def _add_suffix(arr, suffix:str):
    return list(map(lambda e: f'{e}_{suffix}', arr))

def make_stable(
    df: pd.DataFrame, 
    copy: bool = True,
    add_perm: bool = True,
    use_rand: bool = False,
    no_const: bool = True
) -> pd.DataFrame:    
    df_safe = (df.copy() if copy else df)

    if add_perm:
        df_safe = df_safe + LOG_SAFETY
    
    # NOTE: to ensure that no column in df is constant
    if use_rand or no_const:
        rng_mat = np.random.randn(*df.shape) * LOG_SAFETY
        df_safe = df_safe + rng_mat

    return df_safe
    
def add_stability(df):
    return make_stable(df, copy=False, add_perm=True)

def add_non_const(df):
    return make_stable(df, copy=False, add_perm=False, use_rand=True)
    
def apply_log2_fold(df_granger: pd.DataFrame):
    return -np.log2(df_granger + (2 ** -20))    

def apply_signed_correlation(df_tseries: pd.DataFrame, df_granger: pd.DataFrame) -> pd.DataFrame:
    var_names = df_tseries.index.values
    signed = -np.sign(df_tseries.T.corr()) * np.log(add_stability(df_granger.values))
    df_res = pd.DataFrame(signed, index=var_names, columns=var_names)
    return df_res

def apply_standard_scaler(df: pd.DataFrame, return_scaler: bool = False) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler = scaler.fit(df.T)
    df_res = pd.DataFrame(scaler.transform(df.T)).T
    df_res.index = df.index
    if return_scaler:
        df_res, scaler
    return df_res




def make_dummy_genes(n: int):
    return [f'gene_{i}' for i in range(1, n+1)]

def make_dummy_cells(n: int):
    return [f'cell{i}' for i in range(1, n+1)]

def make_mock_genes_x_tbins(
    genes: list = MOCK_GENES,
    tbins: int = 100,
) -> pd.DataFrame:
    
    if isinstance(genes, int):
        genes = make_dummy_genes(genes)        
    n_genes = len(genes)

    forward = lambda : np.sort(np.random.randint(0, n_genes, (tbins)))
    reverse = lambda : forward()[::-1]

    timeseries = [forward() if i % 2 == 0 else reverse() for i in range(n_genes)]
    df_trj = pd.DataFrame(timeseries, index=genes)
    return df_trj