from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import PchipInterpolator
import warnings

try:
    from statsmodels.tsa.stattools import adfuller as sm_adfuller
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

from ml.data.canonical_series import assess_time_series_quality

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
DEFAULT_PLOTS_DIR = DEFAULT_ARTIFACTS_DIR / "time_series_diagnostics"


# ==============================================================================
# 1. Configuration & Data Structures
# ==============================================================================

@dataclass(frozen=True)
class TimeSeriesDiagnosticsConfig:
    """
    Configuration parameters for quantitative time-series diagnostics.
    """
    max_acf_lags: int = 35
    max_pacf_lags: int = 35
    seasonal_period: int = 7
    rolling_window: int = 7
    significance_level: float = 0.05
    min_required_observations: int = 28
    skewness_log_threshold: float = 1.0
    generate_plots: bool = False
    plots_output_dir: Optional[Path | str] = None


@dataclass
class SeriesProfile:
    start_date: Optional[str]
    end_date: Optional[str]
    observations: int
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    zero_demand_days: int
    missing_values: int
    coefficient_of_variation: Optional[float]


@dataclass
class TrendDiagnostics:
    trend_slope: Optional[float]
    trend_intercept: Optional[float]
    r_squared: Optional[float]
    first_half_mean: Optional[float]
    second_half_mean: Optional[float]
    percentage_change: Optional[float]


@dataclass
class StationarityDiagnostics:
    test_statistic: Optional[float]
    p_value: Optional[float]
    used_lag: Optional[int]
    number_of_observations: Optional[int]
    critical_values: Optional[dict[str, float]]
    stationarity_assessment: str  # "stationary_at_5_percent" | "non_stationary_at_5_percent" | "inconclusive"


@dataclass
class DifferencingDiagnostics:
    original_series: StationarityDiagnostics
    first_difference: StationarityDiagnostics
    seasonal_difference_7d: StationarityDiagnostics
    first_difference_changed_stationarity: bool


@dataclass
class AutocorrelationDiagnostics:
    lag_1: Optional[float]
    lag_7: Optional[float]
    lag_14: Optional[float]
    lag_21: Optional[float]
    lag_28: Optional[float]
    acf_values: list[dict[str, Any]]


@dataclass
class PartialAutocorrelationDiagnostics:
    max_evaluated_lag: int
    lag_1: Optional[float]
    lag_7: Optional[float]
    pacf_values: list[dict[str, Any]]


@dataclass
class SeasonalDiagnostics:
    seasonal_period: int
    lag_7_acf: Optional[float]
    lag_14_acf: Optional[float]
    lag_21_acf: Optional[float]
    lag_28_acf: Optional[float]
    seasonal_strength_7d: Optional[float]
    has_monthly_cycle_candidate: bool
    monthly_lag_30_acf: Optional[float]
    has_annual_cycle_candidate: bool
    annual_lag_364_acf: Optional[float]
    notes: list[str]


@dataclass
class RollingStatisticsDiagnostics:
    window: int
    mean_rolling_std: Optional[float]
    max_rolling_std: Optional[float]
    min_rolling_std: Optional[float]
    rolling_std_ratio: Optional[float]
    variance_signal: str  # "stable_variance" | "changing_variance_indicated" | "insufficient_data"


@dataclass
class VarianceTransformationDiagnostics:
    skewness: Optional[float]
    kurtosis: Optional[float]
    has_zero_demand: bool
    recommendation: str  # "consider_log_transform" | "no_transform_indication"
    recommendation_threshold: str


@dataclass
class OutlierDiagnostics:
    q1: Optional[float]
    q3: Optional[float]
    iqr: Optional[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    iqr_outlier_count: int
    iqr_outlier_ratio: Optional[float]
    extreme_max_ratio: Optional[float]
    extreme_min_ratio: Optional[float]


@dataclass
class SufficiencyDiagnostics:
    observations: int
    minimum_required_observations: int
    sufficient_for_diagnostics: bool
    sufficient_for_seasonal_7d: bool
    sufficient_for_annual_seasonal: bool


@dataclass
class DiagnosticConclusion:
    stationarity: str
    trend: str
    weekly_autocorrelation: str
    differencing_signal: str
    variance_signal: str
    observation_sufficiency: str


def _clean_for_json(val: Any) -> Any:
    """Recursively sanitize float NaN and Infinite values for strict JSON serialization."""
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, dict):
        return {k: _clean_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_clean_for_json(v) for v in val]
    if isinstance(val, Path):
        return str(val)
    if hasattr(val, "__dataclass_fields__"):
        return _clean_for_json(asdict(val))
    return val


@dataclass
class TimeSeriesDiagnosticsResult:
    """
    Structured outcome of the time-series diagnostics pipeline.
    """
    config: TimeSeriesDiagnosticsConfig
    profile: SeriesProfile
    trend: TrendDiagnostics
    stationarity: StationarityDiagnostics
    differencing: DifferencingDiagnostics
    autocorrelation: AutocorrelationDiagnostics
    partial_autocorrelation: PartialAutocorrelationDiagnostics
    seasonality: SeasonalDiagnostics
    rolling_statistics: RollingStatisticsDiagnostics
    variance_transformation: VarianceTransformationDiagnostics
    outlier_summary: OutlierDiagnostics
    sufficiency: SufficiencyDiagnostics
    conclusion: DiagnosticConclusion
    plot_artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a clean, JSON-serializable dictionary."""
        raw = {
            "config": asdict(self.config),
            "profile": asdict(self.profile),
            "trend": asdict(self.trend),
            "stationarity": asdict(self.stationarity),
            "differencing": asdict(self.differencing),
            "autocorrelation": asdict(self.autocorrelation),
            "partial_autocorrelation": asdict(self.partial_autocorrelation),
            "seasonality": asdict(self.seasonality),
            "rolling_statistics": asdict(self.rolling_statistics),
            "variance_transformation": asdict(self.variance_transformation),
            "outlier_summary": asdict(self.outlier_summary),
            "sufficiency": asdict(self.sufficiency),
            "conclusion": asdict(self.conclusion),
            "plot_artifacts": self.plot_artifacts,
        }
        return _clean_for_json(raw)

    def save_json(self, path: Path | str) -> Path:
        """Serialize diagnostic results to a JSON file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path


# ==============================================================================
# 2. Statistical Computations (ADF, ACF, PACF, Seasonality)
# ==============================================================================

# MacKinnon (1994, 2010) Dickey-Fuller empirical quantiles and asymptotic t-values for Case 2 (constant, no trend)
_DF_T_GRID = np.array([-5.0, -3.96, -3.43, -3.12, -2.86, -2.57, -2.23, -1.97, -1.57, -1.15, -0.91, -0.44, 0.00, 0.34, 0.60, 1.25, 3.0])
_DF_P_GRID = np.array([0.0001, 0.001, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.999, 0.9999])
_DF_PVALUE_INTERPOLATOR = PchipInterpolator(_DF_T_GRID, _DF_P_GRID)


def approximate_df_pvalue(t_stat: float) -> float:
    """
    Interpolate Dickey-Fuller p-value from MacKinnon empirical distribution.
    Guaranteed bounded strictly in [0.0001, 0.9999].
    """
    if t_stat <= _DF_T_GRID[0]:
        return 0.0001
    if t_stat >= _DF_T_GRID[-1]:
        return 0.9999
    val = float(_DF_PVALUE_INTERPOLATOR(t_stat))
    return float(np.clip(val, 0.0001, 0.9999))


def compute_adf_test(
    series: np.ndarray | Sequence[float],
    max_lag: Optional[int] = None,
    significance_level: float = 0.05,
) -> StationarityDiagnostics:
    """
    Augmented Dickey-Fuller (ADF) unit root test with constant and AIC lag selection.

    Model: Delta y_t = c + gamma * y_{t-1} + sum_{j=1}^p delta_j * Delta y_{t-j} + epsilon_t
    H0: gamma = 0 (unit root / non-stationary)
    H1: gamma < 0 (stationary)
    """
    y = np.asarray(series, dtype=float)
    n = len(y)

    if n < 8:
        return StationarityDiagnostics(
            test_statistic=None,
            p_value=None,
            used_lag=None,
            number_of_observations=n,
            critical_values=None,
            stationarity_assessment="inconclusive",
        )

    # Constant series check
    var_y = float(np.var(y))
    if var_y == 0.0:
        return StationarityDiagnostics(
            test_statistic=0.0,
            p_value=1.0,
            used_lag=0,
            number_of_observations=n,
            critical_values={"1%": -3.43, "5%": -2.86, "10%": -2.57},
            stationarity_assessment="non_stationary_at_5_percent",
        )

    # Primary execution path via statsmodels.tsa.stattools.adfuller
    if _HAS_STATSMODELS:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                sm_res = sm_adfuller(y, maxlag=max_lag, autolag="AIC", regression="c")
            t_stat = float(sm_res[0])
            p_val = float(sm_res[1])
            used_lag = int(sm_res[2])
            nobs = int(sm_res[3])
            crit_vals = {k: round(float(v), 4) for k, v in sm_res[4].items()}
            threshold_cv = crit_vals.get(f"{int(significance_level * 100)}%", crit_vals.get("5%", -2.86))
            if t_stat < threshold_cv or p_val <= significance_level:
                assessment = "stationary_at_5_percent"
            else:
                assessment = "non_stationary_at_5_percent"

            return StationarityDiagnostics(
                test_statistic=round(t_stat, 4),
                p_value=round(p_val, 4),
                used_lag=used_lag,
                number_of_observations=nobs,
                critical_values=crit_vals,
                stationarity_assessment=assessment,
            )
        except Exception:
            pass  # Fallback to custom implementation below

    dy = np.diff(y)
    default_max = int(np.floor((n - 1) ** (1 / 3)))
    if max_lag is None:
        max_lag = default_max
    max_lag = min(max_lag, max(0, (n - 6) // 2))

    best_aic = np.inf
    best_fit: Optional[tuple[float, int, int]] = None

    for p in range(0, max_lag + 1):
        T = len(dy) - p
        if T <= p + 2:
            continue

        Y_reg = dy[p:]
        X_cols = [np.ones(T), y[p : p + T]]
        for j in range(1, p + 1):
            X_cols.append(dy[p - j : p - j + T])
        X = np.column_stack(X_cols)

        beta, residuals, rank, _ = np.linalg.lstsq(X, Y_reg, rcond=None)
        if rank < X.shape[1]:
            continue

        resids = Y_reg - X @ beta
        sse = float(np.sum(resids ** 2))
        k = X.shape[1]
        df_e = T - k
        if df_e <= 0:
            continue

        s2 = sse / df_e
        try:
            cov = s2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            continue

        if se[1] <= 0:
            continue

        t_stat = float(beta[1] / se[1])
        # AIC = T * ln(SSE / T) + 2 * k
        aic = float(T * np.log(max(1e-12, sse / T)) + 2 * k)

        if aic < best_aic:
            best_aic = aic
            best_fit = (t_stat, p, T)

    if best_fit is None:
        return StationarityDiagnostics(
            test_statistic=None,
            p_value=None,
            used_lag=None,
            number_of_observations=n,
            critical_values=None,
            stationarity_assessment="inconclusive",
        )

    t_stat, used_lag, T_obs = best_fit

    # MacKinnon (2010) finite-sample critical values for constant (no trend)
    cv = {
        "1%": round(float(-3.43035 - 6.5393 / T_obs - 16.786 / (T_obs ** 2)), 4),
        "5%": round(float(-2.86154 - 2.8903 / T_obs - 4.234 / (T_obs ** 2)), 4),
        "10%": round(float(-2.56677 - 1.5384 / T_obs - 1.409 / (T_obs ** 2)), 4),
    }

    p_value = approximate_df_pvalue(t_stat)

    # Stationarity decision based on significance level
    threshold_cv = cv.get(f"{int(significance_level * 100)}%", cv["5%"])
    if t_stat < threshold_cv or p_value <= significance_level:
        assessment = "stationary_at_5_percent"
    else:
        assessment = "non_stationary_at_5_percent"

    return StationarityDiagnostics(
        test_statistic=round(t_stat, 4),
        p_value=round(p_value, 4),
        used_lag=used_lag,
        number_of_observations=T_obs,
        critical_values=cv,
        stationarity_assessment=assessment,
    )


def compute_sample_acf(series: np.ndarray, max_lag: int = 35) -> list[dict[str, Any]]:
    """
    Calculate sample autocorrelation coefficients for lags 1 to max_lag.
    """
    n = len(series)
    max_lag = min(max_lag, n - 1)
    if max_lag < 1:
        return []

    y = np.asarray(series, dtype=float)
    mean = float(np.mean(y))
    dev = y - mean
    var = float(np.sum(dev ** 2))
    if var == 0.0:
        return [{"lag": k, "acf": 0.0} for k in range(1, max_lag + 1)]

    # Compute cross-correlation of deviations
    acov = np.correlate(dev, dev, mode="full")
    acov = acov[n - 1 : n + max_lag]
    raw_acf = acov[1:] / var

    return [{"lag": k + 1, "acf": round(float(raw_acf[k]), 6)} for k in range(len(raw_acf))]


def compute_sample_pacf(series: np.ndarray, max_lag: int = 35) -> list[dict[str, Any]]:
    """
    Calculate sample partial autocorrelation coefficients via OLS autoregressions.
    Safely limits maximum evaluated lag to avoid over-parameterization on short series.
    """
    n = len(series)
    safe_max = min(max_lag, max(0, (n // 2) - 1))
    if safe_max < 1:
        return []

    y = np.asarray(series, dtype=float)
    pacf_list: list[dict[str, Any]] = []

    for k in range(1, safe_max + 1):
        Y = y[k:]
        X_cols = [y[k - j : n - j] for j in range(1, k + 1)]
        X_cols.append(np.ones(len(Y)))
        X = np.column_stack(X_cols)

        beta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
        if rank < X.shape[1]:
            break
        pacf_val = float(beta[k - 1])
        # Bound PACF between -1.0 and 1.0
        pacf_val = max(-1.0, min(1.0, pacf_val))
        pacf_list.append({"lag": k, "pacf": round(pacf_val, 6)})

    return pacf_list


def compute_seasonal_strength_7d(y: np.ndarray) -> Optional[float]:
    """
    Calculate Wang-Smith-Hyndman seasonal strength metric for 7-day seasonality:
    F_s = max(0.0, 1.0 - Var(Remainder) / Var(Seasonal + Remainder)).
    """
    n = len(y)
    if n < 14:
        return None

    # Detrend using 7-day rolling mean
    trend = pd.Series(y).rolling(window=7, center=True, min_periods=4).mean().to_numpy()
    detrended = y - trend

    # Day of week component (0 to 6)
    dow = np.arange(n) % 7
    dow_means = np.zeros(7)
    for d in range(7):
        mask = (dow == d) & ~np.isnan(detrended)
        if np.any(mask):
            dow_means[d] = np.mean(detrended[mask])

    # Normalize seasonal component to sum to zero
    dow_means -= np.mean(dow_means)
    seasonal = np.array([dow_means[d % 7] for d in range(n)])

    remainder = detrended - seasonal
    valid = ~np.isnan(remainder) & ~np.isnan(seasonal)
    if np.sum(valid) < 7:
        return None

    var_rem = float(np.var(remainder[valid]))
    var_seas_rem = float(np.var(seasonal[valid] + remainder[valid]))

    if var_seas_rem == 0.0:
        return 0.0

    strength = max(0.0, min(1.0, 1.0 - (var_rem / var_seas_rem)))
    return round(strength, 4)


# ==============================================================================
# 3. Main Diagnostics Engine
# ==============================================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate input contract reusing the P0.2 canonical quality layer.
    Rejects missing calendar dates, is_missing indicators, or empty inputs.
    """
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty or None.")

    work_df = df.copy()

    if "Date" in work_df.columns:
        work_df["Date"] = pd.to_datetime(work_df["Date"], errors="coerce").dt.normalize()
        if work_df.index.name == "Date":
            work_df.index.name = None
    elif isinstance(work_df.index, pd.DatetimeIndex):
        norm_dates = work_df.index.normalize()
        work_df = work_df.copy()
        work_df["Date"] = norm_dates
        work_df.index.name = None
    else:
        raise ValueError("Input data must contain a 'Date' column or DatetimeIndex.")

    if "Quantity" not in work_df.columns:
        raise ValueError("Input data must contain a 'Quantity' column.")

    if work_df["Date"].isna().any():
        raise ValueError("Time series contains invalid or null Date values.")

    # Explicit missing date indicators from P0.2 canonical spine
    if "is_missing" in work_df.columns:
        missing_count = int(work_df["is_missing"].astype(bool).sum())
        if missing_count > 0:
            raise ValueError(
                f"Time series contains {missing_count} missing calendar day indicators (is_missing=True). "
                "Canonical daily continuity is required."
            )

    # Validate canonical daily continuity using P0.2 quality layer when 2 or more rows are present
    if len(work_df) >= 2:
        report = assess_time_series_quality(work_df)
        if not report.is_continuous or report.missing_days > 0:
            raise ValueError(
                f"Time series fails canonical daily continuity: {report.missing_days} missing calendar days detected "
                f"between {report.min_date} and {report.max_date}."
            )

    return work_df.reset_index(drop=True).sort_values("Date").reset_index(drop=True)


def diagnose_time_series(
    df: pd.DataFrame,
    config: Optional[TimeSeriesDiagnosticsConfig] = None,
) -> TimeSeriesDiagnosticsResult:
    """
    Execute comprehensive quantitative time-series diagnostics on a canonical daily series.
    Leaves input DataFrame completely unmutated.
    """
    cfg = config or TimeSeriesDiagnosticsConfig()
    clean_df = validate_input_data(df)

    n = len(clean_df)
    dates = pd.to_datetime(clean_df["Date"])
    y = pd.to_numeric(clean_df["Quantity"], errors="coerce").to_numpy(dtype=float)

    if np.isnan(y).any():
        raise ValueError("Quantity column contains non-numeric or NaN observations.")

    # -------------------------------------------------------------------------
    # 1. Basic Series Profile
    # -------------------------------------------------------------------------
    start_date = dates.iloc[0].strftime("%Y-%m-%d") if n > 0 else None
    end_date = dates.iloc[-1].strftime("%Y-%m-%d") if n > 0 else None
    mean_val = float(np.mean(y)) if n > 0 else None
    median_val = float(np.median(y)) if n > 0 else None
    std_val = float(np.std(y, ddof=1)) if n > 1 else (0.0 if n == 1 else None)
    min_val = float(np.min(y)) if n > 0 else None
    max_val = float(np.max(y)) if n > 0 else None
    zero_demand_days = int(np.sum(y == 0.0))
    missing_values = 0

    cov_val: Optional[float] = None
    if mean_val is not None and mean_val > 0.0 and std_val is not None:
        cov_val = round(float(std_val / mean_val), 4)

    profile = SeriesProfile(
        start_date=start_date,
        end_date=end_date,
        observations=n,
        mean=round(mean_val, 4) if mean_val is not None else None,
        median=round(median_val, 4) if median_val is not None else None,
        std=round(std_val, 4) if std_val is not None else None,
        min=round(min_val, 4) if min_val is not None else None,
        max=round(max_val, 4) if max_val is not None else None,
        zero_demand_days=zero_demand_days,
        missing_values=missing_values,
        coefficient_of_variation=cov_val,
    )

    # -------------------------------------------------------------------------
    # 2. Trend Diagnostics
    # -------------------------------------------------------------------------
    if n >= 2:
        t = np.arange(n, dtype=float)
        t_bar = (n - 1) / 2.0
        var_t = float(np.sum((t - t_bar) ** 2))
        cov_ty = float(np.sum((t - t_bar) * (y - mean_val)))
        slope = cov_ty / var_t if var_t > 0 else 0.0
        intercept = mean_val - slope * t_bar
        y_hat = intercept + slope * t
        ss_tot = float(np.sum((y - mean_val) ** 2))
        ss_res = float(np.sum((y - y_hat) ** 2))
        r_squared = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        mid = n // 2
        first_half_mean = float(np.mean(y[:mid])) if mid > 0 else float(y[0])
        second_half_mean = float(np.mean(y[mid:]))
        pct_change = (
            float((second_half_mean - first_half_mean) / first_half_mean)
            if first_half_mean != 0.0
            else None
        )

        trend = TrendDiagnostics(
            trend_slope=round(slope, 6),
            trend_intercept=round(intercept, 4),
            r_squared=round(r_squared, 4),
            first_half_mean=round(first_half_mean, 4),
            second_half_mean=round(second_half_mean, 4),
            percentage_change=round(pct_change, 4) if pct_change is not None else None,
        )
    else:
        trend = TrendDiagnostics(
            trend_slope=None,
            trend_intercept=None,
            r_squared=None,
            first_half_mean=mean_val,
            second_half_mean=mean_val,
            percentage_change=None,
        )

    # -------------------------------------------------------------------------
    # 3. Stationarity Diagnostics (ADF)
    # -------------------------------------------------------------------------
    stationarity = compute_adf_test(
        series=y,
        significance_level=cfg.significance_level,
    )

    # -------------------------------------------------------------------------
    # 4. Differencing Diagnostics
    # -------------------------------------------------------------------------
    if n >= 2:
        diff_1 = np.diff(y)
        diff_1_adf = compute_adf_test(diff_1, significance_level=cfg.significance_level)
    else:
        diff_1_adf = StationarityDiagnostics(
            test_statistic=None,
            p_value=None,
            used_lag=None,
            number_of_observations=0,
            critical_values=None,
            stationarity_assessment="inconclusive",
        )

    if n > 7:
        diff_7 = y[7:] - y[:-7]
        diff_7_adf = compute_adf_test(diff_7, significance_level=cfg.significance_level)
    else:
        diff_7_adf = StationarityDiagnostics(
            test_statistic=None,
            p_value=None,
            used_lag=None,
            number_of_observations=0,
            critical_values=None,
            stationarity_assessment="inconclusive",
        )

    first_diff_changed = bool(
        stationarity.stationarity_assessment == "non_stationary_at_5_percent"
        and diff_1_adf.stationarity_assessment == "stationary_at_5_percent"
    )

    differencing = DifferencingDiagnostics(
        original_series=stationarity,
        first_difference=diff_1_adf,
        seasonal_difference_7d=diff_7_adf,
        first_difference_changed_stationarity=first_diff_changed,
    )

    # -------------------------------------------------------------------------
    # 5. Autocorrelation Diagnostics (ACF)
    # -------------------------------------------------------------------------
    acf_list = compute_sample_acf(y, max_lag=cfg.max_acf_lags)
    acf_map = {item["lag"]: item["acf"] for item in acf_list}

    autocorr = AutocorrelationDiagnostics(
        lag_1=acf_map.get(1),
        lag_7=acf_map.get(7),
        lag_14=acf_map.get(14),
        lag_21=acf_map.get(21),
        lag_28=acf_map.get(28),
        acf_values=acf_list,
    )

    # -------------------------------------------------------------------------
    # 6. Partial Autocorrelation Diagnostics (PACF)
    # -------------------------------------------------------------------------
    pacf_list = compute_sample_pacf(y, max_lag=cfg.max_pacf_lags)
    pacf_map = {item["lag"]: item["pacf"] for item in pacf_list}

    partial_autocorr = PartialAutocorrelationDiagnostics(
        max_evaluated_lag=len(pacf_list),
        lag_1=pacf_map.get(1),
        lag_7=pacf_map.get(7),
        pacf_values=pacf_list,
    )

    # -------------------------------------------------------------------------
    # 7. Seasonal Diagnostics
    # -------------------------------------------------------------------------
    seasonal_strength = compute_seasonal_strength_7d(y)
    notes: list[str] = []

    # Monthly candidate (lag 30)
    has_monthly = False
    monthly_acf: Optional[float] = None
    if n >= 60:
        monthly_acf_list = compute_sample_acf(y, max_lag=30)
        m_map = {i["lag"]: i["acf"] for i in monthly_acf_list}
        monthly_acf = m_map.get(30)
        if monthly_acf is not None and monthly_acf > 0.25:
            has_monthly = True
            notes.append(f"Candidate monthly cyclical pattern detected at lag 30 (ACF: {monthly_acf:.3f}).")
    else:
        notes.append("Insufficient observations for monthly cycle analysis (requires >= 60 days).")

    # Annual candidate (lag 364 for 52 exact weeks)
    has_annual = False
    annual_acf: Optional[float] = None
    if n >= 730:
        annual_acf_list = compute_sample_acf(y, max_lag=364)
        a_map = {i["lag"]: i["acf"] for i in annual_acf_list}
        annual_acf = a_map.get(364)
        if annual_acf is not None and annual_acf > 0.25:
            has_annual = True
            notes.append(f"Candidate annual cyclical pattern detected at lag 364 (ACF: {annual_acf:.3f}).")
    else:
        notes.append("Insufficient observations for annual cycle analysis (requires >= 730 days).")

    seasonality = SeasonalDiagnostics(
        seasonal_period=cfg.seasonal_period,
        lag_7_acf=acf_map.get(7),
        lag_14_acf=acf_map.get(14),
        lag_21_acf=acf_map.get(21),
        lag_28_acf=acf_map.get(28),
        seasonal_strength_7d=seasonal_strength,
        has_monthly_cycle_candidate=has_monthly,
        monthly_lag_30_acf=monthly_acf,
        has_annual_cycle_candidate=has_annual,
        annual_lag_364_acf=annual_acf,
        notes=notes,
    )

    # -------------------------------------------------------------------------
    # 8. Rolling Statistics
    # -------------------------------------------------------------------------
    if n >= cfg.rolling_window:
        roll_series = pd.Series(y).rolling(window=cfg.rolling_window).std().dropna()
        if len(roll_series) > 0:
            mean_r_std = float(roll_series.mean())
            max_r_std = float(roll_series.max())
            min_r_std = float(roll_series.min())
            r_ratio = float(max_r_std / min_r_std) if min_r_std > 0 else None
            var_sig = "changing_variance_indicated" if (r_ratio is not None and r_ratio > 3.0) else "stable_variance"

            rolling_stats = RollingStatisticsDiagnostics(
                window=cfg.rolling_window,
                mean_rolling_std=round(mean_r_std, 4),
                max_rolling_std=round(max_r_std, 4),
                min_rolling_std=round(min_r_std, 4),
                rolling_std_ratio=round(r_ratio, 4) if r_ratio is not None else None,
                variance_signal=var_sig,
            )
        else:
            rolling_stats = RollingStatisticsDiagnostics(
                window=cfg.rolling_window,
                mean_rolling_std=None,
                max_rolling_std=None,
                min_rolling_std=None,
                rolling_std_ratio=None,
                variance_signal="insufficient_data",
            )
    else:
        rolling_stats = RollingStatisticsDiagnostics(
            window=cfg.rolling_window,
            mean_rolling_std=None,
            max_rolling_std=None,
            min_rolling_std=None,
            rolling_std_ratio=None,
            variance_signal="insufficient_data",
        )

    # -------------------------------------------------------------------------
    # 9. Variance / Transformation Diagnostics
    # -------------------------------------------------------------------------
    var_y = float(np.var(y)) if n > 0 else 0.0
    if n >= 3 and var_y > 0.0:
        skew_val = float(stats.skew(y))
        kurt_val = float(stats.kurtosis(y))
        recommend_transform = "consider_log_transform" if (skew_val > cfg.skewness_log_threshold and min_val >= 0.0) else "no_transform_indication"
    else:
        skew_val = None
        kurt_val = None
        recommend_transform = "no_transform_indication"

    has_zeros = bool(zero_demand_days > 0)
    transform_diag = VarianceTransformationDiagnostics(
        skewness=round(skew_val, 4) if skew_val is not None else None,
        kurtosis=round(kurt_val, 4) if kurt_val is not None else None,
        has_zero_demand=has_zeros,
        recommendation=recommend_transform,
        recommendation_threshold=f"skewness > {cfg.skewness_log_threshold} and min >= 0.0 (use log1p if zeros are present)",
    )

    # -------------------------------------------------------------------------
    # 10. Outlier / Statistical Summary
    # -------------------------------------------------------------------------
    if n >= 4:
        q1 = float(np.percentile(y, 25))
        q3 = float(np.percentile(y, 75))
        iqr = float(q3 - q1)
        lower_bound = float(q1 - 1.5 * iqr)
        upper_bound = float(q3 + 1.5 * iqr)
        outlier_mask = (y < lower_bound) | (y > upper_bound)
        outlier_count = int(np.sum(outlier_mask))
        outlier_ratio = round(float(outlier_count / n), 4)

        ext_max_ratio = round(float(max_val / median_val), 4) if median_val and median_val > 0 else None
        ext_min_ratio = round(float(min_val / median_val), 4) if median_val and median_val > 0 else None

        outlier_diag = OutlierDiagnostics(
            q1=round(q1, 4),
            q3=round(q3, 4),
            iqr=round(iqr, 4),
            lower_bound=round(lower_bound, 4),
            upper_bound=round(upper_bound, 4),
            iqr_outlier_count=outlier_count,
            iqr_outlier_ratio=outlier_ratio,
            extreme_max_ratio=ext_max_ratio,
            extreme_min_ratio=ext_min_ratio,
        )
    else:
        outlier_diag = OutlierDiagnostics(
            q1=min_val,
            q3=max_val,
            iqr=0.0,
            lower_bound=min_val,
            upper_bound=max_val,
            iqr_outlier_count=0,
            iqr_outlier_ratio=0.0,
            extreme_max_ratio=None,
            extreme_min_ratio=None,
        )

    # -------------------------------------------------------------------------
    # 11. Data Sufficiency
    # -------------------------------------------------------------------------
    sufficiency = SufficiencyDiagnostics(
        observations=n,
        minimum_required_observations=cfg.min_required_observations,
        sufficient_for_diagnostics=bool(n >= cfg.min_required_observations),
        sufficient_for_seasonal_7d=bool(n >= 14),
        sufficient_for_annual_seasonal=bool(n >= 730),
    )

    # -------------------------------------------------------------------------
    # 12. Structured Conclusion
    # -------------------------------------------------------------------------
    lag7_val = acf_map.get(7, 0.0)
    if lag7_val >= 0.5:
        weekly_signal = "strong_weekly_autocorrelation"
    elif lag7_val >= 0.25:
        weekly_signal = "moderate_weekly_autocorrelation"
    else:
        weekly_signal = "weak_weekly_autocorrelation"

    if trend.trend_slope is not None and abs(trend.trend_slope) > 0.01 and (trend.r_squared or 0) > 0.1:
        trend_signal = "upward_linear_trend" if trend.trend_slope > 0 else "downward_linear_trend"
    else:
        trend_signal = "flat_trend"

    if first_diff_changed:
        diff_signal = "first_differencing_recommended"
    elif stationarity.stationarity_assessment == "stationary_at_5_percent":
        diff_signal = "no_differencing_indicated"
    else:
        diff_signal = "inconclusive"

    conclusion = DiagnosticConclusion(
        stationarity=stationarity.stationarity_assessment,
        trend=trend_signal,
        weekly_autocorrelation=weekly_signal,
        differencing_signal=diff_signal,
        variance_signal=rolling_stats.variance_signal,
        observation_sufficiency=(
            "sufficient_for_classical_modelling"
            if sufficiency.sufficient_for_diagnostics
            else "insufficient_sample_size"
        ),
    )

    # -------------------------------------------------------------------------
    # 13. Optional Plot Artifacts
    # -------------------------------------------------------------------------
    plot_files: list[str] = []
    if cfg.generate_plots:
        target_dir = Path(cfg.plots_output_dir or DEFAULT_PLOTS_DIR)
        plot_files = generate_diagnostic_plots(clean_df, acf_list, pacf_list, target_dir)

    return TimeSeriesDiagnosticsResult(
        config=cfg,
        profile=profile,
        trend=trend,
        stationarity=stationarity,
        differencing=differencing,
        autocorrelation=autocorr,
        partial_autocorrelation=partial_autocorr,
        seasonality=seasonality,
        rolling_statistics=rolling_stats,
        variance_transformation=transform_diag,
        outlier_summary=outlier_diag,
        sufficiency=sufficiency,
        conclusion=conclusion,
        plot_artifacts=plot_files,
    )


# ==============================================================================
# 4. Optional Plotting Helpers
# ==============================================================================

def generate_diagnostic_plots(
    df: pd.DataFrame,
    acf_list: list[dict[str, Any]],
    pacf_list: list[dict[str, Any]],
    output_dir: Path | str,
) -> list[str]:
    """
    Generate diagnostic visual artifacts using matplotlib in headless mode.
    Does not impact core mathematical computations.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    dates = pd.to_datetime(df["Date"])
    y = df["Quantity"].to_numpy(dtype=float)

    # 1. Original Demand and Rolling Mean/Std
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, y, label="Daily Demand", color="#1f77b4", alpha=0.6, linewidth=1)
    if len(y) >= 7:
        r_mean = pd.Series(y).rolling(7).mean()
        r_std = pd.Series(y).rolling(7).std()
        ax.plot(dates, r_mean, label="7-Day Rolling Mean", color="#ff7f0e", linewidth=2)
        ax.fill_between(dates, r_mean - r_std, r_mean + r_std, color="#ff7f0e", alpha=0.2, label="±1 Rolling Std")
    ax.set_title("Time Series Demand and Rolling Statistics (7-day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p1 = out_dir / "original_and_rolling.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    generated.append(str(p1))

    # 2. ACF and PACF Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    if acf_list:
        lags_a = [x["lag"] for x in acf_list]
        vals_a = [x["acf"] for x in acf_list]
        ax1.stem(lags_a, vals_a, basefmt="k-")
        ax1.axhline(0, color="gray", linestyle="--")
        ax1.axhline(1.96 / np.sqrt(len(y)), color="red", linestyle=":", alpha=0.7)
        ax1.axhline(-1.96 / np.sqrt(len(y)), color="red", linestyle=":", alpha=0.7)
        ax1.set_title("Autocorrelation Function (ACF)")
        ax1.set_xlabel("Lag (Days)")
        ax1.set_ylabel("Autocorrelation")
        ax1.grid(True, linestyle="--", alpha=0.4)

    if pacf_list:
        lags_p = [x["lag"] for x in pacf_list]
        vals_p = [x["pacf"] for x in pacf_list]
        ax2.stem(lags_p, vals_p, basefmt="k-")
        ax2.axhline(0, color="gray", linestyle="--")
        ax2.axhline(1.96 / np.sqrt(len(y)), color="red", linestyle=":", alpha=0.7)
        ax2.axhline(-1.96 / np.sqrt(len(y)), color="red", linestyle=":", alpha=0.7)
        ax2.set_title("Partial Autocorrelation Function (PACF)")
        ax2.set_xlabel("Lag (Days)")
        ax2.set_ylabel("Partial Autocorrelation")
        ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    p2 = out_dir / "acf_pacf.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    generated.append(str(p2))

    # 3. Differenced Series
    if len(y) > 7:
        fig, (ax_d1, ax_d7) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax_d1.plot(dates[1:], np.diff(y), color="#2ca02c", linewidth=1)
        ax_d1.set_title("First Differenced Series (Delta y_t)")
        ax_d1.grid(True, linestyle="--", alpha=0.5)

        ax_d7.plot(dates[7:], y[7:] - y[:-7], color="#9467bd", linewidth=1)
        ax_d7.set_title("Seasonal Differenced Series (Lag 7: Delta_7 y_t)")
        ax_d7.set_xlabel("Date")
        ax_d7.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        p3 = out_dir / "differenced_series.png"
        fig.savefig(p3, dpi=120)
        plt.close(fig)
        generated.append(str(p3))

    return generated


# ==============================================================================
# 5. CLI Execution / Verification Entrypoint
# ==============================================================================

def main():
    """CLI runner executing diagnostics on the production dataset and persisting JSON."""
    from ml.data.data_loader import aggregate_daily_data, load_processed_data

    data_path = PROJECT_ROOT / "data" / "processed_daily_forecasting_features.csv"
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        return

    print("Loading processed sales dataset...")
    df = load_processed_data(data_path)
    daily = aggregate_daily_data(df)

    print(f"Running diagnostics on {len(daily)} continuous daily observations...")
    config = TimeSeriesDiagnosticsConfig(
        max_acf_lags=35,
        max_pacf_lags=35,
        generate_plots=True,
        plots_output_dir=DEFAULT_PLOTS_DIR,
    )

    result = diagnose_time_series(daily, config=config)
    json_path = DEFAULT_ARTIFACTS_DIR / "time_series_diagnostics.json"
    result.save_json(json_path)

    print("\n" + "=" * 70)
    print("TIME SERIES DIAGNOSTICS SUMMARY")
    print("=" * 70)
    p = result.profile
    print(f"Date Range:       {p.start_date} -> {p.end_date} ({p.observations} days)")
    print(f"Demand Profile:   Mean={p.mean}, Median={p.median}, Std={p.std}, Min={p.min}, Max={p.max}")
    print(f"Zero Demand Days: {p.zero_demand_days} | Coeff of Variation: {p.coefficient_of_variation}")

    s = result.stationarity
    print(f"\nStationarity (ADF): {s.stationarity_assessment} (t={s.test_statistic}, p={s.p_value}, lag={s.used_lag})")
    print(f"Critical Values:    1%={s.critical_values.get('1%')}, 5%={s.critical_values.get('5%')}, 10%={s.critical_values.get('10%')}")

    d = result.differencing
    print(f"Differencing:       First-diff stationarity changed: {d.first_difference_changed_stationarity}")

    t = result.trend
    print(f"Trend:              Slope={t.trend_slope}, R^2={t.r_squared}, Half1->Half2 Change={t.percentage_change}")

    seas = result.seasonality
    print(f"Weekly Seasonality: Lag-7 ACF={seas.lag_7_acf}, Lag-14={seas.lag_14_acf}, Strength={seas.seasonal_strength_7d}")
    print(f"Longer Cycles:      Monthly Candidate={seas.has_monthly_cycle_candidate}, Annual Candidate={seas.has_annual_cycle_candidate}")

    c = result.conclusion
    print(f"\nConclusions:        {c.stationarity} | {c.trend} | {c.weekly_autocorrelation} | {c.differencing_signal}")
    print(f"Saved Diagnostics:  {json_path}")
    if result.plot_artifacts:
        print(f"Saved Plots ({len(result.plot_artifacts)}): {DEFAULT_PLOTS_DIR}")


if __name__ == "__main__":
    main()
