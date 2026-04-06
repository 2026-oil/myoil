"""Kalman Filter and LPPL preprocessing for oil price forecasting research."""

__all__ = ["KalmanFilter1D", "LPPLDetector", "EMGMMRegime"]

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class KalmanFilter1D:
    """1D Kalman Filter for time series denoising and trend extraction.

    State-space model:
        x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
        z_t = x_t + v_t,      v_t ~ N(0, R)

    Args:
        observation_noise: R parameter (measurement noise variance).
        process_noise: Q parameter (process noise variance).
        initial_state: Initial state estimate.
        initial_covariance: Initial state covariance.
    """

    def __init__(
        self,
        observation_noise: float = 1.0,
        process_noise: float = 0.01,
        initial_state: Optional[float] = None,
        initial_covariance: float = 1.0,
    ):
        self.R = observation_noise
        self.Q = process_noise
        self.x_init = initial_state
        self.P_init = initial_covariance

    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter over observations.

        Returns:
            filtered_states: Smoothed state estimates.
            filtered_covariances: State covariance at each step.
        """
        n = len(observations)
        x = np.zeros(n)
        P = np.zeros(n)

        x[0] = observations[0] if self.x_init is None else self.x_init
        P[0] = self.P_init

        for t in range(1, n):
            x_pred = x[t - 1]
            P_pred = P[t - 1] + self.Q

            K = P_pred / (P_pred + self.R)
            x[t] = x_pred + K * (observations[t] - x_pred)
            P[t] = (1 - K) * P_pred

        return x, P

    def smooth(self, observations: np.ndarray) -> np.ndarray:
        """Run Kalman filter + Rauch-Tung-Striebel smoother.

        Returns:
            smoothed_states: Two-pass smoothed state estimates.
        """
        x_filt, P_filt = self.filter(observations)
        n = len(observations)
        x_smooth = x_filt.copy()

        for t in range(n - 2, -1, -1):
            C = P_filt[t] / (P_filt[t] + self.Q)
            x_smooth[t] = x_filt[t] + C * (x_smooth[t + 1] - x_filt[t])

        return x_smooth

    def denoise_series(self, series: pd.Series) -> pd.Series:
        """Apply Kalman smoothing to denoise a pandas Series."""
        values = series.values.astype(float)
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return series

        values_clean = values[mask]
        smoothed = self.smooth(values_clean)

        result = values.copy()
        result[mask] = smoothed
        return pd.Series(result, index=series.index, name=series.name)


class LPPLDetector:
    """Log-Periodic Power Law singularity detection.

    Detects critical points (bubbles/crashes) using the LPPL model:
        ln(p(t)) = A + B*(tc-t)^m + C*(tc-t)^m*cos(omega*ln(tc-t)+phi)

    Args:
        t_start: Minimum tc-t distance for fitting.
        t_end: Maximum tc-t distance for fitting.
        m_range: Range for exponent m (typically 0.01-0.99).
        omega_range: Range for angular frequency omega (typically 6-13).
    """

    def __init__(
        self,
        t_start: int = 5,
        t_end: int = 60,
        m_range: Tuple[float, float] = (0.01, 0.99),
        omega_range: Tuple[float, float] = (6.0, 13.0),
    ):
        self.t_start = t_start
        self.t_end = t_end
        self.m_min, self.m_max = m_range
        self.omega_min, self.omega_max = omega_range

    def _lppl_features(
        self, t: np.ndarray, tc: float, m: float, omega: float
    ) -> np.ndarray:
        dt = tc - t
        dt = np.maximum(dt, 1e-10)
        dt_m = dt**m
        cos_term = np.cos(omega * np.log(dt))
        sin_term = np.sin(omega * np.log(dt))
        return np.column_stack(
            [np.ones_like(dt), dt_m, dt_m * cos_term, dt_m * sin_term]
        )

    def _fit_linear(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if len(residuals) == 0:
                return coeffs, np.inf
            return coeffs, residuals[0] / max(len(y), 1)
        except np.linalg.LinAlgError:
            return np.zeros(X.shape[1]), np.inf

    def fit(self, series: pd.Series) -> dict:
        """Fit LPPL model and detect critical time tc.

        Returns:
            Dictionary with best tc, m, omega, A, B, C, phi, and fit quality metrics.
        """
        values = series.dropna().values.astype(float)
        n = len(values)
        if n < self.t_start + 5:
            return {"valid": False, "reason": "insufficient_data"}

        y = np.log(np.maximum(values, 1e-10))
        t = np.arange(n, dtype=float)

        best_residual = np.inf
        best_params = {}

        for tc_offset in range(self.t_start, min(self.t_end, n)):
            tc = t[-1] + tc_offset
            for m in np.linspace(self.m_min, self.m_max, 8):
                for omega in np.linspace(self.omega_min, self.omega_max, 8):
                    X = self._lppl_features(t, tc, m, omega)
                    coeffs, residual = self._fit_linear(X, y)

                    if residual < best_residual:
                        A, B, C_cos, C_sin = coeffs
                        C = math.sqrt(C_cos**2 + C_sin**2)
                        phi = math.atan2(-C_sin, C_cos)
                        best_residual = residual
                        best_params = {
                            "valid": True,
                            "tc": tc,
                            "m": m,
                            "omega": omega,
                            "A": A,
                            "B": B,
                            "C": C,
                            "phi": phi,
                            "residual": residual,
                            "tc_offset": tc_offset,
                        }

        return best_params

    def compute_signal(self, series: pd.Series, window: int = 30) -> pd.Series:
        """Compute rolling LPPL bubble signal.

        Returns:
            Series of LPPL residual (lower = stronger bubble signal).
        """
        n = len(series)
        signals = pd.Series(np.nan, index=series.index, name="lppl_signal")

        for i in range(window, n):
            window_data = series.iloc[max(0, i - window) : i]
            result = self.fit(window_data)
            if result.get("valid"):
                signals.iloc[i] = result["residual"]

        return signals


class EMGMMRegime:
    """Expectation-Maximization Gaussian Mixture Model for regime detection.

    Identifies market regimes (normal/volatile/crisis) using EM-fitted GMM
    on return distributions. Improves upon NEC's built-in GMM by providing
    multi-dimensional regime features.

    Args:
        n_components: Number of Gaussian components.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = None
        self.covars_ = None
        self.weights_ = None

    def _initialize(self, X: np.ndarray):
        n, d = X.shape
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)[: self.n_components]
        self.means_ = X[idx]
        self.covars_ = np.array([np.eye(d) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        log_resp = np.zeros((n, self.n_components))

        for k in range(self.n_components):
            diff = X - self.means_[k]
            log_det = np.log(np.linalg.det(self.covars_[k]) + 1e-10)
            inv_cov = np.linalg.inv(self.covars_[k] + 1e-10 * np.eye(X.shape[1]))
            mahalanobis = np.sum(diff @ inv_cov * diff, axis=1)
            log_resp[:, k] = (
                np.log(self.weights_[k] + 1e-10)
                - 0.5 * X.shape[1] * np.log(2 * np.pi)
                - 0.5 * log_det
                - 0.5 * mahalanobis
            )

        log_sum = np.max(log_resp, axis=1, keepdims=True)
        log_resp -= log_sum
        resp = np.exp(log_resp)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum = np.maximum(resp_sum, 1e-10)
        resp /= resp_sum
        return resp

    def _m_step(self, X: np.ndarray, resp: np.ndarray):
        n = X.shape[0]
        self.weights_ = resp.sum(axis=0) / n
        self.weights_ = np.maximum(self.weights_, 1e-10)
        self.weights_ /= self.weights_.sum()

        for k in range(self.n_components):
            resp_k = resp[:, k : k + 1]
            Nk = resp_k.sum()
            self.means_[k] = (resp_k * X).sum(axis=0) / Nk
            diff = X - self.means_[k]
            self.covars_[k] = (resp_k * diff).T @ diff / Nk + 1e-6 * np.eye(X.shape[1])

    def fit(self, X: np.ndarray) -> "EMGMMRegime":
        """Fit GMM using EM algorithm."""
        self._initialize(X)
        prev_log_likelihood = -np.inf

        for _ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)

            log_likelihood = np.sum(
                np.log(np.maximum((resp * np.log(resp + 1e-10)).sum(), 1e-10))
            )
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

        return self

    def predict_regimes(self, X: np.ndarray) -> np.ndarray:
        """Assign each observation to its most likely regime."""
        resp = self._e_step(X)
        return resp.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return regime probabilities for each observation."""
        return self._e_step(X)

    def fit_transform(self, series: pd.Series, n_features: int = 5) -> pd.DataFrame:
        """Create regime features from a time series.

        Returns:
            DataFrame with regime probability columns.
        """
        values = series.dropna().values.astype(float)
        if len(values) < self.n_components * 3:
            return pd.DataFrame(index=series.index)

        returns = np.diff(np.log(np.maximum(values, 1e-10)))
        vol = np.abs(returns)

        X = np.column_stack([returns, vol])
        for lag in range(1, n_features - 1):
            X = np.column_stack([X, np.roll(returns, lag)])
        X = X[n_features - 1 :]

        self.fit(X)
        proba = self.predict_proba(X)

        regime_cols = {f"em_regime_{k}": proba[:, k] for k in range(self.n_components)}
        regime_df = pd.DataFrame(regime_cols)

        pad_len = len(series) - len(regime_df)
        regime_df.index = series.index[pad_len:]
        return regime_df.reindex(series.index)
