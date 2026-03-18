# Requirements: numpy, pandas, scipy, pycwt
# pip install numpy pandas scipy pycwt

import numpy as np
import pandas as pd
import pycwt as cwt
from scipy import stats

def compute_wavelet_power(series, dt=1.0, dj=1/12, s0=None, J=None, wavelet='morlet'):
    """Compute CWT and return wave, scales, periods, power (|wave|^2). series must be 1D array-like."""
    x = np.asarray(series)
    n = x.size
    # default s0 & J typical choices; s0 = 2*dt (smallest scale)
    if s0 is None:
        s0 = 2 * dt
    if J is None:
        # choose J so scales cover up to ~n*dt / 2
        J = int(np.floor(np.log2((n*dt)/s0) / dj))
    mother = cwt.Morlet(6.0)  # omega0 = 6 is standard
    wave, scales, freqs, coi, fft, fftfreqs = cwt.cwt(x, dt, dj, s0, J, mother)
    power = (np.abs(wave))**2
    periods = 1.0 / freqs  # but pycwt's freqs are cycles per unit time -> periods in time units
    # NOTE: with monthly data and dt=1.0, periods are in months (if dt=1 month)
    return dict(wave=wave,scales=scales,periods=periods,power=power,coi=coi)

def extract_prominent_periods(power, periods, period_min_months, period_max_months, pct=95):
    """
    power: 2D array shape (n_scales, n_times)
    periods: 1D array length n_scales (period for each scale) in months
    returns: 1D array of period values (months) for all times/scales within band whose power >= scale-specific pctth percentile
    """
    # identify scale indices within period band
    band_idx = np.where((periods >= period_min_months) & (periods <= period_max_months))[0]
    picked_periods = []
    for i in band_idx:
        thr = np.percentile(power[i, :], pct)
        # find times where power >= thr (uppermost pct)
        times_idx = np.where(power[i, :] >= thr)[0]
        # append the period (in months) for each such timepoint
        if times_idx.size > 0:
            picked_periods.extend([periods[i]] * times_idx.size)
    return np.array(picked_periods)  # may be empty if no extreme power

def bootstrap_tost(vec_ref, vec_cmp, margin_months, n_boot=9999, seed=None):
    """
    Bootstrapped TOST using bootstrap distribution of (mean_ref - mean_cmp).
    Returns tuple (p_lower, p_upper, mean_diff, ci)
    p_lower = proportion of boot_diffs <= -margin  (one-sided test H0: diff <= -margin)
    p_upper = proportion of boot_diffs >= +margin  (one-sided test H0: diff >= +margin)
    Equivalence if both p_lower < 0.05 and p_upper < 0.05
    """
    rng = np.random.default_rng(seed)
    n1 = len(vec_ref)
    n2 = len(vec_cmp)
    if n1 == 0 or n2 == 0:
        return dict(p_lower=np.nan, p_upper=np.nan, mean_diff=np.nan, ci=(np.nan,np.nan), boot_diffs=None)
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        s1 = rng.choice(vec_ref, size=n1, replace=True)
        s2 = rng.choice(vec_cmp, size=n2, replace=True)
        boot_diffs[b] = np.nanmean(s1) - np.nanmean(s2)
    mean_diff = np.nanmean(vec_ref) - np.nanmean(vec_cmp)
    # p-values defined as proportions in the tails that support the null; for rejection we want small p
    p_lower = np.mean(boot_diffs <= -margin_months)
    p_upper = np.mean(boot_diffs >= margin_months)
    # 95% bootstrap CI:
    ci = (np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5))
    return dict(p_lower=p_lower, p_upper=p_upper, mean_diff=mean_diff, ci=ci, boot_diffs=boot_diffs)

def tost_with_growing_margin(vec_ref, vec_cmp, start_margin=3, step=3, max_margin=60, n_boot=9999, alpha=0.05, seed=None):
    """
    Try increasing equivalence margins (in months) starting at start_margin (months),
    stepping by step months, until both TOST one-sided p-values < alpha, or reach max_margin.
    Returns a list of dicts with results for each margin tried and the margin where equivalence first declared (or None).
    """
    results = []
    equiv_margin = None
    for m in range(start_margin, max_margin+1, step):
        res = bootstrap_tost(vec_ref, vec_cmp, margin_months=m, n_boot=n_boot, seed=seed)
        res['margin'] = m
        res['equivalent'] = (res['p_lower'] < alpha) and (res['p_upper'] < alpha)
        results.append(res)
        if res['equivalent']:
            equiv_margin = m
            break
    return dict(results=results, equiv_margin=equiv_margin)

# ----------------------------
# Example usage (assuming monthly pandas.Series with DatetimeIndex)
# ----------------------------
def run_duffy_style_test(polynya_series, predictor_series, start_years=10, end_years=44,
                         dt_months=1, dj=1/12, s0=None, J=None, pct=95,
                         n_boot=9999, seed=42):
    """
    Main wrapper:
    - polynya_series, predictor_series: pandas.Series (monthly) with no detrending (use raw)
    - returns the extracted period-vectors and TOST table of margins
    """
    # Ensure same length and no NaNs for simplicity (you may want to align / pad instead)
    polynya = polynya_series.dropna()
    predictor = predictor_series.dropna()
    # Use dt=1.0 for monthly (1 month timestep). The pycwt periods will be in months.
    dt = dt_months
    w_pol = compute_wavelet_power(polynya.values, dt=dt, dj=dj, s0=s0, J=J)
    w_pr  = compute_wavelet_power(predictor.values, dt=dt, dj=dj, s0=s0, J=J)
    # Convert year bounds to months
    pmin = start_years * 12
    pmax = end_years * 12
    vec_pol = extract_prominent_periods(w_pol['power'], w_pol['periods'], pmin, pmax, pct=pct)
    vec_pr  = extract_prominent_periods(w_pr['power'], w_pr['periods'], pmin, pmax, pct=pct)
    tost_res = tost_with_growing_margin(vec_pol, vec_pr, start_margin=1, step=1, max_margin=240, n_boot=n_boot, seed=seed)
    return dict(polynya_periods_months=vec_pol, predictor_periods_months=vec_pr, tost=tost_res)



# ----------------------------
# Example call (replace polynya_s, predictor_s with your pandas Series)
# ----------------------------
# out = run_duffy_style_test_old(polynya_s, predictor_s, n_boot=9999)
# print("equivalence margin found (months):", out['tost']['equiv_margin'])
# for rec in out['tost']['results'][:10]:
#     print(rec['margin'], rec['p_lower'], rec['p_upper'], rec['equivalent'])
