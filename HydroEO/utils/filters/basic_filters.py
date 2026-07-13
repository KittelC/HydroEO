"""simple filters that can be applied to sat timeseries objects"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.svm import SVR

from datetime import datetime

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0088


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two points given in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _along_track_distance_km(lats, lons):
    """
    Cumulative along-track distance (km) for points already ordered along
    the ground track (see _order_along_track for how that order is chosen).
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if len(lats) < 2:
        return np.zeros_like(lats)
    step = _haversine_km(lats[:-1], lons[:-1], lats[1:], lons[1:])
    return np.concatenate([[0.0], np.cumsum(step)])


def _order_along_track(lats, lons):
    """
    Return the index order that best approximates along-track order for a
    single pass, given only lat/lon (no reliable sub-daily timestamp to sort
    by). Sorts by whichever of latitude/longitude spans the larger range
    within this group, since a single pass is normally close to monotonic in
    that coordinate (avoids failing near-equatorial, near-east-west passes
    where latitude barely changes).
    """
    lat_range = np.ptp(lats) if len(lats) else 0
    lon_range = np.ptp(lons) if len(lons) else 0
    sort_vals = lats if lat_range >= lon_range else lons
    return np.argsort(sort_vals)


def _resolve_pass_groups(df, date_key, pass_key=None, platform_key=None, orbit_key=None):
    """
    Resolve a grouping key identifying individual physical passes/crossings,
    generic across missions (this function knows nothing about which real
    column names any given mission uses -- that mapping happens when the
    Timeseries object is constructed).

    Resolved per ROW, not per dataframe -- this matters as soon as sources
    from different missions are concatenated together (see
    Timeseries.concat), since one mission's rows may have a valid pass_key
    while another's are NaN. Each row falls back independently through:
      1. `pass_key`, if non-null for that row.
      2. (date_key, platform_key, orbit_key), if platform_key and orbit_key
         are both non-null for that row.
      3. date_key alone.

    Returns a pandas Series of group labels (strings), same index as df.
    """
    n = len(df)
    group = pd.Series(pd.NA, index=df.index, dtype=object)

    if pass_key and pass_key in df.columns:
        has_pass = df[pass_key].notna()
        group.loc[has_pass] = "pass:" + df.loc[has_pass, pass_key].astype(str)

    remaining = group.isna()
    if (
        remaining.any()
        and platform_key
        and orbit_key
        and platform_key in df.columns
        and orbit_key in df.columns
    ):
        has_po = remaining & df[platform_key].notna() & df[orbit_key].notna()
        group.loc[has_po] = (
            "po:"
            + df.loc[has_po, date_key].astype(str)
            + "_"
            + df.loc[has_po, platform_key].astype(str)
            + "_"
            + df.loc[has_po, orbit_key].astype(str)
        )

    remaining = group.isna()
    if remaining.any():
        group.loc[remaining] = "date:" + df.loc[remaining, date_key].astype(str)

    return group


def elevation_filter(timeseries, height_range):
    min_height, max_height = height_range
    timeseries.df = timeseries.df.loc[timeseries.df[timeseries.height_key] > min_height]
    timeseries.df = timeseries.df.loc[timeseries.df[timeseries.height_key] < max_height]

    timeseries.df = timeseries.df.reset_index(drop=True)
    timeseries.df.sort_values(by=timeseries.date_key)


def mad_filter(timeseries, threshold=5):
    # calculate support to make sure we can make a statistical decision, otheriwse leave
    if len(timeseries.df) >= 30:
        # calculate standard deviation and remove obvious outliers
        med = np.median(timeseries.df.height)
        abs_dev = np.abs(timeseries.df.height - med)
        mad = np.median(abs_dev)
        timeseries.df = timeseries.df.loc[abs_dev < threshold * mad]

        timeseries.df = timeseries.df.reset_index(drop=True)
        timeseries.df = timeseries.df.sort_values(by=timeseries.date_key)

    return timeseries


def daily_mad_error(timeseries, reg_weight=0.1, reg_default=0.5, error_key="ADM"):
    # sort inplace the timeseries object
    timeseries.df = timeseries.df.sort_values(by=timeseries.date_key).reset_index(
        drop=True
    )

    # exctract for calculations
    df = timeseries.df.copy()
    date_key = timeseries.date_key
    height_key = timeseries.height_key

    # Assign and use a consistent day key for grouping and mapping.
    # Mixing full timestamps and date-only keys can produce NaN mapped values.
    day_key = df[date_key].dt.floor("D")

    # Group by day and get median/count.
    medval = df.groupby(day_key).median(numeric_only=True)[height_key]
    day_grp = df.groupby(day_key).count()[height_key]

    # Add regularization factor to avoid giving advantage to median:
    reg = reg_weight / day_grp
    reg[day_grp == 1] = reg_default

    # map the median val and regularization to the dates they belong to
    med_map = medval.to_dict()
    reg_map = reg.to_dict()
    df["med"] = day_key.map(med_map)
    df["reg"] = day_key.map(reg_map)

    # calculate error and enforce a positive finite lower bound for Kalman stability
    error = np.abs(df[height_key] - df["med"]) + df["reg"]
    error = (
        error.replace([np.inf, -np.inf], np.nan).fillna(reg_default).clip(lower=1e-6)
    )

    # add error to timeseries
    timeseries.df[error_key] = error.values

    return timeseries


def daily_mean_merge(timeseries):
    # aggregates values that occur on the same day to a mean value

    dct = {
        "number": "mean",
        "object": lambda col: col.mode() if col.nunique() == 1 else np.nan,
    }

    groupby_cols = [timeseries.date_key]
    dct = {
        k: v
        for i in [
            {
                col: agg
                for col in timeseries.df.select_dtypes(tp).columns.difference(
                    groupby_cols
                )
            }
            for tp, agg in dct.items()
        ]
        for k, v in i.items()
    }
    timeseries.df = timeseries.df.groupby(groupby_cols).agg(
        **{k: (k, v) for k, v in dct.items()}
    )

    timeseries.df[timeseries.date_key] = pd.to_datetime(timeseries.df.index)
    timeseries.df = timeseries.df.reset_index(drop=True)
    timeseries.df.sort_values(by=timeseries.date_key)

    return timeseries


def hampel(timeseries, k=7, t0=3):
    """
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    """
    vals = timeseries.df.loc[:, timeseries.height_key]

    # Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(k).median()
    difference = np.abs(rolling_median - vals)
    median_abs_deviation = difference.rolling(k).median()
    threshold = t0 * L * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan

    timeseries.df[timeseries.height_key] = vals.values

    return timeseries


def rolling_median(timeseries, window=7):
    timeseries.df[timeseries.height_key] = (
        timeseries.df[timeseries.height_key].rolling(window).median()
    )

    return timeseries


def _run_svr_linear(heights, err=0.1, epsilon=0.1, max_iter=5000):
    """
    Linear Support Vector Regression outlier filter.

    Fits a free-slope linear SVR through heights along-track and flags
    points that deviate from that fitted line by more than `err` as
    outliers. The fitted slope is used only to decide which points are
    outliers here -- it is not removed from the retained heights (that is
    a separate, later step).

    Parameters
    ----------
    heights : array
        Heights to be fit, already ordered along-track.
    err : Float, optional
        Allowed deviation from the linear fit. The default is 0.1.
    epsilon : Float, optional
        "Epsilon in the epsilon-SVR model.
        It specifies the epsilon-tube within which no penalty is associated in the training loss
        function with points predicted within a distance epsilon from the actual value."
        (from https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
        The default is .1.
    max_iter : int, optional
        Hard cap on the underlying solver's iterations. sklearn's SVR
        defaults to max_iter=-1 (no cap at all) -- on a pathological or
        unexpectedly large/ill-conditioned group, that can make a single
        fit call run for a very long time with no visible progress,
        indistinguishable from a genuine hang. Default 5000; if this is
        hit, a warning is logged and the (not fully converged, but usually
        still reasonable) fit is used rather than blocking indefinitely.

    Returns
    -------
    array
        Integer positions (0-based, within `heights`) of the points that
        survive the filter.
    """
    heights = np.asarray(heights, dtype=float)

    # sequential x axis along track
    x = np.arange(0, len(heights))
    x = np.vstack(
        [x, np.ones(len(x))]
    ).T  # Extend x data to contain another row vector of 1s
    y = heights

    # make SVR kernel and fit with confidence bounds
    svr_rbf = SVR(kernel="linear", epsilon=epsilon, max_iter=max_iter)
    rbf = svr_rbf.fit(x, y)
    if getattr(rbf, "n_iter_", 0) is not None and np.any(
        np.asarray(rbf.n_iter_) >= max_iter
    ):
        logger.warning(
            "_run_svr_linear: SVR hit max_iter=%d without full convergence "
            "on a group of %d points -- result used anyway, but check "
            "whether this group is unexpectedly large (pass-grouping "
            "issue) or has pathological/duplicate values.",
            max_iter, len(heights),
        )
    uconf = rbf.predict(x) + err
    lconf = rbf.predict(x) - err

    # return the positions of the retained points (not the values), so the
    # caller can keep every column for those rows, not just height
    return np.where((y >= lconf) & (y <= uconf))[0]


def svr_linear(timeseries, err=0.1, epsilon=0.1, max_iter=5000, warn_group_size=500):
    """
    Along-track linear-SVR outlier filter, grouped by physical pass (see
    _resolve_pass_groups) rather than calendar day, and preserving every
    column of the surviving rows (not just date/height).

    Parameters
    ----------
    timeseries : Timeseries
    err : float, optional
        Allowed deviation from the local linear fit (m). Default 0.1.
    epsilon : float, optional
        SVR epsilon-tube. Default 0.1.
    max_iter : int, optional
        Passed through to the underlying SVR fit; see _run_svr_linear.
    warn_group_size : int, optional
        Log a warning up front for any single pass/group larger than this,
        before attempting to fit it -- SVR's cost scales badly with group
        size, so an unexpectedly large group (e.g. a pass-grouping fallback
        issue lumping many points together) is worth surfacing immediately
        rather than only being discovered via a slow or capped fit.
    """
    df = timeseries.df.copy()
    date_key = timeseries.date_key
    height_key = timeseries.height_key

    groups = _resolve_pass_groups(
        df,
        date_key,
        getattr(timeseries, "pass_key", None),
        getattr(timeseries, "platform_key", None),
        getattr(timeseries, "orbit_key", None),
    )

    group_items = df.groupby(groups).groups.items()
    group_sizes = {g: len(idx) for g, idx in group_items}
    oversized = {g: n for g, n in group_sizes.items() if n > warn_group_size}
    if oversized:
        logger.warning(
            "svr_linear: %d group(s) exceed warn_group_size=%d (largest: "
            "%s with %d points) -- SVR fit cost scales badly with group "
            "size; if this is unexpected, check pass_key/platform_key/"
            "orbit_key resolution for this timeseries.",
            len(oversized), warn_group_size,
            max(oversized, key=oversized.get), max(oversized.values()),
        )

    keep_idx = []
    for _, idx in group_items:
        idx = np.asarray(idx)
        if len(idx) < 2:
            # nothing to compare a single point against; keep it as-is
            keep_idx.extend(idx)
            continue
        heights = df.loc[idx, height_key].values
        local_keep = _run_svr_linear(heights, err=err, epsilon=epsilon, max_iter=max_iter)
        keep_idx.extend(idx[local_keep])

    timeseries.df = df.loc[np.sort(np.asarray(keep_idx))].reset_index(drop=True)

    return timeseries


"""Kalman filter for estimating state of reservoir from noisy timeseries"""


def _update(obs, xk, cov_xx, height="height", error="ADM", n=1):
    """
    Update function for Kalman filter

    Parameters
    ----------
    obs : array
        DESCRIPTION.
    xk : float
        Previous prediction, x.
    cov_xx : float
        Uncertainty of x.
    height : string, optional
        DESCRIPTION. The default is 'height'.
    error : string, optional
        DESCRIPTION. The default is 'ADM'.
    n : Int, optional
        Grid size - only relevant for large lakes e.g. Not yet implemented. The default is 1.

    Returns
    -------
    xk_plus : Float
        Kalman filter updated value.
    cov_xx_plus : Float
        Covariance of the updated value.

    """
    obs = obs.copy()
    obs = obs.replace([np.inf, -np.inf], np.nan)
    obs = obs.dropna(subset=[height, error])

    # No valid observations for this epoch: keep predicted state unchanged.
    if obs.empty:
        return xk, cov_xx

    obs_list = obs[height].values
    m = len(obs_list)
    lk = np.array(obs_list).reshape(m, n)
    Ak = np.ones((m, n))

    # compute Kalman matrix (weights of the innovation)
    slk = np.maximum((obs[error].values) ** 2, 1e-12)
    cov_lk = np.diag(slk)
    inv = Ak * cov_xx * np.transpose(Ak) + cov_lk
    if not np.all(np.isfinite(inv)):
        return xk, cov_xx
    inv = inv + np.eye(m) * 1e-8
    Kk = cov_xx * np.dot(np.transpose(Ak), np.linalg.pinv(inv))

    # update x
    xk_plus = xk + np.dot(Kk, lk - Ak * xk)

    # update sigma
    cov_xx_plus = cov_xx * (1 - np.dot(Kk, Ak))

    return xk_plus, cov_xx_plus


def _pred(xk, cov_xx, n=1, system_noise=0, system_noise_unc=0.05):
    """
    Prediction function for Kalman filter

    Parameters
    ----------
    xk : float
        Updated prediction, x.
    cov_xx : float
        Uncertainty of x.
    n : Int, optional
        Grid size - only relevant for large lakes e.g. Not yet implemented. The default is 1.
    system_noise : float, optional
        System noise. The default is 0.
    system_noise_unc : float, optional
        Uncertainty of the system noise. The default is 0.05.

    Returns
    -------
    xk_plus : Float
        Kalman filter predicted value.
    cov_xx_plus : Float
        Covariance of the predicted value.

    """
    # Setup dynamic model with no deterministic contribution
    thetak = np.identity(n)
    hatk = np.identity(n)
    Qk = np.array(system_noise_unc).reshape(n, n)
    qk = np.array(system_noise).reshape(n, 1)
    noise = hatk.dot(Qk).dot(hatk.T)

    # pred x
    xk_next = thetak * xk + hatk * qk
    # predict covariance matrix
    cov_xx_next = thetak.dot(cov_xx).dot(thetak.T) + noise

    return xk_next, cov_xx_next


def kalman(timeseries, error_key="ADM", n=1):
    """
    Run Kalman filter

    Parameters
    ----------
    time_series : DataFrame
        Outlier filtered dataframe to be used as input for Kalman filter.
        Must contain height and error columns
    height : string, optional
        Name of height column. The default is 'height_OCOG'.
    error : string, optional
        Name of error column. The default is 'ADM'.
    n : Int, optional
        Grid size - only relevant for large lakes e.g. Not yet implemented. The default is 1.

    Returns
    -------
    xk_plus : array
        Kalman filter updated values of WSE.
    cov_xx_plus : array
        Covariance of the updated value.

    """

    df = timeseries.df.copy()
    date_key = timeseries.date_key
    height_key = timeseries.height_key

    dates = sorted(df[date_key].unique())
    xks = np.ones((n, len(df[date_key].unique()))) * np.nan
    cov_xxs = np.ones((n, n, len(df[date_key].unique()))) * np.nan

    # Initialize prediction and uncertainty matrices
    obs = df.loc[df[date_key] == dates[0]]
    lk = obs[height_key].values
    slk = (obs[error_key].values) ** 2

    # First prediction = value with smallest ADM and identity matrix of size n, n
    xks[:, 0] = lk[np.argmin(slk)]
    cov_xxs[:, :, 0] = np.identity(n)

    # Observation model
    for i, d in enumerate(dates):
        # Update
        obs = df.loc[df[date_key] == d]
        xk_plus, cov_xx_plus = _update(
            obs, xks[:, i], cov_xxs[:, :, i], height=height_key, error=error_key, n=n
        )

        # Predict
        xk1, cov_xx1 = _pred(
            xk_plus, cov_xx_plus, n=n, system_noise=0, system_noise_unc=0.05
        )
        xks[:, i] = xk_plus
        cov_xxs[:, :, i] = cov_xx_plus
        try:
            xks[:, i + 1] = xk1
            cov_xxs[:, :, i + 1] = cov_xx1
        except IndexError:
            pass

    # return a new dataframe with the filtered data
    df_kalman = pd.DataFrame(
        {date_key: dates, height_key: xks[0], "cov": cov_xxs[0][0]}
    )
    return df_kalman


def _run_svr_rbf(dates, heights, err=1.0, rbf_c=10000, gamma=0.0000438, epsilon=0.1,
                  max_iter=-1, max_fit_points=None):
    """
    Radial Base Function outlier filtering post-Kalman filter.
    This is run at virtual station level

    Parameters
    ----------
    dates : array
        Sorted dates of altimetry observations.
    heights : array
        Predicted (and updated) water surface elevation.
    err : Float, optional
        Observation uncertainty. The default is 1 m. As used in DAHITI for rivers
        (0.1 m can be used for lakes)
    C : TYPE, optional
        Regularization parameter in sklearn.svm.RBF. The default here is 10000 (as in DAHITI)
    gamma : TYPE, optional
        Kernel coefficient for ‘rbf’ in sklearn.svm.RBF
        Alternative values include float, 'scale' or 'auto'
        The default here is 0.0000438 (as in DAHITI)
    epsilon : TYPE, optional
        "Epsilon in the epsilon-SVR model.
        It specifies the epsilon-tube within which no penalty is associated in the training loss
        function with points predicted within a distance epsilon from the actual value."
        (from https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
        The default is .1.
    max_iter : int, optional
        Passed to the underlying SVR fit. UNLIKE _run_svr_linear, this
        defaults to -1 (unbounded) -- confirmed empirically that capping
        this specific configuration (RBF kernel + rbf_c=10000, very little
        regularization) produces non-monotonic, unreliable output quality:
        e.g. max_iter=5000 kept 10/5760 points, 50000 kept 827/5760, and
        100000 kept only 30/5760 -- only fully unbounded (23s on that real
        test case) gave the correct 2154/5760. Do not cap this without
        checking output row counts, not just wall-clock time. Use
        max_fit_points instead for a speedup that doesn't have this risk.
    max_fit_points : int, optional
        If set and there are more than this many points, the SVR is fit on
        an evenly-subsampled subset of this size (not truncated -- spread
        across the whole series), then applied via .predict() to ALL
        original points for the outlier decision. This reduces the cost
        driver (problem size going into the O(n^2)-ish solver) directly,
        rather than truncating iterations on the full problem -- much
        safer than capping max_iter for this kernel/C combination, since
        the underlying trend being fit is smooth and a representative
        subsample captures it about as well as the full set. None (default)
        disables this and fits on every point, as before.

    Returns
    -------
    rbf_filter : array
        Filter for Kalman filter predictions of WSE
    """
    # Day since start
    x = np.array(
        [0]
        + list(
            np.cumsum(
                pd.to_datetime(dates)[1:] - pd.to_datetime(dates)[:-1]
            ).days.astype("float32")
        )
    )
    # Kalman heights
    y = heights
    x = np.vstack([x, np.ones(len(x))]).T

    if max_fit_points and len(x) > max_fit_points:
        fit_sel = np.linspace(0, len(x) - 1, max_fit_points).round().astype(int)
        fit_x, fit_y = x[fit_sel], y[fit_sel]
    else:
        fit_x, fit_y = x, y

    svr_rbf = SVR(kernel="rbf", C=rbf_c, gamma=gamma, epsilon=epsilon, max_iter=max_iter)
    rbf = svr_rbf.fit(fit_x, fit_y)
    if getattr(rbf, "n_iter_", 0) is not None and np.any(
        np.asarray(rbf.n_iter_) >= max_iter > 0
    ):
        logger.warning(
            "_run_svr_rbf: SVR hit max_iter=%d without full convergence on "
            "%d fit points -- result used anyway.", max_iter, len(fit_x),
        )
    corr = rbf.predict(x)

    uconf = corr + err
    lconf = corr - err

    rbf_filter = np.where((y >= lconf) & (y <= uconf))

    return rbf_filter


def _year_fraction(dt):
    start = datetime(dt.year, 1, 1).toordinal()
    year_length = datetime(dt.year + 1, 1, 1).toordinal() - start
    return dt.year + float(dt.toordinal() - start) / year_length


def svr_radial(timeseries, err=1.0, rbf_c=10000, gamma=0.0000438, epsilon=0.1,
                max_iter=-1, max_fit_points=None):
    """
    Radial-basis post-Kalman outlier filter, run at virtual station level.

    Parameters
    ----------
    timeseries : Timeseries
        Kalman-filtered timeseries to be filtered.
    err : float, optional
        Observation uncertainty / half-width of the confidence band (m).
        DAHITI uses about 1 m for rivers and 0.1 m for lakes -- since
        flow.py runs separate river vs. reservoir flows, pass the
        appropriate value explicitly from there rather than relying on
        this default. The default here (1.0) is the more conservative
        (river-like) choice.
    rbf_c : float, optional
        Regularization parameter C of the RBF SVR. The default is 10000,
        matching DAHITI (lower values make the fit more regularized 
        / less able to follow the data than intended).
    gamma : float, optional
        RBF kernel coefficient. The default is 0.0000438, as in DAHITI.
    epsilon : float, optional
        Epsilon-tube of the SVR (see sklearn.svm.SVR). The default is 0.1.
    max_iter : int, optional
        Passed through to the underlying SVR fit; see _run_svr_rbf for why
        this defaults to -1 (unbounded) rather than being capped.
    max_fit_points : int, optional
        Passed through to _run_svr_rbf -- the safe speedup for this stage
        (fit on an evenly-subsampled subset, apply to all points), rather
        than capping max_iter.

    Returns
    -------
    vs : DataFrame
        Filtered (date, height) virtual station timeseries.
    """
    df = timeseries.df.copy()
    date_key = timeseries.date_key
    height_key = timeseries.height_key

    rbf_filter = _run_svr_rbf(
        df[date_key].values,
        df[height_key].values,
        err=err,
        rbf_c=rbf_c,
        gamma=gamma,
        epsilon=epsilon,
        max_iter=max_iter,
        max_fit_points=max_fit_points,
    )

    nb_obs = df.groupby(date_key).count().reset_index()

    nb_obs["nb_obs"] = nb_obs[date_key]

    # Create new dataframe:
    vs = pd.DataFrame(
        {
            date_key: df[date_key].values[rbf_filter],
            height_key: df[height_key].values[rbf_filter],
        }
    )

    return vs