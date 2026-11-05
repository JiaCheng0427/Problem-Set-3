"""Difference-in-differences toolkit for ECON 5200 PS3."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

NEVER = 10 ** 9


# Functions from Lab 11
def two_by_two(df, outcome, treated, post):
    g = df.groupby([treated, post])[outcome].mean()
    return (g[1, 1] - g[1, 0]) - (g[0, 1] - g[0, 0])


def att_by_cohort_period(df, time, outcome, cohort):
    out = []
    cohorts = sorted(g for g in df[cohort].unique()
                     if np.isfinite(g) and g < NEVER)

    for g in cohorts:
        tr = df[df[cohort] == g]
        baseline = g - 1

        for t in sorted(df.loc[df[time] >= g, time].unique()):
            co = df[df[cohort] > t]

            if co.empty or not (tr[time] == baseline).any():
                continue

            d_tr = (
                tr.loc[tr[time] == t, outcome].mean()
                - tr.loc[tr[time] == baseline, outcome].mean()
            )
            d_co = (
                co.loc[co[time] == t, outcome].mean()
                - co.loc[co[time] == baseline, outcome].mean()
            )

            if np.isfinite(d_tr) and np.isfinite(d_co):
                out.append({
                    "g": g,
                    "t": t,
                    "e": t - g,
                    "att": d_tr - d_co,
                    "n": int((tr[time] == t).sum())
                })

    return pd.DataFrame(out)


def overall_att(cells):
    return float(np.average(cells["att"], weights=cells["n"]))


def event_study(cells):
    rows = []
    for e, group in cells.groupby("e"):
        rows.append({
            "e": e,
            "att": np.average(group["att"], weights=group["n"])
        })
    return pd.DataFrame(rows)


def clean_att(df):
    cells = att_by_cohort_period(
        df, "year", "y", "adoption_year"
    )
    return cells.rename(columns={
        "g": "cohort",
        "t": "year",
        "n": "n_treated"
    })[["cohort", "year", "att", "n_treated"]]


def cluster_se_ratio(fit_iid, fit_clustered, param):
    return float(fit_clustered.bse[param] / fit_iid.bse[param])


# Clustered inference
def clustered_inference(df, cluster="county"):
    formula = "y ~ treated + C(county) + C(year)"

    iid = smf.ols(formula, data=df).fit()

    clustered = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster]}
    )

    return pd.DataFrame({
        "estimate": [clustered.params["treated"]],
        "iid_se": [iid.bse["treated"]],
        "cluster_se": [clustered.bse["treated"]],
        "p_value": [clustered.pvalues["treated"]]
    })


if __name__ == "__main__":
    print("DiD toolkit ready.")

# Task 4.1: Wild cluster bootstrap
def wild_cluster_bootstrap(
    df: pd.DataFrame,
    cluster: str = "board"
) -> dict:
    """Wild cluster bootstrap with all Rademacher sign vectors."""
    from itertools import product

    groups = sorted(df[cluster].unique())
    G = len(groups)

    assert G == 8, "Expected exactly 8 boards"

    # Unrestricted model and observed cluster-robust t-statistic
    formula = "y ~ treated + C(county) + C(year)"
    full = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster]}
    )

    t_observed = float(full.tvalues["treated"])

    # Restricted model under H0: treatment effect = 0
    restricted = smf.ols(
        "y ~ C(county) + C(year)",
        data=df
    ).fit()

    fitted = restricted.fittedvalues.to_numpy()
    residuals = restricted.resid.to_numpy()

    bootstrap_t = []

    # Enumerate all 2^8 = 256 board-level sign vectors
    for signs in product([-1, 1], repeat=G):
        sign_map = dict(zip(groups, signs))
        weights = df[cluster].map(sign_map).to_numpy()

        boot = df.copy()
        boot["y_star"] = fitted + weights * residuals

        # Unrestricted regression on bootstrap outcome
        fit_star = smf.ols(
            "y_star ~ treated + C(county) + C(year)",
            data=boot
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": boot[cluster]}
        )

        bootstrap_t.append(float(fit_star.tvalues["treated"]))

    bootstrap_t = np.array(bootstrap_t)

    # Two-sided bootstrap p-value
    p_boot = np.mean(
        np.abs(bootstrap_t) >= abs(t_observed) - 1e-12
    )

    assert len(bootstrap_t) == 256

    return {
        "estimate": float(full.params["treated"]),
        "observed_t": t_observed,
        "cluster_p": float(full.pvalues["treated"]),
        "bootstrap_p": float(p_boot),
        "n_vectors": len(bootstrap_t)
    }
