"""
Module for the official evaluation.
"""

import sys
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import classification_report, f1_score

from ..logging import get_logger, log
from .utils import (
    get_team_and_run_name,
    load_submission,
    submission_has_correct_format,
    submission_is_evaluable,
)

_logger = get_logger(__name__)


def submission_is_valid(
    submission_file: Path,
    ground_truth_file: Path,
) -> bool:
    run_df = load_submission(submission_file)
    test_df = load_submission(ground_truth_file)
    is_valid = submission_has_correct_format(
        run_df
    ) and submission_is_evaluable(run_df, test_df)
    if is_valid:
        log(
            _logger.info,
            "Congratulations! Your submission is valid!",
            color="green",
        )
    else:
        log(
            _logger.warning,
            "Some issues were found with your submission :(",
            color="yellow",
        )
    return is_valid


def evaluate_submission(
    run_df: pd.DataFrame,
    team: str,
    run_name: str,
    ground_truth_df: pd.DataFrame,
) -> dict[str, Any]:
    results: dict[str, Any] = {"team": team, "run": run_name}
    run_df = run_df.merge(
        ground_truth_df,
        on="id",
        how="left",
        suffixes=("_pred", "_true"),
    )

    y_true = run_df["label_true"]
    y_pred = run_df["label_pred"]

    results["mf1"] = f1_score(y_true=y_true, y_pred=y_pred, average="macro")

    mf1_cinterval = bootstrap(
        data=[y_true, y_pred],
        statistic=partial(f1_score, average="macro"),
        n_resamples=100,
        paired=True,
        confidence_level=0.95,
        method="basic",
    )
    results["mf1_cinterval"] = (
        float(mf1_cinterval.confidence_interval.low),
        float(mf1_cinterval.confidence_interval.high),
    )

    results["all_metrics"] = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        digits=4,
        output_dict=True,
    )
    return results


def evaluate_submissions(
    subtask: int,
    submissions_dir: Path,
    ground_truth_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Evaluates all the submissions in the submissions path.
    This is the official evaluation code that the organizers will use
    to rank submissions. Computes f1 per-class and macro-averaged,
    accuracy, a classification report and confidence intervals of the
    macro-averaged f1.

    Args:
        subtask (int): either 1 or 2.
        submissions_dir (Path): base path for submissions.
        ground_truth_dir (Path): base path for ground truths.
        output_dir (Path): file to store the ranking and results.

    Returns:
        pd.DataFrame: a ranking of the submissions.
    """
    ground_truth_path = next(
        ground_truth_dir.glob(f"subtask_{subtask}/truth.jsonl")
    )
    ground_truth_df = load_submission(ground_truth_path)
    if not submission_has_correct_format(ground_truth_df):
        log(
            _logger.error,
            f"Ground truth path is incorrect: {ground_truth_path.as_posix()}",
            color="red",
        )
        sys.exit()

    results = []

    for run_path in submissions_dir.glob(f"*/subtask_{subtask}/*.jsonl"):
        team, run_name = get_team_and_run_name(run_path)

        run_df = load_submission(run_path)
        if submission_has_correct_format(run_df) and submission_is_evaluable(
            run_df, ground_truth_df
        ):
            result = evaluate_submission(
                run_df, team, run_name, ground_truth_df
            )
            results.append(result)
        else:
            log(
                _logger.warning,
                f"Skipping {team=} {run_name=} with {run_path=}",
                color="yellow",
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="mf1", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        output_dir / f"subtask_{subtask}.tsv", sep="\t", index=False
    )
    return results_df
