from functools import partial
from pathlib import Path

import pandas as pd

from ..logging import get_logger, log
from ..types import Labels

_logger = get_logger(__name__)
log_error = partial(log, logger_method=_logger.error, color="red")


def load_submission(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_json(
            path,
            lines=True,
            orient="records",
        )
        return df
    except ValueError as e:
        raise RuntimeError(
            f"Could not load submission {path.as_posix()}"
        ) from e


def submission_is_evaluable(
    run_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> bool:
    # Check this to warn the user but the submission is still evaluable
    uses_all_labels(run_df, ground_truth_df)
    return all(
        [
            has_same_number_of_samples(run_df, ground_truth_df),
            no_repeated_ids(run_df),
            has_all_ids(run_df, ground_truth_df),
        ]
    )


def submission_has_correct_format(df: pd.DataFrame) -> bool:
    return all(
        [
            has_correct_columns(df),
            has_no_empty_values(df),
            has_correct_labels(df),
        ]
    )


def uses_all_labels(
    run_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> bool:
    run_labels = set(run_df["label"])
    ground_truth_labels = set(ground_truth_df["label"])
    missing_labels = list(ground_truth_labels - run_labels)
    if missing_labels:
        log(
            _logger.warning,
            text=f"The submission only predicts the labels {list(run_labels)}. Missing {missing_labels} as predictions.",
            color="yellow",
        )

    return len(missing_labels) == 0


def has_correct_columns(df: pd.DataFrame) -> bool:
    if len(df.columns) > 2:
        log_error(text="Too many columns")
        return False

    for column in ["id", "label"]:
        missing = not (column in df.columns)
        if missing:
            log_error(text=f"Missing '{column}' column.")
            return False
    return True


def has_no_empty_values(df: pd.DataFrame) -> bool:
    for column in df.columns:
        if df[column].isna().any():
            log_error(text=f"NA value found in column '{column}'")
            return False
    return True


def has_correct_labels(df: pd.DataFrame) -> bool:
    possible_labels = [label.value for label in Labels]
    unique_labels = set(df["label"])
    for label in unique_labels:
        if label not in possible_labels:
            log_error(
                text=f"Unknown label '{label}'. Should be one of {possible_labels}"
            )
            return False
    return True


def no_repeated_ids(df: pd.DataFrame) -> bool:
    duplicates = df[df["id"].duplicated()]["id"].to_list()
    if duplicates:
        log_error(text=f"Prediction has duplicate ids: {duplicates}")
    return len(duplicates) == 0


def has_same_number_of_samples(
    run_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> bool:
    has_same_length = len(run_df) == len(ground_truth_df)
    if not has_same_length:
        log_error(
            text=f"Prediction length ({len(run_df)}) mismatch with ground truth length ({len(ground_truth_df)})"
        )
    return has_same_length


def has_all_ids(run_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> bool:
    missing_ids_index = ~ground_truth_df["id"].isin(run_df["id"])
    missing_ids = list(ground_truth_df[missing_ids_index]["id"])
    if missing_ids:
        log_error(text=f"Missing ids: {list(missing_ids)}.")
    return len(missing_ids) == 0


def get_team_and_run_name(path: Path) -> tuple[str, str]:
    team = str(path.parents[1]).split("/")[-1]
    run_name = path.stem
    return team, run_name
