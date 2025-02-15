import os
from functools import partial
from pathlib import Path

import typer

from .evaluation import evaluate_submissions, submission_is_valid
from .logging import get_logger, log
from .models import run_experiment as _run_experiment

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

app = typer.Typer()
_logger = get_logger(__name__)
log_info = partial(log, logger_method=_logger.info, color="green")


@app.command()
def run_experiment(
    subtask: int = typer.Option(help="Subtask, either 1 or 2"),
    config_path: Path = typer.Option(
        help="JSON file with model configs.",
        exists=True,
        file_okay=True,
        resolve_path=True,
    ),
    dataset_path: Path = typer.Option(
        help="Path to the dataset folder.",
        exists=True,
        dir_okay=True,
        resolve_path=True,
    ),
    team_name: str = typer.Option(help="Team name"),
    output_path: Path = typer.Option(
        help="Path to store predictions.",
        exists=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Fits models and predicts, saving the predictions into disk, ready to be
    evaluated with the `evaluate` endpoint.

    You can use this endpoint to run the baselines or to run
    your own models, both for subtasks 1 and 2.

    Args:
        subtask (int): Subtask, either 1 or 2.
        config_path (Path): JSON file with model configs.
        dataset_path (Path): Path to the dataset folder.
        team_name (str): Team name.
        output_path (Path): Path to store predictions.
    """
    log_info(
        text=f"Running experiment for subtask {subtask}. Using config {config_path.as_posix()}"
    )
    _run_experiment(
        subtask,
        config_path,
        dataset_path,
        team_name,
        output_path,
    )


@app.command()
def evaluate(
    subtask: int = typer.Option(help="Subtask, either 1 or 2"),
    submissions_dir: Path = typer.Option(
        Path("./evaluation_sample/submissions"),
        help="Path containing submissions.",
        exists=True,
        dir_okay=True,
        resolve_path=True,
    ),
    ground_truth_dir: Path = typer.Option(
        Path("./evaluation_sample/ground_truth"),
        help="Path containing ground truths.",
        exists=True,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        Path("./ranking_results"),
        help="File name to store the ranking dataframe.",
        resolve_path=True,
    ),
) -> None:
    """
    Evaluates submissions for the MIMIC shared task. This is the
    official evaluation code that the organizers will use
    to rank submissions.

    Args:
        subtask (int): the subtask, either 1 or 2.
        submissions_dir (Path): base path for submissions. Defaults to ./evaluation_sample/submissions.
        ground_truth_dir (Path): base path for ground truths. Defaults to ./evaluation_sample/ground_truth.
        output_dir (Path): file to store the ranking and results. Defaults to ./ranking_results.
    """
    log_info(text=f"Evaluating submissions in {submissions_dir.as_posix()}")
    ranking_df = evaluate_submissions(
        subtask, submissions_dir, ground_truth_dir, output_dir
    )
    log_info(text=str(ranking_df))


@app.command()
def check_format(
    submission_file: Path = typer.Option(
        help="File with predictions.",
        file_okay=True,
        resolve_path=True,
    ),
    ground_truth_file: Path = typer.Option(
        help="File with the test dataset.",
        file_okay=True,
        resolve_path=True,
    ),
) -> None:
    """
    Checks whether the submission format is correct.

    Args:
        submission_file (Path): submission file with the predictions.
        ground_truth_file (Path): test dataset to compare the ids of your submission.
    """
    log_info(
        text=f"Checking if submission in {submission_file.as_posix()} is valid."
    )
    submission_is_valid(submission_file, ground_truth_file)


if __name__ == "__main__":
    app()
