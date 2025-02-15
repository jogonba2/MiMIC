from pathlib import Path

from datasets import load_from_disk

from ..logging import get_logger, log
from ..utils import read_yaml, save_predictions
from .functional import FunctionalClassifier
from .late_fusion import LateFusionClassifier
from .unimodal import CombinedUnimodalClassifier
from .vlm import VisionLanguageModelClassifier

_logger = get_logger(__name__)

MODELS = {
    "unimodal": CombinedUnimodalClassifier,
    "functional": FunctionalClassifier,
    "late_fusion": LateFusionClassifier,
    "vlm": VisionLanguageModelClassifier,
}


def run_experiment(
    subtask: int,
    config_path: Path,
    dataset_path: Path,
    team_name: str,
    output_path: Path,
) -> None:
    dataset = load_from_disk(dataset_path)

    # Save the test dataset for easy evals later on
    ground_truth_path = output_path / "ground_truth" / f"subtask_{subtask}"
    ground_truth_path.mkdir(parents=True, exist_ok=True)
    ground_truth_file = ground_truth_path / "truth.jsonl"
    save_predictions(
        ground_truth_file, dataset["test"]["id"], dataset["test"]["label"]
    )

    output_path = output_path / "submissions" / team_name / f"subtask_{subtask}"
    output_path.mkdir(parents=True, exist_ok=True)

    config = read_yaml(config_path)

    # Train and predict for each model in the provided config
    for model_name, model_args in config.items():
        output_file = output_path / f"{model_name}.jsonl"

        # Skip the models if we already have their predictions
        if output_file.is_file():
            log(
                _logger.info,
                f"{output_file.as_posix()} already exists. Skipping experiment.",
                color="green",
            )
            continue

        model_class = MODELS[model_args["type"]]
        model = model_class(**model_args["params"])  # type: ignore

        model.fit(dataset["train"])
        predictions = model.predict(dataset["test"])

        save_predictions(output_file, dataset["test"]["id"], predictions)
