from pathlib import Path

import pandas as pd
import yaml


def save_predictions(
    output_file: Path, ids: list[str], preds: list[str]
) -> None:
    df = pd.DataFrame({"id": ids, "label": preds})
    df.to_json(output_file, lines=True, orient="records")


def read_yaml(path: Path) -> dict:
    """
    Read a yaml file, removing anchors
    specified with the `anchor_` prefix

    Args:
        path (Path): path to a yaml file
    Returns:
        dict: parsed yaml
    """
    with open(path, "r") as f:
        content = yaml.safe_load(f)
    return {
        key: val
        for key, val in content.items()
        if not key.startswith("anchor_")
    }
