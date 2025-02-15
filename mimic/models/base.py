from abc import ABC, abstractmethod

from datasets import Dataset


class Classifier(ABC):
    def __init__(
        self,
        model_params: dict,
        processor_params: dict,
        training_params: dict,
        inference_params: dict,
    ) -> None:
        self.model_params = model_params
        self.processor_params = processor_params
        self.training_params = training_params
        self.inference_params = inference_params

    @abstractmethod
    def fit(self, dataset: Dataset) -> "Classifier": ...

    @abstractmethod
    def predict(self, dataset: Dataset) -> list[str]: ...
