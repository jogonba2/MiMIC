from datasets import Dataset
from sklearn.dummy import DummyClassifier

from .base import Classifier


class FunctionalClassifier(Classifier):
    def __init__(
        self,
        model_params: dict,
        processor_params: dict,
        training_params: dict,
        inference_params: dict,
    ) -> None:
        super().__init__(
            model_params, processor_params, training_params, inference_params
        )
        self.model = DummyClassifier(**self.model_params)

    def fit(self, dataset: Dataset) -> Classifier:
        self.model.fit(dataset["text"], dataset["label"])
        return self

    def predict(self, dataset: Dataset) -> list[str]:
        predictions = self.model.predict(dataset["text"])
        return predictions
