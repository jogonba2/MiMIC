from datasets import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from ..types import Labels
from .base import Classifier


class UnimodalTextClassifier(Classifier):
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
        self.model_name = self.model_params["pretrained_model_name_or_path"]
        self.model = AutoModelForSequenceClassification.from_pretrained(
            **self.model_params
        )
        self.processor = AutoTokenizer.from_pretrained(
            self.model_name, **self.processor_params
        )

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        dataset = dataset.select_columns(["text", "label"])
        generated_text_labels = [Labels.TEXT_GENERATED, Labels.FULLY_GENERATED]

        def prepare_batch(texts, labels):
            tokenized_texts = self.processor(texts, truncation=True)

            def convert_label(label):
                return (
                    "generated" if label in generated_text_labels else "human"
                )

            labels = {
                "label": [
                    self.model_params["label2id"][convert_label(label)]
                    for label in labels
                ]
            }

            return {**tokenized_texts, **labels}

        dataset = dataset.map(
            prepare_batch,
            input_columns=["text", "label"],
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing texts for {self.model_name}",
        )
        return dataset

    def fit(self, dataset: Dataset) -> Classifier:
        dataset = self.prepare_dataset(dataset)

        data_collator = DataCollatorWithPadding(tokenizer=self.processor)
        training_args = TrainingArguments(**self.training_params)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model()
        return self

    def predict(self, dataset: Dataset) -> list[str]:
        dataset = self.prepare_dataset(dataset)
        data_collator = DataCollatorWithPadding(tokenizer=self.processor)
        training_args = TrainingArguments(**self.inference_params)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
        )
        outputs = trainer.predict(dataset)

        # Some encoder-decoder models returns predictions as a tuple.
        if isinstance(outputs.predictions, tuple):
            predictions = outputs.predictions[0].argmax(-1)
        else:
            predictions = outputs.predictions.argmax(-1)

        predicted_labels = [
            self.model_params["id2label"][str(predcition)]
            for predcition in predictions
        ]
        return predicted_labels


class UnimodalImageClassifier(Classifier):
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
        self.model_name = self.model_params["pretrained_model_name_or_path"]
        self.model = AutoModelForImageClassification.from_pretrained(
            **self.model_params
        )
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name, **self.processor_params
        )

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        dataset = dataset.select_columns(["image", "label"])
        generated_image_labels = [
            Labels.IMAGE_GENERATED,
            Labels.FULLY_GENERATED,
        ]

        def prepare_batch(images, labels):
            tokenized_images = self.processor(images)

            def convert_label(label):
                return (
                    "generated" if label in generated_image_labels else "human"
                )

            labels = {
                "label": [
                    self.model_params["label2id"][convert_label(label)]
                    for label in labels
                ]
            }

            return {**tokenized_images, **labels}

        dataset = dataset.map(
            prepare_batch,
            input_columns=["image", "label"],
            batched=True,
            remove_columns=["image"],
            desc=f"Tokenizing images for {self.model_name}",
        )
        return dataset

    def fit(
        self,
        dataset: Dataset,
    ) -> Classifier:
        dataset = self.prepare_dataset(dataset)

        data_collator = DefaultDataCollator()
        training_args = TrainingArguments(**self.training_params)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model()
        return self

    def predict(self, dataset: Dataset) -> list[str]:
        dataset = self.prepare_dataset(dataset)
        data_collator = DefaultDataCollator()
        training_args = TrainingArguments(**self.inference_params)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
        )
        outputs = trainer.predict(dataset)
        predictions = outputs.predictions.argmax(-1)

        predicted_labels = [
            self.model_params["id2label"][str(prediction)]
            for prediction in predictions
        ]
        return predicted_labels


class CombinedUnimodalClassifier(Classifier):
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
        self.model_params["text"].update(self.model_params["label_mappings"])
        self.model_params["image"].update(self.model_params["label_mappings"])

        self.text_model = UnimodalTextClassifier(
            self.model_params["text"],
            self.processor_params["text"],
            self.training_params["text"],
            self.inference_params["text"],
        )
        self.image_model = UnimodalImageClassifier(
            self.model_params["image"],
            self.processor_params["image"],
            self.training_params["image"],
            self.inference_params["image"],
        )

    def fit(
        self,
        dataset: Dataset,
    ) -> Classifier:
        self.text_model.fit(dataset)
        self.image_model.fit(dataset)
        return self

    def predict(self, dataset: Dataset) -> list[str]:
        text_predictions = self.text_model.predict(dataset)
        image_predictions = self.image_model.predict(dataset)
        return self.transform_predictions(text_predictions, image_predictions)

    def transform_predictions(
        self, text_predictions: list[str], image_predictions: list[str]
    ) -> list[str]:
        unimodal_label2label = {
            ("human", "human"): Labels.FULLY_HUMAN,
            ("generated", "human"): Labels.TEXT_GENERATED,
            ("human", "generated"): Labels.IMAGE_GENERATED,
            ("generated", "generated"): Labels.FULLY_GENERATED,
        }

        result = [
            unimodal_label2label[text_label, image_label].value
            for text_label, image_label in zip(
                text_predictions, image_predictions
            )
        ]
        return result
