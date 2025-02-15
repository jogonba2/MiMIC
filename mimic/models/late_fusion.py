from typing import Any

import torch
from datasets import Dataset
from PIL import Image
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPVisionModel,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from .base import Classifier


class LateFusionHead(nn.Module):
    def __init__(
        self, model_params: dict, input_dims: int, output_dims: int
    ) -> None:
        super().__init__()
        self.model_params = model_params

        self.layers = nn.ModuleList(
            [nn.Linear(input_dims, self.model_params["ffn_dim"])]
            + [
                nn.Linear(
                    self.model_params["ffn_dim"], self.model_params["ffn_dim"]
                )
                for _ in range(self.model_params["n_layers"])
            ]
        )

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.model_params["ffn_dim"])
                for _ in range(self.model_params["n_layers"] + 1)
            ]
        )

        self.head = nn.Linear(self.model_params["ffn_dim"], output_dims)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(self.model_params["dropout"])
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self, text_embedding, image_embedding, labels=None
    ) -> torch.Tensor:
        x = torch.concat([text_embedding, image_embedding], axis=-1)

        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}


class LateFusionClassifier(Classifier):
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
        self.text_model_name = self.model_params["text"][
            "pretrained_model_name_or_path"
        ]
        self.image_model_name = self.model_params["image"][
            "pretrained_model_name_or_path"
        ]

        self.text_processor = AutoTokenizer.from_pretrained(
            self.text_model_name, **self.processor_params["text"]
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.image_model_name, **self.processor_params["image"]
        )

        self.text_model = AutoModel.from_pretrained(**self.model_params["text"])
        self.image_model = CLIPVisionModel.from_pretrained(
            **self.model_params["image"]
        )

        text_embedding_dim = self.text_model.config.hidden_size
        image_embedding_dim = self.image_model.config.hidden_size
        self.classifier = LateFusionHead(
            self.model_params["head"],
            text_embedding_dim + image_embedding_dim,
            len(self.model_params["label_mappings"]["label2id"]),
        )

    def embed_texts(self, texts: list[str]) -> dict[str, Any]:
        tokenized_texts = self.text_processor(
            texts, truncation=True, padding=True, return_tensors="pt"
        )
        with torch.inference_mode():
            outputs = self.text_model(**tokenized_texts)
            embeddings = outputs.pooler_output
        return {"text_embedding": embeddings}

    def embed_images(self, images: list[Image.Image]) -> dict[str, Any]:
        tokenized_texts = self.image_processor(images, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.image_model(**tokenized_texts)
            embeddings = outputs.pooler_output
        return {"image_embedding": embeddings}

    def embed(self, dataset: Dataset) -> Dataset:
        dataset = dataset.select_columns(["text", "image", "label"])

        self.text_model.eval()
        self.image_model.eval()

        dataset = dataset.map(
            self.embed_images,
            input_columns=["image"],
            batched=True,
            remove_columns=["image"],
            desc=f"Embedding images with {self.image_model_name}",
            batch_size=self.processor_params["image_encoder"].get(
                "batch_size", 16
            ),
        )

        dataset = dataset.map(
            self.embed_texts,
            input_columns=["text"],
            batched=True,
            remove_columns=["text"],
            desc=f"Embedding texts with {self.text_model_name}",
            batch_size=self.processor_params["text_encoder"].get(
                "batch_size", 16
            ),
        )
        return dataset

    def prepare_labels(self, dataset: Dataset) -> Dataset:
        def prepare(labels):
            return {
                "label": [
                    self.model_params["label_mappings"]["label2id"][label]
                    for label in labels
                ]
            }

        return dataset.map(
            prepare,
            input_columns=["label"],
            batched=True,
            desc="Preparing labels...",
        )

    def fit(
        self,
        dataset: Dataset,
    ) -> Classifier:
        dataset = self.embed(dataset)
        dataset = self.prepare_labels(dataset)

        data_collator = DefaultDataCollator()
        training_args = TrainingArguments(**self.training_params)
        trainer = Trainer(
            model=self.classifier,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model()
        return self

    def predict(self, dataset: Dataset) -> list[str]:
        dataset = self.embed(dataset)
        dataset = self.prepare_labels(dataset)
        data_collator = DefaultDataCollator()
        training_args = TrainingArguments(**self.inference_params)
        trainer = Trainer(
            model=self.classifier,
            args=training_args,
            data_collator=data_collator,
        )
        outputs = trainer.predict(dataset)
        predictions = outputs.predictions.argmax(-1)

        predicted_labels = [
            self.model_params["label_mappings"]["id2label"][str(prediction)]
            for prediction in predictions
        ]
        return predicted_labels
