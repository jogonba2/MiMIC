from dataclasses import dataclass
from typing import Optional, TypeAlias

import torch
from datasets import Dataset
from PIL import Image as Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

from ..logging import get_logger, log
from .base import Classifier

Tensor: TypeAlias = torch.Tensor

_logger = get_logger(__name__)


@dataclass
class Examples:
    """
    Class for keeping in-context examples
    """

    messages: list[dict]
    images: list[Image]
    pixel_values: Optional[Tensor] = None
    image_sizes: Optional[Tensor] = None


def interleave_tensors(X: Tensor, Y: Tensor) -> Tensor:
    """
    Adds the tensor Y before each row (axis=0) in the tensor X
    """
    groups = [torch.cat([Y, x.unsqueeze(0)], dim=0) for x in X]
    return torch.cat(groups, dim=0)


class VisionLanguageModelClassifier(Classifier):
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
        if "bitsandbytes" in self.model_params:
            self.bnb_config = BitsAndBytesConfig(
                **self.model_params["bitsandbytes"]
            )
            self.model_params["model"]["torch_dtype"] = "auto"
            self.model_params["model"]["quantization_config"] = self.bnb_config
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)

        self.model_name = self.model_params["model"][
            "pretrained_model_name_or_path"
        ]
        self.model = AutoModelForImageTextToText.from_pretrained(
            **self.model_params["model"],
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, **self.processor_params
        )
        # Need to specify pad token (is in vocab but not set as pad token)
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.pad_token_id = (
            self.processor.tokenizer.eos_token_id
        )

        # Ensure `predict`` works even without calling `fit` for zero-shot
        self.examples: Optional[Examples] = None

    def get_image_features(self, images: list[Image]) -> tuple[Tensor, Tensor]:
        features = self.processor.image_processor(
            images=images, return_tensors="pt"
        )
        return features["pixel_values"], features["image_sizes"]

    def get_icl_examples(
        self, dataset: Dataset, num_examples_per_label: int
    ) -> Examples:
        icl_messages, icl_images = [], []

        for option, label in self.model_params["option2label"].items():
            label_examples = dataset.filter(
                lambda label_: label_ == label, input_columns=["label"]
            ).select(range(num_examples_per_label))

            for row in label_examples:
                icl_messages.extend(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": row["text"]},
                            ],
                        },
                        {"role": "assistant", "content": option},
                    ]
                )
                icl_images.append(row["image"])

        pixel_values, image_sizes = self.get_image_features(icl_images)
        return Examples(
            messages=icl_messages,
            images=icl_images,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

    def prepare_batch(
        self,
        images: list[Image],
        texts: list[str],
        examples: Optional[Examples],
    ) -> dict:
        messages = [
            [
                {
                    "role": "system",
                    "content": self.inference_params["prompt"]["system"],
                },
                *(examples.messages if examples is not None else []),
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {
                            "type": "text",
                            "text": self.inference_params["prompt"][
                                "user"
                            ].format(text=text),
                        },
                    ],
                },
            ]
            for text in texts
        ]
        prepared_texts = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # When ICL, interleave the ICL images with the input images
        # and pass them in order to the processor.
        # Otherwise, just pass the input images to the processor.
        all_images = images
        if examples is not None:
            all_images = []
            for image in images:
                all_images += examples.images
                all_images.append(image)

        inputs = self.processor(
            text=prepared_texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
        )

        # Remove the ICL images from the processor
        # to avoid shape errors in the Dataset object.
        # ICL images will be sent later in `forward`
        if examples is not None:
            shift = len(examples.images)
            inputs["pixel_values"] = inputs["pixel_values"][shift::shift]
            inputs["image_sizes"] = inputs["image_sizes"][shift::shift]

        return inputs

    def prepare_dataset(
        self, dataset: Dataset, examples: Optional[Examples]
    ) -> Dataset:
        return dataset.map(
            self.prepare_batch,
            input_columns=["image", "text"],
            batched=True,
            desc=f"Preparing inputs for {self.model_name}",
            batch_size=self.inference_params["batch_size"],
            remove_columns=["image", "text"],
            fn_kwargs={"examples": examples},
        )

    def fit(self, dataset: Dataset) -> Classifier:
        num_examples_per_label = self.training_params.get(
            "num_examples_per_label", 0
        )
        if num_examples_per_label > 0:
            self.examples = self.get_icl_examples(
                dataset, self.training_params["num_examples_per_label"]
            )

        return self

    def predict(self, dataset: Dataset) -> list[str]:
        dataset = dataset.select_columns(["image", "text"])
        examples = self.examples
        dataset = self.prepare_dataset(dataset, examples)
        labels = list(self.model_params["option2label"].keys())
        logits = self.forward(
            dataset,
            examples,
            labels,
        )

        predictions = logits.argmax(-1)
        predicted_labels = [
            self.model_params["option2label"][labels[idx]]
            for idx in predictions
        ]
        return predicted_labels

    def forward(
        self,
        tokenized_dataset: Dataset,
        examples: Optional[Examples],
        labels: list[str],
    ) -> Tensor:
        """
        Efficient forward pass for classification.

        Args:
            tokenizer_dataset (Dataset): an already tokenized test dataset.
            examples (Examples): in-context examples.
            labels (list[str]): list of labels in the dataset.
        Returns:
            Tensor: logits produced by the model.
        """
        # Tokenize the labels
        # Temporary set padding to right before tokenizing and restore it
        self.processor.tokenizer.padding_side = "right"
        tokenized_labels = self.processor.tokenizer(
            labels, add_special_tokens=False, padding=True
        )
        self.processor.tokenizer.padding_side = "left"

        # Prepare the labels
        label_ids = torch.LongTensor(tokenized_labels["input_ids"]).to(
            self.model.device
        )
        collator = DataCollatorWithPadding(
            self.processor.tokenizer, padding=True
        )
        data_loader = DataLoader(
            tokenized_dataset,
            batch_size=self.inference_params["batch_size"],
            collate_fn=collator,
        )

        # Forward pass for classification
        self.model.eval()
        with torch.inference_mode():
            output_logits = []
            for batch in tqdm(data_loader, desc="Classifying..."):
                # When doing ICL, interleave example images with input images
                if examples is not None:
                    batch["pixel_values"] = interleave_tensors(
                        batch["pixel_values"], examples.pixel_values
                    )
                    batch["image_sizes"] = interleave_tensors(
                        batch["image_sizes"], examples.image_sizes
                    )
                batch = {
                    k: batch[k].to(self.model.device) for k in batch.keys()
                }
                # Pixel values are float and should be of the same type as the models dtype
                if "pixel_values" in batch:
                    batch["pixel_values"] = batch["pixel_values"].to(
                        self.model.dtype
                    )

                output = self.model(**batch)

                current_batch_size = list(batch.values())[0].shape[0]

                # (current_batch_size, num_labels, label_ids)
                logits_per_label = torch.zeros(
                    (current_batch_size, *label_ids.shape)
                ).to(self.model.device)

                # Go label by label to compute the logits of each label token
                for i, current_label_ids in enumerate(label_ids):
                    past_key_values = output.past_key_values
                    # Compute the logit of the first label token for each sample in the batch
                    logits_per_label[:, i, 0] = output.logits[
                        :, -1, current_label_ids[0]
                    ]

                    if torch.isnan(logits_per_label).any():
                        log(
                            _logger.warning,
                            "The model is returning NaNs for your input",
                            "yellow",
                        )
                    # Compute the logits of the remaining tokens of the label `i` for the whole batch
                    for j in range(1, current_label_ids.shape[0]):
                        current_id = current_label_ids[j]

                        # Skip the padding
                        if current_id == self.processor.tokenizer.pad_token_id:
                            continue

                        label_output = self.model(
                            input_ids=current_id.repeat(
                                current_batch_size, 1
                            ).to(self.model.device),
                            past_key_values=past_key_values,
                        )
                        logits_per_label[:, i, j] = label_output.logits[
                            :, -1, current_id
                        ]

                        past_key_values = label_output.past_key_values

                # Average the logits (exclude 0 from padding)
                mask = logits_per_label != 0
                mean_logits = (logits_per_label * mask).sum(dim=-1) / mask.sum(
                    dim=-1
                )

                # Accumulate outputs
                output_logits.append(mean_logits.detach().cpu())

            # Stack outputs
            output_logits = torch.vstack(output_logits)

        return output_logits
