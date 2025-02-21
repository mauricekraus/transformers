import torch
import torch.nn as nn
from ..bert import (
    BertTokenizerFast,
    BertModel
)
from ...modeling_utils import PreTrainedModel
from .configuration_bert_emotion_ordinal_regression import BertForMultiOutputOrdinalRegressionConfig

from dataclasses import dataclass
from typing import Optional
import torch
from ...modeling_outputs import ModelOutput

@dataclass
class BertForOrdinalRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    cat_loss: Optional[torch.FloatTensor] = None
    dim_loss: Optional[torch.FloatTensor] = None
    categories: torch.LongTensor = None
    dimensions: torch.LongTensor = None


class BertPreprocessor:

    def __init__(self, max_length=256, model_name="bert-base-uncased"):
        self.max_length = max_length
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def __call__(self, example):
        toks = self.tokenizer(
            example["caption"], truncation=True, padding="longest", max_length=self.max_length
        )
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels_categories": example["labels_categories"],
            "labels_dimensions": example["labels_dimensions"],
        }


class OrdinalHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        """
        in_features: Size of the input features (e.g. BERT hidden size).
        num_classes: Total number of ordinal classes (for example, 5 for values 0–4).
                     The head will output num_classes - 1 logits.
        """
        super().__init__()
        self.num_classes = (
            num_classes  # e.g., 5 for categories or 7 for some dimensions.
        )
        self.linear = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, in_features)
        Returns:
          logits: (batch_size, num_classes - 1)
          probabilities: (batch_size, num_classes - 1) after applying sigmoid.
        """
        logits = self.linear(x)
        probabilities = torch.sigmoid(logits)
        return logits, probabilities

class BertForMultiOutputOrdinalRegression(PreTrainedModel):
    def __init__(self, config: BertForMultiOutputOrdinalRegressionConfig):
        super().__init__(config)
        self.bert = BertModel(config.bert_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Create one ordinal head per emotion category.
        self.categories_heads = nn.ModuleList()
        for cat in config.categories_mapping:
            _, min_val, max_val = cat
            num_classes = int(max_val - min_val + 1)
            self.categories_heads.append(OrdinalHead(config.bert_config.hidden_size, num_classes))

        # Create one ordinal head per dimension.
        self.dimensions_heads = nn.ModuleList()
        for dim in config.dimensions_mapping:
            _, min_val, max_val = dim
            num_classes = int(max_val - min_val + 1)
            self.dimensions_heads.append(OrdinalHead(config.bert_config.hidden_size, num_classes))

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels_categories=None,  # shape: (batch, num_categories)
        labels_dimensions=None,  # shape: (batch, num_dimensions)
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use the pooled output (the [CLS] token representation).
        pooled_output = (
            outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[1]
        )
        pooled_output = self.dropout(pooled_output)
        batch_size = pooled_output.size(0)

        # Process emotion categories.
        cat_logits_all = []
        cat_probs_all = []
        for head in self.categories_heads:
            logits, probs = head(pooled_output)
            cat_logits_all.append(logits)
            cat_probs_all.append(probs)

        # Process dimensions.
        dim_logits_all = []
        dim_probs_all = []
        for head in self.dimensions_heads:
            logits, probs = head(pooled_output)
            dim_logits_all.append(logits)
            dim_probs_all.append(probs)

        loss = None
        cat_loss = None
        dim_loss = None

        if labels_categories is not None and labels_dimensions is not None:
            device = pooled_output.device
            cat_loss = 0.0
            bce_loss = nn.BCELoss()

            # Compute loss for each category head.
            for i, head in enumerate(self.categories_heads):
                target = labels_categories[:, i].long()  # shape: (batch,)
                num_thresholds = head.num_classes - 1
                cumulative = torch.zeros((batch_size, num_thresholds), device=device)
                for t in range(1, head.num_classes):
                    cumulative[:, t - 1] = (target >= t).float()
                probs = cat_probs_all[i]
                cat_loss += bce_loss(probs, cumulative)

            # Compute loss for each dimension head.
            dim_loss = 0.0
            for i, head in enumerate(self.dimensions_heads):
                target = labels_dimensions[:, i].long()  # shape: (batch,)
                num_thresholds = head.num_classes - 1
                cumulative = torch.zeros((batch_size, num_thresholds), device=device)
                for t in range(1, head.num_classes):
                    cumulative[:, t - 1] = (target >= t).float()
                probs = dim_probs_all[i]
                dim_loss += bce_loss(probs, cumulative)

            loss = cat_loss + dim_loss

        # Get predictions directly from the probabilities.
        # For each head, we count the number of thresholds passed.
        cat_preds = torch.cat(
            [self.predict_from_head(probs).unsqueeze(1) for probs in cat_probs_all], dim=1
        )
        dim_preds = torch.cat(
            [self.predict_from_head(probs).unsqueeze(1) for probs in dim_probs_all], dim=1
        )

        return BertForOrdinalRegressionOutput(
            loss=loss,
            cat_loss=cat_loss,
            dim_loss=dim_loss,
            categories=cat_preds,
            dimensions=dim_preds,
        )

    @staticmethod
    def predict_from_head(probabilities, threshold=0.5):
        """
        Convert a tensor of probabilities (shape: batch_size x num_thresholds)
        into ordinal predictions by counting how many thresholds are passed.
        """
        binary_preds = (probabilities > threshold).float()
        ordinal_preds = torch.sum(binary_preds, dim=-1)
        return ordinal_preds

    def predict_ordinals(
        self, input_ids, attention_mask=None, token_type_ids=None, threshold=0.5
    ):
        """
        A convenience method for inference that runs a forward pass without labels.
        It returns ordinal predictions for both categories and dimensions.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # Directly return the predictions from the model output.
            return {"categories": outputs.categories, "dimensions": outputs.dimensions}
