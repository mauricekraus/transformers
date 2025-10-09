# coding=utf-8
# MIT License
#
# Copyright (c) 2025 Maurice Kraus
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""PyTorch xLSTM-Mixer model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import OptionalDependencyNotAvailable, is_xlstm_available, logging
from .configuration_xlstm_mixer import xLSTMMixerConfig


logger = logging.get_logger(__name__)


try:
    if not is_xlstm_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable as dependency_error:
    raise OptionalDependencyNotAvailable(
        "xLSTMMixer requires the `xlstm` package. Install it with `pip install xlstm` (adjust the version as needed)."
    ) from dependency_error

try:  # pragma: no cover
    from torch.utils import cpp_extension as _torch_cpp_extension
except ImportError:  # pragma: no cover
    _torch_cpp_extension = None
else:  # pragma: no cover
    if _torch_cpp_extension is not None:
        try:
            _torch_cpp_extension.include_paths(cuda=True)
        except TypeError:
            _original_include_paths = _torch_cpp_extension.include_paths

            def _include_paths_wrapper(*args, **kwargs):
                kwargs.pop("cuda", None)
                return _original_include_paths(*args, **kwargs)

            _torch_cpp_extension.include_paths = _include_paths_wrapper

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, sLSTMBlockConfig, sLSTMLayerConfig


class _MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of a time series.
    """

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Pad both ends of the input sequence
        front = inputs[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = inputs[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        padded = torch.cat([front, inputs, end], dim=1)
        smoothed = self.avg(padded.permute(0, 2, 1))
        return smoothed.permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    """
    Seasonal/trend decomposition implemented with a moving average filter.
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = _MovingAverage(kernel_size, stride=1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(inputs)
        seasonal = inputs - trend
        return seasonal, trend


class RevIN(nn.Module):
    """
    Reversible instance normalization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("mean", None, persistent=False)
        self.register_buffer("stdev", None, persistent=False)
        self.register_buffer("last", None, persistent=False)

    def forward(self, inputs: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._update_statistics(inputs)
            return self._normalize(inputs)
        if mode == "denorm":
            return self._denormalize(inputs)
        raise ValueError("`mode` must be either 'norm' or 'denorm'.")

    def _update_statistics(self, inputs: torch.Tensor) -> None:
        reduce_dims = tuple(range(1, inputs.ndim - 1))
        if self.subtract_last:
            self.last = inputs[:, -1, :].unsqueeze(1)
        else:
            self.mean = inputs.mean(dim=reduce_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(inputs.var(dim=reduce_dims, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return inputs
        if self.subtract_last:
            outputs = inputs - self.last
        else:
            outputs = inputs - self.mean
        outputs = outputs / self.stdev
        if self.affine:
            outputs = outputs * self.affine_weight
            outputs = outputs + self.affine_bias
        return outputs

    def _denormalize(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return inputs
        outputs = inputs
        if self.affine:
            outputs = outputs - self.affine_bias
            outputs = outputs / (self.affine_weight + self.eps * self.eps)
        outputs = outputs * self.stdev
        if self.subtract_last:
            outputs = outputs + self.last
        else:
            outputs = outputs + self.mean
        return outputs


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (tensor * mask).sum(dim=dim, keepdim=True) / denom


def mean_abs_scaling(
    past_values: torch.Tensor,
    observed_mask: Optional[torch.Tensor] = None,
    minimum_scale: float = 1e-5,
) -> torch.Tensor:
    if observed_mask is None:
        scale = past_values.abs().mean(dim=1, keepdim=True)
    else:
        scale = masked_mean(past_values.abs(), observed_mask, dim=1)
    return scale.clamp_min(minimum_scale)


def pinball_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    observed_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target = target.unsqueeze(-1)
    diff = target - predictions
    reshape_dims = (1,) * (diff.dim() - 1) + (-1,)
    q = quantiles.view(*reshape_dims)
    loss = torch.maximum(q * diff, (q - 1.0) * diff).abs()
    if observed_mask is not None:
        weights = observed_mask.unsqueeze(-1)
        loss = loss * weights
        denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
    else:
        denom = predictions.new_tensor(predictions.size(1) * predictions.size(2) * predictions.size(3))
    return loss.sum(dim=(1, 2, 3)) / denom


def aggregate_hidden_states(hidden: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "mean":
        return hidden.mean(dim=1)
    if strategy == "max":
        return hidden.max(dim=1).values
    if strategy == "last":
        return hidden[:, -1]
    raise ValueError(f"Unsupported aggregation strategy: {strategy}")


@dataclass
class xLSTMMixerModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    prediction_features: Optional[torch.FloatTensor] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


@dataclass
class xLSTMMixerForPredictionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: Optional[torch.FloatTensor] = None
    quantile_outputs: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


@dataclass
class xLSTMMixerForTimeSeriesClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    pooled_hidden_state: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class xLSTMMixerForRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: Optional[torch.FloatTensor] = None
    pooled_hidden_state: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class xLSTMMixerForPretrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reconstruction: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class QuantileProjection(nn.Module):
    def __init__(self, num_channels: int, num_quantiles: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_quantiles = num_quantiles
        self.proj = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels * num_quantiles,
            kernel_size=1,
            groups=num_channels,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        feat = features.transpose(1, 2)
        out = self.proj(feat)
        out = out.reshape(features.size(0), self.num_channels, self.num_quantiles, features.size(1))
        return out.permute(0, 3, 1, 2).contiguous()


class xLSTMMixerEncoder(nn.Module):
    def __init__(self, config: xLSTMMixerConfig) -> None:
        super().__init__()
        self.config = config

        self.revin = RevIN(config.num_input_channels, affine=config.revin_affine)
        self.decomposition = SeriesDecomposition(config.decomposition_window)

        self.seasonal_mlp = nn.Linear(config.context_length, config.prediction_length)
        self.trend_mlp = nn.Linear(config.context_length, config.prediction_length)
        self.pre_encoding = nn.Linear(config.prediction_length, config.d_model)

        slstm_layer = sLSTMLayerConfig(
            embedding_dim=config.d_model,
            num_heads=config.num_heads,
            conv1d_kernel_size=config.conv1d_kernel_size,
            backend="vanilla",
            dtype="float32",
            dtype_b="float32",
            dtype_r="float32",
            dtype_w="float32",
            dtype_g="float32",
            dtype_s="float32",
            dtype_a="float32",
            enable_automatic_mixed_precision=False,
        )
        slstm_block = sLSTMBlockConfig(slstm=slstm_layer)

        stack_config = xLSTMBlockStackConfig(
            slstm_block=slstm_block,
            mlstm_block=None,
            num_blocks=config.num_layers,
            embedding_dim=config.d_model,
            context_length=config.num_input_channels,
            dropout=config.dropout,
            bias=True,
        )
        self.xlstm = xLSTMBlockStack(stack_config)
        self.dropout = nn.Dropout(config.dropout)
        self.feature_proj = nn.Linear(config.d_model, config.prediction_length)

    def forward(self, past_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seasonal_inputs, trend_inputs = self.decomposition(past_values)
        seasonal = seasonal_inputs.transpose(1, 2)
        trend = trend_inputs.transpose(1, 2)
        seasonal = self.seasonal_mlp(seasonal)
        trend = self.trend_mlp(trend)
        tokens = seasonal + trend
        tokens_encoded = self.pre_encoding(tokens)
        hidden = self.dropout(self.xlstm(tokens_encoded))
        projected = self.feature_proj(hidden)
        predictions = projected.transpose(1, 2)
        return hidden, predictions


class xLSTMMixerPreTrainedModel(PreTrainedModel):
    config_class = xLSTMMixerConfig
    base_model_prefix = "model"

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, "init_std", 0.02)
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class xLSTMMixerModel(xLSTMMixerPreTrainedModel):
    def __init__(self, config: xLSTMMixerConfig) -> None:
        super().__init__(config)
        self.encoder = xLSTMMixerEncoder(config)
        self.scaling = config.scaling

        self.post_init()

    def _scale_inputs(
        self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observed_mask is None:
            observed_mask = torch.ones_like(past_values)

        if self.scaling == "mean":
            loc = masked_mean(past_values, observed_mask, dim=1)
            centered = past_values - loc
            scale = mean_abs_scaling(centered, observed_mask)
            scaled = centered / scale
        else:
            loc = torch.zeros_like(past_values[:, :1])
            scale = torch.ones_like(past_values[:, :1])
            scaled = past_values

        scaled = self.encoder.revin(scaled, mode="norm")
        return scaled, loc, scale

    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> xLSTMMixerModelOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        scaled, loc, scale = self._scale_inputs(past_values, observed_mask)
        hidden, predictions = self.encoder(scaled)

        if not return_dict:
            return hidden, predictions, loc, scale

        return xLSTMMixerModelOutput(
            last_hidden_state=hidden,
            prediction_features=predictions,
            loc=loc,
            scale=scale,
        )


class xLSTMMixerForPrediction(xLSTMMixerPreTrainedModel):
    def __init__(self, config: xLSTMMixerConfig) -> None:
        super().__init__(config)
        self.model = xLSTMMixerModel(config)
        self.loss_type = config.loss
        self.num_channels = config.num_input_channels

        if self.loss_type == "pinball":
            if not config.quantiles:
                raise ValueError("`quantiles` must be provided when `loss='pinball'`.")
            quantiles = torch.tensor(config.quantiles, dtype=torch.float32)
            self.register_buffer("quantiles", quantiles, persistent=False)
            self._quantiles_list = list(config.quantiles)
            self.quantile_proj = QuantileProjection(config.num_input_channels, len(config.quantiles))
        else:
            self.quantiles = None
            self._quantiles_list = None
            self.quantile_proj = None

        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> xLSTMMixerForPredictionOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.model(
            past_values=past_values,
            observed_mask=observed_mask,
            return_dict=True,
        )

        prediction_features = model_output.prediction_features
        loc = model_output.loc
        scale = model_output.scale

        prediction_outputs = None
        quantile_outputs = None
        loss = None

        if self.loss_type == "mse":
            denorm = self.model.encoder.revin(prediction_features, mode="denorm")
            prediction_outputs = denorm * scale + loc
            if return_loss and future_values is not None:
                mse = (prediction_outputs - future_values) ** 2
                if future_observed_mask is not None:
                    weights = future_observed_mask.to(mse.dtype)
                    mse = mse * weights
                    denom = weights.sum(dim=(1, 2)).clamp_min(1.0)
                else:
                    denom = mse.new_full((mse.size(0),), mse.size(1) * mse.size(2))
                loss = mse.sum(dim=(1, 2)) / denom
        elif self.loss_type == "pinball":
            projections = self.quantile_proj(prediction_features)
            quantile_slices = []
            for qi in range(projections.size(-1)):
                slice_norm = projections[..., qi]
                slice_denorm = self.model.encoder.revin(slice_norm, mode="denorm")
                quantile_slices.append(slice_denorm.unsqueeze(-1))
            quantile_outputs = torch.cat(quantile_slices, dim=-1)
            quantile_outputs = quantile_outputs * scale.unsqueeze(-1) + loc.unsqueeze(-1)
            if self._quantiles_list and 0.5 in self._quantiles_list:
                median_idx = self._quantiles_list.index(0.5)
                prediction_outputs = quantile_outputs[..., median_idx]
            if return_loss and future_values is not None:
                q_tensor = self.quantiles.to(quantile_outputs.device, quantile_outputs.dtype)
                loss = pinball_loss(
                    predictions=quantile_outputs,
                    target=future_values,
                    quantiles=q_tensor,
                    observed_mask=future_observed_mask,
                )

        if isinstance(loss, torch.Tensor):
            loss = loss.mean()

        if not return_dict:
            return tuple(
                item
                for item in (loss, prediction_outputs, quantile_outputs, model_output.last_hidden_state, loc, scale)
                if item is not None
            )

        return xLSTMMixerForPredictionOutput(
            loss=loss,
            prediction_outputs=prediction_outputs,
            quantile_outputs=quantile_outputs,
            last_hidden_state=model_output.last_hidden_state,
            loc=loc,
            scale=scale,
        )


class xLSTMMixerForTimeSeriesClassification(xLSTMMixerPreTrainedModel):
    def __init__(self, config: xLSTMMixerConfig) -> None:
        super().__init__(config)
        self.model = xLSTMMixerModel(config)
        self.dropout = nn.Dropout(config.head_dropout)
        self.classifier = nn.Linear(config.num_input_channels, config.num_labels)

        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> xLSTMMixerForTimeSeriesClassificationOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.model(
            past_values=past_values,
            observed_mask=observed_mask,
            return_dict=True,
        )

        features = model_output.prediction_features
        pooled = aggregate_hidden_states(features, self.config.classification_aggregation)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, reduction="mean")

        if not return_dict:
            return tuple(item for item in (loss, logits, pooled, model_output.last_hidden_state) if item is not None)

        return xLSTMMixerForTimeSeriesClassificationOutput(
            loss=loss,
            logits=logits,
            pooled_hidden_state=pooled,
            last_hidden_state=model_output.last_hidden_state,
        )


class xLSTMMixerForRegression(xLSTMMixerPreTrainedModel):
    def __init__(self, config: xLSTMMixerConfig) -> None:
        super().__init__(config)
        self.model = xLSTMMixerModel(config)
        self.dropout = nn.Dropout(config.head_dropout)
        self.regressor = nn.Linear(config.num_input_channels, config.num_targets)

        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> xLSTMMixerForRegressionOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.model(
            past_values=past_values,
            observed_mask=observed_mask,
            return_dict=True,
        )

        features = model_output.prediction_features
        pooled = aggregate_hidden_states(features, self.config.classification_aggregation)
        pooled = self.dropout(pooled)
        preds = self.regressor(pooled)

        loss = None
        if labels is not None:
            loss = F.mse_loss(preds, labels, reduction="mean")

        if not return_dict:
            return tuple(item for item in (loss, preds, pooled, model_output.last_hidden_state) if item is not None)

        return xLSTMMixerForRegressionOutput(
            loss=loss,
            prediction_outputs=preds,
            pooled_hidden_state=pooled,
            last_hidden_state=model_output.last_hidden_state,
        )


class xLSTMMixerForPretraining(xLSTMMixerPreTrainedModel):
    def __init__(self, config: xLSTMMixerConfig) -> None:
        super().__init__(config)
        self.model = xLSTMMixerModel(config)
        self.reconstruction_proj = nn.Linear(config.prediction_length, config.context_length)

        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> xLSTMMixerForPretrainingOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.model(
            past_values=past_values,
            observed_mask=observed_mask,
            return_dict=True,
        )

        features = model_output.prediction_features
        recon_norm = self.reconstruction_proj(features.transpose(1, 2)).transpose(1, 2)
        recon_denorm = self.model.encoder.revin(recon_norm, mode="denorm")
        reconstruction = recon_denorm * model_output.scale + model_output.loc

        loss = None
        if return_loss:
            diff = (reconstruction - past_values) ** 2
            if observed_mask is not None:
                weights = observed_mask.to(diff.dtype)
                diff = diff * weights
                denom = weights.sum(dim=(1, 2)).clamp_min(1.0)
            else:
                denom = diff.new_full((diff.size(0),), diff.size(1) * diff.size(2))
            loss = (diff.sum(dim=(1, 2)) / denom).mean()

        if not return_dict:
            return tuple(item for item in (loss, reconstruction, model_output.last_hidden_state) if item is not None)

        return xLSTMMixerForPretrainingOutput(
            loss=loss,
            reconstruction=reconstruction,
            last_hidden_state=model_output.last_hidden_state,
        )


__all__ = [
    "xLSTMMixerPreTrainedModel",
    "xLSTMMixerModel",
    "xLSTMMixerForPretraining",
    "xLSTMMixerForPrediction",
    "xLSTMMixerForTimeSeriesClassification",
    "xLSTMMixerForRegression",
]
