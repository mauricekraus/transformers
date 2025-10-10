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
    prediction_outputs: Optional[torch.FloatTensor] = None
    pooled_hidden_state: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None

    def __post_init__(self):
        if self.prediction_outputs is None and self.logits is not None:
            object.__setattr__(self, "prediction_outputs", self.logits)
        elif self.logits is None and self.prediction_outputs is not None:
            object.__setattr__(self, "logits", self.prediction_outputs)


@dataclass
class xLSTMMixerForRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: Optional[torch.FloatTensor] = None
    pooled_hidden_state: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class xLSTMMixerForPretrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: Optional[torch.FloatTensor] = None
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
        self.backcast = config.backcast
        self.ensemble_size = config.ensemble_size
        self.num_memory_tokens = config.num_memory_tokens
        self.num_tokens_per_variate = config.num_tokens_per_variate

        self.hidden_dim = config.d_model
        self.total_views = max(self.ensemble_size, 2 if self.backcast else 1)

        self.num_series_tokens = config.num_input_channels * self.num_tokens_per_variate
        self.token_embedding_dim = self.hidden_dim
        self.view_embedding_dim = self.hidden_dim * self.total_views

        if self.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(self.num_memory_tokens, self.token_embedding_dim) * config.init_std
            )
        else:
            self.memory_tokens = None

        self.nlinear_projection = nn.Linear(config.context_length, config.prediction_length)
        self.pre_encoding = nn.Linear(config.prediction_length, self.hidden_dim)

        slstm_layer = sLSTMLayerConfig(
            embedding_dim=self.token_embedding_dim,
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
            embedding_dim=self.token_embedding_dim,
            context_length=self.num_series_tokens + self.num_memory_tokens,
            dropout=config.dropout,
            bias=True,
        )
        self.xlstm = xLSTMBlockStack(stack_config)
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(self.view_embedding_dim, config.prediction_length)

        self.output_tokens = config.num_input_channels
        self.classifier_feature_dim = self.output_tokens * self.view_embedding_dim

    def _prepare_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_last = inputs[:, -1:, :].detach()
        residual = inputs - seq_last
        projected = self.nlinear_projection(residual.transpose(1, 2)).transpose(1, 2)
        tokens = projected + seq_last
        tokens = tokens.transpose(1, 2)

        if self.num_tokens_per_variate > 1:
            tokens = tokens.repeat_interleave(self.num_tokens_per_variate, dim=1)

        encoded = self.pre_encoding(tokens)
        return encoded

    def _prepend_memory(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.memory_tokens is None:
            return tokens
        mem = self.memory_tokens.unsqueeze(0).expand(tokens.size(0), -1, -1)
        return torch.cat([mem, tokens], dim=1)

    def _remove_memory(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.memory_tokens is None:
            return tokens
        return tokens[:, self.num_memory_tokens :, :]

    def _generate_views(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        views: list[torch.Tensor] = [tokens]
        if self.backcast:
            views.append(torch.flip(tokens, dims=[1]))

        base_indices = torch.arange(tokens.size(1), device=tokens.device)
        while len(views) < self.total_views:
            shift = len(views)  # deterministic shift based on current view count
            permuted_indices = torch.roll(base_indices, shifts=shift, dims=0)
            views.append(tokens[:, permuted_indices, :])
        return views

    def _run_view(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._prepend_memory(tokens)
        processed = self.xlstm(tokens)
        processed = self.dropout(processed)
        processed = self._remove_memory(processed)
        return processed

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_tokens = self._prepare_tokens(inputs)
        views = self._generate_views(encoded_tokens)
        processed_views = [self._run_view(view) for view in views]
        combined = torch.cat(processed_views, dim=-1)

        if self.num_tokens_per_variate > 1:
            batch_size, total_tokens, combined_dim = combined.shape
            combined = combined.view(
                batch_size,
                self.config.num_input_channels,
                self.num_tokens_per_variate,
                combined_dim,
            ).mean(dim=2)

        predictions = self.projection(combined).transpose(1, 2)
        return combined, predictions


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
        self.classifier_feature_dim = self.encoder.classifier_feature_dim

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
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.head_dropout)
        self.classifier = nn.Linear(self.model.classifier_feature_dim, config.num_labels)

        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        target_values: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> xLSTMMixerForTimeSeriesClassificationOutput:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series. These correspond to the inputs fed through the encoder backbone.
            target_values (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Class labels for the sequence. When provided and `return_loss=True`, a cross-entropy loss is returned.
            observed_mask (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`, *optional*):
                Optional binary mask indicating which entries in `past_values` were observed (`1.0`) versus missing
                (`0.0`). Unused entries can be filled with zeros.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Included for API compatibility with other time-series heads. Hidden states are not returned by the
                current implementation.
            return_loss (`bool`, *optional*, defaults to `True`):
                Whether to return the loss value when `target_values` is provided.
            return_dict (`bool`, *optional*):
                Whether to return a [`~transformers.utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Support legacy positional calls where the observed mask was passed as the second argument.
        if (
            target_values is not None
            and observed_mask is None
            and target_values.shape == past_values.shape
            and target_values.dim() == past_values.dim()
        ):
            observed_mask = target_values
            target_values = None

        model_output = self.model(
            past_values=past_values,
            observed_mask=observed_mask,
            return_dict=True,
        )

        hidden = model_output.last_hidden_state
        activated = self.activation(hidden)
        flattened = self.dropout(activated).reshape(activated.size(0), -1)
        logits = self.classifier(flattened)

        loss = None
        if target_values is not None and return_loss:
            loss = F.cross_entropy(logits, target_values, reduction="mean")

        if not return_dict:
            return tuple(item for item in (loss, logits, flattened, model_output.last_hidden_state) if item is not None)

        return xLSTMMixerForTimeSeriesClassificationOutput(
            loss=loss,
            logits=logits,
            prediction_outputs=logits,
            pooled_hidden_state=flattened,
            last_hidden_state=model_output.last_hidden_state,
        )


class xLSTMMixerForRegression(xLSTMMixerPreTrainedModel):
    def __init__(self, config: xLSTMMixerConfig) -> None:
        super().__init__(config)
        self.model = xLSTMMixerModel(config)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.head_dropout)
        self.regressor = nn.Linear(self.model.classifier_feature_dim, config.num_targets)

        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> xLSTMMixerForRegressionOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Support legacy positional calls where the observed mask was passed as the second argument.
        if (
            labels is not None
            and observed_mask is None
            and labels.shape == past_values.shape
            and labels.dim() == past_values.dim()
        ):
            observed_mask = labels
            labels = None

        model_output = self.model(
            past_values=past_values,
            observed_mask=observed_mask,
            return_dict=True,
        )

        hidden = model_output.last_hidden_state
        activated = self.activation(hidden)
        flattened = self.dropout(activated).reshape(activated.size(0), -1)
        preds = self.regressor(flattened)

        loss = None
        if labels is not None:
            loss = F.mse_loss(preds, labels, reduction="mean")

        if not return_dict:
            return tuple(item for item in (loss, preds, flattened, model_output.last_hidden_state) if item is not None)

        return xLSTMMixerForRegressionOutput(
            loss=loss,
            prediction_outputs=preds,
            pooled_hidden_state=flattened,
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
        prediction_outputs = reconstruction

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
            return tuple(
                item for item in (loss, prediction_outputs, model_output.last_hidden_state) if item is not None
            )

        return xLSTMMixerForPretrainingOutput(
            loss=loss,
            prediction_outputs=prediction_outputs,
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
