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
"""xLSTM-Mixer configuration."""

from typing import Iterable, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class xLSTMMixerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`xLSTMMixerModel`]. It is used to instantiate an
    xLSTM-Mixer model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        context_length (`int`, *optional*, defaults to 96):
            Number of historical timesteps provided to the encoder.
        prediction_length (`int`, *optional*, defaults to 96):
            Number of timesteps the model should forecast.
        num_input_channels (`int`, *optional*, defaults to 1):
            Number of parallel input variates. A value of 1 corresponds to univariate forecasting.
        d_model (`int`, *optional*, defaults to 256):
            Hidden size used inside the xLSTM blocks.
        num_layers (`int`, *optional*, defaults to 2):
            Number of stacked xLSTM blocks.
        num_heads (`int`, *optional*, defaults to 8):
            Number of heads used inside each sLSTM block.
        conv1d_kernel_size (`int`, *optional*, defaults to 0):
            Size of the optional convolution used in the sLSTM gate mixing. A value of 0 disables the convolution.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability applied after the xLSTM stack.
        loss (`str`, *optional*, defaults to `"mse"`):
            Loss/head type to use. Supported values are `"mse"` for point forecasts and `"pinball"` for quantile
            regression.
        quantiles (`Iterable[float]`, *optional*, defaults to `(0.1, 0.5, 0.9)`):
            Quantiles to predict when `loss="pinball"`.
        num_labels (`int`, *optional*, defaults to 2):
            Number of classes for the classification head.
        num_targets (`int`, *optional*, defaults to 1):
            Number of regression targets for the regression head.
        head_dropout (`float`, *optional*, defaults to 0.0):
            Dropout applied inside the classification/regression heads.
        classification_aggregation (`str`, *optional*, defaults to `"mean"`):
            Temporal aggregation strategy for classification/regression heads. Supported values: `"mean"`, `"max"`,
            and `"last"`.
        scaling (`str`, *optional*, defaults to `"mean"`):
            Scaling strategy applied to the inputs before feeding them to the encoder. `"mean"` uses mean absolute
            scaling while any other value disables scaling.
        decomposition_window (`int`, *optional*, defaults to 25):
            Window length used by the Autoformer-style seasonal/trend decomposition.
        revin_affine (`bool`, *optional*, defaults to `False`):
            Whether to enable affine parameters inside the RevIN normalisation layer.
        init_std (`float`, *optional*, defaults to 0.02):
            Standard deviation used to initialise linear layers.
        use_return_dict (`bool`, *optional*, defaults to `True`):
            Whether to return model outputs as a [`~transformers.utils.ModelOutput`] instead of a plain tuple.
    """

    model_type = "xlstm_mixer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        context_length: int = 96,
        prediction_length: int = 96,
        num_input_channels: int = 1,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        conv1d_kernel_size: int = 0,
        dropout: float = 0.1,
        loss: str = "mse",
        quantiles: Optional[Iterable[float]] = (0.1, 0.5, 0.9),
        num_labels: int = 2,
        num_targets: int = 1,
        head_dropout: float = 0.0,
        classification_aggregation: str = "mean",
        scaling: str = "mean",
        decomposition_window: int = 25,
        revin_affine: bool = False,
        init_std: float = 0.02,
        use_return_dict: bool = True,
        **kwargs,
    ) -> None:
        if context_length <= 0:
            raise ValueError("`context_length` must be positive.")
        if prediction_length <= 0:
            raise ValueError("`prediction_length` must be positive.")
        if num_input_channels <= 0:
            raise ValueError("`num_input_channels` must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("`d_model` must be divisible by `num_heads`.")
        if loss not in {"mse", "pinball"}:
            raise ValueError("`loss` must be one of {'mse', 'pinball'}.")
        if classification_aggregation not in {"mean", "max", "last"}:
            raise ValueError("`classification_aggregation` must be one of {'mean', 'max', 'last'}.")

        super().__init__(return_dict=use_return_dict, **kwargs)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_input_channels = num_input_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.conv1d_kernel_size = conv1d_kernel_size
        self.dropout = dropout
        self.loss = loss
        self.quantiles = list(quantiles) if quantiles is not None else None
        self.num_labels = num_labels
        self.num_targets = num_targets
        self.head_dropout = head_dropout
        self.classification_aggregation = classification_aggregation
        self.scaling = scaling
        self.decomposition_window = decomposition_window
        self.revin_affine = revin_affine
        self.init_std = init_std


__all__ = ["xLSTMMixerConfig"]
