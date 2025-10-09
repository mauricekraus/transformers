<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# xLSTMMixer

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The xLSTM-Mixer architecture adapts the xLSTM recurrent backbone for multivariate time-series workloads. It combines
seasonal-trend decomposition, reversible instance normalisation (RevIN), and a stack of sLSTM blocks to offer
forecasting, classification, regression, and reconstruction heads within a single backbone. This integration exposes the
reference implementation from the [xLSTM Mixer project](https://github.com/NX-AI/xLSTM-Mixer) through the 🤗 Transformers
APIs.

The model relies on the external [`xlstm`](https://pypi.org/project/xlstm/) package. Install it before instantiating any
of the classes documented below.

## Usage example

```python
from transformers import xLSTMMixerConfig, xLSTMMixerForPrediction

import torch

config = xLSTMMixerConfig(context_length=96, prediction_length=24, num_input_channels=4)
model = xLSTMMixerForPrediction(config)

past_values = torch.randn(2, config.context_length, config.num_input_channels)
future_values = torch.randn(2, config.prediction_length, config.num_input_channels)
future_observed_mask = torch.ones_like(future_values)

outputs = model(
    past_values=past_values,
    future_values=future_values,
    future_observed_mask=future_observed_mask,
)
loss = outputs.loss
```

## xLSTMMixerConfig

[[autodoc]] xLSTMMixerConfig

## xLSTMMixerModel

[[autodoc]] xLSTMMixerModel
    - forward

## xLSTMMixerForPrediction

[[autodoc]] xLSTMMixerForPrediction
    - forward

## xLSTMMixerForTimeSeriesClassification

[[autodoc]] xLSTMMixerForTimeSeriesClassification
    - forward

## xLSTMMixerForRegression

[[autodoc]] xLSTMMixerForRegression
    - forward

## xLSTMMixerForPretraining

[[autodoc]] xLSTMMixerForPretraining
    - forward
