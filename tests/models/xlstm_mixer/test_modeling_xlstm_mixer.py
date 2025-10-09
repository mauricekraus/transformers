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

import unittest

from transformers.testing_utils import require_torch
from transformers.utils import is_xlstm_available


xlstm_deps_available = is_xlstm_available()
if isinstance(xlstm_deps_available, tuple):
    xlstm_deps_available = xlstm_deps_available[0]

if xlstm_deps_available:
    import torch

    from transformers import (
        xLSTMMixerConfig,
        xLSTMMixerForPretraining,
        xLSTMMixerForPrediction,
        xLSTMMixerForRegression,
        xLSTMMixerForTimeSeriesClassification,
    )
    from transformers.models.xlstm_mixer.modeling_xlstm_mixer import pinball_loss


    def _dummy_inputs(batch_size=2, context_length=12, prediction_length=6, channels=2):
        past = torch.randn(batch_size, context_length, channels)
        future = torch.randn(batch_size, prediction_length, channels)
        mask = torch.ones(batch_size, prediction_length, channels)
        return past, future, mask
else:

    def _dummy_inputs(*args, **kwargs):  # pragma: no cover
        raise unittest.SkipTest("xLSTM optional dependencies are not available.")


@require_torch
@unittest.skipUnless(xlstm_deps_available, "Test requires the `xlstm` package.")
class xLSTMMixerModelTest(unittest.TestCase):
    def test_point_forecasting_forward(self):
        past, future, mask = _dummy_inputs()
        config = xLSTMMixerConfig(
            context_length=past.size(1),
            prediction_length=future.size(1),
            num_input_channels=past.size(2),
            d_model=32,
            num_layers=1,
            num_heads=4,
            loss="mse",
        )
        model = xLSTMMixerForPrediction(config)
        outputs = model(
            past_values=past,
            future_values=future,
            future_observed_mask=mask,
        )

        self.assertIsNotNone(outputs.prediction_outputs)
        self.assertEqual(outputs.prediction_outputs.shape, future.shape)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.numel(), 1)

    def test_quantile_forecasting_forward(self):
        past, future, mask = _dummy_inputs(channels=1)
        quantiles = [0.1, 0.5, 0.9]
        config = xLSTMMixerConfig(
            context_length=past.size(1),
            prediction_length=future.size(1),
            num_input_channels=past.size(2),
            d_model=32,
            num_layers=1,
            num_heads=4,
            loss="pinball",
            quantiles=quantiles,
        )
        model = xLSTMMixerForPrediction(config)
        outputs = model(
            past_values=past,
            future_values=future,
            future_observed_mask=mask,
        )

        self.assertIsNotNone(outputs.quantile_outputs)
        self.assertEqual(outputs.quantile_outputs.shape, (future.size(0), future.size(1), future.size(2), len(quantiles)))
        self.assertIsNotNone(outputs.loss)
        manual = pinball_loss(
            outputs.quantile_outputs,
            future,
            torch.tensor(quantiles, dtype=future.dtype, device=future.device),
            mask,
        ).mean()
        self.assertTrue(torch.allclose(outputs.loss, manual, atol=1e-6))

    def test_classification_forward(self):
        batch_size = 3
        past, _, _ = _dummy_inputs(batch_size=batch_size)
        labels = torch.randint(0, 4, (batch_size,))
        config = xLSTMMixerConfig(
            context_length=past.size(1),
            prediction_length=6,
            num_input_channels=past.size(2),
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_labels=4,
        )
        model = xLSTMMixerForTimeSeriesClassification(config)
        outputs = model(
            past_values=past,
            labels=labels,
        )

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape, (batch_size, 4))
        self.assertIsNotNone(outputs.pooled_hidden_state)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.numel(), 1)

    def test_regression_forward(self):
        batch_size = 2
        past, _, _ = _dummy_inputs(batch_size=batch_size, channels=3)
        targets = torch.randn(batch_size, 5)
        config = xLSTMMixerConfig(
            context_length=past.size(1),
            prediction_length=6,
            num_input_channels=past.size(2),
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_targets=5,
        )
        model = xLSTMMixerForRegression(config)
        outputs = model(
            past_values=past,
            labels=targets,
        )

        self.assertIsNotNone(outputs.prediction_outputs)
        self.assertEqual(outputs.prediction_outputs.shape, (batch_size, 5))
        self.assertIsNotNone(outputs.pooled_hidden_state)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.numel(), 1)

    def test_pretraining_forward(self):
        past, _, _ = _dummy_inputs()
        mask = torch.ones_like(past)
        config = xLSTMMixerConfig(
            context_length=past.size(1),
            prediction_length=6,
            num_input_channels=past.size(2),
            d_model=32,
            num_layers=1,
            num_heads=4,
        )
        model = xLSTMMixerForPretraining(config)
        outputs = model(
            past_values=past,
            observed_mask=mask,
        )

        self.assertIsNotNone(outputs.reconstruction)
        self.assertEqual(outputs.reconstruction.shape, past.shape)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.numel(), 1)
