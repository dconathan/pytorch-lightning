# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


def test_eval_inference_mode():
    """Testing overwriting trainer arguments."""

    class BoringModelNoGrad(BoringModel):
        def on_test_epoch_start(self) -> None:
            assert not torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()
            return super().on_test_epoch_start()

    class BoringModelForInferenceMode(BoringModel):
        def on_test_epoch_start(self) -> None:
            assert not torch.is_grad_enabled()
            assert torch.is_inference_mode_enabled()
            return super().on_test_epoch_start()

    trainer = Trainer(logger=False, inference_mode=False, fast_dev_run=True)
    trainer.test(BoringModelNoGrad())
    trainer = Trainer(logger=False, inference_mode=True, fast_dev_run=True)
    trainer.test(BoringModelForInferenceMode())
