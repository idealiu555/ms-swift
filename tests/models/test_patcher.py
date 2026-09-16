from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from swift.model.patcher import _patch_sequence_classification


def test_sequence_classification_uses_four_layer_silu_mlp():
    config = GPT2Config(
        n_embd=16,
        n_head=2,
        n_layer=1,
        n_positions=8,
        num_labels=1,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model_meta = SimpleNamespace(model_arch=SimpleNamespace(language_model=None))

    _patch_sequence_classification(model, model_meta)

    assert [type(module) for module in model.score] == [
        nn.Linear,
        nn.SiLU,
        nn.Linear,
        nn.SiLU,
        nn.Linear,
        nn.SiLU,
        nn.Linear,
    ]
    linear_layers = model.score[::2]
    assert linear_layers[0].in_features == config.n_embd
    assert linear_layers[-1].out_features == config.num_labels

    outputs = model(input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]), labels=torch.tensor([1.0, 2.0]))
    assert outputs.logits.shape == (2, config.num_labels)
    assert outputs.loss.ndim == 0
