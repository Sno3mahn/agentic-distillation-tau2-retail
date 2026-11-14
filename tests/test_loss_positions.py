"""selective_loss must equal the standard HF causal-LM loss (P0: otherwise we train on the wrong objective)."""

import pytest
import torch

pytest.importorskip("transformers")
from transformers import Cohere2Config, Cohere2ForCausalLM

from agentic_distill.train_lora import IGNORE, selective_loss


def _tiny(logit_scale=0.5):
    torch.manual_seed(0)
    cfg = Cohere2Config(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=8,
                        logit_scale=logit_scale, max_position_embeddings=256)
    return Cohere2ForCausalLM(cfg).eval()


def _batch(T=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 101, (1, T), generator=g)
    labels = ids.clone()
    labels[:, :12] = IGNORE            # masked prefix (system/user turns)
    labels[:, 20:27] = IGNORE          # masked middle (tool result)
    return ids, labels


@pytest.mark.parametrize("scale,chunk", [(1.0, 256), (0.5, 256), (0.5, 4)])
def test_matches_hf_loss(scale, chunk):
    model = _tiny(scale)
    ids, labels = _batch()
    with torch.no_grad():
        ref = model(input_ids=ids, labels=labels).loss
        s, n = selective_loss(model, ids, labels, chunk=chunk)
    assert n == int((labels[:, 1:] != IGNORE).sum())
    torch.testing.assert_close(s / n, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("chunk", [256, 4])
def test_gradients_match(chunk):
    model = _tiny()
    ids, labels = _batch(seed=2)
    model(input_ids=ids, labels=labels).loss.backward()
    ref = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    s, n = selective_loss(model, ids, labels, chunk=chunk)
    (s / n).backward()
    for a, b in zip(ref, [p.grad for p in model.parameters()]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_disabling_sdpa_gqa_keeps_outputs():
    """repeat_kv path (patched) and enable_gqa path (default) must give the same hidden states."""
    import transformers.integrations.sdpa_attention as sa
    from agentic_distill.train_lora import disable_sdpa_gqa
    model = _tiny()
    model.config._attn_implementation = "sdpa"
    ids, _ = _batch()
    orig = sa.use_gqa_in_sdpa
    calls = []
    sa.use_gqa_in_sdpa = lambda m, k, v: (calls.append(1), orig(m, k, v))[1]
    try:
        with torch.no_grad():
            ref = model.model(input_ids=ids).last_hidden_state
        assert any(calls), "default path never consulted use_gqa_in_sdpa"
        disable_sdpa_gqa()
        with torch.no_grad():
            patched = model.model(input_ids=ids).last_hidden_state
    finally:
        sa.use_gqa_in_sdpa = orig
    torch.testing.assert_close(patched, ref, rtol=1e-5, atol=1e-6)
