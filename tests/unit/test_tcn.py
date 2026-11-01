import pytest

from mixing_copilot.models.tcn import CausalTCN

torch = pytest.importorskip("torch", reason="install optional [tcn] dependencies for smoke test")


def test_tcn_output_prefix_is_unchanged_by_future_input():
    torch.manual_seed(7)
    model = CausalTCN(input_features=4, channels=8, levels=2, dropout=0).eval()
    first = torch.randn(2, 4, 12)
    second = first.clone()
    second[..., 8:] += 100
    with torch.no_grad():
        first_output = model(first)
        second_output = model(second)
    assert torch.allclose(first_output[..., :8], second_output[..., :8], atol=1e-6)
