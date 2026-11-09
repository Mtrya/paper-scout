"""Check the closed-form last-readout spectrum against the full autograd Jacobian."""
import json

import torch

from models import TTTLayer


def main():
    torch.manual_seed(7)
    layer = TTTLayer(d=4, hidden=3, fast_update="last").double()
    W = {k: v.detach().clone() for k, v in layer.init_state(1).items()}
    W["W2"] = W["W2"].requires_grad_(True)
    k = torch.randn(1, 4, dtype=torch.float64)
    v = torch.randn(1, 4, dtype=torch.float64)
    out = layer._inner_update(W, k, v, create_graph=True)[0]["W2"].reshape(-1)
    rows = [torch.autograd.grad(out[i], W["W2"], retain_graph=True)[0].reshape(-1)
            for i in range(out.numel())]
    J = torch.stack(rows)
    phi = torch.nn.functional.gelu(k @ W["W1"][0] + W["b1"][0])
    radial = 1 - (2 / layer.d) * phi.square().sum() * layer.eta
    expected = max(1.0, radial.abs().max().item())
    actual = torch.linalg.svdvals(J)[0].item()
    if abs(actual - expected) >= 1e-10:
        raise RuntimeError({"actual": actual, "expected": expected})
    print(json.dumps({"autograd_operator_norm": actual,
                      "closed_form_operator_norm": expected,
                      "absolute_error": abs(actual - expected)}, indent=2))


if __name__ == "__main__":
    main()
