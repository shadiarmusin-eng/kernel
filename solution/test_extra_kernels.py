import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ref_rmsnorm(x, w, eps):
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


@cuda
@pytest.mark.parametrize("xdt", [torch.float16, torch.float32])
def test_rmsnorm(xdt):
    from solution.rmsnorm import RMSNormFn
    torch.manual_seed(0)
    x = (torch.randn(3, 37, 1536, device="cuda") * 2).to(xdt).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w = torch.rand(1536, device="cuda") + 0.5
    dy = torch.randn(3, 37, 1536, device="cuda")
    y = RMSNormFn.apply(x, w, 1e-6)
    y_ref = ref_rmsnorm(x_ref, w, 1e-6)
    assert y.dtype == y_ref.dtype
    torch.testing.assert_close(y, y_ref, rtol=2e-3, atol=2e-3)
    y.backward(dy)
    y_ref.backward(dy)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-2, atol=3e-2)


@cuda
@pytest.mark.parametrize("ldt", [torch.float32, torch.float16])
def test_ce_and_entropy(ldt):
    from solution.fused_ce import cross_entropy_rows, fused_entropy
    torch.manual_seed(0)
    V = 151936
    logits = (torch.randn(2 * 19, V, device="cuda") * 3).to(ldt).requires_grad_()
    ref = logits.detach().clone().requires_grad_()
    labels = torch.randint(0, V, (2 * 19,), device="cuda")
    labels[::5] = -100
    rows = cross_entropy_rows(logits, labels)
    ref_rows = F.cross_entropy(ref.float(), labels, ignore_index=-100, reduction="none")
    torch.testing.assert_close(rows, ref_rows, rtol=1e-4, atol=1e-4)
    g = torch.rand_like(rows)
    (rows * g).sum().backward()
    (ref_rows * g).sum().backward()
    torch.testing.assert_close(logits.grad.float(), ref.grad.float(), rtol=1e-2, atol=1e-5)
    p = torch.log_softmax(logits.detach().float(), -1)
    ent_ref = -(p.exp() * p).sum(-1)
    torch.testing.assert_close(fused_entropy(logits.detach()), ent_ref, rtol=1e-4, atol=1e-4)   
