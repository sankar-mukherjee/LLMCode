# - **2D Convolution from Scratch:** Implement a 2D convolution kernel using NumPy. You must handle `stride` and `padding` logic manually using nested loops or `im2col`.
#     - 3.2 Convolution
#     - Implement a **naive 2D convolution** using tensor operations.
#     - Explain **im2col** and how it improves performance.
#     - How would you optimize this convolution for GPU?

# https://chatgpt.com/share/698459a2-6700-800f-94bb-98b606de7e2b

import torch
import torch.nn.functional as F

def conv2d_naive_torch(x, w, bias=None, stride=1, padding=0):
    """
    x: (N, C_in, H, W)
    w: (C_out, C_in, K_h, K_w)
    bias: (C_out,)
    """
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = w.shape
    S, P = stride, padding

    # Pad input
    x_pad = F.pad(x, (P, P, P, P))

    H_out = (H + 2*P - K_h) // S + 1
    W_out = (W + 2*P - K_w) // S + 1

    out = torch.zeros(
        (N, C_out, H_out, W_out),
        device=x.device,
        dtype=x.dtype
    )

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h = i * S
                    w_ = j * S
                    region = x_pad[n, :, h:h+K_h, w_:w_+K_w]
                    out[n, co, i, j] = (region * w[co]).sum()
                    if bias is not None:
                        out[n, co, i, j] += bias[co]

    return out


def im2col_torch(x, kernel_size, stride=1, padding=0):
    """
    x: (N, C, H, W)
    return: (N, C*K_h*K_w, H_out*W_out)
    """
    return F.unfold(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )


def conv2d_im2col_torch(x, w, bias=None, stride=1, padding=0):
    """
    x: (N, C_in, H, W)
    w: (C_out, C_in, K_h, K_w)
    """
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = w.shape

    x_col = F.unfold(
        x,
        kernel_size=(K_h, K_w),
        stride=stride,
        padding=padding
    )
    # x_col: (N, C_in*K_h*K_w, H_out*W_out)

    w_col = w.view(C_out, -1)  # (C_out, C_in*K_h*K_w)

    out = torch.matmul(w_col, x_col)  # (N, C_out, H_out*W_out)
    out = out.view(
        N,
        C_out,
        (H + 2*padding - K_h) // stride + 1,
        (W + 2*padding - K_w) // stride + 1
    )

    if bias is not None:
        out += bias.view(1, -1, 1, 1)

    return out


x = torch.randn(2, 3, 32, 32, requires_grad=True)
w = torch.randn(8, 3, 3, 3, requires_grad=True)

y = conv2d_im2col_torch(x, w, stride=1, padding=1)
loss = y.mean()
loss.backward()

print(x.grad.shape)  # ✅
print(w.grad.shape)  # ✅
