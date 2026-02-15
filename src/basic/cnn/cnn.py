import torch
import torch.nn as nn
import torch.nn.functional as F


# conv 1d
class Conv1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

        self.linear = nn.Linear(
            in_channels * kernel_size,
            out_channels,
            bias=bias
        )

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape

        if self.padding > 0:
            # (B, C, L+2p)
            x = F.pad(x, (self.padding, self.padding))

        out_l = (x.shape[2] - self.kernel_size) // self.stride + 1

        # (B, C, out_l, k)
        patches = x.unfold(2, self.kernel_size, self.stride)
        # (B, out_l, C*k)
        patches = patches.permute(0, 2, 1, 3).reshape(B, out_l, -1)

        # (B, out_l, out_channels)
        out = self.linear(patches)

        # (B, out_channels, out_l)
        return out.permute(0, 2, 1)

class MaxPool1D(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        # x: (B, C, L) -> (B, C, L//p)
        return x.unfold(2, self.pool_size, self.pool_size).amax(dim=-1)
    
class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, pool_size=2):
        super().__init__()

        self.conv = Conv1D(in_c, out_c, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = MaxPool1D(pool_size=pool_size)

    def forward(self, x):
        # x: (B, in_c, L)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class MultiLayerCNN1D(nn.Module):
    def __init__(self,
                 input_size=100,
                 in_channels=1,
                 conv_channels=[32, 64],
                 kernel_size=3,
                 padding=1,
                 pool_size=2,
                 fc_hidden=128,
                 num_classes=10):

        super().__init__()

        # Build conv blocks
        channels = [in_channels] + conv_channels
        self.blocks = nn.ModuleList([
            ConvBlock1D(channels[i], channels[i+1], kernel_size, padding, pool_size)
            for i in range(len(conv_channels))
        ])

        # Calculate output size after all conv+pool layers
        size = input_size
        for _ in conv_channels:
            size = (size + 2*padding - kernel_size + 1) // pool_size

        flattened = conv_channels[-1] * size

        # Fully connected layers
        self.fc1 = nn.Linear(flattened, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, in_channels, input_size)
        
        for block in self.blocks:
            x = block(x)

        # (B, flattened)
        x = x.view(x.size(0), -1)

        # (B, fc_hidden) -> (B, num_classes)
        return self.fc2(self.relu(self.fc1(x)))


# conv 2d
class Conv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

        # (out_channels, in_channels, kernel_size, kernel_size)
        self.linear = nn.Linear(
            in_channels * kernel_size * kernel_size,
            out_channels,
            bias=bias
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        if self.padding > 0:
            # (B, C, H+2p, W+2p)
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        out_h = (x.shape[2] - self.kernel_size) // self.stride + 1
        out_w = (x.shape[3] - self.kernel_size) // self.stride + 1

        # (B, C, out_h, out_w, k, k)
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # (B, C, out_h, out_w, k*k)
        patches = patches.contiguous().view(B, C, out_h, out_w, -1)
        # (B, out_h, out_w, C*k*k)
        patches = patches.permute(0,2,3,1,4).reshape(B, out_h, out_w, -1)

        # (B, out_h, out_w, out_channels)
        out = self.linear(patches)

        # (B, out_channels, out_h, out_w)
        out = out.permute(0,3,1,2)

        return out

class MaxPool2D(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        # x: (B, C, H, W)
        p = self.pool_size
        # (B, C, H//p, W//p, p, p)
        patches = x.unfold(2, p, p).unfold(3, p, p)
        # (B, C, H//p, W//p)
        return patches.amax(dim=(-1, -2))

class ConvBlock2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, pool_size=2):
        super().__init__()

        self.conv = Conv2D(in_c, out_c, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = MaxPool2D(pool_size=pool_size)

    def forward(self, x):
        # x: (B, in_c, H, W)
        # (B, out_c, H', W')
        x = self.conv(x)
        # (B, out_c, H', W')
        x = self.relu(x)
        # (B, out_c, H'//p, W'//p)
        x = self.pool(x)
        return x

class MultiLayerCNN2D(nn.Module):
    def __init__(self,
                 input_size=28,
                 in_channels=1,
                 conv_channels=[32, 64],
                 kernel_size=3,
                 padding=1,
                 pool_size=2,
                 fc_hidden=128,
                 num_classes=10):

        super().__init__()

        # Build conv blocks
        channels = [in_channels] + conv_channels
        self.blocks = nn.ModuleList([
            ConvBlock2D(channels[i], channels[i+1], kernel_size, padding, pool_size)
            for i in range(len(conv_channels))
        ])


        # Calculate output size after all conv+pool layers
        size = input_size
        for _ in conv_channels:
            size = (size + 2*padding - kernel_size + 1) // pool_size

        flattened = conv_channels[-1] * size * size

        # Fully connected layers
        self.fc1 = nn.Linear(flattened, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, in_channels, input_size, input_size)
        
        for block in self.blocks:
            # (B, C_i, H_i, W_i)
            x = block(x)

        # (B, flattened)
        x = x.view(x.size(0), -1)

        # (B, fc_hidden)
        x = self.relu(self.fc1(x))
        # (B, num_classes)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    print("="*50)
    print("Running CNN Tests")
    print("="*50)
    
    # Test 1: Conv2D matches PyTorch implementation
    print("\n[Test 1] Conv2D vs PyTorch Conv2d")
    print("-" * 50)
    x = torch.randn(2, 3, 8, 8)
    custom = Conv2D(3, 16, kernel_size=3, padding=1)
    pytorch = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    
    # Copy weights (reshape from linear to conv2d format)
    pytorch.weight.data = custom.linear.weight.data.view(16, 3, 3, 3).clone()
    pytorch.bias.data = custom.linear.bias.data.clone()
    
    diff = (custom(x) - pytorch(x)).abs().max().item()
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {custom(x).shape}")
    print(f"  Max diff: {diff:.8f} {'✓ PASS' if diff < 1e-5 else '✗ FAIL'}")
    
    # Test 2: MaxPool2D matches PyTorch implementation
    print("\n[Test 2] MaxPool2D vs PyTorch MaxPool2d")
    print("-" * 50)
    x = torch.randn(2, 16, 8, 8)
    custom = MaxPool2D(pool_size=2)  # FIX: was MaxPool
    pytorch = nn.MaxPool2d(kernel_size=2)
    
    diff = (custom(x) - pytorch(x)).abs().max().item()
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {custom(x).shape}")
    print(f"  Max diff: {diff:.8f} {'✓ PASS' if diff < 1e-5 else '✗ FAIL'}")
    
    # Test 3: MultiLayerCNN2D forward pass on MNIST-like data
    print("\n[Test 3] MultiLayerCNN2D on MNIST-like images")
    print("-" * 50)
    
    mnist_batch = torch.randn(4, 1, 28, 28)
    
    model = MultiLayerCNN2D(  # FIX: was MultiLayerCNN
        input_size=28,
        in_channels=1,
        conv_channels=[32, 64],
        kernel_size=3,
        padding=1,
        pool_size=2,
        fc_hidden=128,
        num_classes=10
    )
    
    logits = model(mnist_batch)
    probs = F.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    
    print(f"  Input shape:     {mnist_batch.shape}")
    print(f"  Output shape:    {logits.shape}")
    print(f"  Predictions:     {predictions.tolist()}")
    print(f"  First sample probs: {probs[0].detach().numpy().round(3)}")
    print(f"  {'✓ PASS' if logits.shape == (4, 10) else '✗ FAIL'}")
    
    # Test 4: Different configurations
    print("\n[Test 4] Different network depths")
    print("-" * 50)
    configs = [
        ([32], "Shallow (1 conv layer)"),
        ([32, 64], "Medium (2 conv layers)"),
        ([32, 64, 128], "Deep (3 conv layers)"),
    ]
    
    x = torch.randn(2, 1, 28, 28)
    for channels, desc in configs:
        model = MultiLayerCNN2D(conv_channels=channels)  # FIX: was MultiLayerCNN
        out = model(x)
        print(f"  {desc:25s} -> output: {out.shape} ✓")
    
    # Test 5: Conv1D
    print("\n[Test 5] Conv1D tests")
    print("-" * 50)
    x = torch.randn(4, 1, 100)  # (batch=4, channels=1, length=100)
    model = MultiLayerCNN1D(input_size=100, conv_channels=[32, 64])
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape} {'✓ PASS' if out.shape == (4, 10) else '✗ FAIL'}")
    
    print("\n" + "="*50)
    print("All tests complete!")
    print("="*50)
