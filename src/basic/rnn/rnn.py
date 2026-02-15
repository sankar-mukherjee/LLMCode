import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.hidden_size = hidden_size

        # RNN weights
        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)

        # output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        
        for t in range(seq_len):
            xt = x[:, t, :]
            h = self.tanh(self.Wx(xt) + self.Wh(h))

        out = self.fc(h)
        return out


class MultiLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tanh = nn.Tanh()

        # one Wx + Wh per layer
        self.Wx = nn.ModuleList()
        self.Wh = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.Wx.append(nn.Linear(in_dim, hidden_size))
            self.Wh.append(nn.Linear(hidden_size, hidden_size))

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        B, T, _ = x.shape

        # hidden state for each layer
        h = [torch.zeros(B, self.hidden_size, device=x.device)
             for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t, :]

            for l in range(self.num_layers):
                h[l] = self.tanh(self.Wx[l](inp) + self.Wh[l](h[l]))
                inp = h[l]   # output becomes next layer input

        out = self.fc(h[-1])
        return out

# bi-directinal rnn
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()

        # forward direction
        self.Wx_f = nn.Linear(input_size, hidden_size)
        self.Wh_f = nn.Linear(hidden_size, hidden_size)

        # backward direction
        self.Wx_b = nn.Linear(input_size, hidden_size)
        self.Wh_b = nn.Linear(hidden_size, hidden_size)

        # concat → 2 * hidden
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        B, T, _ = x.shape

        hf = torch.zeros(B, self.hidden_size, device=x.device)
        hb = torch.zeros(B, self.hidden_size, device=x.device)

        # ---- forward pass ----
        for t in range(T):
            hf = self.tanh(self.Wx_f(x[:, t]) + self.Wh_f(hf))

        # ---- backward pass ----
        for t in reversed(range(T)):
            hb = self.tanh(self.Wx_b(x[:, t]) + self.Wh_b(hb))

        # concat both directions
        h = torch.cat([hf, hb], dim=-1)
        return self.fc(h)


# LSTM
class ManualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # one projection for all 4 gates (common interview pattern)
        self.Wx = nn.Linear(input_size, 4 * hidden_size)
        self.Wh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        # x: (batch, input_size)
        gates = self.Wx(x) + self.Wh(h)

        # split into 4 gates
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        g = torch.tanh(g)      # candidate

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super().__init__()
        self.cell = ManualLSTM(input_size, hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)
        c = torch.zeros_like(h)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return self.fc(h)


# GRU
class ManualGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Update and reset gates
        self.Wx_zr = nn.Linear(input_size, 2 * hidden_size)
        self.Wh_zr = nn.Linear(hidden_size, 2 * hidden_size)
        
        # Candidate
        self.Wx_n = nn.Linear(input_size, hidden_size)
        self.Wh_n = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        # x: (batch, input_size)
        
        # Compute update and reset gates
        zr = self.Wx_zr(x) + self.Wh_zr(h)
        z, r = zr.chunk(2, dim=-1)
        
        z = torch.sigmoid(z)  # update gate
        r = torch.sigmoid(r)  # reset gate
        
        # Compute candidate with reset gate applied
        n = torch.tanh(self.Wx_n(x) + self.Wh_n(r * h))
        
        # Update hidden state
        h_next = (1 - z) * n + z * h
        return h_next

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super().__init__()
        self.cell = ManualGRU(input_size, hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)

        for t in range(T):
            h = self.cell(x[:, t], h)

        return self.fc(h)



# Replace Test 6 with this simpler version:

if __name__ == "__main__":
    print("="*50)
    print("Running RNN Tests")
    print("="*50)
    
    # Test config
    B, T, D = 4, 10, 8  # batch, seq_len, input_size
    hidden = 16
    num_classes = 5
    
    # Test 1: SimpleRNN
    print("\n[Test 1] SimpleRNN")
    print("-" * 50)
    x = torch.randn(B, T, D)
    model = SimpleRNN(D, hidden, num_classes)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  {'✓ PASS' if out.shape == (B, num_classes) else '✗ FAIL'}")
    
    # Test 2: MultiLayerRNN
    print("\n[Test 2] MultiLayerRNN")
    print("-" * 50)
    model = MultiLayerRNN(D, hidden, num_classes, num_layers=3)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  {'✓ PASS' if out.shape == (B, num_classes) else '✗ FAIL'}")
    
    # Test 3: BiRNN
    print("\n[Test 3] BiRNN")
    print("-" * 50)
    model = BiRNN(D, hidden, num_classes)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  {'✓ PASS' if out.shape == (B, num_classes) else '✗ FAIL'}")
    
    # Test 4: LSTM
    print("\n[Test 4] LSTMModel")
    print("-" * 50)
    model = LSTMModel(D, hidden, num_classes)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  {'✓ PASS' if out.shape == (B, num_classes) else '✗ FAIL'}")
    
    # Test 5: GRU
    print("\n[Test 5] GRUModel")
    print("-" * 50)
    model = GRUModel(D, hidden, num_classes)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  {'✓ PASS' if out.shape == (B, num_classes) else '✗ FAIL'}")
    
    # Test 6: Verify gradients flow
    print("\n[Test 6] Gradient flow check")
    print("-" * 50)
    x = torch.randn(B, T, D, requires_grad=True)
    target = torch.randint(0, num_classes, (B,))
    
    models = [
        ("SimpleRNN", SimpleRNN(D, hidden, num_classes)),
        ("LSTM", LSTMModel(D, hidden, num_classes)),
        ("GRU", GRUModel(D, hidden, num_classes))
    ]
    
    for name, model in models:
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, target)
        loss.backward()
        has_grad = x.grad is not None and x.grad.abs().sum() > 0
        print(f"  {name:12s} -> grad flows: {'✓' if has_grad else '✗'}")
        x.grad = None  # Reset for next model
    
    print("\n" + "="*50)
    print("All tests complete!")
    print("="*50)
