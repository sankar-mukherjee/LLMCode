# - **Standard Optimization Loop:** Write a training loop including `zero_grad()`, `backward()`, and `optimizer.step()`. 
# Explain the necessity of zeroing gradients and how the optimizer updates weights.


model.train()

for epoch in range(no_epochs):
    for x,y in dataloader:
        
        # forward pass
        preds = model(x)
        loss = criterian(x,y)

        # zero grad
        optimizer.zero_grad(set_to_none=True) # Memory Optimization Trick

        # backword pass
        loss.backword()

        # update parameter
        optimizer.step()


# Optimizers update weights by transforming gradients—optionally through momentum, adaptive scaling, 
# or normalization—into parameter-wise updates, applied in-place during optimizer.step() 
# using internal state accumulated across steps.

# What optimizer.step() Does Internally (PyTorch)
for param in params:
    g = param.grad
    state = optimizer.state[param]

    state = update_state(state, g)
    delta = compute_update(g, state)

    param.data += delta
