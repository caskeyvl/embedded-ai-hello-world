import os
import math

import torch
import executorch.exir as exir

class SineModel(torch.nn.Module):
    """
    Sine MLP approximation model: y = sin(x)

    Architecture: Linear(1, H) -> Tanh -> Linear(H, 1)
    
    Run from project root with: 
        source ~/executorch/et-env/bin/activate.fish
        python3 models/train_and_export.py
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(1, hidden_size)
        self.activation = torch.nn.Tanh()
        self.layer2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

def main():
    os.makedirs("models", exist_ok=True)

    hidden_size = 16
    model = SineModel(hidden_size)

    # Train
    N = 1000
    x_train = torch.linspace(0, 2 * math.pi, N).reshape(N, 1)
    y_train = torch.sin(x_train)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10000
    print("Training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        if(epoch % 200 == 0):
            print(f"  epoch {epoch}: loss = {loss.item():.6f}")

    print(f"Final loss: {loss.item():.6f}")

    # Export
    model.eval()

    # input shape needs to match what is fed at inference time
    example_input = (torch.randn(1,1),)

    print("\nExporting model...")

    ep = torch.export.export(model, example_input)

    # convert to ExecuTorch format
    et_program = (
            exir.to_edge(ep)
            .to_executorch()
    )

    output_path = "models/sine_model.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Exported to: {output_path}")
    print(f"Model size: {len(et_program.buffer)} bytes")

    # Print ops used
    print("\nOps in exported graph (use .out variants for SELECT_OPS_LIST):")
    seen = set()
    for node in exir.to_edge(ep).exported_program().graph.nodes:
        if node.op == 'call_function':
            name = getattr(node.target, '__name__', str(node.target))
            if name not in seen:
                seen.add(name)
                print(f"  {name}")


    # Host sanity check
    print("\nRunning host sanity check...")
    test_angles = torch.tensor([[0.0], [math.pi / 2], [math.pi], [3 * math.pi / 2]])
    with torch.no_grad():
        predicted = model(test_angles)
    expected = torch.sin(test_angles)

    for angle, pred, exp in zip(test_angles, predicted, expected):
        print(f"  angle={angle.item():.4f} predicted={pred.item():.4f} expected={exp.item():.4f}")

    max_error = (predicted - expected).abs().max().item()
    print(f"Max error: {max_error:.4f}")
    assert max_error < 0.05, "Model accuracy too low!"
    print("Sanity check passed!")


if __name__ == "__main__":
    main()
