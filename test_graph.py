import torch

# Load the graph
data = torch.load("data/processed/fraud_graph.pt")

# Print summary
print("Graph loaded successfully!")
print(data)
print(
    f"\nTrain fraud ratio: {(data.y[data.train_mask] == 1).sum().item() / data.train_mask.sum().item() * 100:.2f}%"
)
print(
    f"Test fraud ratio: {(data.y[data.test_mask] == 1).sum().item() / data.test_mask.sum().item() * 100:.2f}%"
)
