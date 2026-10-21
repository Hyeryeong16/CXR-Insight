
import torch
import torchxrayvision as xrv

device = "cuda" if torch.cuda.is_available() else "cpu"

model = xrv.models.DenseNet(
    weights="densenet121-res224-chex"
).to(device)

print("Device:", device)
print("Model loaded successfully!")
print("Pathologies:", model.pathologies)
