#!/usr/bin/env python
# coding: utf-8

# # ARTI 502 — Deep Learning: 1st Assignment
# ## Training-Testing Neural Net with MNIST Dataset using PyTorch
# **Imam Abdulrahman Bin Faisal University — College of Computer Science & IT — Year 2026, Term 1**
# 
# | # | Member Name | Academic ID |
# |---|---|---|
# | 1 |Reena Aloraini |2220002781 |
# | 2 |Maria Aljafri  |2230009064 |
# | 3 |Wala Almukhtar |2230003613 |
# | 4 |Rawan Alzaid   |2230003194 |
# | 5 |Dima Alrahal   |2230009044 |
# | 6 |Sara Alsahli   |2220003277 |  


# ## Task 1: Install PyTorch
# Install PyTorch (and torchvision for the MNIST dataset), then import it into the code.
# This notebook is saved as **`Assigment-1 DL`**.

# In[ ]:


get_ipython().run_line_magic('pip', 'install torch torchvision matplotlib --quiet')


# In[ ]:


# Load PyTorch into the Python source code
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
import time

print("PyTorch version     :", torch.__version__)
print("Torchvision version :", torchvision.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device              :", device)

torch.manual_seed(42)   # reproducible results
import random
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ## Task 2: Download and Load the Dataset
# MNIST: 28×28 grayscale handwritten digits 0–9. torchvision downloads it directly from code.
# 
# *Note:* torchvision's MNIST training file has 60,000 images. To match the assignment's
# **50,000 training / 10,000 test** description, we split the 60,000 into 50,000 for training
# and 10,000 for validation, and keep the official 10,000 test images for testing.

# In[ ]:


# Transform: convert image to tensor [0,1] and normalize with MNIST mean/std
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download directly from Python code
full_train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set   = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 50,000 train / 10,000 validation split
train_set, val_set = random_split(full_train, [50000, 10000],
                                  generator=torch.Generator().manual_seed(42))

BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=1000, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)

print("Dataset loaded successfully!")
print(f"Training images   : {len(train_set)}")
print(f"Validation images : {len(val_set)}")
print(f"Test images       : {len(test_set)}")
print("Classes           :", full_train.classes)


# ## Task 3: Inspect the Dataset
# ### 3.1 Images tensor size

# In[ ]:


# Take one mini-batch from the dataloader
images, labels = next(iter(train_loader))

print("Images tensor size :", images.size())    # [N, C, H, W] = [32, 1, 28, 28]
print("Labels tensor size :", labels.size())    # [32]
print("Data type          :", images.dtype)
print("Pixel min / max    :", images.min().item(), "/", images.max().item())
print("First 10 labels    :", labels[:10].tolist())


# ### 3.2 Data images visualization using matplotlib

# In[ ]:


def show_images(imgs, lbls, n=16, title="MNIST samples"):
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 4))
    for i, ax in enumerate(axes.flat):
        img = imgs[i].squeeze() * 0.3081 + 0.1307   # undo normalization for display
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Label: {lbls[i].item()}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

show_images(images, labels)


# In[ ]:


# Single image in detail + class distribution of the training set
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow((images[0].squeeze() * 0.3081 + 0.1307).clamp(0, 1), cmap="gray")
ax[0].set_title(f"One image (28x28), label = {labels[0].item()}")
ax[0].axis("off")

train_targets = full_train.targets[train_set.indices]
counts = torch.bincount(train_targets, minlength=10)
ax[1].bar(range(10), counts.numpy())
ax[1].set_xticks(range(10))
ax[1].set_xlabel("Digit"); ax[1].set_ylabel("Count")
ax[1].set_title("Training set class distribution")
plt.tight_layout(); plt.show()


# ## Task 4: Creating A Simple Network
# One hidden layer: `Linear(784 → hidden)` → **ReLU** → `Linear(hidden → 10)`.
# `nn.Linear` expects a 2D input (N × I), so the (N × C × H × W) batch is reshaped with
# `torch.flatten`: **(32 × 1 × 28 × 28) → (32 × 784)**.

# In[ ]:


class SimpleNet(nn.Module):
    def __init__(self, num_inputs=784, num_hidden=128, num_outputs=10):
        super().__init__()
        self.fc1  = nn.Linear(num_inputs, num_hidden)    # input  -> hidden
        self.relu = nn.ReLU()                            # nonlinearity
        self.fc2  = nn.Linear(num_hidden, num_outputs)   # hidden -> output

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)   # (N,1,28,28) -> (N,784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)                     # raw scores (logits) for 10 classes
        return x

net = SimpleNet().to(device)
print(net)
print("Trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))


# In[ ]:


# Verify the code against the requirement
x = images.to(device)
print("Input batch shape  :", tuple(x.shape))                           # (32, 1, 28, 28)
print("After flatten      :", tuple(torch.flatten(x, start_dim=1).shape)) # (32, 784)
print("Network output     :", tuple(net(x).shape))                       # (32, 10)

assert torch.flatten(x, start_dim=1).shape == (32, 784)
assert net(x).shape == (32, 10)
print("Requirement verified: (32 x 1 x 28 x 28) -> (32 x 784) -> (32 x 10)")


# ## Task 5: Training Function
# - Inputs: the network and the training dataloader, plus optional parameters
#   (`epochs`, `lr`, `momentum`, `print_every`, `verbose`).
# - Loss: `nn.CrossEntropyLoss` (softmax + negative log-likelihood).
# - Optimizer: SGD with learning rate and momentum.
# - Prints epoch, batch number, current loss and accuracy; returns all mini-batch losses
#   plus per-epoch loss and accuracy.

# In[ ]:


def train(net, dataloader, epochs=5, lr=0.01, momentum=0.9,
          print_every=200, verbose=True):
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    batch_losses = []                  # loss of every mini-batch
    epoch_losses, epoch_accs = [], []  # averages per epoch

    for epoch in range(1, epochs + 1):
        running_loss, running_correct, running_total = 0.0, 0, 0
        epoch_loss_sum, epoch_correct, epoch_total = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()              # 1. reset gradients
            outputs = net(inputs)              # 2. forward pass
            loss = criterion(outputs, targets) # 3. compute loss
            loss.backward()                    # 4. backpropagation
            optimizer.step()                   # 5. update weights

            batch_losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            correct = (preds == targets).sum().item()

            running_loss += loss.item(); running_correct += correct
            running_total += targets.size(0)
            epoch_loss_sum += loss.item() * targets.size(0)
            epoch_correct += correct; epoch_total += targets.size(0)

            if verbose and batch_idx % print_every == 0:
                print(f"Epoch [{epoch}/{epochs}]  Batch [{batch_idx}/{len(dataloader)}]  "
                      f"Loss: {running_loss / print_every:.4f}  "
                      f"Accuracy: {100 * running_correct / running_total:.2f}%")
                running_loss, running_correct, running_total = 0.0, 0, 0

        epoch_losses.append(epoch_loss_sum / epoch_total)
        epoch_accs.append(100 * epoch_correct / epoch_total)
        if verbose:
            print(f"==> Epoch {epoch} finished | Avg loss: {epoch_losses[-1]:.4f} | "
                  f"Accuracy: {epoch_accs[-1]:.2f}%\n")

    return batch_losses, epoch_losses, epoch_accs


# In[ ]:


def evaluate(net, dataloader):
    """Returns (average loss, accuracy %) on a dataloader without updating weights."""
    net.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, targets).item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return total_loss / total, 100 * correct / total

def train_with_validation(model, train_dl, validation_dl, epochs, lr, momentum, verbose=False):
    """Train while recording comparable train/validation curves per epoch."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)
        vloss, vacc = evaluate(model, validation_dl)
        history["train_loss"].append(loss_sum / total)
        history["train_acc"].append(100 * correct / total)
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: train acc {history['train_acc'][-1]:.2f}% | val acc {vacc:.2f}%")
    return history, time.perf_counter() - start


# ## Task 6: Training Network
# ### 6.1 Instantiate a new network and train it

# In[ ]:


EPOCHS, LR, MOMENTUM = 5, 0.01, 0.9

set_seed(42)
net = SimpleNet(num_hidden=128).to(device)
start = time.time()
batch_losses, epoch_losses, epoch_accs = train(net, train_loader,
                                               epochs=EPOCHS, lr=LR, momentum=MOMENTUM,
                                               print_every=200)
print(f"Training time: {time.time() - start:.1f} s")

val_loss, val_acc = evaluate(net, val_loader)
print(f"Validation  -> loss: {val_loss:.4f} | accuracy: {val_acc:.2f}%")


# ### 6.2 Graphs of the overall training process

# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))

ax[0].plot(batch_losses, alpha=0.4, label="mini-batch loss")
w = 100
smooth = np.convolve(batch_losses, np.ones(w) / w, mode="valid")
ax[0].plot(range(w - 1, len(batch_losses)), smooth, label=f"moving avg ({w})")
ax[0].set_title("Loss of every mini-batch"); ax[0].set_xlabel("Batch"); ax[0].set_ylabel("Loss")
ax[0].legend()

ep = range(1, EPOCHS + 1)
ax[1].plot(ep, epoch_losses, "o-"); ax[1].set_title("Loss per epoch")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss"); ax[1].set_xticks(list(ep))

ax[2].plot(ep, epoch_accs, "o-", color="green"); ax[2].set_title("Accuracy per epoch")
ax[2].set_xlabel("Epoch"); ax[2].set_ylabel("Accuracy (%)"); ax[2].set_xticks(list(ep))

plt.tight_layout(); plt.show()


# ## Task 7: Testing Network
# ### 7.1 Test on a few test images

# In[ ]:


net.eval()
test_images, test_labels = next(iter(DataLoader(test_set, batch_size=10, shuffle=True)))
with torch.no_grad():
    outputs = net(test_images.to(device))
    probs = torch.softmax(outputs, dim=1)
    preds = probs.argmax(dim=1).cpu()

print("True labels     :", test_labels.tolist())
print("Predicted labels:", preds.tolist())

fig, axes = plt.subplots(1, 10, figsize=(18, 2.8))
for i, ax in enumerate(axes):
    ax.imshow(test_images[i].squeeze() * 0.3081 + 0.1307, cmap="gray")
    ok = preds[i] == test_labels[i]
    ax.set_title(f"T:{test_labels[i].item()} P:{preds[i].item()}\n{probs[i].max().item():.0%}",
                 color="green" if ok else "red")
    ax.axis("off")
plt.suptitle("Test images (T = true, P = predicted, % = confidence)")
plt.tight_layout(); plt.show()


# **Note on evaluation:** The test set is held out until final model selection. We use the validation set for the baseline diagnostics and all hyperparameter decisions.

# ### 7.2 Performance on the full 10,000-image test set

# ## Task 8: Tuning the Parameters and Model Architecture
# Each configuration is trained on the training set and compared on the **validation set**
# (the test set is kept for the final check only). Fewer epochs are used per run to keep the
# search fast; increase `TUNE_EPOCHS` if you have time/GPU.
# 
# ### 8.1 Tuning learning rate and momentum

# In[ ]:


TUNE_EPOCHS = 3
learning_rates = [0.001, 0.01, 0.05, 0.1]
momentums      = [0.0, 0.5, 0.9]

lr_results = []
for lr in learning_rates:
    for m in momentums:
        set_seed(42)
        model = SimpleNet(num_hidden=128).to(device)
        start = time.perf_counter()
        _, tr_losses, tr_accs = train(model, train_loader, epochs=TUNE_EPOCHS,
                                      lr=lr, momentum=m, verbose=False)
        v_loss, v_acc = evaluate(model, val_loader)
        lr_results.append({"lr": lr, "momentum": m, "train_loss": tr_losses[-1],
                           "val_loss": v_loss, "val_acc": v_acc,
                           "seconds": time.perf_counter() - start})
        print(f"lr={lr:<6} momentum={m:<4} | train loss {tr_losses[-1]:.4f} | "
              f"val loss {v_loss:.4f} | val acc {v_acc:.2f}%")


# In[ ]:


import pandas as pd
lr_df = pd.DataFrame(lr_results).sort_values("val_acc", ascending=False)
display(lr_df.style.format({"train_loss": "{:.4f}", "val_loss": "{:.4f}", "val_acc": "{:.2f}"})
        .highlight_max(subset=["val_acc"], color="lightgreen")
        .highlight_min(subset=["val_loss"], color="lightblue"))

best_lr_row = lr_df.iloc[0]
BEST_LR, BEST_MOMENTUM = float(best_lr_row["lr"]), float(best_lr_row["momentum"])
print(f"Best: lr = {BEST_LR}, momentum = {BEST_MOMENTUM} -> "
      f"val loss {best_lr_row['val_loss']:.4f}, val acc {best_lr_row['val_acc']:.2f}%")

# Heatmap of validation accuracy
pivot = lr_df.pivot(index="lr", columns="momentum", values="val_acc")
plt.figure(figsize=(6, 4))
plt.imshow(pivot.values, cmap="viridis")
plt.colorbar(label="Validation accuracy (%)")
plt.xticks(range(len(pivot.columns)), pivot.columns); plt.xlabel("Momentum")
plt.yticks(range(len(pivot.index)), pivot.index);     plt.ylabel("Learning rate")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        plt.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", color="w")
plt.title("Learning rate vs momentum"); plt.show()


# ### 8.2 Tuning the architecture (number of hidden layers and neurons)
# `FlexibleNet` builds any number of hidden layers, each with its own number of neurons,
# using ReLU after every hidden layer. The best learning rate and momentum from 8.1 are used.

# In[ ]:


class FlexibleNet(nn.Module):
    def __init__(self, hidden_sizes=(128,), num_inputs=784, num_outputs=10):
        super().__init__()
        layers, prev = [], num_inputs
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(torch.flatten(x, start_dim=1))

architectures = [
    (32,), (64,), (128,), (256,), (512,),          # 1 hidden layer, varying neurons
    (128, 64), (256, 128), (512, 256),             # 2 hidden layers
    (256, 128, 64), (512, 256, 128),               # 3 hidden layers
    (512, 256, 128, 64),                           # 4 hidden layers
]

arch_results = []
for hs in architectures:
    set_seed(42)
    model = FlexibleNet(hs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    start = time.perf_counter()
    _, tr_losses, _ = train(model, train_loader, epochs=TUNE_EPOCHS,
                            lr=BEST_LR, momentum=BEST_MOMENTUM, verbose=False)
    v_loss, v_acc = evaluate(model, val_loader)
    arch_results.append({"hidden_layers": len(hs), "neurons": hs, "params": n_params,
                         "train_loss": tr_losses[-1], "val_loss": v_loss, "val_acc": v_acc,
                         "seconds": time.perf_counter() - start})
    print(f"{len(hs)} layer(s) {str(hs):<22} params {n_params:>7} | "
          f"val loss {v_loss:.4f} | val acc {v_acc:.2f}%")


# In[ ]:


arch_df = pd.DataFrame(arch_results).sort_values("val_acc", ascending=False)
display(arch_df.style.format({"train_loss": "{:.4f}", "val_loss": "{:.4f}", "val_acc": "{:.2f}"})
        .highlight_max(subset=["val_acc"], color="lightgreen")
        .highlight_min(subset=["val_loss"], color="lightblue"))

BEST_ARCH = tuple(arch_df.iloc[0]["neurons"])
print("Best architecture (hidden sizes):", BEST_ARCH)

plt.figure(figsize=(10, 4))
plt.bar(arch_df["neurons"].astype(str), arch_df["val_acc"])
plt.ylim(arch_df["val_acc"].min() - 0.5, 100)
plt.xticks(rotation=45, ha="right"); plt.ylabel("Validation accuracy (%)")
plt.title("Architecture comparison"); plt.tight_layout(); plt.show()


# ### 8.3 Controlled architecture experiments
# 
# To distinguish **width** from **depth**, compare single-hidden-layer networks of different widths, then compare networks with the same width (128 units) but different depths. Parameter count still grows with depth, so this does **not** hold total capacity constant. All configurations use the same number of epochs and the selected learning rate/momentum.

# In[ ]:


# Width-only results from the architecture sweep above
width_df = arch_df[arch_df["hidden_layers"] == 1].sort_values("params").copy()
display(width_df[["neurons", "params", "val_acc", "seconds"]])

# Fixed-width depth comparison; reuse (128,) result from original sweep
fixed_width_archs = [(128, 128), (128, 128, 128), (128, 128, 128, 128)]
depth_rows = [dict(r) for _, r in arch_df.iterrows() if r["neurons"] == (128,)]
for hs in fixed_width_archs:
    set_seed(42)
    model = FlexibleNet(hs).to(device)
    start = time.perf_counter()
    _, losses, _ = train(model, train_loader, epochs=TUNE_EPOCHS,
                         lr=BEST_LR, momentum=BEST_MOMENTUM, verbose=False)
    vloss, vacc = evaluate(model, val_loader)
    depth_rows.append({"hidden_layers": len(hs), "neurons": hs,
                       "params": sum(p.numel() for p in model.parameters()),
                       "train_loss": losses[-1], "val_loss": vloss,
                       "val_acc": vacc, "seconds": time.perf_counter()-start})
depth_df = pd.DataFrame(depth_rows).sort_values("hidden_layers")
display(depth_df)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(width_df["params"], width_df["val_acc"], "o-")
ax[0].set(xlabel="Trainable parameters", ylabel="Validation accuracy (%)", title="Increasing width (1 hidden layer)")
ax[1].plot(depth_df["hidden_layers"], depth_df["val_acc"], "o-")
ax[1].set(xlabel="Hidden layers (128 neurons each)", ylabel="Validation accuracy (%)", title="Increasing depth (fixed width)")
ax[1].set_xticks(depth_df["hidden_layers"])
plt.tight_layout(); plt.show()


# ### 8.3 Final model with the best parameters and architecture

# In[ ]:


# Fair comparison: both baseline and selected model train for the SAME number of epochs.
FINAL_EPOCHS = 10
set_seed(42)
baseline_final = SimpleNet(num_hidden=128).to(device)
baseline_history, baseline_seconds = train_with_validation(
    baseline_final, train_loader, val_loader, FINAL_EPOCHS, LR, MOMENTUM, verbose=True)

set_seed(42)
best_net = FlexibleNet(BEST_ARCH).to(device)
best_history, best_seconds = train_with_validation(
    best_net, train_loader, val_loader, FINAL_EPOCHS,
    BEST_LR, BEST_MOMENTUM, verbose=True)

# Only now, after selection, evaluate once on the held-out test set.
baseline_test_loss, baseline_test_acc = evaluate(baseline_final, test_loader)
best_test_loss, best_test_acc = evaluate(best_net, test_loader)
comparison = pd.DataFrame([
    {"model": "Baseline", "architecture": "(128,)", "lr": LR, "momentum": MOMENTUM,
     "params": sum(p.numel() for p in baseline_final.parameters()),
     "seconds": baseline_seconds, "val_acc": baseline_history["val_acc"][-1],
     "test_loss": baseline_test_loss, "test_acc": baseline_test_acc},
    {"model": "Tuned", "architecture": str(BEST_ARCH), "lr": BEST_LR, "momentum": BEST_MOMENTUM,
     "params": sum(p.numel() for p in best_net.parameters()),
     "seconds": best_seconds, "val_acc": best_history["val_acc"][-1],
     "test_loss": best_test_loss, "test_acc": best_test_acc}
])
display(comparison)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for label, hist in [("Baseline", baseline_history), ("Tuned", best_history)]:
    for split, linestyle in [("train", "-"), ("val", "--")]:
        ax[0].plot(range(1, FINAL_EPOCHS+1), hist[f"{split}_loss"], linestyle,
                   label=f"{label} {split}")
        ax[1].plot(range(1, FINAL_EPOCHS+1), hist[f"{split}_acc"], linestyle,
                   label=f"{label} {split}")
ax[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training vs validation loss")
ax[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Training vs validation accuracy")
for a in ax: a.legend(); a.grid(alpha=0.2)
plt.tight_layout(); plt.show()


# ### 8.4 Results discussion
# 
# **Learning rate and momentum.** The heatmap illustrates that these hyperparameters interact: momentum may accelerate convergence at smaller learning rates, but large learning rates combined with high momentum can destabilize optimization. Compare the actual best and worst combinations in the table; do not assume the relationship is monotonic.
# 
# **Width and depth.** The single-layer width experiment tests the effect of increasing neurons, while the fixed-width depth experiment tests adding layers. Inspect validation accuracy alongside parameter counts and runtime. Greater complexity does not guarantee better performance, and increasing depth also increases parameter count.
# 
# **Generalization.** Compare training and validation curves. A growing train–validation gap may suggest overfitting, especially if validation loss rises while training loss continues falling. A gap alone is not conclusive.
# 
# **Fair final comparison.** Both the baseline and tuned networks now train for 10 epochs, so their test results are more directly comparable. However, their architecture and optimizer settings still differ, so improvements cannot be assigned to any single change. Results vary with initialization and minibatch order; repeating the strongest configurations with multiple random seeds would strengthen the conclusions.
# 
# **After executing the notebook:** Add 2–3 sentences quoting the *newly generated* best validation configuration, the depth/width comparisons, and the final test accuracies. Previous run values are not automatically applicable to this revised notebook.

# ## Task 9: Save Your Source Code
# Save the trained model, export this notebook as a `.py` script, and push both to GitHub.

# In[ ]:


# Save trained model weights
torch.save(best_net.state_dict(), "best_mnist_model.pth")
print("Model saved to best_mnist_model.pth")

# Reload to confirm it was saved correctly
reloaded = FlexibleNet(BEST_ARCH).to(device)
reloaded.load_state_dict(torch.load("best_mnist_model.pth", map_location=device))
print("Reloaded model test accuracy: %.2f%%" % evaluate(reloaded, test_loader)[1])


# In[ ]:


