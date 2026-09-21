"""exp05 — AMC: train a 1-D ResNet on the IQ dataset, evaluate.

Produces a confusion matrix (high-SNR eval set) and an accuracy-vs-SNR
curve. CPU-sized: 5 classes x 400 train + 150 test samples.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from _common import save

from phylayer.amc import AMCNet, accuracy_vs_snr, confusion_matrix, predict, train_model
from phylayer.iqdata import CLASSES, generate_dataset

N_TRAIN = 800   # per class
N_TEST = 150
N_SAMPLES = 1024
SNR_TRAIN = (-4.0, 24.0)
EPOCHS = 18


def main():
    t0 = time.time()
    Xtr, ytr, _ = generate_dataset(CLASSES, N_TRAIN, N_SAMPLES, SNR_TRAIN, seed=11)
    Xte, yte, ste = generate_dataset(CLASSES, N_TEST, N_SAMPLES, SNR_TRAIN, seed=99)
    print(f"dataset {Xtr.shape[0]} train / {Xte.shape[0]} test "
          f"({time.time() - t0:.0f}s)")

    model = AMCNet(n_classes=len(CLASSES))
    losses = train_model(model, Xtr, ytr, epochs=EPOCHS)
    print("loss curve:", np.round(losses, 3))

    pred = predict(model, Xte)
    acc = float((pred == yte).mean())
    print(f"test accuracy (SNR -4..24 mixed): {acc:.3f}")

    cm = confusion_matrix(yte, pred, len(CLASSES))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    im = axes[0].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues",
                      vmin=0, vmax=1)
    axes[0].set_xticks(range(len(CLASSES)), CLASSES)
    axes[0].set_yticks(range(len(CLASSES)), CLASSES)
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true")
    axes[0].set_title(f"confusion (acc={acc:.2f})")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            axes[0].text(j, i, cm[i, j], ha="center",
                         va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    edges = np.arange(-4, 26, 4)
    ctr, ac = accuracy_vs_snr(yte, pred, ste, edges)
    axes[1].plot(ctr, ac, "o-")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("accuracy vs SNR")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "amc_results.png")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
