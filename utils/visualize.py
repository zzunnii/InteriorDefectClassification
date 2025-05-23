import os
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """혼동행렬 시각화"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("예측값")
    plt.ylabel("실제값")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_learning_curves(history, save_path=None):
    """학습 곡선 시각화"""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 5))

    # Loss 그래프
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train")
    plt.plot(epochs, history["val_loss"], "r-", label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy 그래프
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], "b-", label="Train")
    plt.plot(epochs, history["val_acc"], "r-", label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # F1 Score 그래프
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["val_f1"], "g-")
    plt.title("Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_class_accuracy(class_accuracy, save_path=None, title="Class-wise Accuracy"):
    """클래스별 정확도 시각화"""
    classes = list(class_accuracy.keys())
    accuracies = list(class_accuracy.values())

    # 정확도에 따라 정렬
    sorted_idx = np.argsort(accuracies)
    classes = [classes[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]

    plt.figure(figsize=(12, 10))
    plt.barh(classes, accuracies, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title(title)
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)

    # 정확도 값 표시
    for i, v in enumerate(accuracies):
        plt.text(v + 0.01, i, f"{v:.3f}", va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_models_comparison(results, metric='accuracy', save_path=None):
    """모델별 성능 비교 시각화"""
    models = list(results.keys())

    if metric == 'accuracy':
        values = [results[m]["test_acc"] for m in models]
        title = "Models Accuracy Comparison"
        ylabel = "Accuracy"
    elif metric == 'f1':
        values = [results[m]["test_f1"] for m in models]
        title = "Models F1 Score Comparison"
        ylabel = "F1 Score"
    elif metric == 'time':
        values = [results[m]["training_time"] / 60 for m in models]
        title = "Models Training Time Comparison"
        ylabel = "Time (min)"

    plt.figure(figsize=(10, 6))
    plt.bar(models, values, color='lightgreen')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.3)

    # 값 표시
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()