import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split


def save_eda_pairplot_like(path):
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        'Area': rng.normal(1500, 350, n),
        'Bedrooms': rng.integers(1, 6, n),
        'Age': rng.integers(0, 40, n),
    })
    df['Price'] = 70000 + df['Area'] * 180 + df['Bedrooms'] * 15000 - df['Age'] * 1200 + rng.normal(0, 30000, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    axes[0].hist(df['Price'], bins=22, color='#0ea5e9', edgecolor='white')
    axes[0].set_title('Price Distribution')
    axes[0].set_xlabel('Price')
    axes[0].set_ylabel('Count')

    scatter = axes[1].scatter(df['Area'], df['Price'], c=df['Bedrooms'], cmap='viridis', alpha=0.8)
    axes[1].set_title('Area vs Price (Colored by Bedrooms)')
    axes[1].set_xlabel('Area')
    axes[1].set_ylabel('Price')
    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label('Bedrooms')

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_confusion_matrix(path):
    X, y = make_classification(
        n_samples=700,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        class_sep=1.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=140)
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap='Blues', ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix (Logistic Regression)')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_roc_curve(path):
    X, y = make_classification(
        n_samples=700,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        class_sep=1.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=140)
    ax.plot(fpr, tpr, color='#16a34a', lw=2.2, label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random baseline')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_training_animation_gif(path):
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception:
        return

    rng = np.random.default_rng(7)
    epochs = np.arange(1, 41)
    train_loss = 1.2 * np.exp(-epochs / 12) + 0.05 * rng.random(len(epochs))
    val_loss = 1.35 * np.exp(-epochs / 11) + 0.08 * rng.random(len(epochs))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, max(val_loss) + 0.15)
    ax.set_title('AEDA Training Progress (Example)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    train_line, = ax.plot([], [], color='#0ea5e9', lw=2.5, label='Train Loss')
    val_line, = ax.plot([], [], color='#f97316', lw=2.5, label='Validation Loss')
    ax.legend()
    ax.grid(alpha=0.22)

    def init():
        train_line.set_data([], [])
        val_line.set_data([], [])
        return train_line, val_line

    def update(i):
        train_line.set_data(epochs[: i + 1], train_loss[: i + 1])
        val_line.set_data(epochs[: i + 1], val_loss[: i + 1])
        return train_line, val_line

    anim = FuncAnimation(fig, update, frames=len(epochs), init_func=init, blit=True, interval=55)
    anim.save(path, writer=PillowWriter(fps=16))
    plt.close(fig)


if __name__ == '__main__':
    save_eda_pairplot_like('docs/images/eda_overview.png')
    save_confusion_matrix('docs/images/confusion_matrix.png')
    save_roc_curve('docs/images/roc_curve.png')
    save_training_animation_gif('docs/images/training_animation.gif')
    print('Generated assets in docs/images')
