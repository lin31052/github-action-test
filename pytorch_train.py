"""
PyTorch CPU 版真实训练 + 可视化演示
在 GitHub Actions (Ubuntu/glibc) 上运行。

任务：多层感知机(MLP) 在合成二分类数据上学习非线性决策边界。
亮点：多层网络 + Dropout + batch 训练循环 + 每 epoch 记录 loss/acc + 可视化曲线。
数据：两个"月亮"形状重叠簇(带噪声) —— 线性不可分，正是深度的意义所在。
"""
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def make_moons(n=1500, noise=0.15, seed=42):
    """生成两弯月亮形状的合成数据（非线性可分二分类）"""
    rng = np.random.default_rng(seed)
    n_half = n // 2
    t = np.linspace(0, 2 * np.pi, n_half) + rng.uniform(-0.5, 0.5, n_half)
    # 上月亮
    x1 = np.cos(t).reshape(-1, 1)
    y1 = np.sin(t).reshape(-1, 1)
    # 下月亮（反转、下移）
    t2 = np.linspace(0, 2 * np.pi, n - n_half) + rng.uniform(-0.5, 0.5, n - n_half)
    x2 = 1 - np.cos(t2).reshape(-1, 1)
    y2 = 0.5 - np.sin(t2).reshape(-1, 1) - 0.5

    X = np.vstack([np.hstack([x1, y1]), np.hstack([x2, y2])])
    y = np.vstack([np.zeros((n_half, 1)), np.ones((n - n_half, 1))])
    X += rng.normal(0, noise, X.shape)
    # 打乱
    idx = rng.permutation(n)
    return X[idx].astype(np.float32), y[idx].astype(np.float32)


class MLP(nn.Module):
    """一个稍复杂的三隐层 MLP（带 Dropout）"""
    def __init__(self, in_dim=2, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)  # 输出 logit


def main():
    out_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(out_dir, exist_ok=True)

    print("===== PyTorch CPU 云端训练演示 =====")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"是否支持 CUDA: {torch.cuda.is_available()}（这里应为 False，纯 CPU）")
    print(f"CPU 核数: {os.cpu_count()}")

    # 数据
    X, y = make_moons()
    n = len(y)
    # 80/20 切分
    n_train = int(0.8 * n)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_te_t = torch.from_numpy(X_te)
    y_te_t = torch.from_numpy(y_te)

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()   # 二分类

    # 训练
    epochs = 200
    batch_size = 128
    train_losses, test_losses, test_accs = [], [], []

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        # 每个 epoch 打乱做 mini-batch
        perm = torch.randperm(len(X_tr_t))
        epoch_losses = []
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            logit = model(xb)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        # 每个 epoch 记录 train loss / test loss / test acc
        model.eval()
        with torch.no_grad():
            tr_loss = np.mean(epoch_losses)
            te_logit = model(X_te_t)
            te_loss = loss_fn(te_logit, y_te_t).item()
            prob = torch.sigmoid(te_logit)
            pred = (prob >= 0.5).float()
            acc = (pred == y_te_t).float().mean().item()
        train_losses.append(round(float(tr_loss), 4))
        test_losses.append(round(float(te_loss), 4))
        test_accs.append(round(float(acc), 4))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"epoch {epoch+1:>3}/{epochs}: train_loss={tr_loss:.4f} "
                  f"test_loss={te_loss:.4f} test_acc={acc:.4f}")

    elapsed = time.time() - t0
    final_acc = test_accs[-1]
    print(f"\n训练完成: {epochs} epochs, 耗时 {elapsed:.2f}s, 最终测试准确率 {final_acc:.4f}")

    # ---------- 可视化 ----------

    # 图1: loss & acc 学习曲线
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x_axis = range(1, epochs + 1)
    ax1.plot(x_axis, train_losses, label="train loss", color="#2f6f8f")
    ax1.plot(x_axis, test_losses, label="test loss", color="#d98e32")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCELoss")
    ax1.set_title("PyTorch MLP 训练曲线（CPU）")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close(fig)

    # 图2: 决策边界可视化 + 数据点
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    model.eval()
    with torch.no_grad():
        Z = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
    ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te[:, 0], cmap="coolwarm",
               edgecolors="k", s=15)
    ax.set_title(f"MLP 决策边界（CPU，测试准确率 {final_acc:.3f}）")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "decision_boundary.png"))
    plt.close(fig)

    print("已保存可视化: loss_curve.png, decision_boundary.png")

    # 图3: MLP 结构可视化(参数统计表即可)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")

    # ---------- 结构化输出 ----------
    result = {
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpus": os.cpu_count(),
        "n_samples": int(n),
        "n_train": int(n_train),
        "n_test": int(n - n_train),
        "epochs": epochs,
        "batch_size": batch_size,
        "train_time_s": round(float(elapsed), 3),
        "param_count": int(param_count),
        "final_test_accuracy": round(float(final_acc), 4),
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accs": test_accs,
    }
    with open(os.path.join(out_dir, "time_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("已保存 JSON: time_result.json")

    # ---------- Markdown 报告 ----------
    lines = []
    lines.append("# PyTorch CPU 云端训练报告")
    lines.append("")
    lines.append(f"- **PyTorch**: {torch.__version__} (CPU)")
    lines.append(f"- **CUDA**: {torch.cuda.is_available()}（应为 False，纯 CPU 训练）")
    lines.append(f"- **CPU 核数**: {os.cpu_count()}")
    lines.append(f"- **模型**: 3 隐层 MLP（每层 64 神经元 + Dropout），参数量 **{param_count:,}**")
    lines.append(f"- **数据**: 双月(moons)合成二分类，{n} 样本（80/20 切分）")
    lines.append(f"- **训练**: {epochs} epochs × batch {batch_size}，耗时 **{elapsed:.2f}s**")
    lines.append(f"- **最终测试准确率**: **{final_acc:.4f}**")
    lines.append("")
    lines.append("## 训练曲线")
    lines.append("")
    lines.append("![loss 曲线](loss_curve.png)")
    lines.append("")
    lines.append("## 决策边界")
    lines.append("")
    lines.append("![决策边界](decision_boundary.png)")
    lines.append("")
    lines.append("## 每 50 epoch 指标")
    lines.append("")
    lines.append("| epoch | train_loss | test_loss | test_acc |")
    lines.append("|---|---|---|---|")
    for e in range(0, epochs, 50):
        idx = min(e, epochs - 1)
        lines.append(f"| {idx+1} | {train_losses[idx]} | {test_losses[idx]} | {test_accs[idx]} |")
    lines.append("")
    lines.append("---")
    lines.append("> 原始数据见 `time_result.json`（含全部 200 个 epoch 的曲线）。")
    with open(os.path.join(out_dir, "RESULT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("已保存 Markdown: RESULT.md")
    print("==== 完成 ====")


if __name__ == "__main__":
    main()
