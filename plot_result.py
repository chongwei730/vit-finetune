import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置根目录和参数
root_dir = 'output'
models = ['vit', 'swin']
schedulers = ['cosine', 'cosine_restart', 'plateau', 'random_plateau']
scheduler_colors = {
    'cosine': 'tab:blue',
    'cosine_restart': 'tab:orange',
    'plateau': 'tab:green',
    'random_plateau': 'tab:red'
}

for model in models:
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'{model.upper()} Training Summary', fontsize=16)

    annotations = []  # 存储注释文本

    for scheduler in schedulers:
        path = os.path.join(root_dir, scheduler, model, 'version_0', 'metrics.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)

        # 提取列
        steps = df.get('step')
        lr = df.get('lr')
        train_acc = df.get('train_acc_epoch')
        val_acc = df.get('val_acc')
        train_loss = df.get('train_loss_epoch')
        val_loss = df.get('val_loss')

        color = scheduler_colors.get(scheduler, None)

        def plot_valid(x, y, ax, label, style):
            if x is None or y is None:
                return
            valid = (~x.isna()) & (~y.isna())
            if valid.any():
                ax.plot(x[valid], y[valid], label=label, linestyle=style, color=color)

        # Plot all metrics
        plot_valid(steps, lr, axs[0], f'{scheduler}', style='-')
        plot_valid(steps, train_acc, axs[1], f'{scheduler} train', style='-')
        plot_valid(steps, val_acc, axs[1], f'{scheduler} val', style='--')
        plot_valid(steps, train_loss, axs[2], f'{scheduler} train', style='-')
        plot_valid(steps, val_loss, axs[2], f'{scheduler} val', style='--')

        # 添加最高 val_acc 注释
        if steps is not None and val_acc is not None:
            valid = (~steps.isna()) & (~val_acc.isna())
            if valid.any():
                valid_steps = steps[valid].astype(int)
                valid_val_acc = val_acc[valid].astype(float)
                max_idx = valid_val_acc.idxmax()
                max_acc = valid_val_acc.loc[max_idx]
                max_step = valid_steps.loc[max_idx]
                annotations.append(f"{scheduler}: {max_acc:.2%} @ {max_step}")

    # 把注释标到第一张图（Learning Rate）
    if annotations:
        annotation_text = "\n".join(annotations)
        axs[0].text(
            0.01, 0.98, annotation_text,
            transform=axs[0].transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    # 设置标题和标签
    axs[0].set_title('Learning Rate')
    axs[0].set_ylabel('LR')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Top-1 Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].set_title('Loss')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{model}_summary.png')
    plt.show()
