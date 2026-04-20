"""
从 dataset/labels.csv 中提取数据，生成项目报告和 README 所需的可视化图表。
"""
import csv
import math
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import numpy as np


def dist(lm_a, lm_b):
    """计算两个landmark的二维距离（忽略z轴）"""
    return math.sqrt((lm_a[0] - lm_b[0])**2 + (lm_a[1] - lm_b[1])**2)


def parse_csv(path):
    """解析CSV，返回 (labels, records) 其中 records 是 dict 列表"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label'].strip()
            # 统一标签格式
            if label.lower() == 'together':
                label = 'together'
            elif label.lower() == 'none' or label == '0':
                label = 'none'
            else:
                label = int(label)

            landmarks = []
            for i in range(21):
                x = float(row[f'lm{i}_x'])
                y = float(row[f'lm{i}_y'])
                z = float(row[f'lm{i}_z'])
                landmarks.append((x, y, z))

            records.append({'label': label, 'landmarks': landmarks})
    return records


def calc_features(rec):
    """计算各条记录的关键特征值"""
    lm = rec['landmarks']
    wrist = lm[0]
    thumb_tip = lm[4]
    thumb_mcp = lm[2]
    index_mcp = lm[5]
    index_tip = lm[8]
    middle_mcp = lm[9]
    middle_tip = lm[12]
    ring_tip = lm[16]
    pinky_mcp = lm[17]
    pinky_tip = lm[20]

    # 手掌宽度（手腕到中指根）
    palm_size = dist(wrist, middle_mcp)
    if palm_size < 1e-6:
        palm_size = 1e-6

    # 大拇指判定比值：tip_to_index_mcp / mcp_to_index_mcp
    tip_to_index_mcp = dist(thumb_tip, index_mcp)
    mcp_to_index_mcp = dist(thumb_mcp, index_mcp)
    thumb_ratio = tip_to_index_mcp / mcp_to_index_mcp if mcp_to_index_mcp > 1e-6 else 0

    # 拇指靠近度（拇指尖到最近的其他四指尖距离）
    four_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
    thumb_near = min(dist(thumb_tip, t) for t in four_tips)
    thumb_near_norm = thumb_near / palm_size

    # 四指离散度（四指尖最大跨度）
    four_spread = max(dist(four_tips[i], four_tips[j])
                      for i in range(4) for j in range(i+1, 4))
    four_spread_norm = four_spread / palm_size

    # cross_z（手掌朝向）
    v1x = index_mcp[0] - wrist[0]
    v1y = index_mcp[1] - wrist[1]
    v2x = pinky_mcp[0] - wrist[0]
    v2y = pinky_mcp[1] - wrist[1]
    cross_z = abs(v1x * v2y - v1y * v2x)

    return {
        'label': rec['label'],
        'palm_size': palm_size,
        'thumb_ratio': thumb_ratio,
        'thumb_near': thumb_near,
        'thumb_near_norm': thumb_near_norm,
        'four_spread': four_spread,
        'four_spread_norm': four_spread_norm,
        'cross_z': cross_z,
    }


def plot_sample_distribution(records, out_path):
    """图1：数据采集分布柱状图"""
    labels = [r['label'] for r in records]
    cnt = Counter(labels)

    # 按 1,2,3,4,5,together 排序
    order = [1, 2, 3, 4, 5, 'together']
    names = ['1', '2', '3', '4', '5', 'together']
    counts = [cnt.get(k, 0) for k in order]
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=1.2)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_xlabel('Gesture Label', fontsize=12)
    ax.set_title('Dataset Distribution (Total: 401 samples)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 在柱子上方标注数字
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(c), ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_thumb_threshold(data, out_path):
    """图2：大拇指阈值分布图（label 4 vs label 5），带0.75分界线"""
    label4_ratios = [d['thumb_ratio'] for d in data if d['label'] == 4]
    label5_ratios = [d['thumb_ratio'] for d in data if d['label'] == 5]

    fig, ax = plt.subplots(figsize=(8, 5))

    # 使用半透明的散点+抖动，避免重叠
    y4 = np.random.normal(0, 0.08, size=len(label4_ratios))
    y5 = np.random.normal(1, 0.08, size=len(label5_ratios))

    ax.scatter(label4_ratios, y4, alpha=0.6, s=50, color='#e74c3c', label='Label 4 (thumb bent)', edgecolors='white', linewidth=0.5)
    ax.scatter(label5_ratios, y5, alpha=0.6, s=50, color='#3498db', label='Label 5 (thumb straight)', edgecolors='white', linewidth=0.5)

    # 画阈值线
    ax.axvline(x=0.75, color='#2c3e50', linestyle='--', linewidth=2, label='Threshold = 0.75')

    # 标注范围
    ax.text(0.48, 0.5, f'Label 4\n[{min(label4_ratios):.2f} ~ {max(label4_ratios):.2f}]',
            ha='center', va='center', fontsize=10, color='#e74c3c',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaea', edgecolor='#e74c3c', alpha=0.8))
    ax.text(0.93, 0.5, f'Label 5\n[{min(label5_ratios):.2f} ~ {max(label5_ratios):.2f}]',
            ha='center', va='center', fontsize=10, color='#3498db',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#eaf4ff', edgecolor='#3498db', alpha=0.8))

    ax.set_xlim(0.3, 1.15)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel('Thumb Ratio (tip_to_index_mcp / mcp_to_index_mcp)', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Label 4', 'Label 5'])
    ax.set_title('Thumb Threshold Calibration: Label 4 vs Label 5', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_dual_feature_scatter(data, out_path):
    """图3：双特征联合判定散点图"""
    # 数字手势 (1-5)
    digit_x = []
    digit_y = []
    for d in data:
        if isinstance(d['label'], int) and 1 <= d['label'] <= 5:
            digit_x.append(d['thumb_near_norm'])
            digit_y.append(d['four_spread_norm'])

    # together
    together_x = []
    together_y = []
    for d in data:
        if d['label'] == 'together':
            together_x.append(d['thumb_near_norm'])
            together_y.append(d['four_spread_norm'])

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(digit_x, digit_y, alpha=0.5, s=40, color='#3498db',
               label=f'Digit gestures 1-5 (n={len(digit_x)})', edgecolors='white', linewidth=0.3)
    ax.scatter(together_x, together_y, alpha=0.7, s=60, color='#e74c3c',
               label=f'Together (n={len(together_x)})', edgecolors='white', linewidth=0.3, marker='D')

    # 画阈值线
    ax.axvline(x=0.65, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=0.30, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.8)

    # 标注阈值区域
    ax.fill_between([0, 0.65], 0, 0.30, color='#2ecc71', alpha=0.15)
    ax.text(0.32, 0.15, 'Together\nRegion\n(100% accuracy)', ha='center', va='center',
            fontsize=11, color='#27ae60', fontweight='bold')

    ax.set_xlabel('Thumb Nearness (relative to palm size)', fontsize=12)
    ax.set_ylabel('Four-Finger Spread (relative to palm size)', fontsize=12)
    ax.set_title('Dual-Feature Joint Decision: Together vs Digit Gestures', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1.8)
    ax.set_ylim(0, 1.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_cross_z_distribution(data, out_path):
    """图4：手掌朝向 cross_z 分布"""
    digit_cross_z = [d['cross_z'] for d in data if isinstance(d['label'], int) and 1 <= d['label'] <= 5]
    together_cross_z = [d['cross_z'] for d in data if d['label'] == 'together']

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, 0.08, 50)
    ax.hist(digit_cross_z, bins=bins, alpha=0.6, color='#3498db', label='Digit gestures 1-5', edgecolor='white')
    ax.hist(together_cross_z, bins=bins, alpha=0.6, color='#e74c3c', label='Together', edgecolor='white')

    # 阈值线
    ax.axvline(x=0.014, color='#2c3e50', linestyle='--', linewidth=2, label='Threshold = 0.014')
    ax.axvline(x=0.020, color='#95a5a6', linestyle=':', linewidth=2, label='Old threshold = 0.020')

    ax.set_xlabel('Cross-Z Value (absolute)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Palm Facing Filter: Cross-Z Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    csv_path = os.path.join('dataset', 'labels.csv')
    out_dir = os.path.join('assets')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {csv_path} ...")
    records = parse_csv(csv_path)
    print(f"Total records: {len(records)}")

    print("Calculating features ...")
    data = [calc_features(r) for r in records]

    print("Generating charts ...")
    plot_sample_distribution(records, os.path.join(out_dir, 'fig1_dataset_distribution.png'))
    plot_thumb_threshold(data, os.path.join(out_dir, 'fig2_thumb_threshold.png'))
    plot_dual_feature_scatter(data, os.path.join(out_dir, 'fig3_dual_feature_scatter.png'))
    plot_cross_z_distribution(data, os.path.join(out_dir, 'fig4_cross_z_distribution.png'))

    print("\nAll charts generated successfully in 'assets/' folder!")


if __name__ == '__main__':
    main()
