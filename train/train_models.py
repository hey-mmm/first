"""
训练车牌字符识别模型
复制 C++ 预处理管线，生成 OpenCV XML 格式的模型文件。
输出：
  1_hu_model.xml, 1_pca_model.xml, 1_svm_model.xml  (中文)
  2_hu_model.xml, 2_pca_model.xml, 2_svm_model.xml  (英文/数字)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).parent / "training_data"
OUTPUT_DIR = Path(__file__).parent.parent / "build"

# 与 C++ 完全一致的 HOG 参数
HOG_WIN_SIZE = (48, 48)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9

HOG_DIM = ((HOG_WIN_SIZE[0] - HOG_BLOCK_SIZE[0]) // HOG_BLOCK_STRIDE[0] + 1) * \
          ((HOG_WIN_SIZE[1] - HOG_BLOCK_SIZE[1]) // HOG_BLOCK_STRIDE[1] + 1) * \
          (HOG_BLOCK_SIZE[0] // HOG_CELL_SIZE[0]) * \
          (HOG_BLOCK_SIZE[1] // HOG_CELL_SIZE[1]) * \
          HOG_NBINS  # = 900
FEAT_DIM = HOG_DIM  # 纯 HOG 特征，不再使用 Hu 矩

PCA_RETAINED_VARIANCE = 0.95
SVM_C = 15.0
SVM_GAMMA = 0.5


# ============================================================
# 预处理（精确复制 C++ preprocessChar + resizeKeepAspectRatio）
# ============================================================
def preprocess_char(src):
    """复制 C++ preprocessChar 逻辑"""
    if src is None or src.size == 0:
        return None

    # 1. 灰度化
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()

    # 2. 裁剪非零区域
    pts = cv2.findNonZero(gray)
    if pts is None or len(pts) == 0:
        return np.zeros(HOG_WIN_SIZE, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(pts)
    x = max(0, x - 2)
    y = max(0, y - 2)
    w = min(gray.shape[1] - x, w + 4)
    h = min(gray.shape[0] - y, h + 4)
    cropped = gray[y:y+h, x:x+w]

    if cropped.size == 0:
        return np.zeros(HOG_WIN_SIZE, dtype=np.uint8)

    # 3. resizeKeepAspectRatio —— 完全复制 C++ 逻辑
    # 计算边界填充值（四条边的平均灰度）
    border_val = (float(np.mean(cropped[0, :])) +
                  float(np.mean(cropped[-1, :])) +
                  float(np.mean(cropped[:, 0])) +
                  float(np.mean(cropped[:, -1]))) / 4.0

    dst = np.full(HOG_WIN_SIZE, int(border_val), dtype=np.uint8)

    src_ratio = cropped.shape[1] / cropped.shape[0]
    dst_ratio = HOG_WIN_SIZE[0] / HOG_WIN_SIZE[1]

    if src_ratio > dst_ratio:
        new_w = HOG_WIN_SIZE[0]
        new_h = max(1, int(round(HOG_WIN_SIZE[0] / src_ratio)))
    else:
        new_h = HOG_WIN_SIZE[1]
        new_w = max(1, int(round(HOG_WIN_SIZE[1] * src_ratio)))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x_off = (HOG_WIN_SIZE[0] - new_w) // 2
    y_off = (HOG_WIN_SIZE[1] - new_h) // 2
    dst[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return dst


# ============================================================
# 特征提取
# ============================================================
def extract_hog_features(img):
    """从原始训练图像提取 HOG 特征 (900 维)"""
    processed = preprocess_char(img)
    if processed is None:
        return np.zeros(HOG_DIM)

    hog = cv2.HOGDescriptor(
        HOG_WIN_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_STRIDE,
        HOG_CELL_SIZE, HOG_NBINS
    )
    desc = hog.compute(processed, winStride=(8, 8), padding=(0, 0)).flatten()

    # L2 归一化 —— 与 C++ getHogOne 一致
    norm = np.sqrt(np.sum(desc ** 2))
    if norm > 1e-6:
        desc = desc / norm

    return desc.astype(np.float64)


# ============================================================
# 训练函数
# ============================================================
def load_training_data(subdir):
    """加载训练数据，返回 features (N x 900), labels (N,)"""
    data_dir = DATA_DIR / subdir
    features_list = []
    labels_list = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = int(class_dir.name)
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        print(f"  Class {class_id}: {len(images)} images")

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feat = extract_hog_features(img)
            features_list.append(feat)
            labels_list.append(class_id)

    features = np.array(features_list, dtype=np.float64)
    labels = np.array(labels_list, dtype=np.int32)
    return features, labels


def train_pca(features, output_path):
    """训练 PCA 降维并保存"""
    # OpenCV PCA 需要 float32
    data = features.astype(np.float32)
    mean_mat = np.mean(data, axis=0).reshape(1, -1).astype(np.float32)

    # 计算协方差矩阵和特征分解
    centered = data - mean_mat
    cov = np.dot(centered.T, centered) / (data.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 确定保留的维度
    total_var = np.sum(eigenvalues)
    cum_var = np.cumsum(eigenvalues) / total_var
    n_components = np.searchsorted(cum_var, PCA_RETAINED_VARIANCE) + 1
    n_components = min(n_components, data.shape[1])

    print(f"  PCA: {data.shape[1]} -> {n_components} dims "
          f"(retained variance: {cum_var[n_components-1]*100:.1f}%)")

    # 保存为 OpenCV PCA 格式
    eigenvectors_top = eigenvectors[:, :n_components].T.copy()  # (n_components, ndims)

    fs = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    fs.write("mean", mean_mat)
    fs.write("eigenvectors", eigenvectors_top.astype(np.float32))
    fs.write("eigenvalues", eigenvalues[:n_components].astype(np.float32).reshape(-1, 1))
    fs.release()

    print(f"  PCA model saved: {output_path}")

    # 投影数据
    reduced = np.dot(centered.astype(np.float64), eigenvectors_top.T.astype(np.float64))
    return reduced, n_components


def train_svm(features_reduced, labels, output_path, n_classes):
    """训练 SVM 并保存"""
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setGamma(SVM_GAMMA)
    svm.setC(SVM_C)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                         1000, 1e-7))

    train_data = cv2.ml.TrainData_create(
        samples=features_reduced.astype(np.float32),
        layout=cv2.ml.ROW_SAMPLE,
        responses=labels.astype(np.int32).reshape(-1, 1)
    )
    svm.train(train_data)

    svm.save(str(output_path))
    print(f"  SVM model saved: {output_path}")
    print(f"    classes: {n_classes}, features: {features_reduced.shape[1]}")
    return svm


def evaluate(svm, features_reduced, labels, label_map):
    """评估模型准确率"""
    correct = 0
    total = len(labels)
    confusion = defaultdict(lambda: defaultdict(int))

    for i in range(total):
        pred = int(round(svm.predict(features_reduced[i:i+1].astype(np.float32))[1][0][0]))
        true = labels[i]
        if pred == true:
            correct += 1
        confusion[true][pred] += 1

    acc = correct / total * 100

    # 打印每类准确率
    print(f"\n  Overall accuracy: {acc:.1f}% ({correct}/{total})")
    print(f"  Per-class accuracy:")
    for class_id in sorted(set(labels)):
        class_total = sum(confusion[class_id].values())
        class_correct = confusion[class_id][class_id]
        class_acc = class_correct / class_total * 100 if class_total > 0 else 0
        char = label_map.get(class_id, "?")
        bar = "#" * int(class_acc / 5) + "-" * (20 - int(class_acc / 5))
        print(f"    {class_id:3d} '{char}': {class_acc:5.1f}% [{bar}] "
              f"({class_correct}/{class_total})")

    # 打印混淆最多的配对
    print(f"\n  Top confusions:")
    confusions = []
    for true_id in confusion:
        for pred_id in confusion[true_id]:
            if true_id != pred_id and confusion[true_id][pred_id] > 0:
                confusions.append((confusion[true_id][pred_id], true_id, pred_id))
    confusions.sort(reverse=True)
    for count, true_id, pred_id in confusions[:10]:
        tc = label_map.get(true_id, "?")
        pc = label_map.get(pred_id, "?")
        print(f"    '{tc}' -> '{pc}': {count} times")

    return acc


# ============================================================
# 主流程
# ============================================================
def train_model(subdir, model_prefix, label_map):
    """训练一个完整模型集（PCA + SVM）"""
    print(f"\n{'='*60}")
    print(f"Training {model_prefix} model ({subdir})")
    print(f"{'='*60}")

    # 1. 加载数据
    print("\n[1/5] Loading training data...")
    features, labels = load_training_data(subdir)
    n_classes = len(set(labels))
    print(f"  Total: {features.shape[0]} samples, {n_classes} classes, "
          f"{features.shape[1]} features")

    # 2. 训练 PCA
    print("\n[2/4] Training PCA...")
    features_reduced, pca_dims = train_pca(
        features, OUTPUT_DIR / f"{model_prefix}_pca_model.xml"
    )

    # 3. 训练 SVM
    print(f"\n[3/4] Training SVM ({pca_dims} dims, C={SVM_C}, gamma={SVM_GAMMA})...")
    svm = train_svm(
        features_reduced, labels,
        OUTPUT_DIR / f"{model_prefix}_svm_model.xml",
        n_classes
    )

    # 4. 评估
    print(f"\n[4/4] Evaluating...")
    acc = evaluate(svm, features_reduced, labels, label_map)

    return acc


def main():
    print("=" * 60)
    print("车牌字符识别模型训练")
    print(f"HOG-only: {FEAT_DIM}d (Hu moments removed)")
    print(f"PCA retain: {PCA_RETAINED_VARIANCE*100}%, SVM: C={SVM_C}, gamma={SVM_GAMMA}")
    print("=" * 60)

    # 中文模型
    CHN_MAP = {
        101: "鲁", 102: "京", 103: "津", 104: "沪", 105: "渝",
        106: "冀", 107: "豫", 108: "云", 109: "辽", 110: "黑",
        111: "湘", 112: "皖", 113: "闽", 114: "新", 115: "苏",
        116: "浙", 119: "吉", 120: "粤", 121: "桂", 122: "琼",
        123: "川", 125: "贵", 127: "陕", 128: "甘", 130: "青",
        131: "藏", 132: "蒙", 133: "宁", 134: "晋",
    }
    zh_acc = train_model("chinese", "1", CHN_MAP)

    # 英文/数字模型
    EN_MAP = {
        48: "0", 49: "1", 50: "2", 51: "3", 52: "4",
        53: "5", 54: "6", 55: "7", 56: "8", 57: "9",
        65: "A", 66: "B", 67: "C", 68: "D", 69: "E",
        70: "F", 71: "G", 72: "H", 74: "J", 75: "K",
        76: "L", 77: "M", 78: "N", 80: "P", 81: "Q",
        82: "R", 83: "S", 84: "T", 85: "U", 86: "V",
        87: "W", 88: "X", 89: "Y", 90: "Z",
    }
    en_acc = train_model("english", "2", EN_MAP)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Chinese accuracy: {zh_acc:.1f}%")
    print(f"  English accuracy: {en_acc:.1f}%")
    print(f"  Models saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
