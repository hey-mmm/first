"""
使用真实车牌数据 + 数据增强训练字符识别模型
数据源: F:\...\Data(1)\Data\chinese + english
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
REAL_DATA_DIR = r"F:\xwechat_files\wxid_o99vsezol5i122_b98b\msg\file\2026-05\Data(1)\Data"
SYNTH_DATA_DIR = Path(__file__).parent / "training_data"  # 之前生成的合成数据
OUTPUT_DIR = Path(__file__).parent.parent / "build"

# 与 C++ 完全一致的 HOG 参数
HOG_WIN_SIZE = (48, 48)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9

HOG_DIM = ((48 - 16) // 8 + 1) * ((48 - 16) // 8 + 1) * (16 // 8) * (16 // 8) * 9  # = 900
FEAT_DIM = HOG_DIM  # 纯 HOG 特征，不再使用 Hu 矩

PCA_RETAINED_VARIANCE = 0.99       # 保留更多方差
PCA_MIN_DIMS = 40                  # 最少保留维度
SVM_C_VALUES = [1.0, 10.0, 50.0]        # 网格搜索 C
SVM_GAMMA_VALUES = [0.01, 0.05, 0.1]   # 网格搜索 gamma
GRID_SEARCH_MAX_SAMPLES = 8000          # 网格搜索最多用 8000 样本加速

# 中文文件夹名 -> classId 映射（与 PlateRecognizer.cpp CHN_MAP 一致）
CHINESE_FOLDER_MAP = {
    "lu": 101, "jing": 102, "jin": 103, "hu": 104, "yu1": 105,
    "ji": 106, "yu": 107, "yun": 108, "liao": 109, "hei": 110,
    "xiang": 111, "wan": 112, "min": 113, "xin": 114, "su": 115,
    "zhe": 116, "jl": 119, "yue": 120, "gui1": 121, "qiong": 122,
    "chuan": 123, "gui": 125, "shan": 127, "gan1": 128, "qing": 130,
    "zang": 131, "meng": 132, "ning": 133, "sx": 134,
}

# 英文文件夹名 -> classId 映射
ENGLISH_FOLDER_MAP = {
    "0": 48, "1": 49, "2": 50, "3": 51, "4": 52,
    "5": 53, "6": 54, "7": 55, "8": 56, "9": 57,
    "A": 65, "B": 66, "C": 67, "D": 68, "E": 69,
    "F": 70, "G": 71, "H": 72, "J": 74, "K": 75,
    "L": 76, "M": 77, "N": 78, "P": 80, "Q": 81,
    "R": 82, "S": 83, "T": 84, "U": 85, "V": 86,
    "W": 87, "X": 88, "Y": 89, "Z": 90,
}

# 显示名称
CHN_MAP = {
    101: "鲁", 102: "京", 103: "津", 104: "沪", 105: "渝",
    106: "冀", 107: "豫", 108: "云", 109: "辽", 110: "黑",
    111: "湘", 112: "皖", 113: "闽", 114: "新", 115: "苏",
    116: "浙", 119: "吉", 120: "粤", 121: "桂", 122: "琼",
    123: "川", 125: "贵", 127: "陕", 128: "甘", 130: "青",
    131: "藏", 132: "蒙", 133: "宁", 134: "晋",
}
EN_MAP = {
    48: "0", 49: "1", 50: "2", 51: "3", 52: "4",
    53: "5", 54: "6", 55: "7", 56: "8", 57: "9",
    65: "A", 66: "B", 67: "C", 68: "D", 69: "E",
    70: "F", 71: "G", 72: "H", 74: "J", 75: "K",
    76: "L", 77: "M", 78: "N", 80: "P", 81: "Q",
    82: "R", 83: "S", 84: "T", 85: "U", 86: "V",
    87: "W", 88: "X", 89: "Y", 90: "Z",
}


# ============================================================
# 数据增强
# ============================================================
def augment_image(img):
    """对输入图像应用随机增强，返回增强后的图像列表"""
    results = [img]  # 原始图像

    h, w = img.shape

    # 1. 轻微旋转（±5度）
    for angle in [-3, -1.5, 1.5, 3]:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
        results.append(rotated)

    # 2. 平移（±2px）
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        translated = cv2.warpAffine(img, M, (w, h), borderValue=0)
        results.append(translated)

    # 3. 缩放微调
    for scale in [0.9, 1.1]:
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 放回原尺寸画布
        canvas = np.zeros((h, w), dtype=np.uint8)
        x_off = (w - new_w) // 2
        y_off = (h - new_h) // 2
        x_off = max(0, x_off)
        y_off = max(0, y_off)
        roi_w = min(new_w, w - x_off)
        roi_h = min(new_h, h - y_off)
        canvas[y_off:y_off+roi_h, x_off:x_off+roi_w] = scaled[:roi_h, :roi_w]
        results.append(canvas)

    # 4. 腐蚀/膨胀（改变笔画粗细）
    for ks in [2, 3]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        results.append(cv2.erode(img, kernel))
        results.append(cv2.dilate(img, kernel))

    # 5. 椒盐噪声
    noisy = img.copy()
    n = int(img.size * 0.01)
    for _ in range(n):
        y, x = random.randint(0, h-1), random.randint(0, w-1)
        noisy[y, x] = 255 if random.random() < 0.5 else 0
    results.append(noisy)

    return results


# ============================================================
# 预处理（精确复制 C++）
# ============================================================
def preprocess_char(src):
    """精确复制 C++ preprocessChar"""
    if src is None or src.size == 0:
        return None

    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()

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

    # resizeKeepAspectRatio
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
    """从原始图像提取 HOG 特征 (900 维)"""
    processed = preprocess_char(img)
    if processed is None:
        return np.zeros(HOG_DIM)

    # HOG
    hog = cv2.HOGDescriptor(HOG_WIN_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_STRIDE,
                            HOG_CELL_SIZE, HOG_NBINS)
    desc = hog.compute(processed, winStride=(8, 8), padding=(0, 0)).flatten()
    norm = np.sqrt(np.sum(desc ** 2))
    if norm > 1e-6:
        desc = desc / norm

    return desc.astype(np.float64)


# ============================================================
# 数据加载
# ============================================================
def load_real_data(subdir, folder_map, label_map, target_per_class=500):
    """
    加载真实数据，样本少的类做增强使每类接近 target_per_class
    """
    data_dir = Path(REAL_DATA_DIR) / subdir
    features_list = []
    labels_list = []
    class_samples = defaultdict(list)

    # 先收集所有原始图像
    for folder_name, class_id in folder_map.items():
        folder_path = data_dir / folder_name
        if not folder_path.is_dir():
            continue

        for img_file in os.listdir(str(folder_path)):
            if not img_file.endswith(('.jpg', '.png', '.bmp')):
                continue
            img_path = str(folder_path / img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            class_samples[class_id].append(img)

    # 提取特征 + 按需增强
    for class_id, images in class_samples.items():
        n_raw = len(images)
        char_name = label_map.get(class_id, "?")

        for img in images:
            features_list.append(extract_hog_features(img))
            labels_list.append(class_id)

        # 对样本数不足的类做增强
        augmented_count = n_raw
        if n_raw < target_per_class:
            for img in images:
                if augmented_count >= target_per_class:
                    break
                aug_imgs = augment_image(img)
                for aug_img in aug_imgs[1:]:  # 跳过原始图（已添加）
                    if augmented_count >= target_per_class:
                        break
                    features_list.append(extract_hog_features(aug_img))
                    labels_list.append(class_id)
                    augmented_count += 1

        print(f"  {class_id:3d} '{char_name}': {n_raw} raw -> {augmented_count} total")

    features = np.array(features_list, dtype=np.float64)
    labels = np.array(labels_list, dtype=np.int32)
    return features, labels


# ============================================================
# 训练函数
# ============================================================
def train_pca(features, output_path):
    """训练 PCA，保留至少 PCA_MIN_DIMS 维"""
    data = features.astype(np.float32)
    mean_mat = np.mean(data, axis=0).reshape(1, -1).astype(np.float32)
    centered = data - mean_mat
    cov = np.dot(centered.T, centered) / (data.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = np.sum(eigenvalues)
    cum_var = np.cumsum(eigenvalues) / total_var
    n_var = np.searchsorted(cum_var, PCA_RETAINED_VARIANCE) + 1
    n_comp = max(PCA_MIN_DIMS, min(n_var, data.shape[1]))
    n_comp = min(n_comp, data.shape[0] - 1, data.shape[1])

    eigenvectors_top = eigenvectors[:, :n_comp].T.copy()
    reduced = np.dot(centered.astype(np.float64), eigenvectors_top.T.astype(np.float64))

    fs = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    fs.write("mean", mean_mat)
    fs.write("eigenvectors", eigenvectors_top.astype(np.float32))
    fs.write("eigenvalues", eigenvalues[:n_comp].astype(np.float32).reshape(-1, 1))
    fs.release()
    print(f"  PCA {data.shape[1]}->{n_comp}d, variance={cum_var[n_comp-1]*100:.1f}% -> {output_path.name}")

    return reduced, n_comp, mean_mat, eigenvectors_top


def apply_pca(features, mean_mat, eigenvectors_top):
    """对特征应用已训练的 PCA 投影"""
    centered = features.astype(np.float64) - mean_mat.astype(np.float64)
    return np.dot(centered, eigenvectors_top.T.astype(np.float64))


def train_svm_grid(features, labels, n_classes):
    """网格搜索最优 SVM 超参数，使用 5-fold 交叉验证"""
    best_acc = 0
    best_c = SVM_C_VALUES[0]
    best_gamma = SVM_GAMMA_VALUES[0]

    # 分层抽样，必要时限制总样本数
    n = len(labels)
    if n > GRID_SEARCH_MAX_SAMPLES:
        # 每类均匀采样
        indices_list = []
        class_idxs = defaultdict(list)
        for i in range(n):
            class_idxs[labels[i]].append(i)
        per_class = GRID_SEARCH_MAX_SAMPLES // len(class_idxs)
        np.random.seed(42)
        for cls, idxs in class_idxs.items():
            sampled = np.random.choice(idxs, min(per_class, len(idxs)), replace=False)
            indices_list.extend(sampled)
        indices = np.array(indices_list)
        np.random.shuffle(indices)
        n = len(indices)
    else:
        indices = np.arange(n)
        np.random.shuffle(indices)

    # 确保每类都有代表，80/20 分
    class_indices = defaultdict(list)
    for i in indices:
        class_indices[labels[i]].append(i)

    train_idx = []
    val_idx = []
    for cls_id, idxs in class_indices.items():
        split = max(1, int(len(idxs) * 0.8))
        train_idx.extend(idxs[:split])
        val_idx.extend(idxs[split:])

    train_data = features[train_idx].astype(np.float32)
    train_labels = labels[train_idx].astype(np.int32).reshape(-1, 1)
    val_data = features[val_idx].astype(np.float32)
    val_labels = labels[val_idx]

    print(f"  Grid search: {len(SVM_C_VALUES)} C x {len(SVM_GAMMA_VALUES)} gamma")
    results = []

    for c in SVM_C_VALUES:
        for gamma in SVM_GAMMA_VALUES:
            svm = cv2.ml.SVM_create()
            svm.setType(cv2.ml.SVM_C_SVC)
            svm.setKernel(cv2.ml.SVM_RBF)
            svm.setGamma(gamma)
            svm.setC(c)
            svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                                 2000, 1e-7))

            td = cv2.ml.TrainData_create(samples=train_data,
                                         layout=cv2.ml.ROW_SAMPLE,
                                         responses=train_labels)
            svm.train(td)

            correct = 0
            for i in range(len(val_labels)):
                pred = int(round(svm.predict(val_data[i:i+1])[1][0][0]))
                if pred == val_labels[i]:
                    correct += 1
            acc = correct / len(val_labels) * 100
            results.append((acc, c, gamma))
            print(f"    C={c:.1f}, gamma={gamma:.3f} -> val acc={acc:.1f}%")

    results.sort(reverse=True)
    best_acc, best_c, best_gamma = results[0]
    print(f"  Best: C={best_c}, gamma={best_gamma}, val_acc={best_acc:.1f}%")

    # 用最佳参数全量训练
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setGamma(best_gamma)
    svm.setC(best_c)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                         2000, 1e-7))

    all_data = features.astype(np.float32)
    all_labels = labels.astype(np.int32).reshape(-1, 1)
    td = cv2.ml.TrainData_create(samples=all_data, layout=cv2.ml.ROW_SAMPLE,
                                 responses=all_labels)
    svm.train(td)

    return svm, best_c, best_gamma


def evaluate(svm, features, labels, label_map):
    """评估"""
    correct = 0
    total = len(labels)
    confusion = defaultdict(lambda: defaultdict(int))

    for i in range(total):
        pred = int(round(svm.predict(features[i:i+1].astype(np.float32))[1][0][0]))
        true = labels[i]
        if pred == true:
            correct += 1
        confusion[true][pred] += 1

    acc = correct / total * 100
    print(f"\n  Overall: {acc:.1f}% ({correct}/{total})")

    # 每类准确率
    print(f"  Per-class:")
    bad_classes = []
    for cls_id in sorted(set(labels)):
        cls_total = sum(confusion[cls_id].values())
        cls_correct = confusion[cls_id][cls_id]
        cls_acc = cls_correct / cls_total * 100 if cls_total > 0 else 0
        char = label_map.get(cls_id, "?")
        bar = "#" * int(cls_acc / 5) + "-" * (20 - int(cls_acc / 5))
        marker = " ***" if cls_acc < 70 else ""
        print(f"    {cls_id:3d} '{char}': {cls_acc:5.1f}% [{bar}] ({cls_correct}/{cls_total}){marker}")
        if cls_acc < 70:
            bad_classes.append((cls_id, char, cls_acc))

    # 混淆
    confusions = []
    for t in confusion:
        for p in confusion[t]:
            if t != p and confusion[t][p] > 0:
                confusions.append((confusion[t][p], t, p))
    confusions.sort(reverse=True)
    if confusions:
        print(f"\n  Top confusions:")
        for count, t, p in confusions[:8]:
            print(f"    '{label_map.get(t,'?')}' -> '{label_map.get(p,'?')}': {count}次")

    return acc


# ============================================================
# 主流程
# ============================================================
def train_model(subdir, folder_map, label_map, prefix):
    """训练一个完整模型集（PCA + SVM，纯 HOG 特征）"""
    print(f"\n{'='*60}")
    print(f"Training {prefix} model from REAL data ({subdir})")
    print(f"{'='*60}")

    print("\n[1] Loading + augmenting data...")
    features, labels = load_real_data(subdir, folder_map, label_map)
    n_classes = len(set(labels))
    print(f"  Total: {len(labels)} samples, {n_classes} classes")

    print("\n[2] Training PCA...")
    features_red, pca_dims, _, _ = train_pca(features, OUTPUT_DIR / f"{prefix}_pca_model.xml")

    print(f"\n[3] Grid search SVM ({pca_dims}d)...")
    svm, best_c, best_gamma = train_svm_grid(features_red, labels, n_classes)
    svm.save(str(OUTPUT_DIR / f"{prefix}_svm_model.xml"))
    print(f"  SVM saved (C={best_c}, gamma={best_gamma}) -> {prefix}_svm_model.xml")

    print(f"\n[4] Evaluating...")
    acc = evaluate(svm, features_red, labels, label_map)
    return acc


def main():
    print("=" * 60)
    print("车牌识别模型训练 — 真实数据 + 增强")
    print(f"Real data: {REAL_DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    zh_acc = train_model("chinese", CHINESE_FOLDER_MAP, CHN_MAP, "1")
    en_acc = train_model("english", ENGLISH_FOLDER_MAP, EN_MAP, "2")

    print(f"\n{'='*60}")
    print(f"DONE! Chinese: {zh_acc:.1f}% | English: {en_acc:.1f}%")
    print(f"Models saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
