"""
生成车牌字符训练数据
模拟字符分割后的效果：白色字符(255) + 黑色背景(0)，尺寸约 19x38
"""

import cv2
import numpy as np
import os
from pathlib import Path

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = Path(__file__).parent / "training_data"

# 中文字符（29 个省份简称），与 PlateRecognizer.cpp 的 CHN_MAP 完全一致
CHINESE_CHARS = {
    101: "鲁", 102: "京", 103: "津", 104: "沪", 105: "渝",
    106: "冀", 107: "豫", 108: "云", 109: "辽", 110: "黑",
    111: "湘", 112: "皖", 113: "闽", 114: "新", 115: "苏",
    116: "浙", 119: "吉", 120: "粤", 121: "桂", 122: "琼",
    123: "川", 125: "贵", 127: "陕", 128: "甘", 130: "青",
    131: "藏", 132: "蒙", 133: "宁", 134: "晋",
}

# 英文/数字字符（34 类），与 enMap 一致
ENGLISH_CHARS = {
    48: "0", 49: "1", 50: "2", 51: "3", 52: "4",
    53: "5", 54: "6", 55: "7", 56: "8", 57: "9",
    65: "A", 66: "B", 67: "C", 68: "D", 69: "E",
    70: "F", 71: "G", 72: "H", 74: "J", 75: "K",
    76: "L", 77: "M", 78: "N", 80: "P", 81: "Q",
    82: "R", 83: "S", 84: "T", 85: "U", 86: "V",
    87: "W", 88: "X", 89: "Y", 90: "Z",
}

SAMPLES_PER_CLASS = 200  # 每类生成样本数
CANVAS_WIDTH = 22        # 略大于分割宽度 19，给字符留边距
CANVAS_HEIGHT = 42       # 略大于分割高度 38
FONT_SIZES = [20, 22, 24, 26, 28, 30, 32]  # 多种字号
ROTATION_RANGE = 5        # 旋转角度范围（度）


def find_font_path(candidates, fallback_patterns):
    """
    在 Windows Fonts 目录查找字体文件。
    返回 (font_name, font_path) 或 (None, None)
    """
    from glob import glob

    font_dirs = [
        r"C:\Windows\Fonts",
        r"C:\Windows\Font",
        os.path.expanduser(r"~\AppData\Local\Microsoft\Windows\Fonts"),
    ]

    all_ttf = []
    for fd in font_dirs:
        if os.path.isdir(fd):
            all_ttf.extend(glob(os.path.join(fd, "*.ttf")))
            all_ttf.extend(glob(os.path.join(fd, "*.ttc")))
            all_ttf.extend(glob(os.path.join(fd, "*.otf")))

    if not all_ttf:
        return None

    name_to_path = {}
    for fp in all_ttf:
        name_to_path[os.path.basename(fp).lower()] = fp

    # 1. 精确匹配候选字体
    for name in candidates:
        # 尝试多种文件名格式
        for ext in [".ttf", ".ttc", ".otf"]:
            key = (name + ext).lower()
            if key in name_to_path:
                return name_to_path[key]
        # 不区分大小写的完整匹配
        for key, fp in name_to_path.items():
            if name.lower() in key:
                return fp

    # 2. 模糊匹配
    for pattern in fallback_patterns:
        for key, fp in name_to_path.items():
            if pattern.lower() in key.lower():
                return fp

    # 3. 返回第一个可用的
    return list(name_to_path.values())[0] if name_to_path else None


def get_available_fonts():
    """返回系统可用的中文字体和英文字体路径"""
    zh_path = find_font_path(
        ["simhei", "msyh", "msyhbd", "kaiti", "simkai",
         "simsun", "fangsong", "STSONG", "STKAITI"],
        ["hei", "kai", "song", "cjk", "chinese", "hans"]
    )
    en_path = find_font_path(
        ["arial", "calibri", "consola", "cour", "segoeui",
         "DejaVuSansMono", "LiberationMono-Regular"],
        ["sans", "mono", "arial", "segoe"]
    )

    if zh_path:
        print(f"  Found Chinese font: {zh_path}")
    if en_path:
        print(f"  Found English font: {en_path}")

    return zh_path, en_path


def render_char_opencv(char, font_path, font_size, canvas_w, canvas_h,
                       thickness=1, rotation=0, offset_x=0, offset_y=0):
    """
    用 PIL 渲染单个字符，模拟分割后的二值图像效果。
    返回：二值图像 (0 背景, 255 字符)
    """
    from PIL import Image, ImageDraw, ImageFont

    # 创建临时画布（大一点保证字符不被裁切）
    img = Image.new("L", (canvas_w * 3, canvas_h * 3), 0)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(font_path, font_size)

    # 计算字符位置（居中）
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_x = (img.width - tw) // 2 - bbox[0]
    text_y = (img.height - th) // 2 - bbox[1]

    draw.text((text_x, text_y), char, fill=255, font=font,
              stroke_width=thickness, stroke_fill=255)

    # 转 OpenCV
    img_cv = np.array(img, dtype=np.uint8)

    # 旋转
    if rotation != 0:
        center = (img_cv.shape[1] // 2, img_cv.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        img_cv = cv2.warpAffine(img_cv, M, (img_cv.shape[1], img_cv.shape[0]),
                                borderValue=0)

    # 裁剪空白区域（模拟字符分割效果）
    pts = cv2.findNonZero(img_cv)
    if pts is not None and len(pts) > 0:
        x, y, w, h = cv2.boundingRect(pts)
        margin = np.random.randint(1, 4)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_cv.shape[1] - x, w + 2 * margin)
        h = min(img_cv.shape[0] - y, h + 2 * margin)
        img_cv = img_cv[y:y+h, x:x+w]

    # 缩放到目标画布
    h, w = img_cv.shape
    scale = min((canvas_w - 4) / w, (canvas_h - 4) / h)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w > 0 and new_h > 0:
        img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 放到黑底画布上
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    x_off = (canvas_w - img_cv.shape[1]) // 2 + offset_x
    y_off = (canvas_h - img_cv.shape[0]) // 2 + offset_y

    # 确保不越界
    x_off = max(0, min(x_off, canvas_w - img_cv.shape[1]))
    y_off = max(0, min(y_off, canvas_h - img_cv.shape[0]))

    canvas[y_off:y_off + img_cv.shape[0], x_off:x_off + img_cv.shape[1]] = img_cv

    return canvas


def add_noise(img, noise_level=0.01):
    """添加椒盐噪声"""
    result = img.copy()
    n = int(img.size * noise_level)
    for _ in range(n // 2):
        y = np.random.randint(0, img.shape[0])
        x = np.random.randint(0, img.shape[1])
        result[y, x] = 255
    for _ in range(n // 2):
        y = np.random.randint(0, img.shape[0])
        x = np.random.randint(0, img.shape[1])
        result[y, x] = 0
    return result


def erode_dilate(img):
    """随机腐蚀/膨胀改变笔画粗细"""
    op = np.random.choice(["none", "erode", "dilate", "both"])
    kernel_size = np.random.choice([2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = img.copy()
    if op in ("erode", "both"):
        result = cv2.erode(result, kernel)
    if op in ("dilate", "both"):
        result = cv2.dilate(result, kernel)
    return result


def generate_samples(chars_dict, font_path, output_subdir, canvas_w, canvas_h):
    """为字符集生成训练样本"""
    if font_path is None:
        print(f"  ERROR: No font available for {output_subdir}, skipping!")
        return

    for class_id, char in chars_dict.items():
        out_dir = OUTPUT_DIR / output_subdir / str(class_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS):
            font_size = np.random.choice(FONT_SIZES)
            rotation = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)
            thickness = np.random.choice([1, 2, 3])

            img = render_char_opencv(
                char, font_path, font_size, canvas_w, canvas_h,
                thickness=thickness, rotation=rotation,
                offset_x=offset_x, offset_y=offset_y
            )

            # 随机变换
            if np.random.random() < 0.3:
                img = erode_dilate(img)
            if np.random.random() < 0.2:
                img = add_noise(img, noise_level=np.random.uniform(0.005, 0.02))

            cv2.imwrite(str(out_dir / f"{i:04d}.png"), img)

        print(f"  Class {class_id} ('{char}'): {SAMPLES_PER_CLASS} samples")


def main():
    print("=" * 60)
    print("车牌字符训练数据生成")
    print("=" * 60)

    # 检查字体
    zh_font, en_font = get_available_fonts()
    print(f"\nChinese font: {zh_font}")
    print(f"English font: {en_font}")

    if zh_font is None:
        print("\nERROR: No Chinese font found! Please install a Chinese font.")
        print("On Windows: Install 'SimHei' or 'Microsoft YaHei'")
        return
    if en_font is None:
        print("\nERROR: No English font found!")
        return

    # 生成中文训练数据
    print(f"\n[1/2] Generating Chinese character samples ({len(CHINESE_CHARS)} classes)...")
    generate_samples(CHINESE_CHARS, zh_font, "chinese", CANVAS_WIDTH, CANVAS_HEIGHT)

    # 生成英文/数字训练数据
    print(f"\n[2/2] Generating English/Number samples ({len(ENGLISH_CHARS)} classes)...")
    generate_samples(ENGLISH_CHARS, en_font, "english", CANVAS_WIDTH, CANVAS_HEIGHT)

    print(f"\nDone! Training data saved to: {OUTPUT_DIR}")
    print(f"  Chinese: {OUTPUT_DIR / 'chinese'} ({len(CHINESE_CHARS)} folders)")
    print(f"  English: {OUTPUT_DIR / 'english'} ({len(ENGLISH_CHARS)} folders)")


if __name__ == "__main__":
    main()
