#include "PlateRecognizer.h"
#include "PlateLocator.h"
#include "PlateProcessor.h"
#include "CharSplitter.h"
#include "HuFeature.h"          // 提供 preprocessChar() 声明
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    // 1. 检查命令行参数
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <plate_image_path>" << endl;
        return -1;
    }

    // 2. 读取原始图像
    Mat src = imread(argv[1]);
    if (src.empty()) {
        cerr << "Failed to load image: " << argv[1] << endl;
        return -1;
    }
    cout << "Original image size: " << src.size() << endl;

    // 3. 创建处理对象并加载模型
    CPlateLocator   locator;
    CPlateProcessor processor;
    CCharSplitter   splitter;
    CPlateRecognizer recognizer;
    if (!recognizer.loadAllModels("", "")) {
        cerr << "Failed to load recognition models (check exe directory)." << endl;
        return -1;
    }
    cout << "Models loaded successfully." << endl;

    // 4. 车牌定位、提取、校正、二值化
    Mat plateLocate, plateExtract, plateCorrect, plateBin;
    locator.colorLocate(src, plateLocate);
    if (plateLocate.empty()) {
        cerr << "Color location failed." << endl;
        return -1;
    }
    imwrite("debug_plateLocate.jpg", plateLocate);

    processor.extractPlate(plateLocate, plateExtract);
    if (plateExtract.empty()) {
        cerr << "Plate extraction failed." << endl;
        return -1;
    }
    imwrite("debug_plateExtract.jpg", plateExtract);

    processor.correctPlate(plateExtract, plateCorrect);
    if (plateCorrect.empty()) {
        cerr << "Plate correction failed." << endl;
        return -1;
    }
    imwrite("debug_plateCorrect.jpg", plateCorrect);

    processor.binarizePlate(plateCorrect, plateBin);
    if (plateBin.empty()) {
        cerr << "Binarization failed." << endl;
        return -1;
    }
    imwrite("debug_plateBin.jpg", plateBin);

    // 5. 字符分割
    vector<Mat> chars;
    splitter.splitChars(plateBin, chars);
    if (chars.size() < 7) {
        cerr << "Warning: Only " << chars.size() << " characters segmented (expected 7)." << endl;
    }

    // 6. 识别每个字符
    // 【修复】不再在外部调用 preprocessChar，因为 recognize() 内部的 getHogOne()
    //        和 Hu 特征提取器已自动调用 preprocessChar，双重调用会导致图像变形。
    //        这里只保存归一化后的图像用于调试。
    string plateNumber;
    for (size_t i = 0; i < chars.size(); ++i) {
        // 保存分割后的原始字符图像（用于调试）
        string rawName = "debug_char_raw_" + to_string(i) + ".jpg";
        imwrite(rawName, chars[i]);

        // 保存归一化后的图像（仅用于调试，不影响识别流程）
        Mat normalized = preprocessChar(chars[i]);
        string normName = "debug_char_norm_" + to_string(i) + ".jpg";
        imwrite(normName, normalized);

        // 【修复】直接传入原始字符图像，让 recognize() 内部自行预处理
        bool isChinese = (i == 0);
        string ch = recognizer.recognize(chars[i], isChinese);
        plateNumber += ch;
        cout << "Char " << i << " (" << (isChinese ? "CHN" : "EN")
             << ") recognized: " << ch << endl;
    }

    // 7. 输出最终车牌
    cout << "\n========================================" << endl;
    cout << "Final plate number: " << plateNumber << endl;
    cout << "========================================" << endl;

    return 0;
}