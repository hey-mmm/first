#include "HogFeature.h"

int main() {

std::vector<cv::Mat> charImgs = { cv::imread("../chp1/picture/char_3.png") };
    std::vector<std::vector<double>> features;

    getHogMoments(charImgs, features);

    std::cout << "样本数: " << features.size()
              << ", 特征维度: " << features[0].size() << std::endl;
    // 输出：样本数: N, 特征维度: 324

    // 接下来可以喂给 SVM / KNN / MLP 训练
    return 0;
}
