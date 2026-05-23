#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "HogFeature.h"
#include "PCAReducer.h"
#include "utils.h"

using namespace cv;
using namespace std;

int main() {
    // ==========================================
    // 【第一步：用 char_3.png 训练 PCA 模型】
    // ==========================================
    Mat trainImg1 = imread("../chp1/picture/char_3.png", IMREAD_GRAYSCALE);
    if (trainImg1.empty()) {
        cerr << "char_3.png 加载失败！" << endl;
        return -1;
    }
 Mat trainImg2 = imread("../chp1/picture/char_1.png", IMREAD_GRAYSCALE);

    vector<Mat> trainImgs;
    trainImgs.push_back(trainImg1);
    trainImgs.push_back(trainImg2);
    vector<vector<double>> trainFeats;
    getHogMoments(trainImgs, trainFeats);
    Mat trainData = vec2dToMatF(trainFeats);

    // 训练 PCA
    CPCAReducer pca;
    pca.fit(trainData);
    cout << "✅ PCA 训练完成（使用 char_3.png）" << endl;

    // ==========================================
    // 【第二步：用 char_2.png 测试降维】
    // ==========================================
    Mat testImg = imread("../chp1/picture/char_2.png", IMREAD_GRAYSCALE);
    if (testImg.empty()) {
        cerr << "char_2.png 加载失败！" << endl;
        return -1;
    }
    resize(testImg, testImg, Size(32, 32));

    vector<Mat> testImgs;
    testImgs.push_back(testImg);

    vector<vector<double>> testFeats;
    getHogMoments(testImgs, testFeats);
    Mat testData = vec2dToMatF(testFeats);

    // 用训练好的PCA降维！
    Mat reduced = pca.transform(testData);

    // ==========================================
    // 输出结果
    // ==========================================
    cout << "\n📊 测试结果（char_2.png 降维）：" << endl;
    cout << "原始特征维度：" << testData.cols << endl;
    cout << "降维后维度：" << reduced.cols << endl;
    cout << "降维后数据：" << reduced << endl;

    return 0;
}