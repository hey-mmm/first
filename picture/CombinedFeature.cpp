#include "CombinedFeature.h"
#include "HuFeature.h"
#include "HogFeature.h"
#include "Per_Size.h"
#include <iostream>

using namespace cv;
using namespace std;

/* ========== 维度查询 ========== */

int getHuFeatureDim()
{
    return 7;
}

int getHogFeatureDim()
{
    // 与 HogFeature.cpp 中的参数保持一致
    static const Size HOG_BLOCK_SIZE(16, 16);
    static const Size HOG_BLOCK_STRIDE(8, 8);
    static const Size HOG_CELL_SIZE(8, 8);
    static const int HOG_NBINS = 9;

    int blocksX = (HOG_WIN_SIZE.width - HOG_BLOCK_SIZE.width) / HOG_BLOCK_STRIDE.width + 1;
    int blocksY = (HOG_WIN_SIZE.height - HOG_BLOCK_SIZE.height) / HOG_BLOCK_STRIDE.height + 1;
    int cellsX = HOG_BLOCK_SIZE.width / HOG_CELL_SIZE.width;
    int cellsY = HOG_BLOCK_SIZE.height / HOG_CELL_SIZE.height;

    return blocksX * blocksY * cellsX * cellsY * HOG_NBINS;
}

int getCombinedFeatureDim()
{
    return getHuFeatureDim() + getHogFeatureDim();
}

/* ========== 单张图像接口 ========== */

bool extractCombinedFeature(const Mat &charImg,
                            const Mat &huMean,
                            const Mat &huStd,
                            vector<double> &feature)
{
    feature.assign(getCombinedFeatureDim(), 0.0);

    if (charImg.empty())
    {
        cerr << "[CombinedFeature] 输入图像为空" << endl;
        return false;
    }

    // 包装成 batch=1，复用批量接口
    vector<Mat> oneImg{charImg};
    vector<vector<double>> huFeats, hogFeats;

    getHuMoments(oneImg, huFeats, huMean, huStd);
    getHogMoments(oneImg, hogFeats);

    if (huFeats.empty() || hogFeats.empty())
    {
        cerr << "[CombinedFeature] 子特征提取失败" << endl;
        return false;
    }

    const auto &hu = huFeats[0];
    const auto &hog = hogFeats[0];

    // 维度校验
    if ((int)hu.size() != getHuFeatureDim() ||
        (int)hog.size() != getHogFeatureDim())
    {
        cerr << "[CombinedFeature] 子特征维度不匹配: hu="
             << hu.size() << ", hog=" << hog.size() << endl;
        return false;
    }

    // 拼接 [Hu | HOG]
    feature.clear();
    feature.reserve(getCombinedFeatureDim());
    feature.insert(feature.end(), hu.begin(), hu.end());
    feature.insert(feature.end(), hog.begin(), hog.end());

    return true;
}

/* ========== 批量图像接口 ========== */

void getCombinedFeatures(const vector<Mat> &charImgs,
                         const Mat &huMean,
                         const Mat &huStd,
                         vector<vector<double>> &features)
{
    features.clear();
    features.reserve(charImgs.size());

    // 一次性批量提取，效率更高
    vector<vector<double>> huFeats, hogFeats;
    getHuMoments(charImgs, huFeats, huMean, huStd);
    getHogMoments(charImgs, hogFeats);

    const int huDim = getHuFeatureDim();
    const int hogDim = getHogFeatureDim();
    const int totalDim = huDim + hogDim;

    // 注意: getHuMoments 在遇到空图时会跳过 (导致大小不一致)
    //       这里以 charImgs.size() 为准, 缺失位置填零
    for (size_t i = 0; i < charImgs.size(); ++i)
    {
        vector<double> feat;
        feat.reserve(totalDim);

        bool huOk = (i < huFeats.size()) && ((int)huFeats[i].size() == huDim);
        bool hogOk = (i < hogFeats.size()) && ((int)hogFeats[i].size() == hogDim);

        if (huOk)
        {
            feat.insert(feat.end(), huFeats[i].begin(), huFeats[i].end());
        }
        else
        {
            feat.insert(feat.end(), huDim, 0.0);
        }

        if (hogOk)
        {
            feat.insert(feat.end(), hogFeats[i].begin(), hogFeats[i].end());
        }
        else
        {
            feat.insert(feat.end(), hogDim, 0.0);
        }

        features.emplace_back(std::move(feat));
    }
}

/* ========== 转 OpenCV 矩阵 ========== */

void featuresToMat(const vector<Mat> &charImgs,
                   const Mat &huMean,
                   const Mat &huStd,
                   Mat &outMat)
{
    if (charImgs.empty())
    {
        outMat.release();
        return;
    }

    // 1) 批量提取拼接特征
    vector<vector<double>> features;
    getCombinedFeatures(charImgs, huMean, huStd, features);

    // 2) 转成 N × D 的 CV_32F 矩阵
    const int rows = (int)features.size();
    const int cols = getCombinedFeatureDim();

    outMat.create(rows, cols, CV_32F);
    outMat.setTo(Scalar(0)); // 异常行先置零, 防御性处理

    for (int i = 0; i < rows; ++i)
    {
        const auto &row = features[i];
        float *p = outMat.ptr<float>(i);
        const int n = std::min(cols, (int)row.size());
        for (int j = 0; j < n; ++j)
        {
            p[j] = (float)row[j];
        }
    }
}
