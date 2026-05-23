#include "HuFeature.h"
#include "utils.h"
#include <cmath>
#include <iostream>

// ─────────────────────────────────────────────
//  私有：原始 Hu 矩计算（含预处理）
// ─────────────────────────────────────────────
void CHuFeatureExtractor::computeRawHu(const cv::Mat &src, double hu[7]) const
{
    CV_Assert(!src.empty());
    cv::Mat processed = preprocessChar(src);
    cv::Moments mo = cv::moments(processed, true);
    cv::HuMoments(mo, hu);
}

// ─────────────────────────────────────────────
//  私有：log 变换
// ─────────────────────────────────────────────
void CHuFeatureExtractor::applyLogTransform(const double hu[7], double logHu[7]) const
{
    for (int i = 0; i < 7; i++)
    {
        double absVal = std::fabs(hu[i]);
        if (absVal < 1e-30)
        {
            logHu[i] = 0.0;
        }
        else
        {
            double sign = (hu[i] >= 0) ? 1.0 : -1.0;
            logHu[i] = -sign * std::log10(absVal);
        }
    }
}

// ─────────────────────────────────────────────
//  私有：z-score 归一化
// ─────────────────────────────────────────────
std::vector<double> CHuFeatureExtractor::zScoreNormalize(const double logHu[7]) const
{
    std::vector<double> feat(7);
    for (int k = 0; k < 7; k++)
    {
        double m = m_huMean.at<double>(0, k);
        double s = m_huStd.at<double>(0, k);
        feat[k] = (s > 1e-8) ? (logHu[7] - m) / s : 0.0;
    }
    return feat;
}

// ─────────────────────────────────────────────
//  训练：学习均值和标准差
// ─────────────────────────────────────────────
void CHuFeatureExtractor::trainNormParams(const std::vector<cv::Mat> &trainImgs)
{
    CV_Assert(!trainImgs.empty());

    const int n = (int)trainImgs.size();
    cv::Mat allHu(n, 7, CV_32F); // n×7，每行一个样本

    for (int i = 0; i < n; i++)
    {
        if (trainImgs[i].empty())
        {
            std::cerr << "[警告] 第 " << i << " 张训练图为空，跳过\n";
            continue;
        }
        double hu[7] = {};
        computeRawHu(trainImgs[i], hu);

        // 直接存 log 变换后的值，与推理阶段保持一致
        double logHu[7] = {};
        applyLogTransform(hu, logHu);

        std::memcpy(allHu.ptr<double>(i), logHu, 7 * sizeof(double));
    }

    // meanStdDev 输出 7×1 列向量，转置成 1×7 行向量
    cv::Mat mean, std;
    cv::meanStdDev(allHu, mean, std);
    m_huMean = mean.t(); // 1×7
    m_huStd = std.t();   // 1×7

    m_isTrained = true;
}

// ─────────────────────────────────────────────
//  持久化：保存
// ─────────────────────────────────────────────
void CHuFeatureExtractor::save(const std::string &filename) const
{
    CV_Assert(m_isTrained && "请先 trainNormParams() 再保存");
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "huMean" << m_huMean;
    fs << "huStd" << m_huStd;
    fs.release();
}

// ─────────────────────────────────────────────
//  持久化：加载
// ─────────────────────────────────────────────
void CHuFeatureExtractor::load(const std::string &filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    CV_Assert(fs.isOpened());
    fs["huMean"] >> m_huMean;
    fs["huStd"] >> m_huStd;
    fs.release();
    m_isTrained = true;
}

// ─────────────────────────────────────────────
//  特征提取：单张图
// ─────────────────────────────────────────────
std::vector<double> CHuFeatureExtractor::extractOne(const cv::Mat &src,
                                                    bool useLog) const
{
    double hu[7] = {};
    computeRawHu(src, hu);

    // 是否做 log 变换
    double finalHu[7] = {};
    if (useLog)
        applyLogTransform(hu, finalHu);
    else
        std::memcpy(finalHu, hu, 7 * sizeof(double));

    // 是否做 z-score 归一化
    if (m_isTrained)
        return zScoreNormalize(finalHu);

    return std::vector<double>(finalHu, finalHu + 7);
}

// ─────────────────────────────────────────────
//  特征提取：批量
// ─────────────────────────────────────────────
void CHuFeatureExtractor::extractBatch(const std::vector<cv::Mat> &charImgs,
                                       std::vector<std::vector<double>> &features,
                                       bool useLog) const
{
    features.clear();
    features.reserve(charImgs.size());

    for (size_t i = 0; i < charImgs.size(); i++)
    {
        if (charImgs[i].empty())
        {
            std::cerr << "[警告] 第 " << i << " 个字符图像为空，跳过\n";
            continue;
        }
        features.push_back(extractOne(charImgs[i], useLog));
    }
}
