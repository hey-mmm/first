#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class CHuFeatureExtractor
{
public:
    CHuFeatureExtractor() = default;

    // ===== 训练接口 =====
    // 从训练图像集学习 Hu 矩的均值和标准差
    void trainNormParams(const std::vector<cv::Mat> &trainImgs);

    // ===== 持久化接口 =====
    void save(const std::string &filename) const;
    void load(const std::string &filename);

    // ===== 特征提取接口 =====
    // 单张图：提取 7 维 Hu 矩特征
    //   useLog = false → 原始 Hu 矩
    //   useLog = true  → log 变换后的 Hu 矩
    // 若已训练，自动做 z-score 归一化；未训练则直接返回原始值
    std::vector<double> extractOne(const cv::Mat &src,
                                   bool useLog = true) const;

    // 批量提取（内部循环调用 extractOne）
    void extractBatch(const std::vector<cv::Mat> &charImgs,
                      std::vector<std::vector<double>> &features,
                      bool useLog = true) const;

    // ===== 查询接口 =====
    bool isTrained() const { return m_isTrained; }
    cv::Mat getMean() const { return m_huMean; }
    cv::Mat getStd() const { return m_huStd; }

private:
    // 计算原始 7 个 Hu 矩
    void computeRawHu(const cv::Mat &src, double hu[7]) const;
    // log 变换
    void applyLogTransform(const double hu[7], double logHu[7]) const;

    // z-score 归一化（需已训练）
    std::vector<double> zScoreNormalize(const double logHu[7]) const;

    cv::Mat m_huMean; // 1×7 CV_32F
    cv::Mat m_huStd;  // 1×7 CV_32F
    bool m_isTrained = false;
};
