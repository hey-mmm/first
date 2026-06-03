#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// ==========================================
// 基础工具函数声明
// ==========================================

// 将二维特征向量组转换为 OpenCV 浮点矩阵
cv::Mat vec2dToMatF(const std::vector<std::vector<double>> &features);

// 字符图像预处理（灰度化、裁剪空白、缩放到固定尺寸）
// 注意：该函数在 HuFeature.cpp 中实现，此处为内联声明
cv::Mat preprocessChar(const cv::Mat &src);

// CHuFeatureExtractor 类已删除 — Hu 矩特征不再使用

// ==========================================
// HOG 特征提取函数声明
// ==========================================

// 提取单张图像的 HOG 特征（内部自动调用 preprocessChar 归一化到固定尺寸）
void getHogOne(const cv::Mat &charImg, std::vector<double> &feature);

// 批量提取 HOG 特征（失败位置填零）
void getHogMoments(const std::vector<cv::Mat> &charImgs,
                   std::vector<std::vector<double>> &features);
// initCombinedFeature / extractCombinedFeatures 已删除 — 组合特征不再使用

// ==========================================
// 类 CPCAReducer：PCA 降维器
// ==========================================
class CPCAReducer
{
public:
    CPCAReducer() = default;

    // ----- 训练接口 -----
    // 二选一：若 dims > 0 则使用固定维度；否则使用 retainedVariance（保留方差比例，0~1）
    void fit(const cv::Mat &data,
             int dims = 0,
             double retainedVariance = 0.95);

    // ----- 使用接口 -----
    // 将高维数据投影到低维（必须先 fit 或 load）
    cv::Mat transform(const cv::Mat &data) const;

    // 反投影：从低维重建高维（可用于可视化）
    cv::Mat inverseTransform(const cv::Mat &lowDimData) const;

    // ----- 持久化接口 -----
    void save(const std::string &filename) const; // 保存 PCA 参数（均值、特征向量、特征值）
    void load(const std::string &filename);       // 加载 PCA 参数

    // ----- 查询接口 -----
    bool isFitted() const { return m_isFitted; }
    int getDims() const { return m_isFitted ? m_pca.eigenvalues.rows : 0; }
    cv::Mat getMean() const { return m_pca.mean; }

private:
    cv::PCA m_pca;
    bool m_isFitted = false;
};