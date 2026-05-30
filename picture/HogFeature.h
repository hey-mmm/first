#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>



// 头文件只声明
 cv::Mat vec2dToMatF(const std::vector<std::vector<double>>& features);
 cv::Mat preprocessChar(const cv::Mat& src);


// ==========================================
// 3. 来自 HogFeature.h 的函数声明
// ==========================================
// 单张图：提取 HOG 特征，失败填零
void getHogOne(const cv::Mat& charImg, std::vector<double>& feature);

// 批量图：内部循环调用 getHogOne，失败位置填零
void getHogMoments(const std::vector<cv::Mat>& charImgs,
    std::vector<std::vector<double>>& features);


// ==========================================
// 5. 来自 PCAReducer.h 的类声明
// ==========================================
class CPCAReducer
{
public:
    CPCAReducer() = default;

    // ===== 训练接口 =====
    // 二选一：dims>0 用固定维度；否则用 retainedVariance（0~1）
    void fit(const cv::Mat& data,
        int dims = 0,
        double retainedVariance = 0.80);

    // ===== 使用接口 =====
    // 把高维数据投影到低维（必须先 fit 或 load）
    cv::Mat transform(const cv::Mat& data) const;

    // 反投影：低维 → 高维（用于重建可视化）
    cv::Mat inverseTransform(const cv::Mat& lowDimData) const;

    // ===== 持久化接口 =====
    void save(const std::string& filename) const;
    void load(const std::string& filename);

    // ===== 查询接口 =====
    bool isFitted() const { return m_isFitted; }
    int getDims() const { return m_isFitted ? m_pca.eigenvalues.rows : 0; }
    cv::Mat getMean() const { return m_pca.mean; }

private:
    cv::PCA m_pca;
    bool m_isFitted = false;
};