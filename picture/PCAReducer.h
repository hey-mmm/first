#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * 对联合特征做 PCA 降维。
 * - fit: 用训练集学出投影矩阵
 * - project: 把任意特征投到低维空间
 * - save/load: 模型持久化
 */
class CPCAReducer
{
public:
    CPCAReducer();

    // 训练
    void fit(const cv::Mat& outMat,
             int dims = 90, double retainedVariance = 0.97);

    void project(const std::vector<std::vector<double>>& features,
                 std::vector<std::vector<double>>& reduced) const;

    void projectOne(const std::vector<double>& feat,
                    std::vector<double>& reduced) const;

    bool save(const std::string& file) const;
    bool load(const std::string& file);

    bool isFitted() const { return !m_pca.eigenvectors.empty(); }
    int  outDim() const { return m_pca.eigenvectors.rows; }

private:
    cv::PCA m_pca;
};
