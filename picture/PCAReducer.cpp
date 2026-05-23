#include "PCAReducer.h"

// ─────────────── 训练 ───────────────
void CPCAReducer::fit(const cv::Mat& data, int dims, double retainedVariance)
{
    CV_Assert(!data.empty());
    CV_Assert(data.type() == CV_32F || data.type() == CV_64F);
    CV_Assert(data.rows > 1);

    if (dims > 0) {
        // 固定维度
        int maxDims = std::min(data.rows, data.cols);
        CV_Assert(dims <= maxDims);
        m_pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, dims);
    }
    else {
        // 按保留方差比例自动确定维度
        CV_Assert(retainedVariance > 0.0 && retainedVariance < 1.0);
        m_pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, retainedVariance);
    }

    m_isFitted = true;
}

// ─────────────── 使用 ───────────────
cv::Mat CPCAReducer::transform(const cv::Mat& data) const
{
    CV_Assert(m_isFitted && "请先 fit() 或 load()");
    CV_Assert(data.cols == m_pca.mean.cols);
    return m_pca.project(data);
}

cv::Mat CPCAReducer::inverseTransform(const cv::Mat& lowDimData) const
{
    CV_Assert(m_isFitted);
    return m_pca.backProject(lowDimData);
}

// ─────────────── 持久化 ───────────────
void CPCAReducer::save(const std::string& filename) const
{
    CV_Assert(m_isFitted);
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "mean"         << m_pca.mean;
    fs << "eigenvectors" << m_pca.eigenvectors;
    fs << "eigenvalues"  << m_pca.eigenvalues;
    fs.release();
}

void CPCAReducer::load(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    CV_Assert(fs.isOpened());
    fs["mean"]         >> m_pca.mean;
    fs["eigenvectors"] >> m_pca.eigenvectors;
    fs["eigenvalues"]  >> m_pca.eigenvalues;
    fs.release();
    m_isFitted = true;
}
