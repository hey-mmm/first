#include <cmath>
#include <iostream>
#include "HuFeature.h"

using namespace cv;
using namespace std;

// ============================================================================
// 1. 来自 HuFeature.cpp 的实现代码
// ============================================================================

// 将二维特征向量组转换为 OpenCV 浮点矩阵
cv::Mat vec2dToMatF(const std::vector<std::vector<double>>& features)
{
    const int rows = (int)features.size();
    const int cols = (int)features[0].size();
    cv::Mat data(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            data.at<float>(i, j) = static_cast<float>(features[i][j]);
    return data;
}

// HOG 检测窗口大小（必须与训练时一致）
static const cv::Size HOG_WIN_SIZE(48, 48);

// 字符预处理：灰度化、裁剪空白、缩放到 HOG_WIN_SIZE
static Mat resizeKeepAspectRatio(const Mat& src, const Size& targetSize)
{
    // --- 计算填充灰度值：四周边界的平均灰度 ---
    double borderVal = 0.0;
    if (src.channels() == 1) {
        cv::Scalar topMean    = cv::mean(src.row(0));
        cv::Scalar bottomMean = cv::mean(src.row(src.rows - 1));
        cv::Scalar leftMean   = cv::mean(src.col(0));
        cv::Scalar rightMean  = cv::mean(src.col(src.cols - 1));
        borderVal = (topMean[0] + bottomMean[0] + leftMean[0] + rightMean[0]) / 4.0;
    } else {
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        cv::Scalar topMean    = cv::mean(gray.row(0));
        cv::Scalar bottomMean = cv::mean(gray.row(gray.rows - 1));
        cv::Scalar leftMean   = cv::mean(gray.col(0));
        cv::Scalar rightMean  = cv::mean(gray.col(gray.cols - 1));
        borderVal = (topMean[0] + bottomMean[0] + leftMean[0] + rightMean[0]) / 4.0;
    }

    Mat dst(targetSize, CV_8UC1, Scalar(static_cast<uchar>(borderVal)));

    double srcRatio = (double)src.cols / src.rows;
    double dstRatio = (double)targetSize.width / targetSize.height;

    int newW, newH;
    if (srcRatio > dstRatio) {
        newW = targetSize.width;
        newH = cvRound(targetSize.width / srcRatio);
    } else {
        newH = targetSize.height;
        newW = cvRound(targetSize.height * srcRatio);
    }

    Mat resized;
    resize(src, resized, Size(newW, newH), 0, 0, INTER_LINEAR);
    int x = (targetSize.width - newW) / 2;
    int y = (targetSize.height - newH) / 2;
    resized.copyTo(dst(Rect(x, y, newW, newH)));
    return dst;
}

 Mat preprocessChar(const Mat& src)
{
    if (src.empty())
        return Mat();
    
    // 1. 灰度化
    Mat gray;
    if (src.channels() == 3)
        cvtColor(src, gray, COLOR_BGR2GRAY);
    else if (src.channels() == 4)
        cvtColor(src, gray, COLOR_BGRA2GRAY);
    else
        gray = src.clone();
    
    // 2. 裁剪空白（保留原逻辑）
    Mat cropped = gray;
    vector<Point> pts;
    findNonZero(gray, pts);
    if (!pts.empty())
    {
        Rect roi = boundingRect(pts);
        roi.x = max(0, roi.x - 2);
        roi.y = max(0, roi.y - 2);
        roi.width = min(gray.cols - roi.x, roi.width + 4);
        roi.height = min(gray.rows - roi.y, roi.height + 4);
        cropped = gray(roi);
    }
    
    // 3. 等比例缩放 + 填充（关键修改）
    Mat result = resizeKeepAspectRatio(cropped, HOG_WIN_SIZE);
    return result;
}

// CHuFeatureExtractor 成员函数已删除 — Hu 矩特征不再使用


// ============================================================================
// 2. 来自 HogFeature.cpp 的实现代码
// ============================================================================

// HOG 参数
static const Size HOG_BLOCK_SIZE(16, 16);
static const Size HOG_BLOCK_STRIDE(8, 8);
static const Size HOG_CELL_SIZE(8, 8);
static const int  HOG_NBINS = 9;

static const HOGDescriptor s_hog(
    HOG_WIN_SIZE,
    HOG_BLOCK_SIZE,
    HOG_BLOCK_STRIDE,
    HOG_CELL_SIZE,
    HOG_NBINS
);

static const size_t HOG_FEAT_DIM =
((HOG_WIN_SIZE.width - HOG_BLOCK_SIZE.width) / HOG_BLOCK_STRIDE.width + 1) *
((HOG_WIN_SIZE.height - HOG_BLOCK_SIZE.height) / HOG_BLOCK_STRIDE.height + 1) *
(HOG_BLOCK_SIZE.width / HOG_CELL_SIZE.width) *
(HOG_BLOCK_SIZE.height / HOG_CELL_SIZE.height) *
HOG_NBINS;

// ----------------------------------------------------------------------------
// 单张图像提取 HOG 特征
// ----------------------------------------------------------------------------
void getHogOne(const Mat& charImg, vector<double>& feature)
{
    Mat normImg = preprocessChar(charImg);
    if (normImg.empty() || normImg.size() != HOG_WIN_SIZE)
    {
        feature.assign(HOG_FEAT_DIM, 0.0);
        return;
    }

    vector<float> desc;
    s_hog.compute(normImg, desc, Size(8, 8), Size(0, 0));

    // float 转 double
    feature.assign(desc.begin(), desc.end());

    // L2 归一化
    double norm = 0.0;
    for (double val : feature) norm += val * val;
    norm = sqrt(norm);

    if (norm > 1e-6)
    {
        for (double& val : feature) val /= norm;
    }
}

// ----------------------------------------------------------------------------
// 批量提取 HOG 特征
// ----------------------------------------------------------------------------
void getHogMoments(const vector<Mat>& charImgs, vector<vector<double>>& features)
{
    features.clear();
    features.reserve(charImgs.size());

    for (const auto& img : charImgs)
    {
        vector<double> feat;
        getHogOne(img, feat);
        features.push_back(feat);
    }
}

// initCombinedFeature / extractCombinedFeatures 已删除 — 组合特征不再使用


// ============================================================================
// 4. 来自 PCAReducer.cpp 的实现代码
// ============================================================================

// ------------------------- 训练 -------------------------
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

// ------------------------- 降维 / 重建 -------------------------
cv::Mat CPCAReducer::transform(const cv::Mat& data) const
{
    CV_Assert(m_isFitted && "请先调用 fit() 或 load()");
    CV_Assert(data.cols == m_pca.mean.cols);
    return m_pca.project(data);
}

cv::Mat CPCAReducer::inverseTransform(const cv::Mat& lowDimData) const
{
    CV_Assert(m_isFitted);
    return m_pca.backProject(lowDimData);
}

// ------------------------- 持久化 -------------------------
void CPCAReducer::save(const std::string& filename) const
{
    CV_Assert(m_isFitted);
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    CV_Assert(fs.isOpened());

    fs << "mean" << m_pca.mean;
    fs << "eigenvectors" << m_pca.eigenvectors;
    fs << "eigenvalues" << m_pca.eigenvalues;

    fs.release();
}

void CPCAReducer::load(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    CV_Assert(fs.isOpened());

    fs["mean"] >> m_pca.mean;
    fs["eigenvectors"] >> m_pca.eigenvectors;
    fs["eigenvalues"] >> m_pca.eigenvalues;

    fs.release();
    m_isFitted = true;
}