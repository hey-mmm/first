#include "PCAReducer.h"
#include"CombinedFeature.h"
using namespace cv;
using namespace std;

CPCAReducer::CPCAReducer() {}




void CPCAReducer::fit(const cv::Mat& dataMat, int dims, double retainedVariance)
{
    if (retainedVariance > 0.0 && retainedVariance < 1.0) {
        // 自动保留 99% 信息，PCA自己算维度
        m_pca = cv::PCA(dataMat, cv::Mat(), cv::PCA::DATA_AS_ROW, retainedVariance);
    } else {
        // 手动指定维度
        m_pca = cv::PCA(dataMat, cv::Mat(), cv::PCA::DATA_AS_ROW, dims);
    }
}

void CPCAReducer::project(const vector<vector<double>>& features,
                          vector<vector<double>>& reduced) const
{
    CV_Assert(isFitted());
    Mat data = vec2mat(features);
    Mat proj = m_pca.project(data);     // N x outDim, CV_64F

    reduced.clear();
    reduced.reserve(proj.rows);
    for (int i = 0; i < proj.rows; ++i) {
        vector<double> r(proj.cols);
        for (int j = 0; j < proj.cols; ++j)
            r[j] = proj.at<double>(i, j);
        reduced.emplace_back(std::move(r));
    }
}

void CPCAReducer::projectOne(const vector<double>& feat,
                             vector<double>& reduced) const
{
    CV_Assert(isFitted());
    Mat row(1, (int)feat.size(), CV_64F);
    for (int j = 0; j < (int)feat.size(); ++j)
        row.at<double>(0, j) = feat[j];
    Mat proj = m_pca.project(row);
    reduced.assign((double*)proj.datastart, (double*)proj.dataend);
}

bool CPCAReducer::save(const string& file) const
{
    if (!isFitted()) return false;
    FileStorage fs(file, FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "mean" << m_pca.mean;
    fs << "eigenvectors" << m_pca.eigenvectors;
    fs << "eigenvalues" << m_pca.eigenvalues;
    fs.release();
    return true;
}

bool CPCAReducer::load(const string& file)
{
    FileStorage fs(file, FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["mean"] >> m_pca.mean;
    fs["eigenvectors"] >> m_pca.eigenvectors;
    fs["eigenvalues"] >> m_pca.eigenvalues;
    fs.release();
    return isFitted();
}
