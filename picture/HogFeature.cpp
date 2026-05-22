#include "HogFeature.h"
#include "Per_Size.h"
using namespace cv;
using namespace std;

// ===== HOG 参数（针对字符调优）=====

static const Size HOG_BLOCK_SIZE(16, 16);
static const Size HOG_BLOCK_STRIDE(8, 8);
static const Size HOG_CELL_SIZE(8, 8);
static const int  HOG_NBINS = 9;

/**
 * 单张字符预处理：灰度化 → 去噪 → 自适应二值化 → 居中 → resize
 * 这样能让相同字符不同尺寸/光照下特征更稳定。
 */


void getHogMoments(const vector<Mat> &charImgs,
                   vector<vector<double>> &features)
{
    features.clear();
    features.reserve(charImgs.size());

    // 构建 HOG 描述子（构造一次复用，效率高）
    HOGDescriptor hog(
        HOG_WIN_SIZE,
        HOG_BLOCK_SIZE,
        HOG_BLOCK_STRIDE,
        HOG_CELL_SIZE,
        HOG_NBINS
    );

    // 预先计算特征维度，便于异常时填零向量保持索引对齐
    const size_t featDim =
        ((HOG_WIN_SIZE.width  - HOG_BLOCK_SIZE.width)  / HOG_BLOCK_STRIDE.width  + 1) *
        ((HOG_WIN_SIZE.height - HOG_BLOCK_SIZE.height) / HOG_BLOCK_STRIDE.height + 1) *
        (HOG_BLOCK_SIZE.width  / HOG_CELL_SIZE.width)  *
        (HOG_BLOCK_SIZE.height / HOG_CELL_SIZE.height) *
        HOG_NBINS;

    for (const Mat &img : charImgs) {
        Mat normImg = preprocessChar(img);
        if (normImg.empty() || normImg.size() != HOG_WIN_SIZE) {
            // 输入异常时填零，保持与 charImgs 一一对应
            features.emplace_back(featDim, 0.0);
            continue;
        }

        vector<float> desc;
        hog.compute(normImg, desc, Size(8, 8), Size(0, 0));

        // float -> double
        vector<double> featD(desc.begin(), desc.end());

        // 可选：L2 归一化（让分类器更稳定）
        double norm = 0.0;
        for (double v : featD) norm += v * v;
        norm = sqrt(norm) + 1e-7;
        for (double &v : featD) v /= norm;

        features.emplace_back(std::move(featD));
    }
}
