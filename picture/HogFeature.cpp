#include "HogFeature.h"
#include "utils.h"
using namespace cv;
using namespace std;

// ===== HOG 参数 =====
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
    ((HOG_WIN_SIZE.width  - HOG_BLOCK_SIZE.width)  / HOG_BLOCK_STRIDE.width  + 1) *
    ((HOG_WIN_SIZE.height - HOG_BLOCK_SIZE.height) / HOG_BLOCK_STRIDE.height + 1) *
    (HOG_BLOCK_SIZE.width  / HOG_CELL_SIZE.width)  *
    (HOG_BLOCK_SIZE.height / HOG_CELL_SIZE.height) *
    HOG_NBINS;

// ─────────────────────────────────────────
// 单张图
// ─────────────────────────────────────────
void getHogOne(const Mat &charImg, vector<double> &feature)
{
    Mat normImg = preprocessChar(charImg);
    if (normImg.empty() || normImg.size() != HOG_WIN_SIZE)
    {
        feature.assign(HOG_FEAT_DIM, 0.0);
        return;
    }

    vector<float> desc;
    s_hog.compute(normImg, desc, Size(8, 8), Size(0, 0));

    // float → double
    feature.assign(desc.begin(), desc.end());

    // L2 归一化
    double norm = 0.0;
    for (double v : feature) norm += v * v;
    norm = sqrt(norm) + 1e-7;
    for (double &v : feature) v /= norm;
}

// ─────────────────────────────────────────
// 批量图：循环调用 getHogOne
// ─────────────────────────────────────────
void getHogMoments(const vector<Mat> &charImgs,
                   vector<vector<double>> &features)
{
    features.clear();
    features.reserve(charImgs.size());

    for (const Mat &img : charImgs)
    {
        vector<double> feat;
        getHogOne(img, feat);
        features.emplace_back(std::move(feat));
    }
}
