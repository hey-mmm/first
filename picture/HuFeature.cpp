
#include "HuFeature.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "Per_Size.h"




void getcommonHuMoments(const Mat& src, double hu[7])
{
    // 预处理
    Mat processed=preprocessChar(src);

    Moments mo = moments(processed, true);

    double huRaw[7] = { 0 };
    HuMoments(mo, huRaw);

    for (int i = 0; i < 7; i++)
    {
        hu[i] = huRaw[i];
    }
}



void getLogHuMoments(const Mat& src, double logHu[7])
{
    double hu[7] = { 0 };
    getcommonHuMoments(src, hu);

    for (int i = 0; i < 7; i++)
    {
        double absVal = std::fabs(hu[i]);

        if (absVal < 1e-30)
        {
            // 值极小，视为0，避免log溢出
            logHu[i] = 0.0;
        }
        else
        {
            double sign = (hu[i] >= 0) ? 1.0 : -1.0;
            logHu[i] = -sign * std::log10(absVal);
        }
    }
}



void getstandHuMoments(const std::vector<Mat>& charImgs, 
                  std::vector<std::vector<double>>& features,
                  const Mat& mean,   // 新增参数：均值行向量 (1×7)
                  const Mat& std)    // 新增参数：标准差行向量 (1×7)
{
    features.clear();
    features.reserve(charImgs.size());

    for (size_t i = 0; i < charImgs.size(); i++)
    {
        if (charImgs[i].empty())
        {
            std::cerr << "[警告] 第 " << i << " 个字符图像为空，跳过" << std::endl;
            continue;
        }

        double logHu[7] = { 0 };
        getLogHuMoments(charImgs[i], logHu);

        // ── 新增：对 7 个 Hu 特征做 z-score 标准化 ──────────────────
        std::vector<double> feat;
        feat.reserve(7);
        for (int k = 0; k < 7; ++k)
        {
            double m = mean.at<double>(0, k);
            double s = std.at<double>(0, k);
            double z = (s > 1e-8) ? (logHu[k] - m) / s : 0.0;
            feat.push_back(z);
        }
        // ─────────────────────────────────────────────────────────────

        features.push_back(std::move(feat));
    }
}
