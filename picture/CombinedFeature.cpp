#include "CombinedFeature.h"
#include "HuFeature.h"
#include "HogFeature.h"
#include <iostream>

static constexpr int HU_DIM       = 7;
static constexpr int HOG_DIM      = 900;
static constexpr int COMBINED_DIM = HU_DIM + HOG_DIM;

// 模块内部持有，外部完全不可见
static CHuFeatureExtractor s_huExtractor;

void initCombinedFeature(const std::string& huParamFile)
{
    s_huExtractor.load(huParamFile);
}

void extractCombinedFeatures(const std::vector<cv::Mat>& charImgs,
                             std::vector<std::vector<double>>& features)
{
    features.clear();
    features.reserve(charImgs.size());

    for (size_t i = 0; i < charImgs.size(); i++)
    {
        std::vector<double> feat(COMBINED_DIM, 0.0);

        if (charImgs[i].empty())
        {
            std::cerr << "[CombinedFeature] 第 " << i << " 张图为空，填零\n";
            features.emplace_back(std::move(feat));
            continue;
        }

        // ── Hu 矩（7维）──
        std::vector<double> huFeat = s_huExtractor.extractOne(charImgs[i], true);
        if ((int)huFeat.size() != HU_DIM)
        {
            std::cerr << "[CombinedFeature] 第 " << i << " 张 Hu 维度异常，填零\n";
            features.emplace_back(std::move(feat));
            continue;
        }

        // ── HOG 特征（900维）──
        std::vector<double> hogFeat;
        getHogOne({charImgs[i]}, hogFeat);
        if ((int)hogFeat.size() != HOG_DIM)
        {
            std::cerr << "[CombinedFeature] 第 " << i << " 张 HOG 维度异常，填零\n";
            features.emplace_back(std::move(feat));
            continue;
        }

        // ── 拼接 [Hu | HOG] ──
        feat.clear();
        feat.reserve(COMBINED_DIM);
        feat.insert(feat.end(), huFeat.begin(),  huFeat.end());
        feat.insert(feat.end(), hogFeat.begin(), hogFeat.end());

        features.emplace_back(std::move(feat));
    }
}
