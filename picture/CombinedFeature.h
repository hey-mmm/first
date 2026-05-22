#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @file CombinedFeature.h
 * @brief 将 Hu 矩(7维) 与 HOG 特征 拼接为统一特征向量
 *
 * 拼接顺序固定为: [ Hu(7) | HOG(N) ]
 * 训练 / 推理务必使用相同的 Hu 均值方差(mean/std) 以保证特征一致。
 */


/* ========== 批量图像接口 ========== */


void getCombinedFeatures(const std::vector<cv::Mat>& charImgs,
                         const cv::Mat& huMean,
                         const cv::Mat& huStd,
                         std::vector<std::vector<double>>& features);//输入图像组，得到标准化特征向量




void featuresToMat(const std::vector<cv::Mat>& charImgs,
                   const cv::Mat& huMean,
                   const cv::Mat& huStd,
                   cv::Mat& outMat);//接口函数，输入图像组，得到特征mat