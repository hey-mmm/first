#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// 单张图：提取 HOG 特征，失败填零
void getHogOne(const cv::Mat &charImg, std::vector<double> &feature);

// 批量图：内部循环调用 getHogOne，失败位置填零
void getHogMoments(const std::vector<cv::Mat> &charImgs,
                   std::vector<std::vector<double>> &features);
