#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

void getLogHuMoments(const Mat& src, double logHu[7]);

void getHuMoments(const std::vector<cv::Mat>& charImgs, 
                  std::vector<std::vector<double>>& features,
                  const cv::Mat& mean,   // 新增参数：均值行向量 (1×7)
                  const cv::Mat& std);   // 新增参数：标准差行向量 (1×7)
