#pragma once

#include <opencv2/opencv.hpp>
#include <vector>


void getHogMoments(const std::vector<cv::Mat> &charImgs,
                   std::vector<std::vector<double>> &features);


