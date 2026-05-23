#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>



// 加载 Hu 归一化参数,本质为hu类的load
void initCombinedFeature(const std::string& huParamFile);

// 输入一组字符图像，输出对应的 907 维拼接特征向量组
// 空图或提取失败的位置，对应特征向量填零
void extractCombinedFeatures(const std::vector<cv::Mat>& charImgs,
                             std::vector<std::vector<double>>& features);
