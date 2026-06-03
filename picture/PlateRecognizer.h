#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "HuFeature.h"

std::string getExeDirectory();   // 获取 exe 所在目录（Windows）

class CPlateRecognizer
{
public:
    bool loadAllModels(const std::string& zhPrefix, const std::string& enNumPrefix);
    // 直接返回识别的字符串，不再需要 CString 参数
    std::string recognize(const cv::Mat& charImg, bool isChinese);

private:
    std::string getChineseString(int labelId);
    std::string getEnNumString(int labelId);

    CPCAReducer         m_zhPca;
    cv::Ptr<cv::ml::SVM> m_zhSvm;

    CPCAReducer         m_enPca;
    cv::Ptr<cv::ml::SVM> m_enSvm;
};