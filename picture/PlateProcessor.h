#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
class CPlateProcessor
{
public:
    //��ȡ����
    void extractPlate(const Mat& src, Mat& dst);
    //��ֵ��
    void binarizePlate(const Mat& src, Mat& dst);
    //��бУ��
    void correctPlate(const cv::Mat& src, cv::Mat& dst);
};

