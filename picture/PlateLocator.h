#pragma once
#include <opencv2/opencv.hpp>
#include"PlateLocator.h"
using namespace cv;
class CPlateLocator
{
public:
    // �ӿڣ�������������ȫƥ��
    void colorLocate(const Mat& src, Mat& dst);

    bool _IsGreen = false;

};

