#ifndef CHAR_SPLITTER_H
#define CHAR_SPLITTER_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class CCharSplitter
{
	


public:
    // ���ָ�ӿ�
    void splitChars(const Mat& src, std::vector<Mat>& charImgs);
    void splitChars(const Mat& src, std::vector<Mat>& charImgs, bool isGreen);

    // �Զ��ж��ǲ�������
    bool detectGreenPlate(const Mat& src);

private:
    // Ԥ�������
    void removeBorder(Mat& img);
    void removeRivet(Mat& img);
    Mat preprocessPlate(const Mat& src);

    // �ָ��㷨
    vector<Mat> splitByProjection(const Mat& img);
    vector<Mat> splitGreenByProj(const Mat& img);

    // ��������
    void splitWideCharacter(const Mat& img, vector<Mat>& result,
        int start, int width, int height);
};

#endif // CHAR_SPLITTER_H