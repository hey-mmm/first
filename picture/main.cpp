#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "utils.h"

using namespace cv;
using namespace std;

int main() {

    Mat trainImg1 = imread("char_1.png", IMREAD_GRAYSCALE);
    if (trainImg1.empty()) {
    cout << "❌ 图片加载失败！路径：" <<  endl;
    return -1;
}
    Mat img=preprocessChar(trainImg1);
    namedWindow("预处理结果");
imshow("预处理结果", img);
waitKey(0);
  
    return 0;
}