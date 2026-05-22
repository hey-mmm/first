#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

static const Size HOG_WIN_SIZE(48, 48);     // 字符归一化尺寸

static Mat preprocessChar(const Mat &src)
{
    if (src.empty()) return Mat();

    // 1. 灰度化
    Mat gray;
    if (src.channels() == 3)      cvtColor(src, gray, COLOR_BGR2GRAY);
    else if (src.channels() == 4) cvtColor(src, gray, COLOR_BGRA2GRAY);
    else                          gray = src.clone();

    
    // 4. 找到字符外接矩形并裁剪，减少空白带来的尺度差异
    Mat cropped = gray;
    vector<Point> pts;
    findNonZero(gray, pts);
    if (!pts.empty()) {
        Rect roi = boundingRect(pts);
        // 适当向外扩 2 像素，避免笔画贴边
        roi.x      = max(0, roi.x - 2);
        roi.y      = max(0, roi.y - 2);
        roi.width  = min(gray.cols - roi.x, roi.width  + 4);
        roi.height = min(gray.rows - roi.y, roi.height + 4);
        cropped = gray(roi).clone();
    }

    // 5. 保持长宽比缩放后居中放到 32x32 画布
    int side = max(cropped.cols, cropped.rows);
    Mat square = Mat::zeros(side, side, CV_8UC1);
    int dx = (side - cropped.cols) / 2;
    int dy = (side - cropped.rows) / 2;
    cropped.copyTo(square(Rect(dx, dy, cropped.cols, cropped.rows)));

    Mat resized;
    resize(square, resized, HOG_WIN_SIZE, 0, 0, INTER_AREA);

    return resized;
}