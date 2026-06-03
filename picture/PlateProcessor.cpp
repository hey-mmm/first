
#include "PlateProcessor.h"
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// ===================== ��ȷ��ȡ���� =====================
// ���룺��ɫ��λ�������ͼ�����ܴ�������
// �������ȷ�ü��Ĵ���������ͼ
void CPlateProcessor::extractPlate(const Mat& src, Mat& dst)
{
    if (src.empty()) { dst = Mat(); return; }

    Mat gray, blur, sobel, binary, closed;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 0);
    Sobel(blur, sobel, CV_8U, 1, 0, 3);
    threshold(sobel, binary, 35, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(35, 7));
    morphologyEx(binary, closed, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(closed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) { dst = src.clone(); return; }

    Rect bestRect;
    double bestScore = 0;
    bool foundPlate = false;
    // �����Ѿ�����ɫ��λ�������ͼ����ԭͼС�ࣩܶ����������ʵ��µ�
    double minArea = src.cols * src.rows * 0.05;

    for (const auto& cnt : contours)
    {
        double area = contourArea(cnt);
        if (area < minArea) continue;

        Rect rect = boundingRect(cnt);
        float ratio = (float)rect.width / rect.height;
        if (ratio < 1.5 || ratio > 7.0) continue;

        double score = area * (1.0 / abs(ratio - 3.5));
        if (score > bestScore)
        {
            bestScore = score;
            bestRect = rect;
            foundPlate = true;
        }
    }

    if (!foundPlate) { dst = src.clone(); return; }

    // ȷ����Խ��
    bestRect &= Rect(0, 0, src.cols, src.rows);
    if (bestRect.width <= 0 || bestRect.height <= 0) { dst = src.clone(); return; }

    // ������չ4%�߾�
    int dw = bestRect.width * 0.04;
    int dh = bestRect.height * 0.04;
    bestRect.x = max(0, bestRect.x - dw);
    bestRect.y = max(0, bestRect.y - dh);
    bestRect.width = min(src.cols - bestRect.x, bestRect.width + 2 * dw);
    bestRect.height = min(src.rows - bestRect.y, bestRect.height + 2 * dh);

    dst = src(bestRect).clone();
}

// ===================== ��б���� =====================
// ���룺��ɫ����ͼ����ȷ��λ��Ľ����
// �����͸�ӽ��� + Radonб�������Ĳ�ɫ����
void CPlateProcessor::correctPlate(const cv::Mat& src, cv::Mat& dst)
{

    if (src.empty()) { dst = src.clone(); return; }
    /*
    // ---------- ��ȷ��ȡ�����ı��� ----------
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    Mat maskBlue, maskGreen, maskYellow, mask;
    // ����ʵ�ʳ�����ɫ������ֵ
    inRange(hsv, Scalar(100, 80, 80), Scalar(130, 255, 255), maskBlue);
    inRange(hsv, Scalar(40, 60, 60), Scalar(90, 255, 255), maskGreen);
    inRange(hsv, Scalar(15, 100, 100), Scalar(40, 255, 255), maskYellow);
    mask = maskBlue | maskGreen | maskYellow;
    */

    Mat mask = Mat::zeros(src.size(), CV_8UC1);


    // ��̬ѧȥ�� + ����С����
    Mat kernel_close = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel_close);
    Mat kernel_open = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel_open);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) { dst = src.clone(); return; }

    // ȡ�����ͨ��
    int maxIdx = 0;
    double maxArea = contourArea(contours[0]);
    for (int i = 1; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > maxArea) { maxArea = area; maxIdx = i; }
    }

    // ʹ��͹�� + ����αƽ����õ�����ȷ���ı���
    vector<Point> hull;
    convexHull(contours[maxIdx], hull);
    vector<Point> approx;
    double epsilon = arcLength(hull, true) * 0.02;
    approxPolyDP(hull, approx, epsilon, true);

    // ȷ�����ı���
    if (approx.size() != 4) {
        // ��������ı��Σ�����С��Ӿ��ζ���
        RotatedRect rect = minAreaRect(contours[maxIdx]);
        Point2f vertices[4];
        rect.points(vertices);
        approx = vector<Point>(vertices, vertices + 4);
    }

    // �������ϡ����ϡ����¡�����
    vector<Point2f> srcPoints(approx.begin(), approx.end());
    sort(srcPoints.begin(), srcPoints.end(), [](Point2f a, Point2f b) {
        return (a.x + a.y) < (b.x + b.y);
        });
    Point2f tl = srcPoints[0], br = srcPoints[3];
    vector<Point2f> left = { srcPoints[1], srcPoints[2] };
    Point2f tr = left[0].x > left[1].x ? left[0] : left[1];
    Point2f bl = left[0].x > left[1].x ? left[1] : left[0];
    srcPoints = { tl, tr, br, bl };

    float widthTop = norm(tr - tl);
    float widthBottom = norm(br - bl);
    int dstWidth = static_cast<int>(min(widthTop, widthBottom)); // ȡ��Сֵ�ɱ�������
    float heightLeft = norm(bl - tl);
    float heightRight = norm(br - tr);
    int dstHeight = static_cast<int>(min(heightLeft, heightRight));
    // ��֤�ߴ粻Ϊ��
    dstWidth = max(dstWidth, 100);
    dstHeight = max(dstHeight, 30);

    vector<Point2f> dstPoints = {
        Point2f(0, 0),
        Point2f(dstWidth - 1, 0),
        Point2f(dstWidth - 1, dstHeight - 1),
        Point2f(0, dstHeight - 1)
    };

    Mat M = getPerspectiveTransform(srcPoints, dstPoints);
    Mat perspImg;
    warpPerspective(src, perspImg, M, Size(dstWidth, dstHeight));

    // ---------- Radon б����� ----------
    Mat gray;
    cvtColor(perspImg, gray, COLOR_BGR2GRAY);
    Mat edges;
    Canny(gray, edges, 50, 150);

    auto radonVariance = [](const Mat& bin, double thetaDeg) -> double {
        double theta = thetaDeg * CV_PI / 180.0;
        double cos_t = cos(theta), sin_t = sin(theta);
        int rows = bin.rows, cols = bin.cols;
        int projLen = int(abs(cols * cos_t) + abs(rows * sin_t)) + 1;
        vector<double> proj(projLen, 0.0);
        for (int y = 0; y < rows; ++y) {
            const uchar* ptr = bin.ptr<uchar>(y);
            for (int x = 0; x < cols; ++x) {
                if (ptr[x] > 0) {
                    double t = x * cos_t + y * sin_t;
                    int idx = int(t + projLen / 2.0);
                    if (idx >= 0 && idx < projLen) proj[idx] += 1.0;
                }
            }
        }
        double mean = 0.0;
        for (double v : proj) mean += v;
        mean /= proj.size();
        double var = 0.0;
        for (double v : proj) var += (v - mean) * (v - mean);
        return var / proj.size();
    };

    double bestAngle = 0.0, maxVar = 0.0;
    for (double ang = -20.0; ang <= 20.0; ang += 0.5) {
        double var = radonVariance(edges, ang);
        if (var > maxVar) {
            maxVar = var;
            bestAngle = ang;
        }
    }

    double shearAngle = bestAngle;
    double shearFactor = tan(shearAngle * CV_PI / 180.0);
    int h = perspImg.rows, w = perspImg.cols;
    Mat M_shear = (Mat_<double>(2, 3) <<
        1, shearFactor, -shearFactor * h / 2,
        0, 1, 0);
    Mat corrected;
    warpAffine(perspImg, corrected, M_shear, Size(w, h),
        INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

    dst = corrected;




    /*
    if (src.empty()) { dst = src.clone(); return; }

    // ---------- HSV��ɫ�ָ���ȡ�ı��� ----------
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    Mat maskBlue, maskGreen, maskYellow, mask;
    inRange(hsv, Scalar(100, 80, 80), Scalar(130, 255, 255), maskBlue);
    inRange(hsv, Scalar(40, 60, 60), Scalar(90, 255, 255), maskGreen);
    inRange(hsv, Scalar(15, 100, 100), Scalar(40, 255, 255), maskYellow);
    mask = maskBlue | maskGreen | maskYellow;

    // ��̬ѧȥ�� + ����С����
    Mat kernel_close = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel_close);
    Mat kernel_open = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel_open);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) { dst = src.clone(); return; }

    // ȡ�����ͨ��
    int maxIdx = 0;
    double maxArea = contourArea(contours[0]);
    for (int i = 1; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > maxArea) { maxArea = area; maxIdx = i; }
    }

    // ͹�� + ����αƽ� �� �ı���
    vector<Point> hull;
    convexHull(contours[maxIdx], hull);
    vector<Point> approx;
    double epsilon = arcLength(hull, true) * 0.02;
    approxPolyDP(hull, approx, epsilon, true);

    // ��������ı��Σ�����С��Ӿ��ζ���
    if (approx.size() != 4) {
        RotatedRect rect = minAreaRect(contours[maxIdx]);
        Point2f vertices[4];
        rect.points(vertices);
        approx = vector<Point>(vertices, vertices + 4);
    }

    // �������ϡ����ϡ����¡�����
    vector<Point2f> srcPoints(approx.begin(), approx.end());
    sort(srcPoints.begin(), srcPoints.end(), [](Point2f a, Point2f b) {
        return (a.x + a.y) < (b.x + b.y);
    });
    Point2f tl = srcPoints[0], br = srcPoints[3];
    vector<Point2f> left = { srcPoints[1], srcPoints[2] };
    Point2f tr = left[0].x > left[1].x ? left[0] : left[1];
    Point2f bl = left[0].x > left[1].x ? left[1] : left[0];
    srcPoints = { tl, tr, br, bl };

    // ����ӦĿ��ߴ�
    float widthTop = norm(tr - tl);
    float widthBottom = norm(br - bl);
    int dstWidth = static_cast<int>(min(widthTop, widthBottom));
    float heightLeft = norm(bl - tl);
    float heightRight = norm(br - tr);
    int dstHeight = static_cast<int>(min(heightLeft, heightRight));
    dstWidth = max(dstWidth, 100);
    dstHeight = max(dstHeight, 30);

    vector<Point2f> dstPoints = {
        Point2f(0, 0),
        Point2f(dstWidth - 1, 0),
        Point2f(dstWidth - 1, dstHeight - 1),
        Point2f(0, dstHeight - 1)
    };

    Mat M = getPerspectiveTransform(srcPoints, dstPoints);
    Mat perspImg;
    warpPerspective(src, perspImg, M, Size(dstWidth, dstHeight));

    // ---------- Radon б����� ----------
    Mat gray;
    cvtColor(perspImg, gray, COLOR_BGR2GRAY);
    Mat edges;
    Canny(gray, edges, 50, 150);

    auto radonVariance = [](const Mat& bin, double thetaDeg) -> double {
        double theta = thetaDeg * CV_PI / 180.0;
        double cos_t = cos(theta), sin_t = sin(theta);
        int rows = bin.rows, cols = bin.cols;
        int projLen = int(abs(cols * cos_t) + abs(rows * sin_t)) + 1;
        vector<double> proj(projLen, 0.0);
        for (int y = 0; y < rows; ++y) {
            const uchar* ptr = bin.ptr<uchar>(y);
            for (int x = 0; x < cols; ++x) {
                if (ptr[x] > 0) {
                    double t = x * cos_t + y * sin_t;
                    int idx = int(t + projLen / 2.0);
                    if (idx >= 0 && idx < projLen) proj[idx] += 1.0;
                }
            }
        }
        double mean = 0.0;
        for (double v : proj) mean += v;
        mean /= proj.size();
        double var = 0.0;
        for (double v : proj) var += (v - mean) * (v - mean);
        return var / proj.size();
    };

    double bestAngle = 0.0, maxVar = 0.0;
    for (double ang = -20.0; ang <= 20.0; ang += 0.5) {
        double var = radonVariance(edges, ang);
        if (var > maxVar) {
            maxVar = var;
            bestAngle = ang;
        }
    }

    double shearAngle = bestAngle;
    double shearFactor = tan(shearAngle * CV_PI / 180.0);
    int h = perspImg.rows, w = perspImg.cols;
    Mat M_shear = (Mat_<double>(2, 3) <<
        1, shearFactor, -shearFactor * h / 2,
        0, 1, 0);
    Mat corrected;
    warpAffine(perspImg, corrected, M_shear, Size(w, h),
        INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

    dst = corrected;*/
}

// ===================== ��ֵ�� =====================
// ���룺������Ĳ�ɫ����
// �������ֵͼ���ڵװ��֣�
void CPlateProcessor::binarizePlate(const Mat& src, Mat& dst)
{

    if (src.empty()) { dst = src.clone(); return; }

    // �������ͼ��
    int plate_type = 0;
    if (src.channels() == 3)
    {
        Mat hsv;
        cvtColor(src, hsv, COLOR_BGR2HSV);

        // �ֱ����������ơ�����ɫ����
        Mat maskBlue, maskYellow, maskGreen;
        inRange(hsv, Scalar(100, 43, 46), Scalar(124, 255, 255), maskBlue);
        inRange(hsv, Scalar(26, 43, 46), Scalar(34, 255, 255), maskYellow);
        inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), maskGreen);

        // ͳ�ƶ�Ӧ��ɫ��������
        int blue = countNonZero(maskBlue);
        int yellow = countNonZero(maskYellow);
        int green = countNonZero(maskGreen);

        // �жϳ��Ƶ�ɫ
        if (blue > yellow && blue > green)
            plate_type = 0;
        else if (yellow > blue && yellow > green)
            plate_type = 1;
        else
            plate_type = 2;
    }

    // ת�Ҷ�ͼ
    Mat gray;
    if (src.channels() == 3)
        cvtColor(src, gray, COLOR_BGR2GRAY);
    else
        gray = src.clone();


    // ��˹ģ��
    GaussianBlur(gray, gray, Size(3, 3), 0.3, 0.3);

    Mat binary;
    if (plate_type == 0)
    {
        // ���ƣ�ԭ���� �� ����OTSU �� ֱ�ӵõ��ڵװ���
        threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    }
    else
    {
        // ����/���ƣ�ԭ���� �� ����OTSU�õ��׵׺��� �� ����Ϊ�ڵװ���
        threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
        bitwise_not(binary, binary); // �ؼ���ͳһ����Ϊ�ڵװ���
    }

    // ���պڵװ���ǿ��У��
    int total_pixels = binary.rows * binary.cols;
    int white_pixels = countNonZero(binary);
    double white_ratio = static_cast<double>(white_pixels) / total_pixels;

    if (white_ratio > 0.65 || white_ratio < 0.2)
    {
        bitwise_not(binary, binary);
    }

    //  ��Ե�޳�
    int rows = binary.rows;
    int cols = binary.cols;
    int midRow = rows / 2;
    const int JUMP_THRESH = 7;

    // ȥ���ϱ�Ե
    int topEdge = -1;
    for (int r = 0; r <= midRow; r++)
    {
        int jump = 0;
        uchar pre = binary.at<uchar>(r, 0);
        for (int c = 1; c < cols; c++)
        {
            uchar cur = binary.at<uchar>(r, c);
            if (pre == 255 && cur == 0) jump++;
            pre = cur;
        }
        if (jump < JUMP_THRESH) topEdge = r;
    }
    if (topEdge >= 0) binary.rowRange(0, topEdge + 1).setTo(0);

    // ȥ���±�Ե
    int bottomEdge = -1;
    for (int r = rows - 1; r >= midRow; r--)
    {
        int jump = 0;
        uchar pre = binary.at<uchar>(r, 0);
        for (int c = 1; c < cols; c++)
        {
            uchar cur = binary.at<uchar>(r, c);
            if (pre == 255 && cur == 0) jump++;
            pre = cur;
        }
        if (jump < JUMP_THRESH) bottomEdge = r;
    }
    if (bottomEdge >= 0) binary.rowRange(bottomEdge, rows).setTo(0);

    // ȥ�����ұ�Ե
    const int EDGE_WIDTH_THRESH = 5;

    // ȥ�����Ե
    int leftEdge = -1;
    for (int c = 0; c < cols; c++)
    {
        int whiteCnt = 0;
        for (int r = 0; r < rows; r++)
            if (binary.at<uchar>(r, c) == 255) whiteCnt++;
        if (whiteCnt < EDGE_WIDTH_THRESH) leftEdge = c;
        else break;
    }
    if (leftEdge >= 0) binary.colRange(0, leftEdge + 1).setTo(0);

    // ȥ���ұ�Ե
    int rightEdge = cols;
    for (int c = cols - 1; c >= 0; c--)
    {
        int whiteCnt = 0;
        for (int r = 0; r < rows; r++)
            if (binary.at<uchar>(r, c) == 255) whiteCnt++;
        if (whiteCnt < EDGE_WIDTH_THRESH) rightEdge = c;
        else break;
    }
    if (rightEdge < cols) binary.colRange(rightEdge, cols).setTo(0);

    dst = binary;
}
