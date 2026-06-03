
#include "PlateLocator.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

// ==================== Internal helpers (public interface unchanged) ====================

// Sort RotatedRect corners to: topLeft, topRight, bottomRight, bottomLeft
static void sortCorners(Point2f corners[4])
{
    sort(corners, corners + 4, [](const Point2f& a, const Point2f& b) {
        return a.y < b.y;
    });

    Point2f tl, tr, bl, br;
    if (corners[0].x < corners[1].x) { tl = corners[0]; tr = corners[1]; }
    else                              { tl = corners[1]; tr = corners[0]; }
    if (corners[2].x < corners[3].x) { bl = corners[2]; br = corners[3]; }
    else                              { bl = corners[3]; br = corners[2]; }

    corners[0] = tl;
    corners[1] = tr;
    corners[2] = br;
    corners[3] = bl;
}

// Extract rotated plate region via perspective transform (fixes tilt + reduces background)
static Mat extractRotatedPlate(const Mat& src, Point2f corners[4])
{
    double w1 = norm(corners[1] - corners[0]);
    double w2 = norm(corners[2] - corners[3]);
    int W = max(20, (int)cvRound((w1 + w2) * 0.5));

    double h1 = norm(corners[3] - corners[0]);
    double h2 = norm(corners[2] - corners[1]);
    int H = max(10, (int)cvRound((h1 + h2) * 0.5));

    Point2f dstPts[4] = {
        Point2f(0, 0),
        Point2f((float)(W - 1), 0),
        Point2f((float)(W - 1), (float)(H - 1)),
        Point2f(0, (float)(H - 1))
    };

    Mat M = getPerspectiveTransform(corners, dstPts);
    Mat result;
    warpPerspective(src, result, M, Size(W, H));

    return result;
}

// Edge projection refinement: trim residual background after perspective transform
static void refineByProjection(const Mat& plate, Rect& roi)
{
    Mat gray;
    if (plate.channels() == 3)
        cvtColor(plate, gray, COLOR_BGR2GRAY);
    else
        gray = plate.clone();

    // Horizontal edges (Sobel Y) for top/bottom boundaries
    Mat edgeH;
    Sobel(gray, edgeH, CV_32F, 0, 1);
    edgeH = abs(edgeH);

    // Vertical edges (Sobel X) for left/right boundaries
    Mat edgeV;
    Sobel(gray, edgeV, CV_32F, 1, 0);
    edgeV = abs(edgeV);

    // ---- Horizontal projection ----
    vector<float> hProj(edgeH.rows);
    for (int r = 0; r < edgeH.rows; r++) {
        float sum = 0;
        const float* row = edgeH.ptr<float>(r);
        for (int c = 0; c < edgeH.cols; c++) sum += row[c];
        hProj[r] = sum / edgeH.cols;
    }

    Scalar hMean, hStd;
    meanStdDev(Mat(hProj), hMean, hStd);
    float hThresh = (float)(hMean[0] + hStd[0] * 0.5);

    int top = 0, bottom = edgeH.rows - 1;
    for (int r = 0; r < (int)hProj.size(); r++) {
        if (hProj[r] > hThresh) { top = r; break; }
    }
    for (int r = (int)hProj.size() - 1; r >= 0; r--) {
        if (hProj[r] > hThresh) { bottom = r; break; }
    }

    // ---- Vertical projection ----
    vector<float> vProj(edgeV.cols);
    for (int c = 0; c < edgeV.cols; c++) {
        float sum = 0;
        for (int r = 0; r < edgeV.rows; r++)
            sum += edgeV.at<float>(r, c);
        vProj[c] = sum / edgeV.rows;
    }

    Scalar vMean, vStd;
    meanStdDev(Mat(vProj), vMean, vStd);
    float vThresh = (float)(vMean[0] + vStd[0] * 0.5);

    int left = 0, right = edgeV.cols - 1;
    for (int c = 0; c < (int)vProj.size(); c++) {
        if (vProj[c] > vThresh) { left = c; break; }
    }
    for (int c = (int)vProj.size() - 1; c >= 0; c--) {
        if (vProj[c] > vThresh) { right = c; break; }
    }

    // Guard against projection failure
    if (bottom - top < 10) { top = 0; bottom = plate.rows - 1; }
    if (right - left < 30) { left = 0; right = plate.cols - 1; }

    // Small margin
    top = max(0, top - 2);
    bottom = min(plate.rows - 1, bottom + 2);
    left = max(0, left - 2);
    right = min(plate.cols - 1, right + 2);

    roi = Rect(left, top, right - left, bottom - top);
}

// Green plate verification: check vertical color profile to reject vegetation
static bool verifyGreenPlate(const Mat& plate)
{
    if (plate.rows < 15 || plate.cols < 40) return false;

    int h = plate.rows;
    Mat topPart = plate(Rect(0, 0, plate.cols, h / 4));
    Mat midPart = plate(Rect(0, h / 4, plate.cols, h / 2));
    Mat botPart = plate(Rect(0, h * 3 / 4, plate.cols, h - h * 3 / 4));

    Scalar topMean = mean(topPart);
    Scalar midMean = mean(midPart);
    Scalar botMean = mean(botPart);

    // Top should be noticeably brighter (white strip)
    float topBrightness = (float)(topMean[0] + topMean[1] + topMean[2]);
    float botBrightness = (float)(botMean[0] + botMean[1] + botMean[2]);
    if (topBrightness < botBrightness * 1.15f) return false;

    // Mid-bottom should be green (G > R and G > B)
    float midG = (float)midMean[1], midR = (float)midMean[0], midB = (float)midMean[2];
    if (midG < midR + 10 || midG < midB + 10) return false;

    return true;
}



void CPlateLocator::colorLocate(const Mat& src, Mat& dst)
{
    dst = src.clone();
    if (src.empty()) return;

    int cx = src.cols / 2;
    int cy = src.rows / 2;

    //==================== 1. HSV color masks ========================
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    Mat maskBlue, maskYellow, maskGreen, maskColor;
    // Blue plates
    inRange(hsv, Scalar(95, 80, 40), Scalar(135, 255, 255), maskBlue);
    // Yellow plates
    inRange(hsv, Scalar(10, 80, 80), Scalar(40, 255, 255), maskYellow);
    // Green plates: raised saturation lower bound (100->120) to reduce vegetation
    inRange(hsv, Scalar(40, 120, 50), Scalar(80, 255, 220), maskGreen);
    maskColor = maskBlue | maskYellow | maskGreen;

    //==================== 2. Edge detection ========================
    Mat gray, canny;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Canny(gray, canny, 30, 100);

    //==================== 3. Color + edge combination ========================
    Mat combined;
    bitwise_and(maskColor, canny, combined);

    //==================== 4. Morphology ========================
    Mat kernelClose = getStructuringElement(MORPH_RECT, Size(12, 4));
    Mat kernelOpen = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(combined, combined, MORPH_CLOSE, kernelClose);
    morphologyEx(combined, combined, MORPH_OPEN, kernelOpen);

    //==================== 5. Find contours + scoring ========================
    vector<vector<Point>> contours;
    findContours(combined, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    RotatedRect bestRotRect;
    double bestScore = 0;
    bool bestIsGreen = false;

    for (size_t i = 0; i < contours.size(); i++)
    {
        const vector<Point>& cnt = contours[i];

        RotatedRect rr = minAreaRect(cnt);
        float w = max(rr.size.width, rr.size.height);
        float h = min(rr.size.width, rr.size.height);

        if (w < 30 || h < 10) continue;
        if (h <= 1) continue;

        double ratio = w / h;
        if (ratio < 1.5 || ratio > 8.0) continue;

        double area = contourArea(cnt);
        double rectArea = w * h;
        if (rectArea < 1) continue;
        double fillRate = area / rectArea;
        if (fillRate < 0.25) continue;

        // ---- Green candidate: texture check to filter out vegetation ----
        Rect br = rr.boundingRect();
        br &= Rect(0, 0, src.cols, src.rows);
        if (br.width <= 0 || br.height <= 0) continue;

        Mat roi = src(br);
        Scalar meanVal = mean(roi);
        bool isGreen = (meanVal[1] > meanVal[2] + 25 && meanVal[1] > meanVal[0] + 25);

        if (isGreen) {
            _IsGreen = true;
            Mat roiGray, roiEdge;
            cvtColor(roi, roiGray, COLOR_BGR2GRAY);
            Sobel(roiGray, roiEdge, CV_32F, 1, 0);
            roiEdge = abs(roiEdge);

            // Column-wise projection: peaks = character edges, flat = vegetation
            Mat colProj;
            reduce(roiEdge, colProj, 0, REDUCE_AVG);

            Scalar projMean, projStd;
            meanStdDev(colProj, projMean, projStd);
            double maxVal;
            minMaxLoc(colProj, nullptr, &maxVal);
            double peakRatio = maxVal / (projMean[0] + 1e-6);

            if (peakRatio < 2.0) continue;
        }

        // ---- Position scoring ----
        Point2f center = rr.center;
        double dist_score = 1.0
            - 0.0005 * abs(center.x - cx)
            - 0.001 * abs(center.y - (cy + 60));

        if (dist_score < 0.2) continue;

        double score = w * h * fillRate * dist_score;

        if (score > bestScore) {
            bestScore = score;
            bestRotRect = rr;
            bestIsGreen = isGreen;
        }
    }

    //==================== 6. Output: perspective transform + boundary refinement ========================
    if (bestScore > 0)
    {
        // 6a. Get sorted corners from the best rotated rect
        Point2f corners[4];
        bestRotRect.points(corners);
        sortCorners(corners);

        // 6b. Compute direction vectors for corner expansion
        Point2f topCenter = (corners[0] + corners[1]) * 0.5f;
        Point2f botCenter = (corners[2] + corners[3]) * 0.5f;
        Point2f leftCenter = (corners[0] + corners[3]) * 0.5f;
        Point2f rightCenter = (corners[1] + corners[2]) * 0.5f;

        Point2f upDir = topCenter - botCenter;
        Point2f rightDir = rightCenter - leftCenter;

        float upLen = (float)norm(upDir);
        float rightLen = (float)norm(rightDir);
        if (upLen > 1e-3f) upDir /= upLen;
        if (rightLen > 1e-3f) rightDir /= rightLen;

        // Default expansion margins
        float expandAll = 3.0f;
        float expandTop = expandAll;

        // Green plate: extend top significantly to recover the white strip
        if (bestIsGreen) {
            float plateH = min(bestRotRect.size.width, bestRotRect.size.height);
            expandTop = max(10.0f, plateH * 0.25f);
        }

        // Expand four corners outward
        corners[0] += -rightDir * expandAll + upDir * expandTop;   // topLeft
        corners[1] += rightDir * expandAll + upDir * expandTop;   // topRight
        corners[2] += rightDir * expandAll - upDir * expandAll;   // bottomRight
        corners[3] += -rightDir * expandAll - upDir * expandAll;   // bottomLeft

        // Clamp to image bounds
        for (int i = 0; i < 4; i++) {
            corners[i].x = max(0.0f, min((float)(src.cols - 1), corners[i].x));
            corners[i].y = max(0.0f, min((float)(src.rows - 1), corners[i].y));
        }

        // 6c. Perspective transform extraction
        Mat plate = extractRotatedPlate(src, corners);

        if (plate.rows >= 10 && plate.cols >= 30)
        {
            // 6d. Projection-based refinement
            Rect refineRoi;
            refineByProjection(plate, refineRoi);

            if (refineRoi.width > 20 && refineRoi.height > 8)
                plate = plate(refineRoi).clone();

            // 6e. Green plate final check
            if (bestIsGreen && !verifyGreenPlate(plate)) {
                dst = src.clone();
            }
            else {
                dst = plate;
            }
        }
    }
}
