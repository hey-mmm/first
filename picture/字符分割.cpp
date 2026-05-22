#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

// 查找字符结束位置
// 优化：降低阈值到 0.2，只要像素明显下降就认为字符结束
int findEnd(int start, bool arg, const vector<int>& black, const vector<int>& white,
            int width, int black_max, int white_max) {
    int end = start + 1;
    // 阈值改为 0.2，原代码 0.95 太高了，会导致字符粘连分不开
    double threshold_ratio = 0.2; 
    int max_val = arg ? black_max : white_max;

    for (int m = start + 1; m < width - 1; m++) {
        int pixel_count = arg ? black[m] : white[m];
        // 当像素数量跌落到最大值的 20% 以下时，认为字符结束
        if (pixel_count < threshold_ratio * max_val) {
            end = m;
            break;
        }
    }
    return end;
}

// 字符分割主函数
void charSegmentation(const Mat& thresh) {
    int height = thresh.rows;
    int width = thresh.cols;

    vector<int> white(width, 0);
    vector<int> black(width, 0);
    int white_max = 0;
    int black_max = 0;

    // 1. 统计每一列的黑白像素总数（垂直投影）
    for (int i = 0; i < width; i++) {
        int line_white = 0;
        int line_black = 0;
        for (int j = 0; j < height; j++) {
            uchar pixel = thresh.at<uchar>(j, i);
            if (pixel == 255) line_white++;
            if (pixel == 0)   line_black++;
        }
        white_max = max(white_max, line_white);
        black_max = max(black_max, line_black);
        white[i] = line_white;
        black[i] = line_black;
    }

    // 2. 判断字符颜色模式
    // arg = true  → 黑底白字（黑色背景多，字符是白色，所以看 white_max）
    // arg = false → 白底黑字（白色背景多，字符是黑色，所以看 black_max）
    bool arg = true;
    if (black_max > white_max) {
        arg = false; // 白底黑字
    }

    cout << "颜色模式: " << (arg ? "黑底白字 (看白色像素)" : "白底黑字 (看黑色像素)") << endl;
    cout << "white_max = " << white_max << ", black_max = " << black_max << endl;

    // 用来保存找到的所有字符区域，最后统一排序
    vector<Rect> charRects;

    // 3. 分割车牌字符
    int n = 0; // 从第0列开始扫描
    while (n < width - 2) {
        n++;
        // 判断当前列是否属于字符区域的起点
        // 黑底白字(arg=true)：看白色像素是否超过阈值
        // 白底黑字(arg=false)：看黑色像素是否超过阈值
        double start_threshold = 0.05; // 起点阈值保持较低
        int max_val = arg ? white_max : black_max;
        int pixel_count = arg ? white[n] : black[n];

        if (pixel_count > start_threshold * max_val) {
            int start = n;
            int end = findEnd(start, arg, black, white, width, black_max, white_max);
            n = end; // 跳过当前字符，继续往后找

            int charWidth = end - start;

            // 4. 过滤条件优化
            // - 宽度必须大于5（过滤噪点）
            // - 宽度必须小于总宽度的 1/4（过滤掉连在一起的多个字符或大边框）
            // - 高度必须大于图像高度的 1/3（过滤掉车牌上下的细边框）
            if (charWidth > 5 && charWidth < width / 4) {
                // 裁剪字符区域
                int x1 = max(0, start - 1);
                int x2 = min(width - 1, end + 1);
                Rect cropRect(x1, 0, x2 - x1 + 1, height);
                charRects.push_back(cropRect);
            }
        }
    }

    // 5. 按从左到右的顺序排序（防止窗口乱跳）
    sort(charRects.begin(), charRects.end(), [](const Rect& a, const Rect& b) {
        return a.x < b.x;
    });

    // 6. 统一展示分割结果
    cout << "成功分割出 " << charRects.size() << " 个字符" << endl;
    for (size_t i = 0; i < charRects.size(); i++) {
        Mat cropImg = thresh(charRects[i]).clone();
        Mat resized;
        resize(cropImg, resized, Size(34, 56));

        // 窗口名使用序号 i，清晰明了
        string winName = "Char_" + to_string(i + 1);
        namedWindow(winName, WINDOW_NORMAL | WINDOW_KEEPRATIO);
        imshow(winName, resized);
        
        // 窗口排成一排，方便查看
        moveWindow(winName, 300 + i * 100, 100);
    }
}

int main() {
    Mat src = imread("image.png", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "无法加载图像 image.png！" << endl;
        return -1;
    }

    // 确保是二值图像
    Mat binary;
    threshold(src, binary, 128, 255, THRESH_BINARY);

    namedWindow("原始二值图像", WINDOW_NORMAL | WINDOW_KEEPRATIO);
    imshow("原始二值图像", binary);

    charSegmentation(binary);

    waitKey(0);
    destroyAllWindows();
    return 0;
}