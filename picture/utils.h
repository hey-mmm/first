#include <opencv2/opencv.hpp>
#include <vector>
cv::Mat vec2dToMatF(const std::vector<std::vector<double>> &features)
{
    const int rows = (int)features.size();
    const int cols = (int)features[0].size();
    cv::Mat data(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            data.at<float>(i, j) = static_cast<float>(features[i][j]);
    return data;
}
