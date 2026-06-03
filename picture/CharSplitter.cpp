
#include "CharSplitter.h"
#include <algorithm>
#include <cmath>

// ===================== �������� =====================
namespace {
    const int PLATE_RESIZE_WIDTH = 136;
    const int PLATE_RESIZE_HEIGHT = 44;
    const int RIVET_KERNEL_SIZE = 2;
    const int EXPECT_CHAR_COUNT = 8;
    const int SEARCH_RADIUS = 6;
    const int MIN_CHAR_WIDTH = 7;
    const int MAX_CHAR_WIDTH = 22;
    const int GAP_THRESHOLD = 1;
    const float OVERSPLIT_THRESHOLD = 1.5f;

    // ���ұ߿�ȥ����ֵ��ͶӰֵС�ڴ�ֵ��Ϊ�Ǳ߿�
    const int BORDER_PROJ_THRESHOLD = 5;

    // ��ֱͶӰ����
    std::vector<int> computeVerticalProjection(const cv::Mat& img) {
        std::vector<int> projection(img.cols, 0);
        for (int x = 0; x < img.cols; ++x) {
            projection[x] = cv::countNonZero(img.col(x));
        }
        return projection;
    }

    // ��׼���ߴ�
    cv::Mat normalizePlate(const cv::Mat& src) {
        cv::Mat plate;
        cv::resize(src, plate, cv::Size(PLATE_RESIZE_WIDTH, PLATE_RESIZE_HEIGHT));
        return plate;
    }
}

// ===================== ȥ�����ұ߿� =====================
void CCharSplitter::removeBorder(cv::Mat& img) {
    if (img.empty()) return;

    auto projection = computeVerticalProjection(img);
    int leftBorder = 0;
    int rightBorder = img.cols - 1;

    // ���������ҵ�һ����Ч�ַ�����ʼλ��
    for (int x = 0; x < img.cols; ++x) {
        if (projection[x] > BORDER_PROJ_THRESHOLD) {
            leftBorder = x;
            break;
        }
    }

    // �������������һ����Ч�ַ��Ľ���λ��
    for (int x = img.cols - 1; x >= 0; --x) {
        if (projection[x] > BORDER_PROJ_THRESHOLD) {
            rightBorder = x;
            break;
        }
    }

    // �ü����ұ߿�
    if (leftBorder < rightBorder) {
        cv::Rect roi(leftBorder, 0, rightBorder - leftBorder + 1, img.rows);
        img = img(roi).clone();
    }
}

// ===================== ��̬ѧȥí�� =====================
void CCharSplitter::removeRivet(cv::Mat& img) {
    if (img.empty()) return;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(RIVET_KERNEL_SIZE, RIVET_KERNEL_SIZE));
    cv::morphologyEx(img, img, cv::MORPH_OPEN, kernel);
}

// ===================== Ԥ����ͳһ���� =====================
cv::Mat CCharSplitter::preprocessPlate(const cv::Mat& src) {
    cv::Mat plate = normalizePlate(src);
    removeBorder(plate);  // ֻȥ�����ұ߿�
    removeRivet(plate);
    return plate;
}

// ===================== �����ƾ��ַָ� =====================
std::vector<cv::Mat> CCharSplitter::splitByProjection(const cv::Mat& img) {
    std::vector<cv::Mat> chars;
    const int width = img.cols;
    const int height = img.rows;

    auto projection = computeVerticalProjection(img);
    std::vector<int> splitPositions;

    int expectedCharWidth = width / EXPECT_CHAR_COUNT;

    // Ѱ�����ŷָ�㣨ͶӰ��Сֵ��
    for (int i = 1; i < EXPECT_CHAR_COUNT; ++i) {
        int center = i * expectedCharWidth;
        int bestPos = center;
        int minValue = std::numeric_limits<int>::max();

        int searchStart = std::max(0, center - SEARCH_RADIUS);
        int searchEnd = std::min(width - 1, center + SEARCH_RADIUS);

        for (int p = searchStart; p <= searchEnd; ++p) {
            if (projection[p] < minValue) {
                minValue = projection[p];
                bestPos = p;
            }
        }
        splitPositions.push_back(bestPos);
    }

    // ���ݷָ���и��ַ�
    int left = 0;
    for (int pos : splitPositions) {
        if (pos > left + SEARCH_RADIUS - 1) {
            chars.emplace_back(img(cv::Rect(left, 0, pos - left, height)).clone());
        }
        left = pos;
    }

    // �������һ���ַ�
    if (width - left > SEARCH_RADIUS - 1) {
        chars.emplace_back(img(cv::Rect(left, 0, width - left, height)).clone());
    }

    return chars;
}

// ===================== ����ͶӰ�ָ� =====================
std::vector<cv::Mat> CCharSplitter::splitGreenByProj(const cv::Mat& img) {
    std::vector<cv::Mat> result;
    const int width = img.cols;
    const int height = img.rows;

    auto projection = computeVerticalProjection(img);

    bool inCharacter = false;
    int start = 0;

    for (int x = 0; x < width; ++x) {
        if (!inCharacter && projection[x] > GAP_THRESHOLD) {
            start = x;
            inCharacter = true;
        }
        else if (inCharacter && projection[x] <= GAP_THRESHOLD) {
            int charWidth = x - start;
            if (charWidth > MIN_CHAR_WIDTH) {
                splitWideCharacter(img, result, start, charWidth, height);
            }
            inCharacter = false;
        }
    }

    // �������һ���ַ�
    if (inCharacter) {
        int charWidth = width - start;
        if (charWidth > MIN_CHAR_WIDTH) {
            splitWideCharacter(img, result, start, charWidth, height);
        }
    }

    return result;
}

// �������������������ַ������ָܷ�
void CCharSplitter::splitWideCharacter(const cv::Mat& img, std::vector<cv::Mat>& result,
    int start, int width, int height) {
    if (width > MAX_CHAR_WIDTH * OVERSPLIT_THRESHOLD) {
        int splitCount = std::max(2, static_cast<int>(std::round(static_cast<double>(width) / MAX_CHAR_WIDTH)));
        int subWidth = width / splitCount;

        for (int s = 0; s < splitCount; ++s) {
            cv::Rect subRect(start + s * subWidth, 0, subWidth, height);
            result.emplace_back(img(subRect).clone());
        }
    }
    else {
        result.emplace_back(img(cv::Rect(start, 0, width, height)).clone());
    }
}

// ===================== ���ָ�ӿ� =====================
void CCharSplitter::splitChars(const cv::Mat& src, std::vector<cv::Mat>& charImages, bool isGreen) {
    charImages.clear();

    cv::Mat plate = preprocessPlate(src);

    std::vector<cv::Mat> parts = isGreen ? splitGreenByProj(plate) : splitByProjection(plate);

    // ���ȡǰ8����Ч�ַ�
    for (size_t i = 0; i < parts.size() && charImages.size() < EXPECT_CHAR_COUNT; ++i) {
        charImages.push_back(parts[i]);
    }

    // ��ȫ����8����λ��
    while (charImages.size() < EXPECT_CHAR_COUNT) {
        charImages.emplace_back(cv::Mat::zeros(cv::Size(16, 38), CV_8UC1));
    }
}

// ���ݾɽӿ�
void CCharSplitter::splitChars(const cv::Mat& src, std::vector<cv::Mat>& charImages) {
    splitChars(src, charImages, false);
}

// ===================== �����Զ���� =====================
bool CCharSplitter::detectGreenPlate(const cv::Mat& src) {
    cv::Mat plate = preprocessPlate(src);
    auto parts = splitGreenByProj(plate);
    return parts.size() >= EXPECT_CHAR_COUNT;
}