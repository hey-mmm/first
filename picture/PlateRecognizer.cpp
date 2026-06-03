#include "PlateRecognizer.h"
#include <windows.h> // 用于 GetModuleFileNameW
#include <cstring>
#include <map>
static const std::map<int, std::string> CHN_MAP = {
    {101, "鲁"}, {102, "京"}, {103, "津"}, {104, "沪"}, {105, "渝"}, {106, "冀"}, {107, "豫"}, {108, "云"}, {109, "辽"}, {110, "黑"}, {111, "湘"}, {112, "皖"}, {113, "闽"}, {114, "新"}, {115, "苏"}, {116, "浙"}, {119, "吉"}, {120, "粤"}, {121, "桂"}, {122, "琼"}, {123, "川"}, {125, "贵"}, {127, "陕"}, {128, "甘"}, {130, "青"}, {131, "藏"}, {132, "蒙"}, {133, "宁"}, {134, "晋"}};
static const std::map<int, std::string> enMap = {
    {48, "0"}, {49, "1"}, {50, "2"}, {51, "3"}, {52, "4"}, {53, "5"}, {54, "6"}, {55, "7"}, {56, "8"}, {57, "9"}, {65, "A"}, {66, "B"}, {67, "C"}, {68, "D"}, {69, "E"}, {70, "F"}, {71, "G"}, {72, "H"}, {74, "J"}, {75, "K"}, {76, "L"}, {77, "M"}, {78, "N"}, {80, "P"}, {81, "Q"}, {82, "R"}, {83, "S"}, {84, "T"}, {85, "U"}, {86, "V"}, {87, "W"}, {88, "X"}, {89, "Y"}, {90, "Z"}};
static std::string getExeDir()
{
    wchar_t wpath[MAX_PATH] = {};
    GetModuleFileNameW(NULL, wpath, MAX_PATH);
    std::wstring wstr(wpath);
    size_t pos = wstr.find_last_of(L"\\/");
    if (pos != std::wstring::npos)
    {
        wstr = wstr.substr(0, pos + 1);
    }
    int len = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    std::string path(len - 1, 0);
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &path[0], len, NULL, NULL);
    return path;
}

std::string getExeDirectory() { return getExeDir(); }

bool CPlateRecognizer::loadAllModels(const std::string &zhPrefix, const std::string &enNumPrefix)
{
    try
    {
        std::string baseDir = getExeDir();
        m_zhPca.load(baseDir + zhPrefix + "1_pca_model.xml");
        m_zhSvm = cv::ml::SVM::load(baseDir + zhPrefix + "1_svm_model.xml");
        m_enPca.load(baseDir + enNumPrefix + "2_pca_model.xml");
        m_enSvm = cv::ml::SVM::load(baseDir + enNumPrefix + "2_svm_model.xml");
        return (!m_zhSvm.empty() && !m_enSvm.empty());
    }
    catch (...)
    {
        return false;
    }
}

// 识别函数：直接返回 std::string
// 使用纯 HOG 特征（900 维）→ PCA 降维 → SVM 分类
std::string CPlateRecognizer::recognize(const cv::Mat &charImg, bool isChinese)
{
    if (charImg.empty())
        return "?";

    // 1. 提取 HOG 特征（900 维，内部自动调用 preprocessChar）
    std::vector<double> hogFeat;
    getHogOne(charImg, hogFeat);
    if (hogFeat.size() != 900)
        return "?";

    // 2. 转换为 OpenCV 矩阵 (1 x 900, CV_32F)
    cv::Mat featureMat(1, 900, CV_32F);
    for (int j = 0; j < 900; ++j)
        featureMat.at<float>(0, j) = static_cast<float>(hogFeat[j]);

    // 3. PCA 降维 + SVM 分类
    cv::Mat reduced;
    int classId;

    if (isChinese)
    {
        if (m_zhSvm.empty())
            return "?";
        reduced = m_zhPca.transform(featureMat);
        classId = cvRound(m_zhSvm->predict(reduced));
        std::cout << "  [DEBUG ZH] SVM classId=" << classId << " -> " << getChineseString(classId) << std::endl;
        return getChineseString(classId);
    }
    else
    {
        if (m_enSvm.empty())
            return "?";
        reduced = m_enPca.transform(featureMat);
        classId = cvRound(m_enSvm->predict(reduced));
        std::cout << "  [DEBUG EN] SVM classId=" << classId << " -> " << getEnNumString(classId) << std::endl;
        return getEnNumString(classId);
    }
}

// 辅助映射函数保持不变（只需返回 std::string）
std::string CPlateRecognizer::getChineseString(int labelId)
{

    auto it = CHN_MAP.find(labelId);
    if (it != CHN_MAP.end())
        return it->second;
    else
        return "?";
}
std::string CPlateRecognizer::getEnNumString(int labelId)
{

    auto it = enMap.find(labelId);
    if (it != enMap.end())
        return it->second;
    else
        return "?";
}
