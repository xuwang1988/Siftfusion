#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
cv::Mat readValuesFromTxt(char* path);
vector<string> dirSmart(char* path);