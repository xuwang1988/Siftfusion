#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;
Mat depth2XYZcamera(Mat K, Mat depth);
void align2view(int frameID_i, Mat image_i, Mat XYZcam_i, int frameID_j, Mat image_j, Mat XYZcam_j);
void matchSIFTdesImages(Mat descriptors_i, Mat descriptors_j, vector<int>* matchPointsID_i, vector<int>* matchPointsID_j);
