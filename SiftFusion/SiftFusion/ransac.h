
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class ransac
{
	vector<vector<double>> P3D_i, P3D_j;
	double error;
	Mat Rt;
	ransac(vector<vector<double>> P1,vector<vector<double>> P2);
	void ransacfitRt();
	void estimateRt();
	Mat crossTimesMatrix(Mat V);
	Mat quat2rot(Mat quat);
};