#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "read_input.h"
#include "util.h"
using namespace std;
using namespace cv;
int main(){

	Mat K;
	
	K = readValuesFromTxt("data/intrinsics.txt");
	//std::cout << K;
	//char** imageFiles=new char*[];
	vector<string> imageFiles;
	imageFiles = dirSmart("data/image/*.jpg");
	vector<string> depthFiles;
	depthFiles = dirSmart("data/depth/*.png");
	Mat R = Mat::eye(3,3,CV_64F);
	Mat T = Mat::zeros(3,1,CV_64F);
	Mat cameraRtC2W;
	hconcat(R, T, cameraRtC2W);
	vector<Mat> IMGcamera,XYZcamera;
	Mat IMGcam1,XYZcam1;
	for (int frameID = 0; frameID < 2; frameID++)
	{
		string filename = "./data/image/";
		filename = filename + imageFiles[frameID];
		Mat IMGcam = imread(filename.c_str());
		IMGcamera.push_back(IMGcam);
		if (frameID == 0)
			IMGcam1 = IMGcam;
		filename = "./data/depth/";
		filename = filename + depthFiles[frameID];
		Mat depth = imread(filename, CV_LOAD_IMAGE_ANYDEPTH);
		//std::cout << depth.channels() << std::endl;
//		imshow("depth", depth);
//		cvWaitKey(0);
		Mat XYZcam = depth2XYZcamera(K,depth);
		//std::cout << XYZcam.at<double>(100,100,1) << std::endl;
		XYZcamera.push_back(XYZcam);
		if (frameID == 0)
			XYZcam1 = XYZcam;
		if (frameID == 0)
		{
			R = Mat::eye(3, 3, CV_64F);
			T = Mat::zeros(3, 1, CV_64F);
			Mat camRtC2W;
			hconcat(R, T, camRtC2W);
		}
		else
		{
			align2view(0, IMGcam1, XYZcam1, frameID, IMGcamera[frameID], XYZcamera[frameID]);
		}
	}
	
	system("pause");
	return 1;
}