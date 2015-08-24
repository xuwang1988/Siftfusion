#include <iostream>

#include"util.h"
#include"ransac.h"
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

Mat depth2XYZcamera(Mat K, Mat depth){
	int sizes[3] = {424, 512 , 4};
	Mat XYZcamera(3,sizes,CV_64F,cv::Scalar(0));
	
	for(int i = 0; i < depth.rows; i++)
	{
		for(int j = 0; j < depth.cols; j++)
		{
			//double m = (i - K.at<double>(0, 2))*depth.at<short>(i, j) / K.at<double>(0, 0);
			XYZcamera.at<double>(i, j, 0) = (i - K.at<double>(0, 2))*depth.at<short>(i, j) / K.at<double>(0, 0)/20000.0;
			XYZcamera.at<double>(i, j, 1) = (j - K.at<double>(1, 2))*depth.at<short>(i, j) / K.at<double>(1, 1)/20000.0;
			XYZcamera.at<double>(i, j, 2) = depth.at<short>(i, j)/20000.0;
			XYZcamera.at<double>(i, j, 3) = depth.at<short>(i, j) == 0 ? 0 : 1;
		}
	}

	return XYZcamera;
}

void align2view(int frameID_i, Mat image_i, Mat XYZcam_i, int frameID_j, Mat image_j, Mat XYZcam_j){

	double error3D_threshold = 0.05;
	double error3D_threshold2 = error3D_threshold *error3D_threshold;
	int minHessian = 400;
	SiftFeatureDetector detector(minHessian);
	vector<KeyPoint> keypoints_i, keypoints_j;
	detector.detect(image_i,keypoints_i);
	detector.detect(image_j, keypoints_j);
	SiftDescriptorExtractor extractor;

	Mat descriptors_i, descriptors_j;

	extractor.compute(image_i, keypoints_i, descriptors_i);
	extractor.compute(image_j, keypoints_j, descriptors_j);
	vector<int> matchPointsID_i, matchPointsID_j;
	matchSIFTdesImages(descriptors_i, descriptors_j, &matchPointsID_i, &matchPointsID_j);

	vector<KeyPoint> SIFTloc1_i, SIFTloc1_j;
	for (int i = 0; i < matchPointsID_i.size(); i++)
	{
		SIFTloc1_i.push_back(keypoints_i[matchPointsID_i[i]]);
		SIFTloc1_j.push_back(keypoints_j[matchPointsID_j[i]]);
	}
	vector<KeyPoint> posSIFT_i, posSIFT_j;
	vector<int> valid;
	for (int i = 0; i < SIFTloc1_i.size(); i++)
	{
		Point2f pt1 = SIFTloc1_i[i].pt;
		Point2f pt2 = SIFTloc1_j[i].pt;
		if (round(pt1.x) >= 0 && round(pt1.x) < image_i.cols && round(pt1.y) >= 0 && round(pt1.y) < image_i.rows && round(pt2.x) >= 0 && round(pt2.x) < image_j.cols && round(pt2.y) >= 0 && round(pt2.y) < image_j.rows)
		{
			valid.push_back(i);
		}
	}
	vector<KeyPoint> SIFTloc_i, SIFTloc_j;
	for (int i = 0; i < valid.size(); i++)
	{
		//cout << valid[i] << endl;
		SIFTloc_i.push_back(SIFTloc1_i[valid[i]]);
		SIFTloc_j.push_back(SIFTloc1_j[valid[i]]);
		posSIFT_i.push_back(SIFTloc1_i[valid[i]]);
		posSIFT_j.push_back(SIFTloc1_j[valid[i]]);
	}
	for (int i = 0; i < valid.size(); i++)
	{
		posSIFT_i[i].pt.x = round(posSIFT_i[i].pt.x);
		posSIFT_i[i].pt.y = round(posSIFT_i[i].pt.y);
		posSIFT_j[i].pt.x = round(posSIFT_j[i].pt.x);
		posSIFT_j[i].pt.y = round(posSIFT_j[i].pt.y);
	}
	Mat Xcam_i, Ycam_i, Zcam_i, validM_i;
	Xcam_i = Mat::zeros(XYZcam_i.size[0], XYZcam_j.size[1], CV_64F);
	Xcam_i.copyTo(Ycam_i);
	Xcam_i.copyTo(Zcam_i);
	validM_i = Mat::zeros(XYZcam_i.size[0], XYZcam_j.size[1], CV_16U);
	Mat Xcam_j, Ycam_j, Zcam_j, validM_j;
	Xcam_i.copyTo(Xcam_j);
	Xcam_i.copyTo(Ycam_j);
	Xcam_i.copyTo(Zcam_j);
	validM_i.copyTo(validM_j);
	//vector<int> valid;
	for (int i = 0; i < XYZcam_i.size[0]; ++i)
	{
		for (int j = 0; j < XYZcam_i.size[1]; ++j)
		{
			Xcam_i.at<double>(i, j) = XYZcam_i.at<double>(i, j, 0);
			Ycam_i.at<double>(i, j) = XYZcam_i.at<double>(i, j, 1);
			Zcam_i.at<double>(i, j) = XYZcam_i.at<double>(i, j, 2);
			validM_i.at<short>(i, j) = XYZcam_i.at<double>(i, j, 3) > 0.5 ? 1 : 0;
			Xcam_j.at<double>(i, j) = XYZcam_j.at<double>(i, j, 0);
			Ycam_j.at<double>(i, j) = XYZcam_j.at<double>(i, j, 1);
			Zcam_j.at<double>(i, j) = XYZcam_j.at<double>(i, j, 2);
			validM_j.at<short>(i, j) = XYZcam_j.at<double>(i, j, 3) > 0.5 ? 1 : 0;
			//if (validM_j.at<short>(i, j) == 1 && validM_i.at<short>(i, j))
				//cout << i << "  " << j << endl;
		}
	}
	vector<KeyPoint> ind_i, ind_j;
	for (int i = 0; i < posSIFT_i.size(); ++i)
	{
		ind_i.push_back(posSIFT_i[i]);
		ind_j.push_back(posSIFT_j[i]);
	}
	//Mat valid_ind = Mat::zeros(XYZcam_i.size[0], XYZcam_j.size[1], CV_16U);
	vector<vector<double>> P3D_i, P3D_j;
	vector<KeyPoint> SIFT_i, SIFT_j;
	for (int i = 0; i < ind_i.size(); i++)
	{
		//valid_ind.at<unsigned short>(i, j) = 1;
		if (validM_i.at<short>(ind_i[i].pt.y, ind_i[i].pt.x) == 1 && validM_j.at<short>(ind_j[i].pt.y, ind_j[i].pt.x) == 1)
		{
			vector<double> pt1;
			pt1.push_back(XYZcam_i.at<double>(ind_i[i].pt.y, ind_i[i].pt.x, 0));
			pt1.push_back(XYZcam_i.at<double>(ind_i[i].pt.y, ind_i[i].pt.x, 1));
			pt1.push_back(XYZcam_i.at<double>(ind_i[i].pt.y, ind_i[i].pt.x, 2));
			P3D_i.push_back(pt1);
			vector<double> pt2;
			pt2.push_back(XYZcam_j.at<double>(ind_j[i].pt.y, ind_j[i].pt.x, 0));
			pt2.push_back(XYZcam_j.at<double>(ind_j[i].pt.y, ind_j[i].pt.x, 1));
			pt2.push_back(XYZcam_j.at<double>(ind_j[i].pt.y, ind_j[i].pt.x, 2));
			P3D_j.push_back(pt2);
			SIFT_i.push_back(SIFTloc_i[i]);
			SIFT_j.push_back(SIFTloc_j[i]);
			Mat RtRANSAC= ransacfitRt(P3D_i,P3D_j, error3D_threshold, 0);
		}
	}

}

void matchSIFTdesImages(Mat descriptors_i, Mat descriptors_j, vector<int>* matchPointsID_i, vector<int>* matchPointsID_j){
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_i, descriptors_j, matches);
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_i.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_i.rows; i++)
	{
		if (matches[i].distance < 10 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		matchPointsID_i->push_back(good_matches[i].queryIdx);
		matchPointsID_j->push_back(good_matches[i].trainIdx);
	}
	//printf("-- num of matches : %d \n", good_matches.size());
}