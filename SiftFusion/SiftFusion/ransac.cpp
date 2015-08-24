#include "ransac.h"

ransac::ransac(vector<vector<double>> P1, vector<vector<double>> P2)
{
	P3D_i = P1;
	P3D_j = P2;
	Mat R = Mat::eye(3, 3, CV_64F);
	Mat T = Mat::zeros(3, 1, CV_64F);
	hconcat(R, T, Rt);
}

void ransac::ransacfitRt()
{
	int s = 3;
	if (P3D_i.size() == s)
		estimateRt();
}

void ransac::estimateRt()
{
	double x1_centroid, x2_centroid, y1_centroid, y2_centroid, z1_centroid, z2_centroid;
	x1_centroid = 0;
	x2_centroid = 0;
	y1_centroid = 0;
	y2_centroid = 0;
	z1_centroid = 0;
	z2_centroid = 0;

	for (int i = 0; i < P3D_i.size(); ++i)
	{
		x1_centroid += P3D_i[i][0];
		x2_centroid += P3D_j[i][0];
		y1_centroid += P3D_i[i][1];
		y2_centroid += P3D_j[i][1];
		z1_centroid += P3D_i[i][2];
		z2_centroid += P3D_j[i][2];
	}
	x1_centroid /= P3D_i.size();
	x2_centroid /= P3D_j.size();
	y1_centroid /= P3D_i.size();
	y2_centroid /= P3D_j.size();
	z1_centroid /= P3D_i.size();
	z2_centroid /= P3D_j.size();

	Mat P1_centrized = Mat::zeros(P3D_i.size(),3,CV_64F);
	Mat P2_centrized = Mat::zeros(P3D_i.size(),3,CV_64F);

	for (int i = 0; i < P3D_i.size(); i++)
	{
		P1_centrized.at<double>(i, 1) = P3D_i[i][0] - x1_centroid;
		P1_centrized.at<double>(i, 2) = P3D_i[i][1] - y1_centroid;
		P1_centrized.at<double>(i, 3) = P3D_i[i][2] - z1_centroid;
		P2_centrized.at<double>(i, 1) = P3D_j[i][0] - x2_centroid;
		P2_centrized.at<double>(i, 2) = P3D_j[i][1] - y2_centroid;
		P2_centrized.at<double>(i, 3) = P3D_j[i][2] - z2_centroid;
	}
	Mat R12 = P2_centrized - P1_centrized;
	transpose(R12,R12);
	Mat R21 = P1_centrized - P2_centrized;
	Mat R22_1 = P1_centrized + P2_centrized;
	Mat R22 = crossTimesMatrix(R22_1);
	Mat B = Mat::zeros(4,4,CV_64F);
	int sizes[3] = { 4, 4};
	Mat A(2, sizes, CV_64F, cv::Scalar(0));
	for (int i = 0; i < P3D_i.size(); i++)
	{
		A.at<double>(0, 1) = R12.at<double>(i, 0);
		A.at<double>(0, 2) = R12.at<double>(i, 1);
		A.at<double>(0, 3) = R12.at<double>(i, 2);
		A.at<double>(1, 0) = R21.at<double>(0, i);
		A.at<double>(2, 0) = R21.at<double>(1, i);
		A.at<double>(3, 0) = R21.at<double>(2, i);
		A.at<double>(1, 1) = R22.at<double>(0, 0, i);
		A.at<double>(1, 2) = R22.at<double>(0, 1, i);
		A.at<double>(1, 3) = R22.at<double>(0, 2, i);
		A.at<double>(2, 1) = R22.at<double>(1, 0, i);
		A.at<double>(2, 2) = R22.at<double>(1, 1, i);
		A.at<double>(2, 3) = R22.at<double>(1, 2, i);
		A.at<double>(3, 1) = R22.at<double>(2, 0, i);
		A.at<double>(3, 2) = R22.at<double>(2, 1, i);
		A.at<double>(3, 3) = R22.at<double>(2, 2, i);
		Mat tranA;
		transpose(A,tranA);
		B = B + tranA*A;
	}
	Mat S, U, V;
	SVD::compute(B, S, U, V);
	Mat tmp = V.rowRange(3, 3).rowRange(0,3);
	Mat quat;
	tmp.copyTo(quat);
	Mat rot = quat2rot(quat);


}

Mat ransac::crossTimesMatrix(Mat V)
{
	int sizes[3] = {V.rows,3,V.cols};
	Mat V_times(3, sizes, CV_64F, cv::Scalar(0));
	for (int i = 0; i < V.cols; i++)
	{
		V_times.at<double>(0, 1, i) = -V.at<double>(2,i);
		V_times.at<double>(0, 2, i) = V.at<double>(1,i);
		V_times.at<double>(1, 0, i) = V.at<double>(2,i);
		V_times.at<double>(1, 2, i) = -V.at<double>(0, i);
		V_times.at<double>(2, 0, i) = -V.at<double>(1, i);
		V_times.at<double>(2, 1, i) = V.at<double>(0, i);
	}
	return V_times;
}

Mat ransac::quat2rot(Mat quat){
	Mat R = Mat::zeros(3,3,CV_64F);
	double q0 = quat.at<double>(0, 0);
	double q1 = quat.at<double>(0, 1);
	double q2 = quat.at<double>(0, 2);
	double q3 = quat.at<double>(0, 3);

	R.at<double>(0, 0) = q0*q0 + q1*q1 - q2*q2 - q3*q3;
	R.at<double>(0, 1) = 2 * (q1*q2 - q0*q3);
	R.at<double>(0, 2) = 2 * (q1*q3 + q0*q2);

	R.at<double>(1, 0) = 2 * (q1*q2 + q0*q3);
	R.at<double>(1, 1) = q0*q0 - q1*q1 + q2*q2 - q3*q3;
	R.at<double>(1, 2) = 2 * (q2*q3 - q0*q1);

	R.at<double>(2, 0) = 2 * (q1*q3 - q0*q2);
	R.at<double>(2, 1) = 2 * (q2*q3 + q0*q1);
	R.at<double>(2, 2) = q0*q0 - q1*q1 - q2*q2 + q3*q3;

	return R;
}