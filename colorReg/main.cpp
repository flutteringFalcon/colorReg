#include<iostream>
#include<opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat imOrin = imread("..\\image\\clock.jpg");
	Size imOrinSize = imOrin.size();

	const int THR = 110;              // 红色阈值
	const int THG = 60;               // 绿色阈值
	const int THB = 60;               // 蓝色阈值
	int count = 1;                    // 计数器

	Mat imRedMark(imOrinSize, CV_8UC3, Scalar::all(0));         // 存放红色表针图像
	for (int i = 0; i < imOrinSize.height; ++i) {               // 确定红色表针位置
		for (int j = 0; j < imOrinSize.width; ++j) {
			((imOrin.at<Vec3b>(i, j)[2] > THR) \
				&& (imOrin.at<Vec3b>(i, j)[1] < THG) \
				&& (imOrin.at<Vec3b>(i, j)[0] < THB)) \
				? imRedMark.at<Vec3b>(i, j) = Vec3b(0, 0, 255) \
				: imRedMark.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	}


	Mat imGrayRedMark(imOrinSize, CV_8UC1, Scalar::all(0));
	Mat imGrayRedMarkEdge(imOrinSize, CV_8UC1, Scalar::all(0));
	cvtColor(imRedMark, imGrayRedMark, COLOR_BGR2GRAY);
	Canny(imGrayRedMark, imGrayRedMarkEdge, 0.4, 0.8);                        // 边缘检测

	int labelNum;
	Mat imLabel(imOrinSize, CV_16UC1, Scalar::all(0));
	labelNum = connectedComponents(imGrayRedMarkEdge, imLabel, 8, CV_16U);    // 标记区域数目包括背景区域

	Mat_<int> stats(labelNum, 5);                                             // 检测标记区域相关信息
	Mat centroids(labelNum, 2, CV_32FC1, Scalar::all(0));
	connectedComponentsWithStats(imGrayRedMarkEdge, imLabel, stats, centroids, 8, CV_32S);

	stats.row(0) = stats.row(0) * 0;                                          // 标记0代表背景区域，清零
	int ptNum[4] = {0, 0, 0, 0};                                              // 存放最大四个标记区域下标
	int temp;
	for (int i = 1; i < labelNum; ++i) {                                      // 获取最大四个标记区域下标
		if (stats(i, CC_STAT_AREA) > stats(ptNum[3], CC_STAT_AREA)) {
			ptNum[3] = i;
			for (int j = 2; j >= 0; --j) {
				if (stats(ptNum[j], CC_STAT_AREA) < stats(ptNum[j + 1], CC_STAT_AREA)) {
					temp = ptNum[j];
					ptNum[j] = ptNum[j + 1];
					ptNum[j + 1] = temp;
				}
				else {
					break;
				}
			}
		}
	}



 	namedWindow("The original image", WINDOW_AUTOSIZE);
 	imshow("The original image", imOrin);
 	namedWindow("The red mark image", WINDOW_AUTOSIZE);
 	imshow("The red mark image", imRedMark);
 	namedWindow("The edge image of red mark", WINDOW_AUTOSIZE);
 	imshow("The edge image of red mark", imGrayRedMarkEdge);
 	namedWindow("The label image", WINDOW_AUTOSIZE);
 	imshow("The label image", imLabel);
 
 	waitKey();
 
 	return 0;
 }
