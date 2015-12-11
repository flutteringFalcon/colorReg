#include<iostream>
#include<cmath>
#include<opencv.hpp>

using namespace std;
using namespace cv;

const double PI = 3.1415926;

int main()
{
	Mat imOrin = imread("..\\image\\clock.jpg");
	Size imOrinSize = imOrin.size();

	const int THR = 110;              // 红色阈值
	const int THG = 60;               // 绿色阈值
	const int THB = 60;               // 蓝色阈值

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
	Mat_<double> centroids(labelNum, 2);
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

	int minLeftSub, tempLeft;
	for (int i = 0; i < 3; ++i) {            // 四个标记区域按照X方向坐标从左向右排序（选择排序）
		minLeftSub = i;
		for (int j = i + 1; j < 4; ++j) {
			if (stats(ptNum[j], CC_STAT_LEFT) < stats(ptNum[minLeftSub], CC_STAT_LEFT)) {
				minLeftSub = j;
			}
		}
		if (minLeftSub != i) {
			tempLeft = ptNum[i];
			ptNum[i] = ptNum[minLeftSub];
			ptNum[minLeftSub] = tempLeft;
		}
	}

	const int ADDWIDTH = 5;                                      // 区域边界展宽
	double squareEucDisMaxTemp, squareEucDisMax;                 // 平方欧式距离最大值
	double ptScale[4];                                           // 存放四个指针刻度
	Point2i ptFarthest;                                          // ROI内部距离中心最远点
	double angle;

	for (int i = 0; i < 4; ++i) {
		squareEucDisMax = 0;
		Rect_<int> ROI(stats(ptNum[i], CC_STAT_LEFT) - ADDWIDTH, stats(ptNum[i], CC_STAT_TOP) - ADDWIDTH,\
					   stats(ptNum[i], CC_STAT_WIDTH) + ADDWIDTH * 2, stats(ptNum[i], CC_STAT_HEIGHT) + ADDWIDTH * 2);
		Mat imROIGrayRedMarkEdge(imGrayRedMarkEdge, ROI);

		int ROILabelNum;
		Mat imROILabel(imROIGrayRedMarkEdge.size(), CV_16UC1, Scalar::all(0));
		ROILabelNum = connectedComponents(imROIGrayRedMarkEdge, imROILabel, 8, CV_16U);          // ROI标记区域数目包括背景区域

		Mat_<int> ROIStats(ROILabelNum, 5);                                                      // 检测ROI标记区域相关信息
		Mat_<double> ROICentroids(ROILabelNum, 2);
		connectedComponentsWithStats(imROIGrayRedMarkEdge, imROILabel, ROIStats, ROICentroids, 8, CV_32S);
		for (int j = 0; j < ROILabelNum; ++j) {
			if (ROIStats(j, CC_STAT_AREA) > stats(ptNum[i], CC_STAT_AREA) - 1) {                 // 若是表针则进行后续计算
				for (int m = 0; m < imROILabel.rows; ++m) {
					for (int n = 0; n < imROILabel.cols; ++n) {
						if (j == imROILabel.at<int>(m, n)) {
							squareEucDisMaxTemp = pow(abs(m - ROICentroids(j, 0)), 2)\
												  + pow(abs(n - ROICentroids(j, 1)), 2);
							if (squareEucDisMaxTemp > squareEucDisMax) {
								squareEucDisMax = squareEucDisMaxTemp;
								ptFarthest = Point2i(n, m);
							}
						}
					}
				}

				angle = atan2(ptFarthest.y - ROICentroids(j, 0),\
							  ptFarthest.x - ROICentroids(j, 1));
				ptScale[i] = -1 * (angle - PI / 2) / (PI / 5);
				ptScale[i] < 0 ? ptScale[i] += 10 : ptScale[i] += 0;
			}
		}

		cout << ptFarthest << endl;
	}


	const double K0 = 0.0001;
	const double K1 = 0.001;
	const double K2 = 0.01;
	const double K3 = 0.1;
	double result;
	result = K0 * ptScale[0] + K1 * ptScale[1] + K2 * ptScale[2] + K3 * ptScale[3];
	cout << "The result is: " << result << endl;

	//namedWindow("The original image", WINDOW_AUTOSIZE);
	//imshow("The original image", imOrin);
	//namedWindow("The red mark image", WINDOW_AUTOSIZE);
	//imshow("The red mark image", imRedMark);
	//namedWindow("The edge image of red mark", WINDOW_AUTOSIZE);
	//imshow("The edge image of red mark", imGrayRedMarkEdge);

	//waitKey();
 
 	return 0;
 }
