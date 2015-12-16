#include<iostream>
#include<string>
#include<cmath>
#include<opencv.hpp>

using namespace std;
using namespace cv;

const double PI = 3.1415926;

int main(int argc, char *argv[])
{
	Mat imOrin = imread("..\\image\\clock.jpg");
	Size imOrinSize = imOrin.size();

	const int THR = 110;              // ��ɫ��ֵ
	const int THG = 60;               // ��ɫ��ֵ
	const int THB = 60;               // ��ɫ��ֵ

	Mat imRedMark(imOrinSize, CV_8UC3, Scalar::all(0));          // ��ź�ɫ����ͼ��
	for (int i = 0; i < imOrinSize.height; ++i) {                // ȷ����ɫ����λ��
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
	Canny(imGrayRedMark, imGrayRedMarkEdge, 0.2, 0.8, 5);                     // ��Ե���

	int labelNum;
	Mat imLabel(imOrinSize, CV_16UC1, Scalar::all(0));
	labelNum = connectedComponents(imGrayRedMarkEdge, imLabel, 8, CV_16U);    // ���������Ŀ������������

	Mat_<int> stats(labelNum, 5);                                             // ��������������Ϣ
	Mat_<double> centroids(labelNum, 2);
	connectedComponentsWithStats(imGrayRedMarkEdge, imLabel, stats, centroids, 8, CV_16U);

	stats.row(0) = stats.row(0) * 0;                                          // ���0��������������
	int ptNum[4] = {0, 0, 0, 0};                                              // �������ĸ���������±�
	int temp;
	for (int i = 1; i < labelNum; ++i) {                                      // ��ȡ����ĸ���������±�
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
	for (int i = 0; i < 3; ++i) {                                // �ĸ����������X�������������������ѡ������
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

	const int ADDWIDTH = 5;                                      // ����߽�չ��
	double squareEucDisMaxTemp, squareEucDisMax;                 // ƽ��ŷʽ�������ֵ
	double ptScale[4];                                           // ����ĸ�ָ��̶�
	Point2d ptFarthest, ROICenter;                               // ROI�ڲ�����������Զ��
	double angle;                                                // ����Ƕ�

	for (int i = 0; i < 4; ++i) {
		squareEucDisMax = 0;
		Rect_<int> ROI(stats(ptNum[i], CC_STAT_LEFT) - ADDWIDTH, stats(ptNum[i], CC_STAT_TOP) - ADDWIDTH,\
					   stats(ptNum[i], CC_STAT_WIDTH) + ADDWIDTH * 2, stats(ptNum[i], CC_STAT_HEIGHT) + ADDWIDTH * 2);
		Mat imROILabel(imLabel, ROI);

		ROICenter.x = centroids(ptNum[i], 0) - stats(ptNum[i], CC_STAT_LEFT) + ADDWIDTH;      // ����ROI��������
		ROICenter.y = centroids(ptNum[i], 1) - stats(ptNum[i], CC_STAT_TOP) + ADDWIDTH;

		for (int m = 0; m < imROILabel.rows; ++m) {                                           // ����������λ��
			for (int n = 0; n < imROILabel.cols; ++n) {

				if (ptNum[i] == imROILabel.at<short>(m, n)) {
					imROILabel.at<short>(m, n) = 32767;                                       // ����۲�ROI����
					squareEucDisMaxTemp = pow(abs(n - ROICenter.x), 2) + pow(abs(m - ROICenter.y), 2);

					if (squareEucDisMaxTemp > squareEucDisMax) {
						squareEucDisMax = squareEucDisMaxTemp;
						ptFarthest.x = n;
						ptFarthest.y = m;
					}

				}

			}
		}

		angle = atan2(-(ptFarthest.y - ROICenter.y), ptFarthest.x - ROICenter.x);             // �������ָ����ֵ
		(angle - PI / 2) < 0 ? ptScale[i] = -1 * (angle - PI / 2) / (PI / 5) : ptScale[i] = 10 - (angle - PI / 2) / (PI / 5);

		imROILabel.at<short>(short(ROICenter.y), short(ROICenter.x)) = 32767;                 // ��ʾ���ĵ�

		string ROIName = "..\\image\\";                                                       // ����ROI����ͼ��
		char s[2];
		_itoa_s(i, s, 10);ROIName.push_back(s[0]);ROIName.append(".bmp");
		imwrite(ROIName, imROILabel);
	}


	const double K0 = 0.0001;
	const double K1 = 0.001;
	const double K2 = 0.01;
	const double K3 = 0.1;
	double result = K0 * ptScale[0] + K1 * floor(ptScale[1]) + K2 * floor(ptScale[2]) + K3 * floor(ptScale[3]);

	cout << "The result is: "\
		 << "0.1*" << floor(ptScale[3]) << " + "\
		 << "0.01*" << floor(ptScale[2]) << " + "\
		 << "0.001*" << floor(ptScale[1]) << " + "\
		 << "0.0001*" << ptScale[0] << "= " << result << endl;
 
 	return 0;
 }
