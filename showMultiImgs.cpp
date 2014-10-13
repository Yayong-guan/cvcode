#include <opencv2\opencv.hpp>
#include <iostream>
#include <sstream>
#include <io.h>

using namespace std;
using namespace cv;

// ����pathĿ¼�µ������ļ����洢��files������
void getFiles(string path, vector<string>& files)
{
	//�ļ����
	long   hFile = 0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


// ��imgvec�е�����ͼƬ��ʾ��һ��ͼƬ�ϣ�rowsΪ��ʾͼƬ������
Mat showMultiImgs(std::vector<Mat> imgVec, int rows)
{
	Mat dst;

	int size = imgVec.size();
	int cols = (size + rows - 1) / rows;
	// ��϶�Ŀ��
	int gapWidth = 5;
	// ����ͼƬ�ĳ��ȺͿ�ȣ������޸������ֵ�ı���ʾͼƬ�Ĵ�С
	int singleImgWidth = 200;
	int singleImgHeigh = 200;
	// ����һ��ͼƬ�ĳ��ȺͿ��
	int rowImgRows = singleImgHeigh + 2 * gapWidth;
	int rowImgCols = cols * singleImgWidth + (cols + 1) * gapWidth;

	int imgCounts = 0;
	for (int i = 0; i < rows; i++)
	{
		Mat rowImg = Mat(rowImgRows, rowImgCols, imgVec[0].type());
		rowImg.setTo(128);
		for (int j = 0; j < cols; j++)
		{
			if (imgCounts < size)
			{
				Mat img = imgVec[imgCounts];
				int imgRows = img.rows;
				int imgCols = img.cols;
				// ������ʾͼƬ�ĳ߶ȣ������ʵ��ķ���
				double rate = float(imgRows) / imgCols;
				(imgRows > imgCols) ? resize(img, img, Size(singleImgHeigh / rate, singleImgHeigh))
					: resize(img, img, Size(singleImgWidth, singleImgWidth* rate));
				int centerx = j*(singleImgWidth + gapWidth) + singleImgWidth / 2 + gapWidth;
				int centery = gapWidth + singleImgHeigh / 2;
				// ��ʾͼƬ��ʾ������
				int x = centerx - img.cols / 2;
				int y = centery - img.rows / 2;
				Mat imgROI = rowImg(Rect(x, y, img.cols, img.rows));
				stringstream str;
				str << imgCounts ;
				string strMsg = str.str();
				putText(img, strMsg, Point(20, 20), FONT_HERSHEY_COMPLEX, 0.6, Scalar(CV_RGB(255, 0, 0)));
				img.copyTo(imgROI);
				
				imgCounts++;
			}
			else{
				break;
			}
		}
		// ��ÿһ�еĴ�ͼƬ��ʾ�����յ�Ŀ��ͼƬ��
		dst.push_back(rowImg);
	}

	cout << "merge images over!" << endl;
	return dst;
}

int main()
{
	string	imagePath = "images\\";
	std::vector<string> filelist;
	getFiles(imagePath, filelist);
	std::vector<Mat> imgVec;

	for (size_t i = 0; i < filelist.size(); i++)
	{
		Mat img = imread(filelist[i]);
		imgVec.push_back(img);
	}

	Mat dst;
	dst = showMultiImgs(imgVec, 3);
	imshow("dst image", dst);
	waitKey();
}