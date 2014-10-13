#include <opencv2\opencv.hpp>
#include <iostream>
#include <sstream>
#include <io.h>

using namespace std;
using namespace cv;

// 遍历path目录下的所有文件，存储在files容器中
void getFiles(string path, vector<string>& files)
{
	//文件句柄
	long   hFile = 0;
	//文件信息
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


// 将imgvec中的所有图片显示在一张图片上，rows为显示图片的行数
Mat showMultiImgs(std::vector<Mat> imgVec, int rows)
{
	Mat dst;

	int size = imgVec.size();
	int cols = (size + rows - 1) / rows;
	// 缝隙的宽度
	int gapWidth = 5;
	// 单个图片的长度和宽度，可以修改这里的值改变显示图片的大小
	int singleImgWidth = 200;
	int singleImgHeigh = 200;
	// 整个一行图片的长度和宽度
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
				// 设置显示图片的尺度，进行适当的放缩
				double rate = float(imgRows) / imgCols;
				(imgRows > imgCols) ? resize(img, img, Size(singleImgHeigh / rate, singleImgHeigh))
					: resize(img, img, Size(singleImgWidth, singleImgWidth* rate));
				int centerx = j*(singleImgWidth + gapWidth) + singleImgWidth / 2 + gapWidth;
				int centery = gapWidth + singleImgHeigh / 2;
				// 表示图片显示的区域
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
		// 将每一行的大图片显示在最终的目标图片上
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