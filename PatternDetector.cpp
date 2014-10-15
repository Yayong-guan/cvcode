#include "PatternDetector.h"


PatternDetector::PatternDetector(string feature_type) :enableRatio(false)
{
	if (!feature_type.compare("SUFR") | !feature_type.compare("surf"))
	{
		m_detector = new cv::SurfFeatureDetector(10);
		m_extractor = new cv::SurfDescriptorExtractor;
		m_matcher = new cv::BFMatcher(NORM_L2, false);
	}
	else if (!feature_type.compare("SIFT") | !feature_type.compare("sift"))
	{
		m_detector = new cv::SiftFeatureDetector(500);
		m_extractor = new cv::SiftDescriptorExtractor;
		m_matcher = new  cv::BFMatcher(NORM_L2, false);
	}
	else if (!feature_type.compare("FAST") | !feature_type.compare("fast"))
	{
		m_detector = new cv::FastFeatureDetector;
		m_extractor = new cv::OrbDescriptorExtractor;
		m_matcher = new cv::FlannBasedMatcher();
	}
	else if (!feature_type.compare("ORB") | !feature_type.compare("orb"))
	{
		m_detector = new cv::OrbFeatureDetector(10);
		m_extractor = new cv::OrbDescriptorExtractor;
		m_matcher = new BFMatcher(NORM_L2, false);
	}
	else
	{
		std::cerr << "please input the feature detector" << std::endl;
	}
}

void PatternDetector::buildPatternFromImage(const cv::Mat& image, Pattern& pattern) const
{
	assert(image.data);
	pattern.image = image.clone();
	pattern.size = image.size();
	if (image.channels() == 3)
		cvtColor(image, pattern.grayImg, CV_BGR2GRAY);
	else if (image.channels() == 4)
		cvtColor(image, pattern.grayImg, CV_BGRA2GRAY);
	else
	{
		pattern.grayImg = image.clone();
	}

	// 建立模式的特征描述子
	extractFeatures(pattern.grayImg, pattern.keypoints, pattern.descriptors);
}

void PatternDetector::InitializequeryImg(const Mat& image)
{
	if (image.channels() == 1)
	{
		queryImg = image.clone();
	}
	else if (image.channels() == 3)
	{
		cvtColor(image, queryImg, CV_BGR2GRAY);
	}
	else if (image.channels() == 4)
	{
		cvtColor(image, queryImg, CV_BGRA2GRAY);
	}
	else
	{
		std::cerr << "image type is invalid" << std::endl;
	}
}

bool PatternDetector::extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const
{
	assert(image.channels() == 1);
	assert(!image.empty());

	m_detector->detect(image, keypoints);
	m_extractor->compute(image, keypoints, descriptors);
	if (keypoints.empty())
	{
		return false;
	}

	return true;
}
// 训练模式，便于快速的查找
void PatternDetector::train(const Pattern& pattern)
{
	m_pattern = pattern;
	m_matcher->clear();
	std::vector<cv::Mat> descriptors(1);
	descriptors[0] = pattern.descriptors.clone();
	m_matcher->add(descriptors);
	m_matcher->train();
}

void PatternDetector::RatioFilter(const Mat& descriptors, std::vector<vector<DMatch>>& knnMatches)
{
	knnMatches.clear();
	m_knnMatches.clear();
	const float minRatio = 1.f / 1.5f;
	cv::BruteForceMatcher<cv::L2<float>> matcher;
	Mat trainDescriptors = m_matcher->getTrainDescriptors()[0];
	matcher.knnMatch(descriptors, trainDescriptors, knnMatches, 2);
	std::cout <<"before the RatioFilter, the numbers of Matches is "<< knnMatches.size() << std::endl;
	for (size_t i = 0; i < knnMatches.size(); i++)
	{
		if (knnMatches[i].size() > 1)
		{
			float distanceRatio = knnMatches[i][0].distance / knnMatches[i][1].distance;
			if (distanceRatio < minRatio)
			{
				m_knnMatches.push_back(knnMatches[i]);
			}
		}
	}
	std::cout << "after the RatioFilter, the numbers of Matches is " << m_knnMatches.size() << std::endl;
	enableRatio = true;
}

void PatternDetector::SymmetryFilter(const Mat& descriptors, std::vector<vector<DMatch>>& knnMatches, vector<DMatch>& matches)
{
	matches.clear();
	std::vector<vector<DMatch>> matches1;
	std::vector<vector<DMatch>> matches2;
	Mat trainDescriptors = m_matcher->getTrainDescriptors()[0];
	cv::BruteForceMatcher<cv::L2<float>> matcher;
	if (enableRatio)
	{
		matcher.knnMatch(trainDescriptors, descriptors, matches2, 2);
		for (size_t i = 0; i < knnMatches.size(); i++)
		{
			if (knnMatches[i].size() < 2)
				continue;
			for (size_t j = 0; j < matches2.size(); j++)
			{
				if (matches2[j].size() < 2)
					continue;
				if (knnMatches[i][0].queryIdx == matches2[j][0].trainIdx
					&& knnMatches[i][0].trainIdx == matches2[j][0].queryIdx)
				{
					matches.push_back(knnMatches[i][0]);
					break;
				}
			}
		}

	    std::cout << "after the SymmetryFilter, the numbers of Matches is " << matches.size() << std::endl;
	}
	else
	{
		matcher.knnMatch(m_descriptors, trainDescriptors, matches1, 2);
		matcher.knnMatch(trainDescriptors, m_descriptors, matches2, 2);
		matches.clear();
		for (size_t i = 0; i < matches1.size(); i++)
		{
			if (matches1[i].size() < 2)
				continue;
			for (size_t j = 0; j < matches2.size(); j++)
			{
				if (matches2[j].size() < 2)
					continue;
				if (matches1[i][0].queryIdx == matches2[j][0].trainIdx
					&& matches1[i][0].trainIdx == matches2[j][0].queryIdx)
				{
					matches.push_back(matches1[i][0]);
					break;
				}
			}
		}
		std::cout << "after the SymmetryFilter, the numbers of Matches is " << matches.size() << std::endl;
	}
}

void KeyPointsToPoints(vector<KeyPoint>& keypoint, vector<Point2f>& point)
{
	point.resize(keypoint.size());
	for (size_t i = 0; i < keypoint.size(); i++)
	{
		point.push_back(keypoint[i].pt);
	}
}

void PatternDetector::OpticalFlowFilter(const Mat& queryImg, std::vector<KeyPoint>& keypoints, vector<DMatch>& matches)
{
	// keypoint 转化为 point
	KeyPointsToPoints(m_pattern.keypoints, m_pattern.points);
	vector<Point2f> nextPoints(m_pattern.points.size());

	// 对两幅图片进行光流分析， points
	vector<uchar> vstatus;
	vector<float> verror;
	cv::calcOpticalFlowPyrLK(m_pattern.grayImg, queryImg, m_pattern.points, nextPoints, vstatus, verror);

	vector<Point2f> validNextPoints;
	vector<int> nextPointsBackIndex;
	for (unsigned i = 0; i < vstatus.size(); i++)
	{
		if (vstatus[i] && verror[i] < 12.0)
		{
			nextPointsBackIndex.push_back(i);
			validNextPoints.push_back(nextPoints[i]);
		}
		else
		{
			vstatus[i] = 0;
		}
	}

	Mat validNextPointsFlat = Mat(validNextPoints).reshape(1, validNextPoints.size());
	vector<Point2f> queryPoints;
	KeyPointsToPoints(keypoints, queryPoints);
	Mat queryPointsFlat = Mat(queryPoints).reshape(1, queryPoints.size());

	BFMatcher matcher(CV_L2);
	std::vector<vector<DMatch>> nearestNeighbors;
	matcher.radiusMatch(validNextPointsFlat, queryPointsFlat, nearestNeighbors, 2.0f);
    
	std::set<int> PointsIndex;
	double minRatio = 0.7;
	for (size_t i = 0; i < nearestNeighbors.size(); i++)
	{
		DMatch m;
		if (nearestNeighbors[i].size() == 1)
			m = nearestNeighbors[i][0];
		else if (nearestNeighbors[i].size() > 1)
		{
			double ratio = nearestNeighbors[i][0].distance / float(nearestNeighbors[i][1].distance);
			if (ratio < minRatio)
				m = nearestNeighbors[i][0];
			else
				continue;
		}
		else
		{
			continue;
		}

		if (PointsIndex.find(m.trainIdx) == PointsIndex.end())
		{
			m.queryIdx = nextPointsBackIndex[m.queryIdx];
			matches.push_back(m);
			PointsIndex.insert(m.trainIdx);
		}
	}
}


// 消除几何上不匹配的特征点
void PatternDetector::refineHomographyFilter(std::vector<KeyPoint>& queryKeypoints, vector<DMatch>& matches)
{

	const int minNumberMatches = 8;
	if (matches.size() < minNumberMatches)
	{
		std::cerr << "the pairs of features is less than 8" << std::endl;
		return;
	}

	// 计算单应矩阵
	std::vector<cv::Point2f> srcPoints(matches.size());
	std::vector<cv::Point2f> dstPoints(matches.size());
	for (int i = 0; i < matches.size(); i++)
	{
		srcPoints[i] = m_pattern.keypoints[matches[i].trainIdx].pt;
		dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
	}

	std::vector<unsigned char> inliersMask(srcPoints.size());
	homography = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC, 3.0, inliersMask);
	
	std::vector<cv::DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++)
	{
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}
	matches.swap(inliers);
	std::cout << "after the refineHomographyFilter, the numbers of Matches is " << matches.size() << std::endl;
#if 1
	int size = inliers.size();
	Mat src_points(1, size, CV_32FC2);
	Mat dst_points(1, size, CV_32FC2);

	for (size_t i = 0; i < matches.size(); i++)
	{
		src_points.at<Point2f>(0, i) = m_pattern.keypoints[matches[i].trainIdx].pt;
		dst_points.at<Point2f>(0, i) = queryKeypoints[matches[i].queryIdx].pt;
	}
	// 计算单应矩阵
	homography = findHomography(src_points, dst_points, CV_LMEDS);
#endif

#if 0
	srcPoints.clear();
	dstPoints.clear();
	srcPoints.resize(matches.size());
	dstPoints.resize(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		srcPoints[i] = m_pattern.keypoints[matches[i].trainIdx].pt;
		dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
	}

	homography = findHomography(srcPoints, dstPoints, CV_RANSAC);
#endif
	// 去掉透视投影的效果
	Mat warpImg;
	cv::warpPerspective(queryImg, warpImg, homography, m_pattern.size);
	imshow("warpImg", warpImg);
}

std::vector<DMatch> PatternDetector::getMatches()
{
	extractFeatures(queryImg, m_keyPoints, m_descriptors);
	std::vector<vector<DMatch>> knnMatches;
	RatioFilter(m_descriptors, knnMatches);
	SymmetryFilter(m_descriptors, m_knnMatches, m_matches);
	Mat dst;
	drawMatches(queryImg, m_keyPoints, m_pattern.grayImg, m_pattern.keypoints, m_matches, dst);
	imshow("dst", dst);
	refineHomographyFilter(m_keyPoints, m_matches);
	return m_matches;
}

int main()
{
	//Mat image1 = imread("PyramidPattern.jpg");
	//Mat image2 = imread("PyramidPatternTest.bmp");
	Mat image1 = imread("A2.jpg");
	Mat image2 = imread("A1.jpg");
	Pattern pattern;
	PatternDetector detector("sift");
	detector.buildPatternFromImage(image1, pattern);
	detector.train(pattern);
	detector.InitializequeryImg(image2);
	std::vector<DMatch> matches = detector.getMatches();
	waitKey();
}