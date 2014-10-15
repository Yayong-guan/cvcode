#include <opencv2\opencv.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\legacy\legacy.hpp>

using namespace cv;

struct Pattern
{
	cv::Size size;
	cv::Mat image;
	cv::Mat grayImg;
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::Point2f> points;
	cv::Mat descriptors;
};

class PatternDetector
{
public:
	PatternDetector(string feature_type);
	void buildPatternFromImage(const cv::Mat& image, Pattern& pattern) const;
	void train(const Pattern& pattern);
	void InitializequeryImg(const Mat& image);
	bool extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
	void RatioFilter(const Mat& descriptors, std::vector<vector<DMatch>>& knnMatches);
	void SymmetryFilter(const Mat& descriptors, std::vector<vector<DMatch>>& knnMatches, vector<DMatch>& matches);
	void OpticalFlowFilter(const Mat& queryImg, std::vector<KeyPoint>& keypoints, vector<DMatch>& matches);
	void refineHomographyFilter(std::vector<KeyPoint>& queryKeypoints, vector<DMatch>& matches);
	std::vector<DMatch> getMatches();
private:
	Pattern m_pattern;
	Mat queryImg;
	Mat homography;

	std::vector<cv::KeyPoint> m_keyPoints;
	Mat m_descriptors;
	std::vector<std::vector<cv::DMatch>> m_knnMatches;
	std::vector<DMatch> m_matches;

	cv::Ptr<cv::FeatureDetector> m_detector;
	cv::Ptr<cv::DescriptorExtractor> m_extractor;
	cv::Ptr<cv::DescriptorMatcher> m_matcher;

	bool enableRatio;
	enum FERTURE_TYPE
	{
		SURF,
		SIFT,
		ORB,
		FAST
	};
	enum MATCHER_TYPE
	{
		BF,
		FLANN
	};
};