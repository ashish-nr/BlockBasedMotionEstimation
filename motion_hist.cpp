#include "motion_hist.h"

void MHist::calc_hist(cv::Mat &m_field)
{

	//m_field = motion_field;

 //extract the y-component of the MV into a separate image
	cv::Mat y_comp(m_field.rows, m_field.cols, CV_32F);
	
	int from_to[] = { 1, 0 };
	cv::mixChannels(&m_field, 1, &y_comp, 1, from_to, 1);
	
	/*std::cout << y_comp << std::endl;
	float error = 0;
	for (int i = 0; i < m_field.rows; i++)
	{
		for (int j = 0; j < m_field.cols; j++)
		{
			cv::Vec2f temp = m_field.at<cv::Vec2f>(i, j);
			error += abs(temp[1] - y_comp.at<float>(i, j));			
		}
	}
	std::cout << "Error is " << error << std::endl;*/

	//Establish the number of bins
	int histSize = 201;

	//Set the ranges 
	float range[] = { -100, 100 };
	const float* histRange = { range };

	bool uniform = true; //each bin has the same width
	bool accumulate = true; //clear histograms at beginning or not?

	//Compute the histogram
	cv::calcHist(&y_comp, 1, 0, cv::Mat(), y_hist, 1, &histSize, &histRange, uniform, accumulate);

  //Create mask which will be used to filter out the zero y-component
	//cv::Mat mask = cv::Mat::ones(y_hist.rows, y_hist.cols, CV_8UC1);
	//mask.at<uchar>(0, 10) = 0;

	//Print top five bins
	//void minMaxLoc(InputArray src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, InputArray mask=noArray())
	/*double min_value;
	double max_value;
	cv::Point min_idx;
	cv::Point max_idx;
	cv::minMaxLoc(y_hist, &min_value, &max_value, &min_idx, &max_idx, mask);

	std::cout << "max value is " << max_value << std::endl;
	std::cout << "max index is " << max_idx << std::endl;
	std::cout << "min value is " << min_value << std::endl; 
	std::cout << "min index is " << min_idx << std::endl;*/

	
	//Draw the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	//Normalize the result to [ 0, histImage.rows ]
	normalize(y_hist, y_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	//Draw line
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(y_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(y_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);		
	}
	

	//Display
	cv::namedWindow("Histogram", cv::WINDOW_AUTOSIZE);
	cv::imshow("Histogram", histImage);
	//cv::waitKey(0);

	

}
void MHist::draw_hist()
{


}

MHist::~MHist()
{

}