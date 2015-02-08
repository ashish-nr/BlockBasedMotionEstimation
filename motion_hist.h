#ifndef MOTION_HIST_H
#define MOTION_HIST_H

#include "opencv_headers.h"
#include "standard_headers.h"

class MHist
{
public:
	//MHist(cv::Mat &motion_field) : m_field(motion_field) { }
	MHist(cv::Mat y_histin) : y_hist(y_histin) {}
	void calc_hist(cv::Mat &m_field);
	void draw_hist();
	~MHist();

private:
	//cv::Mat& m_field;
	cv::Mat y_hist;
};

#endif //MOTION_HIST_H