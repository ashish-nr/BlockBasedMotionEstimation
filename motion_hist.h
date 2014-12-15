#ifndef MOTION_HIST_H
#define MOTION_HIST_H

#include "opencv_headers.h"
#include "standard_headers.h"

class MHist
{
public:
	MHist(cv::Mat &motion_field) : m_field(motion_field) { }
	void calc_hist();
	void draw_hist();
	~MHist();

private:
	cv::Mat m_field;
};

#endif //MOTION_HIST_H