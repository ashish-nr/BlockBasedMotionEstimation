
#ifndef PYRAMID_LEVEL_H
#define PYRAMID_LEVEL_H

#include "opencv_headers.h"

class PyramidLevel
{
public:
	cv::Mat level_flow;
	int block_size;
	int search_size;
	cv::Mat image1;
	cv::Mat image2;
};

#endif //PYRAMID_LEVEL_H