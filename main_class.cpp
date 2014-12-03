#include "standard_headers.h"
#include "opencv_headers.h"
#include "motion_framework.h"
#include "rw_flow.h"

int main()
{

	//place to hold images
	cv::Mat image1, image2;

	//place to hold color-coded output image;
	cv::Mat flow_img;
		
	//WARNING:  Image size at each level must be divisble by the block size!
	int search_size[] = { 30, 40, 40 }; //params for block matching
	int block_size[] = { 15, 30, 30 };
	int num_levels = 3;
	
	//read first image
	image1 = cv::imread("test1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//read second image
	image2 = cv::imread("test2.png", CV_LOAD_IMAGE_GRAYSCALE);

	MF motion_pair(image1, image2, search_size, block_size, num_levels);
  cv::Mat flow_res = motion_pair.calcMotionBlockMatching();	 //check smoothness by showing MVs before and after regularization

	//Draw a color-coded image for the motion vectors
	Flow file;
	file.MotionToColor(flow_res, flow_img, -1); //the last parameter is if you want to set a maximum motion vector value.  Any value < 0 ignores this option
	cv::imwrite("flow.png", flow_img);

	return 0;
}