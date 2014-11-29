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
		
	int search_size[] = { 40, 40, 40 }; //params for block matching
	int block_size[] = { 30, 30, 30 };
	//int search_size[] = { 68, 68, 68 }; //params for block matching
	//int block_size[] = { 64, 64, 64 };
	int num_levels = 3;
	
	//read first image
	image1 = cv::imread("test1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//read second image
	image2 = cv::imread("test2.png", CV_LOAD_IMAGE_GRAYSCALE);

	MF motion_pair(image1, image2, search_size, block_size, num_levels);
  cv::Mat flow_res = motion_pair.calcMotionBlockMatching();	

	//Flow file;
	//file.ReadFlowFile(flow_img, "test_small.flo");
	//file.WriteFlowFile(flow_img, "test_small2.flo");
	//file.MotionToColor(flow_img, flow_img2, -1);
	//std::cout << flow_file << std::endl;
	//cv::cvtColor(flow_img, flow_img, cv::COLOR_RGB2BGR, 3);
	//cv::imwrite("test_color.png", flow_img);

	return 0;
}