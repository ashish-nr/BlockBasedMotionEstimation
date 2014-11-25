#include "standard_headers.h"
#include "opencv_headers.h"
#include "motion_framework.h"
#include "rw_flow.h"

int main()
{

	//place to hold images
	cv::Mat image1, image2;

	//place to hold flow vectors from file read in
	cv::Mat flow_file;
		
	int search_size[] = { 30, 30, 40 }; //params for block matching
	int block_size[] = { 15, 15, 15 };
	int num_levels = 3;
	
	//read first image
	image1 = cv::imread("test1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//read second image
	image2 = cv::imread("test2.png", CV_LOAD_IMAGE_GRAYSCALE);

	//MF motion_pair(image1, image2, search_size, block_size, num_levels);
	//motion_pair.calcMotionBlockMatching();	

	Flow file;
	file.ReadFlowFile(flow_file, "test.flo");
	file.WriteFlowFile(flow_file, "testwrite.flo");

	//std::cout << flow_file << std::endl;

	return 0;
}