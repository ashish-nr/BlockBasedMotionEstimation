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
	//int search_size[] = { 30, 40, 40 }; //params for block matching
	//int block_size[] = { 15, 30, 30 };

	int search_size[] = { 32, 32, 42 }; //params for block matching
	int block_size[] = { 16, 16, 32 };
	int num_levels = 3;
	
	//read first image
	image1 = cv::imread(".\\middlebury\\data-gray\\Dimetrodon\\frame10.png", CV_LOAD_IMAGE_GRAYSCALE);
	//read second image
	image2 = cv::imread(".\\middlebury\\data-gray\\Dimetrodon\\frame11.png", CV_LOAD_IMAGE_GRAYSCALE);

	int orig_height = image1.rows;
	int orig_width = image1.cols;

	//pad the images so that we can use the block sizes above -- need block size to divide evenly into the width and height
	//need to make sure that the block sizes are multiples of each other for this to work right at all levels of the hierarchy
	//this needs to be done manually by the user right now, but it's on the TODO list.
	//C++: void copyMakeBorder(InputArray src, OutputArray dst, int top, int bottom, int left, int right, int borderType, const Scalar& value = Scalar())
	//584 x 388
	//640 x 640
	int pad_x = 28;
	int pad_y = 126;
	cv::Mat image1_pad = cv::Mat(image1.rows + pad_y*2, image1.cols + pad_x*2, CV_8UC1);
	cv::Mat image2_pad = cv::Mat(image1.rows + pad_y*2, image1.cols + pad_x*2, CV_8UC1);
	cv::copyMakeBorder(image1, image1_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::copyMakeBorder(image2, image2_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));

	if (!image1.data || !image2.data)
	{
		std::cout << "Could not open one of the images" << std::endl;
		getchar();
		exit(1);
	}

	MF motion_pair(image1_pad, image2_pad, search_size, block_size, num_levels);
  cv::Mat flow_res = motion_pair.calcMotionBlockMatching();	 //check smoothness by showing MVs before and after regularization

	//We have to truncate flow_res because we are only interested in the middle part, not the padded part.
	cv::Mat flow_res_trunc = flow_res(cv::Rect(pad_x, pad_y, orig_width, orig_height));

	//Draw a color-coded image for the motion vectors
	Flow file;
	file.MotionToColor(flow_res_trunc, flow_img, -1); //the last parameter is if you want to set a maximum motion vector value.  Any value < 0 ignores this option
	cv::imwrite("flow.png", flow_img);

	//Calculate Mean-square error for our calculated MV field and the ground truth motion vector field (.flo file provided by Middlebury)
	cv::Mat gtruth;
	file.ReadFlowFile(gtruth, ".\\middlebury\\gt-flow\\Dimetrodon\\flow10.flo");
	double MSE = file.CalculateMSE(gtruth, flow_res_trunc);

	std::cout << "Calculated MSE is " << MSE << std::endl;

	return 0;
}