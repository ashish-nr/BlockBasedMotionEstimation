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
		
	int search_size[] = { 32, 32, 42 }; //params for block matching
	int block_size[] = { 16, 16, 32 };
	int num_levels = 3;

	//int search_size[] = { 64, 64, 64, 64 }; //searchsizes for middlebury most recent -- interpolation factor of 4 -- 4 levels of hierarchy instead of 3 -- smaller block sizes
	//int block_size[] = { 32, 32, 32, 32 };
	//int num_levels = 4;
	
	//read first image
	image1 = cv::imread(".\\middlebury\\data-gray\\Dimetrodon\\frame10.png", 0);
	//read second image
	image2 = cv::imread(".\\middlebury\\data-gray\\Dimetrodon\\frame11.png", 0);

	int orig_height = image1.rows;
	int orig_width = image1.cols;

	//Need to interpolate by a factor of 4.  
	/*cv::resize(image1, image1, cv::Size(), 4, 4, cv::INTER_CUBIC);
	cv::resize(image2, image2, cv::Size(), 4, 4, cv::INTER_CUBIC);*/
					
	/*int biggest_height = image1.rows;
	int biggest_width = image1.cols;*/

	if (!image1.data || !image2.data)
	{
		std::cout << "Could not open one of the images" << std::endl;
		getchar();
		exit(1);
	}

	MF motion_pair(image1, image2, search_size, block_size, num_levels);

	clock_t t1, t2;
	t1 = clock();

  cv::Mat flow_res = motion_pair.calcMotionBlockMatching();	

	t2 = clock();
	float diff = ((float)t2 - (float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout << "Seconds: " << seconds << std::endl;

	////Need to divide all MVs by a factor of 4.  
	//cv::Mat subpix_MVs(orig_height, orig_width, CV_32FC2);
	//for (int i = pad_y; i < biggest_height-pad_y; i+=4)
	//{
	//	for (int j = pad_x; j < biggest_width-pad_x; j+=4)
	//	{
	//		subpix_MVs.at<cv::Vec2f>((i - pad_y) / 4, (j - pad_x) / 4)[0] = flow_res.at<cv::Vec2f>(i, j)[0] / 4;
	//		subpix_MVs.at<cv::Vec2f>((i - pad_y) / 4, (j - pad_x) / 4)[1] = flow_res.at<cv::Vec2f>(i, j)[1] / 4;
	//	}
	//}

	////We have to truncate flow_res because we are only interested in the middle part, not the padded part.
	//cv::Mat flow_res_trunc = subpix_MVs.clone();// (cv::Rect(pad_x / 4, pad_y / 4, orig_width, orig_height));

	//Draw a color-coded image for the motion vectors
	Flow file;
	file.MotionToColor(flow_res, flow_img, -1); //the last parameter is if you want to set a maximum motion vector value.  Any value < 0 ignores this option
	cv::imwrite("flow.png", flow_img);

	//Calculate Mean-square error for our calculated MV field and the ground truth motion vector field (.flo file provided by Middlebury)
	cv::Mat gtruth;
	file.ReadFlowFile(gtruth, ".\\middlebury\\gt-flow\\Dimetrodon\\flow10.flo");
	double MSE = file.CalculateMSE(gtruth, flow_res);

	std::cout << "Calculated MSE is " << MSE << std::endl;

	return 0;
}