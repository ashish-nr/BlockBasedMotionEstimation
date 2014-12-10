#include "standard_headers.h"
#include "opencv_headers.h"
#include "motion_framework.h"
#include "rw_flow.h"

int main()
{	
	int search_size[] = { 40, 40, 42 }; //params for block matching
	int block_size[] = { 32, 32, 32 };
	int num_levels = 3;

	//image padding so block sizes are even multiples of height and width at all levels
	int pad_x = 0;
	int pad_y = 100; //amount to pad on top and bottom

	//place to hold color-coded output image;
	cv::Mat flow_img;

	//video capture object.
	cv::VideoCapture capture;

	capture.open("C:\\Users\\MJ\\Desktop\\Desktop_Computer\\Work_Related\\Consulting\\Styliff\\Code\\BlockBasedME\\test_still.avi");

	if (!capture.isOpened()){
		std::cout << "ERROR ACQUIRING VIDEO FEED\n";
		getchar();
		return -1;
	}

	cv::Mat curr_frame;
	cv::Mat prev_frame;
	
	//curr_frame = cv::imread(".\\vid_images\\image00.png", 0);
	//prev_frame = cv::imread(".\\vid_images\\image192.png", 0);

	//cv::Mat image1_pad = cv::Mat(prev_frame.rows + pad_y * 2, prev_frame.cols + pad_x * 2, CV_8UC1);
	//cv::Mat image2_pad = cv::Mat(prev_frame.rows + pad_y * 2, prev_frame.cols + pad_x * 2, CV_8UC1);

	//cv::copyMakeBorder(prev_frame, image1_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));
	//cv::copyMakeBorder(curr_frame, image2_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));

	//MF motion_pair(image1_pad, image2_pad, search_size, block_size, num_levels);
	//cv::Mat flow_res = motion_pair.calcMotionBlockMatching();	 //check smoothness by showing MVs before and after regularization

	////Draw a color-coded image for the motion vectors
	//Flow file;
	//file.MotionToColor(flow_res, flow_img, -1); //the last parameter is if you want to set a maximum motion vector value.  Any value < 0 ignores this option
	//cv::imwrite("flow.png", flow_img);

	//-----------

	//Capture the first frame
	capture.read(prev_frame);
	//cv::transpose(prev_frame, prev_frame);
	cv::cvtColor(prev_frame, prev_frame, cv::COLOR_BGR2GRAY);
	cv::Mat image1_pad = cv::Mat(prev_frame.rows + pad_y * 2, prev_frame.cols + pad_x * 2, CV_8UC1);
	cv::Mat image2_pad = cv::Mat(prev_frame.rows + pad_y * 2, prev_frame.cols + pad_x * 2, CV_8UC1);
	
	while (capture.read(curr_frame))
	{
		//convert frame to gray scale for frame differencing
		cv::cvtColor(curr_frame, curr_frame, cv::COLOR_BGR2GRAY);
		//cv::transpose(curr_frame, curr_frame);

		//pad images
		cv::copyMakeBorder(prev_frame, image1_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));
		cv::copyMakeBorder(curr_frame, image2_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));

		clock_t t1, t2;
		t1 = clock();

		MF motion_pair(image1_pad, image2_pad, search_size, block_size, num_levels);
		cv::Mat flow_res = motion_pair.calcMotionBlockMatching();	 //check smoothness by showing MVs before and after regularization

		t2 = clock();
		float diff = ((float)t2 - (float)t1);
		float seconds = diff / CLOCKS_PER_SEC;
		std::cout << "Seconds: " << seconds << std::endl;
		
		//Draw a color-coded image for the motion vectors
		Flow file;
		file.MotionToColor(flow_res, flow_img, -1); //the last parameter is if you want to set a maximum motion vector value.  Any value < 0 ignores this option
		cv::imwrite("flow.png", flow_img);

		prev_frame = curr_frame.clone();

		//TODO: need to clear the speedup array before going to the next frame
	  exit(1);
	}	

	return 0;
}