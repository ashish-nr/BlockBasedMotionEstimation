#include "standard_headers.h"
#include "opencv_headers.h"
#include "motion_framework.h"
#include "rw_flow.h"
#include "motion_hist.h"

int main()
{
	
	//place to hold color-coded output image;
	cv::Mat flow_img;

	//video capture object.
	cv::VideoCapture capture;

	capture.open("C:\\Users\\MJ\\Desktop\\Desktop_Computer\\Work_Related\\Consulting\\Styliff\\Code\\BlockBasedME\\arm.mpeg");// test_still.avi");

	if (!capture.isOpened()){
		std::cout << "ERROR ACQUIRING VIDEO FEED\n";
		getchar();
		return -1;
	}

	cv::Mat curr_frame;
	cv::Mat prev_frame;
	
	//Capture the first frame
	capture.read(prev_frame);
	//Change from full HD to 720p
	cv::resize(prev_frame, prev_frame, cv::Size(1280, 720));
	//cv::transpose(prev_frame, prev_frame);
	cv::cvtColor(prev_frame, prev_frame, cv::COLOR_BGR2GRAY);

	cv::Mat flow_res;
	
	while (capture.read(curr_frame))
	{
		//change from full HD to 720p
		cv::resize(curr_frame, curr_frame, cv::Size(1280, 720));
		//convert frame to gray scale for frame differencing
		cv::cvtColor(curr_frame, curr_frame, cv::COLOR_BGR2GRAY);
		//cv::transpose(curr_frame, curr_frame);

		clock_t t1, t2;
		t1 = clock();

		cv::calcOpticalFlowFarneback(prev_frame, curr_frame, flow_res, 0.5, 3, 15, 3, 5, 1.2, 0);		

		t2 = clock();
		float diff = ((float)t2 - (float)t1);
		float seconds = diff / CLOCKS_PER_SEC;
		std::cout << "Seconds: " << seconds << std::endl;

		//Draw a color-coded image for the motion vectors
		Flow file;
		file.MotionToColor(flow_res, flow_img, -1); //the last parameter is if you want to set a maximum motion vector value.  Any value < 0 ignores this option
		cv::imwrite("flow.png", flow_img);

		MHist m(flow_res);
		m.calc_hist();

		prev_frame = curr_frame.clone();

		cv::namedWindow("current image");
		cv::imshow("current image", prev_frame);
		cv::waitKey(0);

		//TODO: need to clear the speedup array before going to the next frame
		//exit(1);
	}

	return 0;
}