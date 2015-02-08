#include "standard_headers.h"
#include "opencv_headers.h"
#include "motion_framework.h"
#include "rw_flow.h"
#include "motion_hist.h"

int main()
{	
	int search_size[] = { 64, 64, 64, 64 }; //params for block matching
	int block_size[] = { 32, 32, 32, 32 };
	int num_levels = 4;

	//image padding so block sizes are even multiples of height and width at all levels
	int pad_x = 0;
	int pad_y = 24;//full HD 100; //amount to pad on top and bottom

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
	
	//curr_frame = cv::imread("curr.png", 0);
	//prev_frame = cv::imread("prev.png", 0);

	//cv::resize(prev_frame, prev_frame, cv::Size(1280, 720));
	//cv::resize(curr_frame, curr_frame, cv::Size(1280, 720));

	//cv::Mat image1_pad = cv::Mat(prev_frame.rows + pad_y * 2, prev_frame.cols + pad_x * 2, CV_8UC1);
	//cv::Mat image2_pad = cv::Mat(prev_frame.rows + pad_y * 2, prev_frame.cols + pad_x * 2, CV_8UC1);

	//cv::copyMakeBorder(prev_frame, image1_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));
	//cv::copyMakeBorder(curr_frame, image2_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));

	//MF motion_pair(image1_pad, image2_pad, search_size, block_size, num_levels);

	//clock_t t1, t2;
	//t1 = clock();

	//cv::Mat flow_res = motion_pair.calcMotionBlockMatching();	 //check smoothness by showing MVs before and after regularization

	//t2 = clock();
	//float diff = ((float)t2 - (float)t1);
	//float seconds = diff / CLOCKS_PER_SEC;
	//std::cout << "Seconds: " << seconds << std::endl;

	////Draw a color-coded image for the motion vectors
	//Flow file;
	//file.MotionToColor(flow_res, flow_img, -1); //the last parameter is if you want to set a maximum motion vector value.  Any value < 0 ignores this option
	//cv::imwrite("flow.png", flow_img);

	//-----------
		
	//Capture the first frame
	capture.read(prev_frame);
	//Change from full HD to 720p
	cv::resize(prev_frame, prev_frame, cv::Size(1280, 720));
	//cv::transpose(prev_frame, prev_frame);
	cv::cvtColor(prev_frame, prev_frame, cv::COLOR_BGR2GRAY);
	cv::Mat image1_pad = cv::Mat(prev_frame.rows + pad_y * 2, prev_frame.cols + pad_x * 2, CV_8UC1);
	cv::Mat image2_pad = cv::Mat(prev_frame.rows + pad_y * 2, prev_frame.cols + pad_x * 2, CV_8UC1);

	int count = 0;

	//cv::Mat y_hist;
	//MHist m(y_hist);

	while (capture.read(curr_frame))
	{
		count++;
		//change from full HD to 720p
		cv::resize(curr_frame, curr_frame, cv::Size(1280, 720));
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
		//cv::Mat flow_res;
		//cv::calcOpticalFlowFarneback(prev_frame, curr_frame, flow_res, 0.5, 3, 15, 3, 5, 1.2, 0);

		t2 = clock();
		float diff = ((float)t2 - (float)t1);
		float seconds = diff / CLOCKS_PER_SEC;
		std::cout << "Seconds: " << seconds << std::endl;

		//Split MVs into x and y components
		//C++: void split(const Mat& src, Mat* mvbegin)
		//cv::Mat channels[2];
		//cv::split(flow_res, channels);

		////Moving average of the motion vectors
		//cv::accumulate(channels[1], movingAvg);

		//cv::scaleAdd(movingAvg, 1.0f / count, zero_array, temp);
		//
		////Copy movingAvg to a new image, and let all the x components of the MV be zero
		//int from_to[] = { 0, 0 };
		//cv::mixChannels(&temp, 1, &movingAvgColor, 1, from_to, 1);		
		//

		//cv::Vec2f temp(0.0f, 0.0f);

		/*for (int i = 0; i < flow_res.rows; i++)
		{
			for (int j = 0; j < flow_res.cols; j++)
			{
				if (flow_res.at<cv::Vec2f>(i, j)[1] >= 0)
					flow_res.at<cv::Vec2f>(i, j)[1] = 0;

				flow_res.at<cv::Vec2f>(i, j)[0] = 0;

			}
		}*/


		//Draw a color-coded image for the motion vectors
		//Flow file;
		//file.MotionToColor(flow_res, flow_img, -1); //the last parameter is if you want to set a maximum motion vector value.  Any value < 0 ignores this option
		///cv::namedWindow("flow image");
		//cv::imshow("flow image", flow_img);
		//cv::imwrite("flowimg.png", flow_img);
		//cv::waitKey(0);

		//MHist m(flow_res);
		//m.calc_hist(flow_res);

		//cv::imwrite("prev.png", prev_frame);
		//cv::imwrite("curr.png", curr_frame);

		prev_frame = curr_frame.clone();

		//cv::namedWindow("current image");
		//cv::imshow("current image", prev_frame);
		//cv::waitKey(0);

		//TODO: need to clear the speedup array before going to the next frame
	  //exit(1);
	}	

	return 0;
}