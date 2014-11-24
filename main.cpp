#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

class BlockPosition
{
  public:
	  int pos_x;
	  int pos_y;
};

void calcMotionBlockMatching(cv::Mat &image1, cv::Mat &image2, cv::Mat &flow, const int search_size, const int block_size);
BlockPosition find_min_block(int i, int j, cv::Mat &image1, cv::Mat &image2, const int search_size, const int block_size);
void fill_block_MV(int i, int j, int block_size, cv::Mat &flow, cv::Vec2f mv);
int min(int elem1, int elem2); 
int max(int elem1, int elem2);

int main()
{
	
	//store frames captured from video
	cv::Mat frame1, frame2;
	//to hold grayscale version of frames
	cv::Mat grayImage1, grayImage2;
	//video capture object.
	cv::VideoCapture capture;

	cv::Mat flow; //to hold optical flow results
	std::vector<cv::Mat> channels(2); //for splitting the MV components into channels
	cv::Mat flowX, flowY;
	cv::Mat thresh_image; //to display where motion is changing
	int search_size = 80; //params for block matching
	int block_size = 60;


	//open video
	capture.open("test.MTS");
	if (!capture.isOpened()){
		std::cout << "ERROR ACQUIRING VIDEO FEED\n";
		getchar();
		return -1;
	}

	//cv::VideoWriter outputVideo;
	//cv::Size S = cv::Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),    //Acquire input size
	//	(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	//outputVideo.open("sample.avi", -1, capture.get(CV_CAP_PROP_FPS), S, true);

	//check if the video has reach its last frame.
	//we add '-1' because we are reading two frames from the video at a time.
	while (capture.get(CV_CAP_PROP_POS_FRAMES)<capture.get(CV_CAP_PROP_FRAME_COUNT) - 1){

		//read first frame
		capture.read(frame1);
		//convert frame1 to gray scale for frame differencing
		cv::cvtColor(frame1, grayImage1, cv::COLOR_BGR2GRAY);
		//copy second frame
		capture.read(frame2);
		//convert frame2 to gray scale for frame differencing
		cv::cvtColor(frame2, grayImage2, cv::COLOR_BGR2GRAY);

	  //compute motion
		calcMotionBlockMatching(grayImage1, grayImage2, flow, search_size, block_size);
		//calcOpticalFlowSF(frame1, frame2, flow, 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
		//calcOpticalFlowFarneback(grayImage1, grayImage2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

		//Create threshold image that shows where motion is changing
		split(flow, channels);
		flowX = channels[0];
		flowY = channels[1];

		cv::threshold(flowX.mul(flowX)+flowY.mul(flowY), thresh_image, 2, 255, CV_THRESH_BINARY);
		

		//show captured frame
		cv::transpose(frame1, frame1);
		cv::flip(frame1, frame1, 0);
		cv::imshow("Frame1", frame1);
		cv::transpose(thresh_image, thresh_image);
		cv::flip(thresh_image, thresh_image, 0);
		cv::imshow("ThreshFrame", thresh_image);
			
		//this 10ms delay is necessary for proper operation of this program
		//if removed, frames will not have enough time to referesh and a blank 
		//image will appear.
		cv::waitKey(10);

	}
	
	//release the capture before re-opening and looping again.
	capture.release();

	return 0;
}

void calcMotionBlockMatching(cv::Mat &image1, cv::Mat &image2, cv::Mat &flow, const int search_size, const int block_size) //greyscale images only
{
	//Make sure that height and width of image are a multiple of the block size
	if (image1.rows % block_size != 0 || image1.cols % block_size != 0)
	{
		std::cout << "Image height AND width must be a multiple of the block size" << std::endl;
		getchar();
	}

	//Check that images only have one channel
	assert(image1.channels() == 1 && image2.channels() == 1);

	//Intialize flow Mat
	flow = cv::Mat::zeros(image1.rows, image1.cols, CV_32FC2);
	
	for (int i = 0; i < image1.rows; i+=block_size)
	{
		for (int j = 0; j < image1.cols; j+=block_size)
		{
			BlockPosition result = find_min_block(i, j, image1, image2, search_size, block_size); //returns i, j position of block found
			//Calculate MV
			cv::Vec2f mv = cv::Vec2f(result.pos_x - j, result.pos_y - i);
			fill_block_MV(i, j, block_size, flow, mv); //assign MV to every pixel in block
		}
		
	}
}

BlockPosition find_min_block(int i, int j, cv::Mat &image1, cv::Mat &image2, const int search_size, const int block_size)
{
  //form search window
	int start_pos = ((search_size - block_size) >> 1); //assuming square block size
	int SAD_min = std::numeric_limits<int>::max(); //max value an integer can take -- used to initialize SAD value
	int min_x = j; //initalizing the positions of the block which we will calculate below -- set to center block initially
	int min_y = i;
	cv::Mat curr_diff; //absolute difference block
	cv::Scalar SAD_value; //current SAD value

	for (int k = max(0, i - start_pos); k < min(image1.rows - block_size + 1, i + start_pos); k++)
	{
		for (int l = max(0, j - start_pos); l < min(image1.cols - block_size + 1, j + start_pos); l++)
		{
		  //calculate difference between block i,j in image1 and block k,l in image 2
			cv::absdiff(image1(cv::Rect(j, i, block_size, block_size)), image2(cv::Rect(l, k, block_size, block_size)), curr_diff);
			SAD_value = cv::sum(curr_diff);
			if ((int)SAD_value.val[0] < SAD_min)
			{
				SAD_min = (int)SAD_value.val[0]; //we need val[0] because of the way that the cv::Scalar is set up
				min_x = l;
				min_y = k;
			}			
		}
	}

	BlockPosition pos; //Create class object to return values
	pos.pos_x = min_x;
	pos.pos_y = min_y;

	return pos;

}

void fill_block_MV(int i, int j, int block_size, cv::Mat &flow, cv::Vec2f mv)
{
	for (int k = i; k < i + block_size; k++)
	{
		for (int l = j; l < j + block_size; l++)
		{
			flow.at<cv::Vec2f>(k, l) = mv;
		}
	}
}

int max(int elem1, int elem2)
{
	return ((elem1 > elem2) ? elem1 : elem2);
}

int min(int elem1, int elem2)
{
	return ((elem1 < elem2) ? elem1 : elem2);
}