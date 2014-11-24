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

class PyramidLevel
{
   public:
		 cv::Mat level_flow;
		 int block_size;
		 int search_size;
		 cv::Mat image1;
		 cv::Mat image2;
};

void calcMotionBlockMatching(cv::Mat &image1, cv::Mat &image2, cv::Mat &flow, const int search_size, const int block_size, const int num_levels);
void calcHBM(std::vector<PyramidLevel> &levels);
void calcLevelBM(cv::Mat &image1, cv::Mat &image2, cv::Mat &flow, int block_size, int search_size);
BlockPosition find_min_block(int image1_ypos, int image1_xpos, int image2_ypos, int image2_xpos, cv::Mat &image1, cv::Mat &image2, int search_size, int block_size);
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
	int num_levels = 3;


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
		calcMotionBlockMatching(grayImage1, grayImage2, flow, search_size, block_size, num_levels);
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

void calcMotionBlockMatching(cv::Mat &image1, cv::Mat &image2, cv::Mat &flow, const int search_size, const int block_size, const int num_levels) //greyscale images only
{
	//TO DO:  Make sure that num_levels > 0
  //the search_size and block_size are for the highest level of the hierarchy

	//Initialize flow matrix for highest resolution level
	flow = cv::Mat::zeros(image1.rows, image1.cols, CV_32FC2);

	//create class for each level of pyramid that we will create
	std::vector<PyramidLevel> level_data;
		
	//save highest level of hierarchy to vector
	PyramidLevel temp;
	temp.image1 = image1; //store image pointers
	temp.image2 = image2;
	temp.level_flow = flow; //store MV pointer
	temp.block_size = block_size; //store block size
	temp.search_size = search_size; //store search size
	level_data.push_back(temp);

	//Keep track of the previous image so we can apply the pyrDown operation on previous image
	cv::Mat prev_image1 = image1.clone();
	cv::Mat prev_image2 = image2.clone();
	//Keep track of previous block sizes and search sizes
	int prev_bsize = block_size;
	int prev_ssize = search_size;
	for (int i = 0; i < num_levels-1; i++)
	{
		PyramidLevel temp;
		pyrDown(prev_image1, temp.image1, cv::Size(prev_image1.cols / 2, prev_image1.rows / 2)); //create downsampled images
		pyrDown(prev_image2, temp.image2, cv::Size(prev_image2.cols / 2, prev_image2.rows / 2));
		temp.level_flow = cv::Mat::zeros(prev_image1.rows / 2, prev_image1.cols / 2, CV_32FC2); //create space to store the computed MVs for the level
		temp.block_size = prev_bsize << 1; //set it so that the block size increases by a factor of two as the resolution decreases by a factor of two
		temp.search_size = prev_ssize << 1; //set it so that the search size increases by a factor of two as the resolution decreases by a factor of two

		prev_image1 = temp.image1.clone(); //save previous values for next iteration
		prev_image2 = temp.image2.clone();
		prev_bsize = temp.block_size;
		prev_ssize = temp.search_size;

		level_data.push_back(temp); //save current level data to vector
	}

	//Call function to perform hierarchical block matching
	calcHBM(level_data);

	////Make sure that height and width of image are a multiple of the block size
	//if (image1.rows % block_size != 0 || image1.cols % block_size != 0)
	//{
	//	std::cout << "Image height AND width must be a multiple of the block size" << std::endl;
	//	getchar();
	//}

	////Check that images only have one channel
	//assert(image1.channels() == 1 && image2.channels() == 1);

	////Intialize flow Mat
	////flow = cv::Mat::zeros(image1.rows, image1.cols, CV_32FC2);
	//
	//for (int i = 0; i < image1.rows; i+=block_size)
	//{
	//	for (int j = 0; j < image1.cols; j+=block_size)
	//	{
	//		BlockPosition result = find_min_block(i, j, image1, image2, search_size, block_size); //returns i, j position of block found
	//		//Calculate MV
	//		cv::Vec2f mv = cv::Vec2f(result.pos_x - j, result.pos_y - i);
	//		fill_block_MV(i, j, block_size, flow, mv); //assign MV to every pixel in block
	//	}
	//	
	//}
}

void calcHBM(std::vector<PyramidLevel> &levels)
{
	for (int i = (int)levels.size() - 1; i >= 0; i++)
	{
		//perform block matching on each level, starting with the lowest resolution level
		if (i == levels.size() - 1) //means we don't have any previous motion field to use
		{
			//don't need to copy MVs from previous level
			calcLevelBM(levels[i].image1, levels[i].image2, levels[i].level_flow, levels[i].block_size, levels[i].search_size);
		}
		else
		{
			//TO DO:  need to copy MVs to next level
			calcLevelBM(levels[i].image1, levels[i].image2, levels[i].level_flow, levels[i].block_size, levels[i].search_size);
		}
	}
}

void calcLevelBM(cv::Mat &image1, cv::Mat &image2, cv::Mat &flow, int block_size, int search_size)
{
	int image2_xpos, image2_ypos;
	for (int i = 0; i < image1.rows; i+=block_size) //i and j here correspond to the y and x position in image 1
	{
		for (int j = 0; j < image1.cols; j+=block_size)
		{
			image2_xpos = j + (int)flow.at<cv::Vec2f>(i, j)[0];
			image2_ypos = i + (int)flow.at<cv::Vec2f>(i, j)[1];
			BlockPosition result = find_min_block(i, j, image2_ypos, image2_xpos, image1, image2, search_size, block_size); //returns i, j position of block found
			//Calculate MV
			cv::Vec2f mv = cv::Vec2f(result.pos_x - j, result.pos_y - i);
			fill_block_MV(i, j, block_size, flow, mv); //assign MV to every pixel in block
		}		
	}
}

BlockPosition find_min_block(int image1_ypos, int image1_xpos, int image2_ypos, int image2_xpos, cv::Mat &image1, cv::Mat &image2, int search_size, int block_size)
{
  //form search window
	int start_pos = ((search_size - block_size) >> 1); //assuming square block size
	int SAD_min = std::numeric_limits<int>::max(); //max value an integer can take -- used to initialize SAD value
	int min_x = image2_xpos; //initalizing the positions of the block which we will calculate below 
	int min_y = image2_ypos;
	cv::Mat curr_diff; //absolute difference block
	cv::Scalar SAD_value; //current SAD value

	for (int k = max(0, image2_ypos - start_pos); k < min(image1.rows - block_size + 1, image2_ypos + start_pos); k++)
	{
		for (int l = max(0, image2_xpos - start_pos); l < min(image1.cols - block_size + 1, image2_xpos + start_pos); l++)
		{
		  //calculate difference between block i,j in image1 and block k,l in image 2
			cv::absdiff(image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), image2(cv::Rect(l, k, block_size, block_size)), curr_diff);
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