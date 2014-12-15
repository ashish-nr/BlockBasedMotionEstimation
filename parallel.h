#ifndef PARALLEL_H
#define PARALLEL_H

#include "opencv_headers.h"
#include "pyramid_level.h"
#include "block_position.h"
#include "motion_framework.h"

class Parallel_process : public cv::ParallelLoopBody
{

private:
	PyramidLevel& level;
	cv::Mat& fast_array;

public:
	Parallel_process(PyramidLevel& inputLevel, cv::Mat& inputFastArray)
		: level(inputLevel), fast_array(inputFastArray) {}

	void process_half(int thread) const
	{
		if (thread == 0)
		{			
			int image2_xpos, image2_ypos;
			for (int i = 0; i < level.image1.rows; i += level.block_size) //i and j here correspond to the y and x position in image 1
			{
				for (int j = 0; j < level.image1.cols/2; j += level.block_size)
				{
					image2_xpos = j + (int)level.level_flow.at<cv::Vec2f>(i, j)[0];
					image2_ypos = i + (int)level.level_flow.at<cv::Vec2f>(i, j)[1];
					//BlockPosition result = find_min_block(i, j, image2_ypos, image2_xpos); //returns i, j position of block found
					BlockPosition result = find_min_block_spiral(i, j, image2_ypos, image2_xpos); //returns i, j position of block found using spiral search
					//Calculate MV
					cv::Vec2f mv = cv::Vec2f((float)result.pos_x - j, (float)result.pos_y - i);
					level.level_flow.at<cv::Vec2f>(i, j) = mv;

				}
			}
		}
		else
		{
			int image2_xpos, image2_ypos;
			for (int i = 0; i < level.image1.rows; i += level.block_size) //i and j here correspond to the y and x position in image 1
			{
				for (int j = level.image1.cols / 2; j < level.image1.cols; j += level.block_size)
				{
					image2_xpos = j + (int)level.level_flow.at<cv::Vec2f>(i, j)[0];
					image2_ypos = i + (int)level.level_flow.at<cv::Vec2f>(i, j)[1];
					//BlockPosition result = find_min_block(i, j, image2_ypos, image2_xpos); //returns i, j position of block found
					BlockPosition result = find_min_block_spiral(i, j, image2_ypos, image2_xpos); //returns i, j position of block found using spiral search
					//Calculate MV
					cv::Vec2f mv = cv::Vec2f((float)result.pos_x - j, (float)result.pos_y - i);
					level.level_flow.at<cv::Vec2f>(i, j) = mv;

				}
			}
		}
	}

	BlockPosition find_min_block_spiral(int image1_ypos, int image1_xpos, int image2_ypos, int image2_xpos) const
	{
		//form search window
		int shift = level.search_size - level.block_size; //assuming square block size
		int block_size = level.block_size; //speed up
		int width = level.image1.cols;
		int height = level.image1.rows;

		if (image2_xpos < 0 || image2_ypos < 0 || (image2_xpos + block_size) > width || (image2_ypos + block_size) > height) //prevent spiral search from going outside of image
		{
			BlockPosition temp;
			temp.pos_x = image1_xpos; //these will cause the MV to be zero for blocks going outside of the image
			temp.pos_y = image1_ypos;
			return temp;
		}

		int min_x = image2_xpos; //initalizing the positions of the block which we will calculate below 
		int min_y = image2_ypos;
		int SAD_value;
		int SAD_min = (int)cv::norm(level.image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level.image2(cv::Rect(min_x, min_y, block_size, block_size)), cv::NORM_L1); //current SAD value

		int l = min_x;
		int k = min_y;
		int m, t;

		//This first outer loop is used to do the spiral search
		//We are repeating patterns of moving right,down, left, then up.
		//At the very end, we go right once more to finish things off.
		//The algorithm is basically:
		//right, down, left(m+1), up(m+1).  And then right(m+1) at the very end.
		for (m = 1; m < shift; m += 2)
		{
			//the variable m will tell us how much to shift each time
			//the variable t is a counter.  if we have to shift 5 times, we will
			//shift one position at a time and calculate the SAD for each shift.
			for (t = 0; t < m; t++)
			{
				l = l + 1; //m;

				if (l < 0 || k < 0 || (l + block_size) > width || (k + block_size) > height) //prevent spiral search from going outside of image
					continue;

				SAD_value = (int)cv::norm(level.image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level.image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
				if (SAD_value < SAD_min)
				{
					SAD_min = SAD_value;
					min_x = l;
					min_y = k;
				}
			}

			for (t = 0; t < m; t++)
			{
				k = k + 1; //m;

				if (l < 0 || k < 0 || (l + block_size) > width || (k + block_size) > height) //prevent spiral search from going outside of image
					continue;

				SAD_value = (int)cv::norm(level.image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level.image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
				if (SAD_value < SAD_min)
				{
					SAD_min = SAD_value;
					min_x = l;
					min_y = k;
				}
			}

			for (t = 0; t < m + 1; t++)
			{
				l = l - 1; //(m + 1);

				if (l < 0 || k < 0 || (l + block_size) > width || (k + block_size) > height) //prevent spiral search from going outside of image
					continue;

				SAD_value = (int)cv::norm(level.image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level.image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
				if (SAD_value < SAD_min)
				{
					SAD_min = SAD_value;
					min_x = l;
					min_y = k;
				}
			}

			for (t = 0; t < m + 1; t++)
			{
				k = k - 1; //(m + 1);

				if (l < 0 || k < 0 || (l + block_size) > width || (k + block_size) > height) //prevent spiral search from going outside of image
					continue;

				SAD_value = (int)cv::norm(level.image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level.image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
				if (SAD_value < SAD_min)
				{
					SAD_min = SAD_value;
					min_x = l;
					min_y = k;
				}
			}
		}

		//This is what we do at the end to move across the top row.
		for (t = 0; t < (m - 1); t++)
		{
			l = l + 1; //m;

			if (l < 0 || k < 0 || (l + block_size) > width || (k + block_size) > height) //prevent spiral search from going outside of image
				continue;

			SAD_value = (int)cv::norm(level.image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level.image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
			if (SAD_value < SAD_min)
			{
				SAD_min = SAD_value;
				min_x = l;
				min_y = k;
			}
		}

		//store frame2 position, SAD value, and block side in the fast array for future quick lookup
		fast_array.at<cv::Vec4i>(image1_ypos, image1_xpos) = cv::Vec4i(min_x, min_y, SAD_min, block_size);

		BlockPosition pos; //Create class object to return values
		pos.pos_x = min_x;
		pos.pos_y = min_y;

		return pos;

	}

	virtual void operator()(const cv::Range& range) const
	{
		for (int i = range.start; i < range.end; i++)
		{
			process_half(i);
		}
	}

	
};

#endif //PARALLEL_H