#include "motion_framework.h"

MF::MF(cv::Mat &image1, cv::Mat &image2, const int search_size[], const int block_size[], const int num_levels)
{
	//TODO: make sure number of levels > 1

	//save highest level of hierarchy to vector
	PyramidLevel temp;
	temp.image1 = image1; //store image pointers
	temp.image2 = image2;
	temp.level_flow = cv::Mat::zeros(image1.rows, image1.cols, CV_32FC2);; //store MV pointer
	temp.block_size = block_size[0]; //store block size
	temp.search_size = search_size[0]; //store search size
	level_data.push_back(temp);

	//Keep track of the previous image so we can apply the pyrDown operation on previous image
	cv::Mat prev_image1 = image1.clone();
	cv::Mat prev_image2 = image2.clone();

	for (int i = 1; i < num_levels; i++)
	{
		PyramidLevel temp;
		pyrDown(prev_image1, temp.image1, cv::Size(prev_image1.cols / 2, prev_image1.rows / 2)); //create downsampled images
		pyrDown(prev_image2, temp.image2, cv::Size(prev_image2.cols / 2, prev_image2.rows / 2));
		temp.level_flow = cv::Mat::zeros(prev_image1.rows / 2, prev_image1.cols / 2, CV_32FC2); //create space to store the computed MVs for the level
		temp.block_size = block_size[i];
		temp.search_size = search_size[i];

		prev_image1 = temp.image1.clone(); //save previous values for next iteration
		prev_image2 = temp.image2.clone();

		level_data.push_back(temp); //save current level data to vector
	}

}

void MF::calcMotionBlockMatching()
{
	for (int i = (int)level_data.size() - 1; i >= 0; i--)
	{
		curr_level = i;
		//perform block matching on each level, starting with the lowest resolution level
		if (i == level_data.size() - 1) //means we don't have any previous motion field to use
		{
			//don't need to copy MVs from previous level
			calcLevelBM();
		}
		else
		{
			//TO DO:  need to copy MVs to next level
			copyMVs();
			calcLevelBM();
		}
	}
}

void MF::calcLevelBM()
{
	int image2_xpos, image2_ypos;
	for (int i = 0; i < level_data[curr_level].image1.rows; i += level_data[curr_level].block_size) //i and j here correspond to the y and x position in image 1
	{
		for (int j = 0; j < level_data[curr_level].image1.cols; j += level_data[curr_level].block_size)
		{
			image2_xpos = j + (int)level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[0];
			image2_ypos = i + (int)level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[1];
			BlockPosition result = find_min_block(i, j, image2_ypos, image2_xpos); //returns i, j position of block found
			//Calculate MV
			cv::Vec2f mv = cv::Vec2f((float)result.pos_x - j, (float)result.pos_y - i);
			fill_block_MV(i, j, level_data[curr_level].block_size, mv); //assign MV to every pixel in block
		}
	}
}

BlockPosition MF::find_min_block(int image1_ypos, int image1_xpos, int image2_ypos, int image2_xpos)
{
	//form search window
	int start_pos = ((level_data[curr_level].search_size - level_data[curr_level].block_size) >> 1); //assuming square block size
	int SAD_min = std::numeric_limits<int>::max(); //max value an integer can take -- used to initialize SAD value
	int min_x = image2_xpos; //initalizing the positions of the block which we will calculate below 
	int min_y = image2_ypos;
	cv::Mat curr_diff; //absolute difference block
	cv::Scalar SAD_value; //current SAD value

	for (int k = max(0, image2_ypos - start_pos); k < min(level_data[curr_level].image1.rows - level_data[curr_level].block_size + 1, image2_ypos + start_pos); k++)
	{
		for (int l = max(0, image2_xpos - start_pos); l < min(level_data[curr_level].image1.cols - level_data[curr_level].block_size + 1, image2_xpos + start_pos); l++)
		{
			//calculate difference between block i,j in image1 and block k,l in image 2
			cv::absdiff(level_data[curr_level].image1(cv::Rect(image1_xpos, image1_ypos, level_data[curr_level].block_size, level_data[curr_level].block_size)), level_data[curr_level].image2(cv::Rect(l, k, level_data[curr_level].block_size, level_data[curr_level].block_size)), curr_diff);
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

void MF::fill_block_MV(int i, int j, int block_size, cv::Vec2f mv)
{
	for (int k = i; k < i + level_data[curr_level].block_size; k++)
	{
		for (int l = j; l < j + level_data[curr_level].block_size; l++)
		{
			level_data[curr_level].level_flow.at<cv::Vec2f>(k, l) = mv;
		}
	}
}

void MF::copyMVs()
{
	for (int i = 0; i < level_data[curr_level + 1].image1.rows; i += level_data[curr_level + 1].block_size)
	{
		for (int j = 0; j < level_data[curr_level + 1].image1.cols; j += level_data[curr_level + 1].block_size)
		{
			//get the MV for the current position
			cv::Vec2f new_MV = level_data[curr_level + 1].level_flow.at<cv::Vec2f>(i, j).mul(cv::Vec2f(2, 2)); //need to check that this works

			//we will fill the new_MV from new_i = 2*i, and new_j = 2*j and for size 2*levels[prev_level].block_size
			fill_block_MV(i << 1, j << 1, level_data[curr_level + 1].block_size << 1, new_MV);
		}
	}
}

int MF::max(int elem1, int elem2)
{
	return ((elem1 > elem2) ? elem1 : elem2);
}

int MF::min(int elem1, int elem2)
{
	return ((elem1 < elem2) ? elem1 : elem2);
}

void MF::colorMVs()
{


}

MF::~MF()
{

}