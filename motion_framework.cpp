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
	temp.lambda = (3 * block_size[0]) >> 2;
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
		temp.lambda = (3 * block_size[i]) >> 2;

		prev_image1 = temp.image1.clone(); //save previous values for next iteration
		prev_image2 = temp.image2.clone();

		level_data.push_back(temp); //save current level data to vector
	}

	//open debugging file
	file.open("debug.txt");

}

cv::Mat MF::calcMotionBlockMatching()
{
	for (int i = (int)level_data.size() - 1; i >= 0; i--)
	{
		curr_level = i;
		//perform block matching on each level, starting with the lowest resolution level
		if (i == level_data.size() - 1) //means we don't have any previous motion field to use
		{
			//don't need to copy MVs from previous level
			calcLevelBM();
			//print_debug(); 
			//cv::Mat test_img = level_data[curr_level].image1.clone();
			//draw_MVs(test_img);
			//cv::imwrite("mv_image.png", test_img);
			regularize_MVs(); //perform regularization on eight-connected spatial neighbors
			//cv::Mat test_img2 = level_data[curr_level].image1.clone();
			//draw_MVs(test_img2);
			//cv::imwrite("mv_image2.png", test_img2);
		}
		else
		{
			//TO DO:  need to copy MVs to next level
			copyMVs();
			calcLevelBM();
			regularize_MVs(); //perform regularization on eight-connected spatial neighbors
		}
	}
	return level_data[curr_level].level_flow;
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
			fill_block_MV(i, j, level_data[curr_level].block_size, mv); //assign MV to every pixel in block -- TODO:  only assign MV to top left pixel in block
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

void MF::regularize_MVs()
{
	int block_size = level_data[curr_level].block_size; //these are temp variables to speed up the computation
	int height = level_data[curr_level].image1.rows;
	int width = level_data[curr_level].image1.cols;

	//create nine candidate MVs which include the current MV and the eight-connected neighbors			
	std::vector<cv::Vec2f> candidates;

	for (int i = 0; i < height; i+=block_size)
	{
		for (int j = 0; j < width; j+=block_size)
		{			
			//this is the normal case and where the loop will spend most of the time -- the case where we are not on any borders
			if (i - block_size >= 0 && j - block_size >= 0 && j + block_size < width && i + block_size < height)
			{
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j - block_size));
			}
			//Handle the case of the top row
			else if (j - block_size >= 0 && j + block_size < width && i == 0)
			{
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j - block_size));
			}
			//Handle the case of the bottom row
			else if (j - block_size >= 0 && j + block_size < width && i == height - block_size)
			{
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j));
			}
			//Handle the case of the left column
			else if (j == 0 && i - block_size >= 0 && i + block_size < height)
			{
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j));
			}
			//Handle the case of the right column
			else if (j == width - block_size && i - block_size >= 0 && i + block_size < height)
			{
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j - block_size));
			}
			//Handle the case of the top left corner
			else if (i == 0 && j == 0)
			{
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j));
			}
			//Handle the case of the top right corner
			else if (i == 0) //this may seem strange, but it is the order of the if statements that matters.
			{
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i + block_size, (float)j - block_size));
			}
			//Handle the case of the bottom left corner
			else if (i == height - block_size && j == 0)
			{
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j + block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j));
			}
			//Handle the case of the bottom right corner
			else 
			{
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j - block_size));
				candidates.push_back(level_data[curr_level].level_flow.at<cv::Vec2f>((float)i - block_size, (float)j));
			}

			find_min_candidate(j, i, candidates); //finds the best MV (based on smoothness notion and SAD criteria) and assigns this MV to the current position

			//clear candidates vector
			candidates.clear();
		}
	}
}

void MF::find_min_candidate(int pos_x1, int pos_y1, std::vector<cv::Vec2f> &candidates)
{
	//store all the energies computed.  An energy is the SAD + lambda*Smoothness term
	std::vector<float> energy;
	
	//positions in image2
	int pos_x2, pos_y2;

	//place to hold SAD value
	cv::Scalar SAD_value;

	//place to hold smoothness value
	float Smoothness;

	//Hold the overall energy - SAD + Smoothness
	float Energy;

	//to speed up the loop
	int block_size = level_data[curr_level].block_size;

	//store lambda value
	float lambda = level_data[curr_level].lambda;

	cv::Mat curr_diff; //absolute difference block

	int min_pos; //used to store candidate that has minimum energy

	int height = level_data[curr_level].image1.rows; //store height and width for bounds checking
	int width = level_data[curr_level].image1.cols;
	
	for (int i = 0; i < candidates.size(); i++)
	{
		//block position in image2
		pos_x2 = pos_x1 + (int)candidates[i][0]; 
		pos_y2 = pos_y1 + (int)candidates[i][1];

		if (pos_x2 < 0 || pos_x2 > (width - block_size) || pos_y2 < 0 || pos_y2 > (height - block_size)) //need to make sure that position doesn't go outside of image
		{
			Energy = std::numeric_limits<float>::max(); //force this candidate not to be chosen
			energy.push_back(Energy);
		}
		else
		{
			//Calculate SAD term
			cv::absdiff(level_data[curr_level].image1(cv::Rect(pos_x1, pos_y1, block_size, block_size)), level_data[curr_level].image2(cv::Rect(pos_x2, pos_y2, block_size, block_size)), curr_diff);
			SAD_value = cv::sum(curr_diff);

			//Calculate smoothness term - pass current MV and the candidate structure
			Smoothness = calculate_smoothness(i, candidates);

			Energy = (float)SAD_value.val[0] + lambda*Smoothness;
			energy.push_back(Energy);
		}
	}

	//Call function to return the position in energy vector that has minimum value
	min_pos = min_energy_candidate(energy);

	//Assign candidate at min_pos to be the new MV
	fill_block_MV(pos_y1, pos_x1, block_size, candidates[min_pos]); //TODO:  don't fill the MVs for the whole block before regularization

}

float MF::calculate_smoothness(int current_candidate, std::vector<cv::Vec2f> &candidates)
{
	float cost = 0;

	float MVx = candidates[current_candidate][0];
	float MVy = candidates[current_candidate][1];

	for (int i = 0; i < candidates.size(); i++) //we could remove one of the candidates since its value will be zero if we want a small speedup
	{
		cost += abs(candidates[i][0] - MVx) + abs(candidates[i][1] - MVy); //uses L1 norm
	}

	return cost;
}

int MF::min_energy_candidate(std::vector<float> &energy)
{
	float min_val = energy[0];
	int min_pos = 0;

	for (int i = 1; i < energy.size(); i++)
	{
		if (energy[i] < min_val)
		{
			min_val = energy[i];
			min_pos = i;
		}
	}
	return min_pos;
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

void MF::copyMVs() //TODO:  Only copy MVs for the top left position in the block
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

void MF::print_debug()
{
	for (int i = 0; i < level_data[curr_level].image1.rows; i++)
	{
		for (int j = 0; j < level_data[curr_level].image1.cols; j++)
		{
			file << level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[0] << std::endl;
			file << level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[1] << std::endl;
		}
	}
}

void MF::draw_MVs(cv::Mat &test_img)
{
	for (int i = 0; i < level_data[curr_level].image1.rows; i += level_data[curr_level].block_size)
	{
		for (int j = 0; j < level_data[curr_level].image1.cols; j += level_data[curr_level].block_size)
		{
			cv::line(test_img, cv::Point(j, i), cv::Point(max(0, min(j + level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[0], level_data[curr_level].image1.cols)), max(0, min(i + level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[1], level_data[curr_level].image1.rows))), cv::Scalar(255, 0, 0));
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

MF::~MF()
{

}