#include "motion_framework.h"
#include "parallel.h"

MF::MF(cv::Mat &image1, cv::Mat &image2, const int search_size[], const int block_size[], const int num_levels)// , int pad_x, int pad_y)
{
	//TODO: make sure number of levels > 0
	assert(num_levels > 0);
	assert(image1.size() == image2.size());

	//TODO:  check that the number of elements in search size and block size is the same as the number of levels
	orig_height = image1.rows;
	orig_width = image1.cols;

	//Determine how much padding to add to images at highest resolution level
	double temp_h = (double)orig_height;
	double temp_w = (double)orig_width;

	int done = 0;
	while (!done)
	{
		if (temp_h == 2 * orig_height || temp_w == 2 * orig_width) //the 2*orig_width and height is there just in case we can't find a multiple -- we need to stop somewhere
		{
			std::cout << "Could not find any multiples of the block size that match padded image dimensions" << std::endl;
			getchar();
			exit(1);
		}

		double rem_h = 0; //keep track of the remainder from the mod for the height
		double rem_w = 0; //same as above but for the width

		for (int i = 0; i < num_levels; i++)
		{
			rem_h += fmod(temp_h, pow(2, i)*block_size[i]); //we add together all the reaminders to check if it is nonzero
			rem_w += fmod(temp_w, pow(2, i)*block_size[i]);
		}

		if (rem_h == 0 && rem_w == 0) //means that we are done -- we found the right width and height that is divisible by all the block sizes
			done = 1;
		else
		{
			if (rem_h != 0)
				temp_h++;
			if (rem_w != 0)
				temp_w++;
		}
	}

	padded_height = temp_h; //save padded height and width
	padded_width = temp_w;
	int pad_x = ((int)temp_w - orig_width) / 2;
	int pad_y = ((int)temp_h - orig_height) / 2;

	padding_x = pad_x; //save these since they will be used at the end of the block matching to truncate the MV field to get rid of padded regions
	padding_y = pad_y;

	//Pad the images so that there are an integer number of blocks
	cv::Mat image1_pad = cv::Mat(image1.rows + pad_y * 2, image1.cols + pad_x * 2, CV_8UC1);
	cv::Mat image2_pad = cv::Mat(image1.rows + pad_y * 2, image1.cols + pad_x * 2, CV_8UC1);

	cv::copyMakeBorder(image1, image1_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::copyMakeBorder(image2, image2_pad, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, cv::Scalar(0));

	//initialize lambda_multiplier to '1' - means no multiply
	lambda_multiplier = 1;

	//save highest level of hierarchy to vector
	PyramidLevel temp;
	temp.image1 = image1_pad; //store image pointers
	temp.image2 = image2_pad;
	temp.level_flow = cv::Mat::zeros(image1_pad.rows, image1_pad.cols, CV_32FC2); //store MV pointer
	temp.block_size = block_size[0]; //store block size
	temp.search_size = search_size[0]; //store search size
	temp.lambda = (float)((block_size[0]) / 2);
	level_data.push_back(temp);

	//For speed up code -- save SAD values so they don't have to be recalculated
	cv::Mat temparr(image1_pad.rows, image1_pad.cols, CV_32SC4, cv::Scalar(0, 0, 0, 0)); //initialize memory to hold fast array
	fast_array.push_back(temparr); //reserve images equal to number of levels
	//cv::Mat temparr2(image1.rows, image1.cols, CV_32FC4, cv::Scalar(0, 0, 0, 0)); //initialize memory to hold fast array of MVs
	//fast_array_MV.push_back(temparr2); 

	//Keep track of the previous image so we can apply the pyrDown operation on previous image
	cv::Mat prev_image1 = image1_pad.clone();
	cv::Mat prev_image2 = image2_pad.clone();

	for (int i = 1; i < num_levels; i++)
	{
		PyramidLevel temp;
		pyrDown(prev_image1, temp.image1, cv::Size(prev_image1.cols / 2, prev_image1.rows / 2)); //create downsampled images
		pyrDown(prev_image2, temp.image2, cv::Size(prev_image2.cols / 2, prev_image2.rows / 2));

		temp.level_flow = cv::Mat::zeros(prev_image1.rows / 2, prev_image1.cols / 2, CV_32FC2); //create space to store the computed MVs for the level
		temp.block_size = block_size[i];
		temp.search_size = search_size[i];
		temp.lambda = (float)((block_size[i]) / 2);

		cv::Mat temparr(prev_image1.rows / 2, prev_image1.cols / 2, CV_32SC4, cv::Scalar(0, 0, 0, 0)); //initialize memory to hold fast array
		fast_array.push_back(temparr); //reserve images equal to number of levels
		//cv::Mat temparr2(image1.rows / 2, image1.cols / 2, CV_32FC4, cv::Scalar(0, 0, 0, 0)); //initialize memory to hold fast array of MVs
		//fast_array_MV.push_back(temparr2);

		prev_image1 = temp.image1.clone(); //save previous values for next iteration
		prev_image2 = temp.image2.clone();

		level_data.push_back(temp); //save current level data to vector
	}

	//open debugging file -- can safely comment this out if not being used
	//file.open("debug.txt");

}

cv::Mat MF::calcMotionBlockMatching()
{
	for (int i = (int)level_data.size() - 1; i >= 0; i--)
	{
		curr_level = i;
		//perform block matching on each level, starting with the lowest resolution level
		if (i == level_data.size() - 1) //means we don't have any previous motion field to use
		{
			//don't need to copy MVs from previous level since there are none
			calcLevelBM();
			//tbb::task_group tg;
			//tg.run([&]() { calcLevelBM_Parallel(); });
			//tg.wait();


			overlap_img = calculate_level_overlap();			//HERE: Location-1 for overlap creation

			//calcLevelBM_Parallel();
			//print_debug(); //you can put this line and the three lines below back in for testing/debugging purposes
			//cv::Mat test_img = level_data[curr_level].image1.clone();
			//draw_MVs(test_img);
			//cv::imwrite("mv_image.png", test_img);

			int init_bsize = level_data[curr_level].block_size;
			float init_lambda = level_data[curr_level].lambda;
			for (int k = 0; k < 1; k++)
			{
				level_data[curr_level].block_size = init_bsize;
				level_data[curr_level].lambda = init_lambda;

				//perform iterative regularization			
				while (level_data[curr_level].block_size > 1)
				{
					for (int l = 0; l < 2; l++) //perform four iterations of regularization
					{
						lambda_multiplier = l + 1;
						regularize_MVs(); //perform regularization on eight-connected spatial neighbors, use previous results for speedup 
					}
					//need to assign MVs to smaller blocks here
					divide_blocks(); //block size will be reduced by half
					level_data[curr_level].block_size = (level_data[curr_level].block_size >> 1);
					level_data[curr_level].lambda = level_data[curr_level].lambda * 2;
				}
			}
			level_data[curr_level].block_size = init_bsize; //need to reset the block size so that copyMVs() for the next level of the pyramid works with the right size

			//cv::Mat test_img = level_data[curr_level].image1.clone(); //this line and the three lines above are for testing/debugging purposes			
			//draw_MVs(test_img);
			//cv::imwrite("mv_imageL3.png", test_img);

			////draw MC image
			//cv::Mat test_img2 = cv::Mat(level_data[curr_level].image1.rows, level_data[curr_level].image1.cols, CV_8UC1);
			//draw_MVimage(test_img2);
			//cv::imwrite("MC_imageL3.png", test_img2);

		}
		else
		{
			copyMVs(); //copy MVs from previous level to next level in hierarchy (and multiply their magnitude by a factor of two)
			calcLevelBM();
			//calcLevelBM_Parallel();
			//print_debug(); //you can put this line in for testing/debugging purposes


			overlap_img = calculate_level_overlap();			//HERE: Location-1 for overlap creation

			//perform iterative regularization
			int init_bsize = level_data[curr_level].block_size;
			float init_lambda = level_data[curr_level].lambda;
			for (int k = 0; k < 1; k++)
			{
				level_data[curr_level].block_size = init_bsize;
				level_data[curr_level].lambda = init_lambda;

				//perform iterative regularization			
				while (level_data[curr_level].block_size > 1)
				{
					for (int l = 0; l < 2; l++) //perform four iterations of regularization
					{
						lambda_multiplier = l + 1;
						regularize_MVs(); //perform regularization on eight-connected spatial neighbors, use previous results for speedup 
					}
					//need to assign MVs to smaller blocks here
					divide_blocks(); //block size will be reduced by half
					level_data[curr_level].block_size = (level_data[curr_level].block_size >> 1);
					level_data[curr_level].lambda = level_data[curr_level].lambda * 2;
				}
			}
			level_data[curr_level].block_size = init_bsize; //need to reset the block size so that copyMVs() for the next level of the pyramid works with the right size

			//if (i == 1)
			//{
			//	cv::Mat test_img = level_data[curr_level].image1.clone(); //this line and the three lines above are for testing/debugging purposes			
			//	draw_MVs(test_img);
			//	cv::imwrite("mv_imageL2.png", test_img);
			//}
		}
	}
	level_data[curr_level].block_size = 2; //set back to 2x2 blocks so we can call the function below
	copy_to_all_pixels(); //copy the MV for the block to all pixels in the block

	//cv::Mat test_img = level_data[curr_level].image1.clone(); //this line and the three lines above are for testing/debugging purposes
	//level_data[curr_level].block_size = 32; //just to draw MVs with some spacing
	//draw_MVs(test_img);
	//cv::imwrite("mv_imageL1.png", test_img);

	////draw MC image
	//cv::Mat test_img2 = cv::Mat(level_data[curr_level].image1.rows, level_data[curr_level].image1.cols, CV_8UC1);
	//draw_MVimage(test_img2);
	//cv::imwrite("MC_imageL1.png", test_img2);

	return level_data[curr_level].level_flow;
}

void MF::calcLevelBM_Parallel()
{
	cv::parallel_for_(cv::Range(0, 2), Parallel_process(level_data[curr_level], fast_array[curr_level]));
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
			//BlockPosition result = find_min_block(i, j, image2_ypos, image2_xpos); //returns i, j position of block found
			BlockPosition result = find_min_block_spiral(i, j, image2_ypos, image2_xpos); //returns i, j position of block found using spiral search
			//Calculate MV
			cv::Vec2f mv = cv::Vec2f((float)result.pos_x - j, (float)result.pos_y - i);
			level_data[curr_level].level_flow.at<cv::Vec2f>(i, j) = mv;

			//fill_block_MV(i, j, level_data[curr_level].block_size, mv); //assign MV to every pixel in block -- this is necessary because of the padding of images.  We can't guarantee that the the block at the next level will start on a position where there's a motion vector if we just assign a motion vector to the top left corner of the block on previous level.
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
	int SAD_value; //current SAD value
	//cv::Scalar SAD_value; //current SAD value

	int l1_dist = std::numeric_limits<int>::max(); //keep track of the L1 distance between block to choose a block closer to the center block
	int block_size = level_data[curr_level].block_size; //speed up

	for (int k = max(0, image2_ypos - start_pos); k < min(level_data[curr_level].image1.rows - block_size + 1, image2_ypos + start_pos + 1); k++)
	{
		for (int l = max(0, image2_xpos - start_pos); l < min(level_data[curr_level].image1.cols - block_size + 1, image2_xpos + start_pos + 1); l++)
		{
			//calculate difference between block i,j in image1 and block k,l in image 2
			SAD_value = (int)cv::norm(level_data[curr_level].image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level_data[curr_level].image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);

			//cv::absdiff(level_data[curr_level].image1(cv::Rect(image1_xpos, image1_ypos, level_data[curr_level].block_size, level_data[curr_level].block_size)), level_data[curr_level].image2(cv::Rect(l, k, level_data[curr_level].block_size, level_data[curr_level].block_size)), curr_diff);
			//SAD_value = cv::sum(curr_diff);
			if (SAD_value/*.val[0]*/ < SAD_min)
			{
				SAD_min = SAD_value/*.val[0]*/; //we need val[0] because of the way that the cv::Scalar is set up
				min_x = l;
				min_y = k;
				l1_dist = abs(image1_xpos - l) + abs(image1_ypos - k);
			}
			else if (SAD_value/*.val[0]*/ == SAD_min && (abs(image1_xpos - l) + abs(image1_ypos - k)) < l1_dist) //this will choose the block that is closest to the center block
			{
				min_x = l;
				min_y = k;
				l1_dist = abs(image1_xpos - l) + abs(image1_ypos - k);
			}
		}
	}

	//store frame2 position, SAD value, and block side in the fast array for future quick lookup
	fast_array[curr_level].at<cv::Vec4i>(image1_ypos, image1_xpos) = cv::Vec4i(min_x, min_y, SAD_min, block_size);

	BlockPosition pos; //Create class object to return values
	pos.pos_x = min_x;
	pos.pos_y = min_y;

	return pos;

}

BlockPosition MF::find_min_block_spiral(int image1_ypos, int image1_xpos, int image2_ypos, int image2_xpos)
{
	//form search window
	int shift = level_data[curr_level].search_size - level_data[curr_level].block_size; //assuming square block size
	int block_size = level_data[curr_level].block_size; //speed up
	int width = level_data[curr_level].image1.cols;
	int height = level_data[curr_level].image1.rows;

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
	int SAD_min = (int)cv::norm(level_data[curr_level].image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level_data[curr_level].image2(cv::Rect(min_x, min_y, block_size, block_size)), cv::NORM_L1); //current SAD value

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

			SAD_value = (int)cv::norm(level_data[curr_level].image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level_data[curr_level].image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
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

			SAD_value = (int)cv::norm(level_data[curr_level].image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level_data[curr_level].image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
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

			SAD_value = (int)cv::norm(level_data[curr_level].image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level_data[curr_level].image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
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

			SAD_value = (int)cv::norm(level_data[curr_level].image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level_data[curr_level].image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
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

		SAD_value = (int)cv::norm(level_data[curr_level].image1(cv::Rect(image1_xpos, image1_ypos, block_size, block_size)), level_data[curr_level].image2(cv::Rect(l, k, block_size, block_size)), cv::NORM_L1);
		if (SAD_value < SAD_min)
		{
			SAD_min = SAD_value;
			min_x = l;
			min_y = k;
		}
	}

	//store frame2 position, SAD value, and block side in the fast array for future quick lookup
	fast_array[curr_level].at<cv::Vec4i>(image1_ypos, image1_xpos) = cv::Vec4i(min_x, min_y, SAD_min, block_size);

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
	candidates.reserve(9);

	for (int i = 0; i < height; i += block_size)
	{
		for (int j = 0; j < width; j += block_size)
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
			else if (j == 0) //i == height - block_size && j == 0)
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
	energy.reserve(9);

	//positions in image2
	//int pos_x2, pos_y2;

	//place to hold SAD value
	int SAD_value;
	//cv::Scalar SAD_value;

	//place to hold smoothness value
	float Smoothness;

	//place to hold Overlap value
	float Overlap;

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

	int csize = (int)candidates.size(); //store to speed things up

	cv::Vec2f pos1 = cv::Vec2f((float)pos_x1, (float)pos_y1);
	cv::Vec2f pos2;

	//check if there is a min_candidate already stored in the fast_array_MV for this block size (block size != 0), next compare the current candidates to the MVs in the fast_array_MV
	//if (fast_array_MV[curr_level].at<cv::Vec4f>(pos_y1, pos_x1)[3] == (float)block_size && check_prev_match(pos_x1, pos_y1)) //means that a match exists
	//return;

	for (int i = 0; i < csize; i++)
	{
		//block position in image2
		pos2 = pos1 + candidates[i];

		if ((int)pos2[0] < 0 || (int)pos2[0] > (width - block_size) || (int)pos2[1] < 0 || (int)pos2[1] > (height - block_size)) //need to make sure that position doesn't go outside of image
		{
			Energy = std::numeric_limits<float>::max(); //force this candidate not to be chosen
			energy.push_back(Energy);
		}
		else
		{
			//Calculate SAD term
			//cv::ocl::oclMat term1(level_data[curr_level].image1(cv::Rect(pos_x1, pos_y1, block_size, block_size)));
			//cv::ocl::oclMat term2(level_data[curr_level].image2(cv::Rect(pos_x2, pos_y2, block_size, block_size)));
			//cv::ocl::oclMat diff;
			//cv::ocl::absdiff(term1, term2, diff);

			//cv::absdiff(level_data[curr_level].image1(cv::Rect(pos_x1, pos_y1, block_size, block_size)), level_data[curr_level].image2(cv::Rect(pos_x2, pos_y2, block_size, block_size)), curr_diff);
			//SAD_value = cv::sum(curr_diff);

			cv::Vec4i temp = fast_array[curr_level].at<cv::Vec4i>(pos_y1, pos_x1);
			if (temp[0] == (int)pos2[0] && temp[1] == (int)pos2[1] && temp[3] == block_size)
				SAD_value = temp[2];
			else
			{
				SAD_value = (int)cv::norm(level_data[curr_level].image1(cv::Rect(pos_x1, pos_y1, block_size, block_size)), level_data[curr_level].image2(cv::Rect((int)pos2[0], (int)pos2[1], block_size, block_size)), cv::NORM_L1);
				//store frame2 position, SAD value, and block side in the fast array for future quick lookup
				fast_array[curr_level].at<cv::Vec4i>(pos_y1, pos_x1) = cv::Vec4i((int)pos2[0], (int)pos2[1], SAD_value, block_size);
			}

			//Calculate smoothness term - pass current MV and the candidate structure
			Smoothness = calculate_smoothness(i, candidates);
			
			Overlap = calculate_candidate_overlap(i, candidates, pos_x1, pos_y1);				//Calculate Overlap of the candidate (average of overlap_image pixels within block)

			//std::cout << Overlap << std::endl;
			Energy = (float)SAD_value/*.val[0]*/ + lambda*(float)lambda_multiplier*Smoothness + Overlap;	//works better without SAD multiplied. Lower endpoint error
			//Energy = (float)SAD_value/*.val[0]*/ + lambda*(float)lambda_multiplier*Smoothness + Overlap*(float)SAD_value;	
			//Energy = (float)SAD_value/*.val[0]*/ + lambda*(float)lambda_multiplier*Smoothness;
			energy.push_back(Energy); //this is super inefficient and should be fixed
		}
	}

	//Call function to return the position in energy vector that has minimum value
	min_pos = min_energy_candidate(energy);

	//Assign candidate at min_pos to be the new MV
	level_data[curr_level].level_flow.at<cv::Vec2f>(pos_y1, pos_x1) = candidates[min_pos];

	//Assign candidate at min_pos to the fast_array_MVs for speedups
	//fast_array_MV[curr_level].at<cv::Vec4f>(pos_y1, pos_x1) = cv::Vec4f(candidates[min_pos][0], candidates[min_pos][1], (float)min_pos, (float)block_size);

}



cv::Mat MF::calculate_level_overlap()
{
	overlap_img = cv::Mat::zeros(level_data[curr_level].level_flow.size(), CV_8U);	


	//cv::Vec2f pos1 = cv::Vec2f((float)j, (float)i);
	cv::Point2f pos2;
	//int block_size = level_data[curr_level + 1].block_size;


	//for (int i = 0; i < level_data[curr_level].image1.rows; i += level_data[curr_level].block_size) //for each block in image 1,
	//{
	//	for (int j = 0; j < level_data[curr_level].image1.cols; j += level_data[curr_level].block_size)
	//	{
	for (int i = 0; i < level_data[curr_level].level_flow.rows; i += level_data[curr_level].block_size) //for each block,
	{
		for (int j = 0; j < level_data[curr_level].level_flow.cols; j += level_data[curr_level].block_size)
		{
			//pos2 = cv::Point2f(j,i) + level_data[curr_level].level_flow.at<cv::Vec2f>(i, j);
			pos2.y = i + level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[1];			//pos2 = pos1 + mv(of current block)
			pos2.x = j + level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[0];

			//std::cout << level_data[curr_level].image1.cols << "," << level_data[curr_level].image1.rows << std::endl;
			//cv::waitKey(0);
			if ((pos2.x < level_data[curr_level].level_flow.cols) && (pos2.y < level_data[curr_level].level_flow.rows))
			{
				//std::cout << pos2.x << "," << pos2.y << std::endl;
				for (int y_co = max(0,(int)pos2.y); y_co < min((int)pos2.y + level_data[curr_level].block_size, level_data[curr_level].level_flow.rows); y_co++)			//For all pixels within block
					for (int x_co = max(0,(int)pos2.x); x_co < min((int)pos2.x + level_data[curr_level].block_size, level_data[curr_level].level_flow.cols); x_co++)
						overlap_img.at<uchar>(y_co, x_co)++;
			}
		}
	}

	//// DISPLAY PURPOSES ONLY!!   making overlaps visible. Curr_level goes from 3 -> 0.
	//for (int y_co = 0; y_co < overlap_img.rows; y_co++)
	//	for (int x_co = 0; x_co < overlap_img.cols; x_co++)
	//		overlap_img.at<uchar>(y_co, x_co) = 20 * overlap_img.at<uchar>(y_co, x_co) + 128;
	
	//if (curr_level == 0)
	//{
	//	cv::namedWindow("curr_Level 0", 1);
	//	cv::imshow("curr_Level 0", overlap_img);
	//	//cv::imwrite("C:\\Users\\ashish\\Desktop\\curr_level 0.jpg", overlap_img);
	//	//cv::waitKey(0);
	//}

	//if (curr_level == 1)
	//{
	//	cv::namedWindow("curr_Level 1", 1);
	//	cv::imshow("curr_Level 1", overlap_img);
	//	//cv::imwrite("C:\\Users\\ashish\\Desktop\\curr_level 1.jpg", overlap_img);
	//}

	//if (curr_level == 2)
	//{
	//	cv::namedWindow("curr_Level 2", 1);
	//	cv::imshow("curr_Level 2", overlap_img);
	//	//cv::imwrite("C:\\Users\\ashish\\Desktop\\curr_level 2.jpg", overlap_img);
	//}

	//if (curr_level == 3)
	//{
	//	cv::namedWindow("curr_Level 3", 1);
	//	cv::imshow("curr_Level 3", overlap_img);
	//	//cv::imwrite("C:\\Users\\ashish\\Desktop\\curr_level 3.jpg", overlap_img);
	//}

	return overlap_img;
}


float MF::calculate_candidate_overlap(int current_candidate, std::vector<cv::Vec2f> &candidates, int pos_X1, int pos_Y1)
{
	//std::cout << "candover started \t" << std::endl;
	float overlap = 0;
	cv::Vec2f MV = candidates[current_candidate];
	cv::Point2f currBlockPos = cv::Point2f(pos_X1,pos_Y1);
	//cv::Mat tempOvlapStore = cv::Mat::zeros(level_data[curr_level].block_size, level_data[curr_level].block_size,CV_8U);
	//int row = 0, col = 0, count1 = 0, count2 = 0, count3 = 0;

	//for (int row = max(0, currBlockPos.y); row < min(row + level_data[curr_level].block_size, level_data[curr_level].level_flow.rows); row++)	//clear prev from overlap_img. Will compensate for the same at the end
	//	for (int col = max(0, currBlockPos.x); col < min(col + level_data[curr_level].block_size, level_data[curr_level].level_flow.cols); col++)
	//	{
	for (int row = max(0, currBlockPos.y); row < min(currBlockPos.y + level_data[curr_level].block_size, level_data[curr_level].level_flow.rows); row++)	//clear prev from overlap_img. Will compensate for the same at the end
		for (int col = max(0, currBlockPos.x); col < min(currBlockPos.x + level_data[curr_level].block_size, level_data[curr_level].level_flow.cols); col++)
		{
			//std::cout << count1 << std::endl;
			//count1++;
			//tempOvlapStore.at<uchar>(row, col) = overlap_img.at<uchar>(row + currBlockPos.y + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[1], col + currBlockPos.x + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[0]);
			overlap_img.at<uchar>(max(0, min(level_data[curr_level].level_flow.rows, row + currBlockPos.y + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[1])), max(0, min(level_data[curr_level].level_flow.cols, col + currBlockPos.x + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[0])))--;											//EXCEPTION AT THIS LINE. Accessing pixels outside range
		}

	cv::Vec2f curr_MV;								//not really useful in this version of code
	curr_MV = candidates[current_candidate];
		
	//for (int row = max(0, currBlockPos.y); row < min(row + level_data[curr_level].block_size, level_data[curr_level].level_flow.rows); row++)	//Putting in new candidate's MVs. Getting them out after accumulation. Probably saves time of another 2 nested for-loops
	//	for (int col = max(0, currBlockPos.x); col < min(col + level_data[curr_level].block_size, level_data[curr_level].level_flow.cols); col++)
	//	{
	for (int row = max(0, currBlockPos.y); row < min(currBlockPos.y + level_data[curr_level].block_size, level_data[curr_level].level_flow.rows); row++)	//Putting in new candidate's MVs. Getting them out after accumulation. Probably saves time of another 2 nested for-loops
		for (int col = max(0, currBlockPos.x); col < min(currBlockPos.x + level_data[curr_level].block_size, level_data[curr_level].level_flow.cols); col++)
		{
			//std::cout << count2 << std::endl;
			//count2++;
			//overlap_img.at<uchar>(max(0, min(level_data[curr_level].level_flow.rows, row + currBlockPos.y + curr_MV[1])), max(0, min(level_data[curr_level].level_flow.cols, col + currBlockPos.x + curr_MV[0])))++;
			//overlap += (float)overlap_img.at<uchar>(max(0, min(level_data[curr_level].level_flow.rows, row + currBlockPos.y + curr_MV[1])), max(0, min(level_data[curr_level].level_flow.cols, col + currBlockPos.x + curr_MV[0])));
			//overlap_img.at<uchar>(max(0, min(level_data[curr_level].level_flow.rows, row + currBlockPos.y + curr_MV[1])), max(0, min(level_data[curr_level].level_flow.cols, col + currBlockPos.x + curr_MV[0])))--;
			overlap += overlap_img.at<uchar>(max(0, min(level_data[curr_level].level_flow.rows, row + currBlockPos.y + curr_MV[1])), max(0, min(level_data[curr_level].level_flow.cols, col + currBlockPos.x + curr_MV[0]))) + 1;
		}

	//for (int row = max(0, currBlockPos.y); row < min(row + level_data[curr_level].block_size, level_data[curr_level].level_flow.rows); row++)	//Compensation
	//	for (int col = max(0, currBlockPos.x); col < min(col + level_data[curr_level].block_size, level_data[curr_level].level_flow.cols); col++)
	//	{
	for (int row = max(0, currBlockPos.y); row < min(currBlockPos.y + level_data[curr_level].block_size, level_data[curr_level].level_flow.rows); row++)	//Compensation
		for (int col = max(0, currBlockPos.x); col < min(currBlockPos.x + level_data[curr_level].block_size, level_data[curr_level].level_flow.cols); col++)
		{
			//std::cout << count3 << std::endl;
			//count3++;
			//overlap_img.at<uchar>(row + currBlockPos.y + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[1], col + currBlockPos.x + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[0]) = tempOvlapStore.at<uchar>(row, col);
			//overlap_img.at<uchar>(row + currBlockPos.y + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[1], col + currBlockPos.x + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[0])++;
			overlap_img.at<uchar>(max(0, min(level_data[curr_level].level_flow.rows, row + currBlockPos.y + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[1])), max(0, min(level_data[curr_level].level_flow.cols, col + currBlockPos.x + level_data[curr_level].level_flow.at<cv::Vec2f>(currBlockPos.y, currBlockPos.x)[0])))++;
		}
		
	overlap /= (level_data[curr_level].block_size * level_data[curr_level].block_size); //divide cumulative overlap in block by the number of pixeld
	//overlap++;

	//std::cout << overlap << std::endl;

	return overlap;
}



float MF::calculate_smoothness(int current_candidate, std::vector<cv::Vec2f> &candidates)
{
	//TODO: Can we speed this up using the L1 norm function in OpenCV?
	float cost = 0;

	cv::Vec2f MV = candidates[current_candidate];

	//float MVx = candidates[current_candidate][0];
	//float MVy = candidates[current_candidate][1];

	int csize = (int)candidates.size();

	cv::Vec2f curr_MV;

	for (int i = 0; i < csize; i++) //we could remove one of the candidates since its value will be zero if we want a small speedup
	{
		curr_MV = candidates[i];
		cost += abs(curr_MV[0] - MV[0]) + abs(curr_MV[1] - MV[1]); //uses L1 norm
	}

	return cost;
}

int MF::min_energy_candidate(std::vector<float> &energy)
{
	float min_val = energy[0];
	int min_pos = 0;

	int esize = (int)energy.size();

	for (int i = 1; i < esize; i++)
	{
		if (energy[i] < min_val)
		{
			min_val = energy[i];
			min_pos = i;
		}
	}
	return min_pos;
}

//bool MF::check_prev_match(int pos_x1, int pos_y1)
//{
//	int width = level_data[curr_level].image1.cols;
//	int height = level_data[curr_level].image1.rows;
//	int block_size = level_data[curr_level].block_size;
//
//	//Split the first two channels from fast_array_MV so we can compare them with level_flow MVs below
//	/*cv::Mat fast_MVs = cv::Mat(height, width, CV_32FC2, cv::Scalar(0, 0, 0, 0));
//	cv::Mat nada = cv::Mat(height, width, CV_32FC2, cv::Scalar(0, 0, 0, 0));
//
//	cv::Mat out[] = { fast_MVs, nada };
//	int from_to[] = { 0, 0, 1, 1, 2, 2, 3, 3};
//	cv::mixChannels(&fast_array_MV[curr_level], 1, out, 2, from_to, 4);*/
//
//	//TODO:  for all of the if statements below, compare the level_data's level_flow values to the fast_array_MV values
//
//	//this is the normal case and where the loop will spend most of the time -- the case where we are not on any borders
//	if (pos_y1 - block_size >= 0 && pos_x1 - block_size >= 0 && pos_x1 + block_size < width && pos_y1 + block_size < height)
//	{
//		if ((level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 - block_size)))
//			return true;
//		else
//			return false;
//	}
//
//	//Handle the case of the top row
//	else if (pos_x1 - block_size >= 0 && pos_x1 + block_size < width && pos_y1 == 0)
//	{
//		if ((level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 - block_size)))
//			return true;
//		else
//			return false;
//	}
//
//	//Handle the case of the bottom row
//	else if (pos_x1 - block_size >= 0 && pos_x1 + block_size < width && pos_y1 == height - block_size)
//	{
//		if ((level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1)))
//			return true;
//		else
//			return false;
//	}
//
//	//Handle the case of the left column
//	else if (pos_x1 == 0 && pos_y1 - block_size >= 0 && pos_y1 + block_size < height)
//	{
//		if ((level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1)))
//			return true;
//		else
//			return false;
//	}
//
//	//Handle the case of the right column
//	else if (pos_x1 == width - block_size && pos_y1 - block_size >= 0 && pos_y1 + block_size < height)
//	{
//		if ((level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 - block_size)))
//			return true;
//		else
//			return false;
//	}
//
//	//Handle the case of the top left corner
//	else if (pos_y1 == 0 && pos_x1 == 0)
//	{
//		if ((level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1)))
//			return true;
//		else
//			return false;
//	}
//
//	//Handle the case of the top right corner
//	else if (pos_y1 == 0) //this may seem strange, but it is the order of the if statements that matters.
//	{
//		if ((level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 + block_size, (float)pos_x1 - block_size)))
//			return true;
//		else
//			return false;
//	}
//
//	//Handle the case of the bottom left corner
//	else if (pos_x1 == 0) //pos_y1 == height - block_size && pos_x1 == 0)
//	{
//		if ((level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 + block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 + block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1)))
//			return true;
//		else
//			return false;
//	}
//
//	//Handle the case of the bottom right corner
//	else
//	{
//		if ((level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 - block_size) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1 - block_size)) &&
//			(level_data[curr_level].level_flow.at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1) == fast_array_MV[curr_level].at<cv::Vec2f>((float)pos_y1 - block_size, (float)pos_x1)))
//			return true;
//		else
//			return false;
//	}
//
//}

void MF::fill_block_MV(int i, int j, int block_size, cv::Vec2f mv)
{
	//Can this be speed up by using the assignment function in OpenCV?
	for (int k = i; k < i + block_size; k++)
	{
		for (int l = j; l < j + block_size; l++)
		{
			level_data[curr_level].level_flow.at<cv::Vec2f>(k, l) = mv;
		}
	}
}

void MF::copy_to_all_pixels()
{
	int block_size = level_data[curr_level].block_size;

	for (int i = 0; i < level_data[curr_level].level_flow.rows; i += block_size)
	{
		for (int j = 0; j < level_data[curr_level].level_flow.cols; j += block_size)
		{
			fill_block_MV(i, j, block_size, level_data[curr_level].level_flow.at<cv::Vec2f>(i, j));
		}
	}
}

void MF::copyMVs()
{
	int block_size = level_data[curr_level + 1].block_size;
	for (int i = 0; i < level_data[curr_level + 1].image1.rows; i += block_size)
	{
		for (int j = 0; j < level_data[curr_level + 1].image1.cols; j += block_size)
		{
			//get the MV for the current position
			cv::Vec2f new_MV = level_data[curr_level + 1].level_flow.at<cv::Vec2f>(i, j).mul(cv::Vec2f(2, 2));

			//we will fill the new_MV from new_i = 2*i, and new_j = 2*j 
			//level_data[curr_level].level_flow.at<cv::Vec2f>(i << 1, j << 1) = new_MV;
			fill_block_MV(i << 1, j << 1, block_size << 1, new_MV);
		}
	}
}

void MF::divide_blocks()
{
	int block_size_orig = level_data[curr_level].block_size;
	int bsize_new = level_data[curr_level].block_size >> 1;
	int height = level_data[curr_level].image1.rows;
	int width = level_data[curr_level].image1.cols;

	for (int i = 0; i < height; i += block_size_orig)
	{
		for (int j = 0; j < width; j += block_size_orig)
		{
			cv::Vec2f curr_MV = level_data[curr_level].level_flow.at<cv::Vec2f>(i, j); //we will assign the MVs to three of the smaller blocks within the large block (the top left block is already assigned)
			level_data[curr_level].level_flow.at<cv::Vec2f>(i + bsize_new, j) = curr_MV;
			level_data[curr_level].level_flow.at<cv::Vec2f>(i, j + bsize_new) = curr_MV;
			level_data[curr_level].level_flow.at<cv::Vec2f>(i + bsize_new, j + bsize_new) = curr_MV;
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
			cv::line(test_img, cv::Point(j, i), cv::Point(max(0, min((int)(j + level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[0]), level_data[curr_level].image1.cols)), max(0, min((int)(i + level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[1]), level_data[curr_level].image1.rows))), cv::Scalar(255, 0, 0));
		}
	}
}

void MF::draw_MVimage(cv::Mat &test_img)
{
	int image2_xpos;
	int image2_ypos;

	for (int i = 0; i < level_data[curr_level].image1.rows; i += level_data[curr_level].block_size)
	{
		for (int j = 0; j < level_data[curr_level].image1.cols; j += level_data[curr_level].block_size)
		{
			image2_xpos = j + (int)level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[0];
			image2_ypos = i + (int)level_data[curr_level].level_flow.at<cv::Vec2f>(i, j)[1];

			if (image2_xpos < 0 || image2_xpos >(level_data[curr_level].image1.cols - level_data[curr_level].block_size) || image2_ypos < 0 || image2_ypos >(level_data[curr_level].image1.rows - level_data[curr_level].block_size))
				continue;

			level_data[curr_level].image2(cv::Rect(image2_xpos, image2_ypos, level_data[curr_level].block_size, level_data[curr_level].block_size)).copyTo(test_img(cv::Rect(j, i, level_data[curr_level].block_size, level_data[curr_level].block_size)));
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
