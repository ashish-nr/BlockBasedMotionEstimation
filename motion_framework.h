#ifndef MOTION_FRAMEWORK_H
#define MOTION_FRAMEWORK_H

#include "opencv_headers.h"
#include "pyramid_level.h"
#include "block_position.h"
#include "standard_headers.h"

class MF
{
public:
	MF(cv::Mat &image1, cv::Mat &image2, const int search_size[], const int block_size[], const int num_levels);// , int pad_x, int pad_y);
	cv::Mat calcMotionBlockMatching(); //Perform block matching for the whole hierarchy/pyramid
	~MF();

	int padded_height;
	int padded_width;
	int padding_x;
	int padding_y;

private:
	void copyMVs(); //Copy scaled MVs (by a factor of two) from previous level of hierarchy to next highest resolution level of hierarchy
	void calcLevelBM(); //Calculate block matching for a single level of the hierarchy
	void calcLevelBM_Parallel(); //Calculate block matching for a single level of the hierarchy using multi-threaded implementation
	BlockPosition find_min_block(int image1_ypos, int image1_xpos, int image2_ypos, int image2_xpos); //find the block with the minimum SAD in the search window
	BlockPosition find_min_block_spiral(int image1_ypos, int image1_xpos, int image2_ypos, int image2_xpos); //find the block with the minimum SAD in the search window with a spiral search
	void fill_block_MV(int i, int j, int block_size, cv::Vec2f mv); //fill the whole block with the MV -- so we have a MV for each pixel
	int min(int elem1, int elem2); //for finding the minimum value between two elements
	int max(int elem1, int elem2); //for finding the maximum value between two elements
	void regularize_MVs(); //speed up is to do a SAD lookup so we don't have to recalculate value
	void divide_blocks(); //used to assign MVs to blocks of half the size of current block
	void find_min_candidate(int pos_x1, int pos_y1, std::vector<cv::Vec2f> &candidates);
	float calculate_smoothness(int current_candidate, std::vector<cv::Vec2f> &candidates);
	int min_energy_candidate(std::vector<float> &energy);
	void copy_to_all_pixels();
	cv::Mat calculate_level_overlap();
	float calculate_candidate_overlap(int current_candidate, std::vector<cv::Vec2f> &candidates, int pos_X1, int pos_Y1);

	std::vector<PyramidLevel> level_data;
	int curr_level; //used to keep track of current level in hierarchy that we are processing
	int lambda_multiplier; //used to keep track of the integer that multiplies lambda when doing iterative regularization

	//variable that will be used to truncate the padded MVs at the end
	int orig_height;
	int orig_width;

	//overlap
	cv::Mat overlap_img;

	//variables used to speed up computation
	std::vector<cv::Mat> fast_array;
	//std::vector<cv::Mat> fast_array_MV;
	//bool check_prev_match(int pos_x1, int pos_y1);

	std::ofstream file; //file for debugging purposes
	void print_debug(); //print out MVs for debugging/verification purposes
	void draw_MVs(cv::Mat &test_img); //for debugging - draws MVs on image
	void draw_MVimage(cv::Mat &test_img); //for debugging - draws the motion compensated frame
};

#endif //MOTION_FRAMEWORK_H
