#ifndef MOTION_FRAMEWORK_H
#define MOTION_FRAMEWORK_H

#include "opencv_headers.h"
#include "pyramid_level.h"
#include "block_position.h"
#include "standard_headers.h"

class MF
{
  public:
		MF(cv::Mat &image1, cv::Mat &image2, const int search_size[], const int block_size[], const int num_levels);
	  cv::Mat calcMotionBlockMatching(); //Perform block matching for the whole hierarchy/pyramid
		~MF();

  private:
		void copyMVs(); //Copy scaled MVs (by a factor of two) from previous level of hierarchy to next highest resolution level of hierarchy
		void calcLevelBM(); //Calculate block matching for a single level of the hierarchy
		BlockPosition find_min_block(int image1_ypos, int image1_xpos, int image2_ypos, int image2_xpos); //find the block with the minimum SAD in the search window
		void fill_block_MV(int i, int j, int block_size, cv::Vec2f mv); //fill the whole block with the MV -- so we have a MV for each pixel
		int min(int elem1, int elem2); //for finding the minimum value between two elements
		int max(int elem1, int elem2); //for finding the maximum value between two elements
		void regularize_MVs();
		void divide_blocks(); //used to assign MVs to blocks of half the size of current block
		void find_min_candidate(int pos_x1, int pos_y1, std::vector<cv::Vec2f> &candidates);
		float calculate_smoothness(int current_candidate, std::vector<cv::Vec2f> &candidates);
		int min_energy_candidate(std::vector<float> &energy);
		void copy_to_all_pixels();

		std::vector<PyramidLevel> level_data;
		int curr_level; //used to keep track of current level in hierarchy that we are processing
		int lambda_multiplier; //used to keep track of the integer that multiplies lambda when doing iterative regularization
	
		std::ofstream file; //file for debugging purposes
		void print_debug(); //print out MVs for debugging/verification purposes
		void draw_MVs(cv::Mat &test_img); //for debugging - draws MVs on image
};

#endif //MOTION_FRAMEWORK_H