#ifndef RW_FLOW_H
#define RW_FLOW_H

//max cols for color wheel
#define MAXCOLS 60

#include "opencv_headers.h"

class Flow
{
  public:
		//Flow();
		// read a flow file into 2-band image
		void ReadFlowFile(cv::Mat &img, const char* filename);
		// write a 2-band image into flow file 
		void WriteFlowFile(cv::Mat img, const char* filename);
		//Color code motion vectors for easier visualization
		void MotionToColor(cv::Mat &input_img, cv::Mat &output_img, float maxmotion);
		//Calculate the mean-squared error between our MVs and ground truth MVs from middlebury or .flo file
		double CalculateMSE(cv::Mat &gtruth, cv::Mat &flow);
		//~Flow();

  private:
		//set function for color wheel
		void setcols(int r, int g, int b, int k);
		//for creating color wheel
		void makecolorwheel();
		//function for getting the color from the color wheel
		void computeColor(float fx, float fy, cv::Vec3b *pix);
		// return whether flow vector is unknown
		bool unknown_flow(float u, float v);
		bool unknown_flow(float *f);
		//variable to hold color wheel components
		int colorwheel[MAXCOLS][3];
		int ncols = 0;
};

#endif //RW_FLOW_H
