#ifndef RW_FLOW_H
#define RW_FLOW_H

#include "opencv_headers.h"

class Flow
{
  public:
		Flow();
		// read a flow file into 2-band image
		void ReadFlowFile(cv::Mat &img, const char* filename);
		// write a 2-band image into flow file 
		void WriteFlowFile(cv::Mat img, const char* filename);
		~Flow();

  private:
		// return whether flow vector is unknown
		bool unknown_flow(float u, float v);
		bool unknown_flow(float *f);

};

#endif //RW_FLOW_H
