#include "rw_flow.h"
#include "standard_headers.h"

// read and write our simple .flo flow file format

// ".flo" file format used for optical flow evaluation
//
// Stores 2-band float image for horizontal (u) and vertical (v) flow components.
// Floats are stored in little-endian order.
// A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
//
//  bytes  contents
//
//  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
//


// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

// the "official" threshold - if the absolute value of either 
// flow component is greater, it's considered unknown
#define UNKNOWN_FLOW_THRESH 1e9

// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10

// return whether flow vector is unknown
bool Flow::unknown_flow(float u, float v) {
	return (fabs(u) >  UNKNOWN_FLOW_THRESH)
		|| (fabs(v) >  UNKNOWN_FLOW_THRESH)
		|| isnan(u) || isnan(v);
}

bool Flow::unknown_flow(float *f) {
	return unknown_flow(f[0], f[1]);
}

// read a flow file into 2-band image
void Flow::ReadFlowFile(cv::Mat &img, const char* filename)
{
	if (filename == NULL)
	{
		std::cout << "ReadFlowFile: empty filename" << std::endl;
		exit(1);
	}

	const char *dot = strrchr(filename, '.');
	if (strcmp(dot, ".flo") != 0)
	{
		std::cout << "ReadFlowFile extension .flo expected" << std::endl;
		exit(1);
	}

	FILE *stream = fopen(filename, "rb");
	if (stream == 0)
	{
		std::cout << "ReadFlowFile: could not open file" << std::endl;
		exit(1);
	}

	int width, height;
	float tag;

	if ((int)fread(&tag, sizeof(float), 1, stream) != 1 ||
		(int)fread(&width, sizeof(int), 1, stream) != 1 ||
		(int)fread(&height, sizeof(int), 1, stream) != 1)
	{
		std::cout << "ReadFlowFile: problem reading file" << std::endl;
		exit(1);
	}
	if (tag != TAG_FLOAT) // simple test for correct endian-ness
	{
		std::cout << "ReadFlowFile: wrong tag (possibly due to big-endian machine?)" << std::endl;
		exit(1);
	}
	// another sanity check to see that integers were read correctly (99999 should do the trick...)
	if (width < 1 || width > 99999)
	{
		std::cout << "ReadFlowFile: illegal width" << std::endl;
		exit(1);
	}

	if (height < 1 || height > 99999)
	{
		std::cout << "ReadFlowFile: illegal height" << std::endl;
		exit(1);
	}

	int nBands = 2;
	img = cv::Mat(height, width, CV_32FC2);

	int n = 1;
	float *ptr;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			ptr = &img.at<cv::Vec2f>(i, j)[0]; //x component

			if ((int)fread(ptr, sizeof(float), n, stream) != n) //reads in x component
			{
				std::cout << "ReadFlowFile: file is too short" << std::endl;
				exit(1);
			}

			ptr = &img.at<cv::Vec2f>(i, j)[1]; //x component

			if ((int)fread(ptr, sizeof(float), n, stream) != n) //reads in y component
			{
				std::cout << "ReadFlowFile: file is too short" << std::endl;
				exit(1);
			}

		}
	}

	if (fgetc(stream) != EOF)
	{
		std::cout << "ReadFlowFile: file is too long" << std::endl;
		exit(1);
	}

	fclose(stream);
}

// write a 2-band image into flow file 
void Flow::WriteFlowFile(cv::Mat img, const char* filename)
{
	if (filename == NULL)
	{
		std::cout << "WriteFlowFile: empty filename" << std::endl;
		exit(1);
	}

	const char *dot = strrchr(filename, '.');
	if (dot == NULL)
	{
		std::cout << "WriteFlowFile: extension required in filename" << std::endl;
		exit(1);
	}

	if (strcmp(dot, ".flo") != 0)
	{
		std::cout << "WriteFlowFile: filename should have extension '.flo'" << std::endl;
		exit(1);
	}

	FILE *stream = fopen(filename, "wb");
	if (stream == 0)
	{
		std::cout << "WriteFlowFile: could not open file" << std::endl;
		exit(1);
	}

	// write the header
	fprintf(stream, TAG_STRING);
	if ((int)fwrite(&img.cols, sizeof(int), 1, stream) != 1 ||
		(int)fwrite(&img.rows, sizeof(int), 1, stream) != 1)
	{
		std::cout << "WriteFlowFile: problem writing header" << std::endl;
		exit(1);
	}

	float *ptr;
	// write the rows
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			ptr = &img.at<cv::Vec2f>(i, j)[0];
			if ((int)fwrite(ptr, sizeof(float), 1, stream) != 1)
			{
				std::cout << "WriteFlowFile: problem writing data" << std::endl;
				exit(1);
			}

			ptr = &img.at<cv::Vec2f>(i, j)[1];
			if ((int)fwrite(ptr, sizeof(float), 1, stream) != 1)
			{
				std::cout << "WriteFlowFile: problem writing data" << std::endl;
				exit(1);
			}

		}
	}
	
	fclose(stream);
}