#include<opencv2\opencv.hpp>
#include<iostream>
#include<string>
#include<cmath>
#define pi 3.1415
#define eps 2.2204e-16
using namespace std;
using namespace cv;

Mat sqrt2Mat(Mat a, Mat b);
Mat atan2Mat(Mat a, Mat b);
Mat BingHogFeature(Mat currFeat, Mat b_mag, Mat b_orient, int cell_size, int nblock, int bin_num);
void ExtractHogFeature(Mat Image, Rect rect, int cell_size, int nblock, int overlap, int angle, int bin_num);

int main(int argc, char* argv[])
{
	String imagepath = "D:\work\CompHogFeature\CompHogFeature\test.jpg";
	
	int x = atoi(argv[1]);
	int y = atoi(argv[2]);
	int w = atoi(argv[3]);
	int h = atoi(argv[4]);
	
	Mat Image = imread(imagepath);
	Rect rect(x, y, w, h);
	//Rect rect(1,1,32,32);
	//ExtractHogFeature(Image, rect, 8, 2, 0.5, 180, 9);
	Mat c = Image(rect);
	imshow("tux", c);
	waitKey(0);

	return 0;
}

void ExtractHogFeature(Mat Image, Rect rect, int cell_size, int nblock, int overlap, int angle, int bin_num)
{
	float a[1][3] = {-1, 0, 1};
	Point anchor = Point(-1, -1);
	double delta =0 ;
	int ddepth = -1;

	Mat grad_x, grad_y, grad_angle, grad_mag;
	Mat hx(1, 3, CV_32FC1, a);  
	Mat hy = -hx.t();

	Mat imageRoi = Image(rect);
	cvtColor(imageRoi, imageRoi, CV_BGR2GRAY);
	filter2D(imageRoi, grad_x, ddepth, hx, anchor, delta);
    filter2D(imageRoi, grad_y, ddepth, hy, anchor, delta);

	grad_mag = sqrt2Mat(grad_x, grad_y);
	grad_angle = (atan2Mat(grad_y, grad_x)+(pi/2))*180/pi; 
	double bin_angle = angle/bin_num;
    Mat grad_orient = grad_angle/bin_angle;

	int block_size = cell_size * nblock;
	int skip_step = block_size * overlap;
	int w = imageRoi.cols;
	int h = imageRoi.rows;
	int x_step_num = (w-block_size)/skip_step + 1;
	int y_step_num = (h-block_size)/skip_step +1;

	int feat_dim = bin_num * nblock * nblock;
	Mat HogFeature = Mat::zeros(feat_dim, x_step_num * y_step_num, CV_32FC1);

	for(int k = 1;k <= y_step_num; k++)
	{
		for(int j= 1;j <= x_step_num; j++)
		{
			int x_off = (j-1) * skip_step + 1;
			int y_off = (k-1) * skip_step + 1;

			Mat b_mag = grad_mag(Rect(x_off, y_off, block_size, block_size));
			Mat b_orient = grad_orient(Rect(x_off, y_off, block_size, block_size));
			Mat currFeat(feat_dim, 1, CV_32FC1);
			currFeat = BingHogFeature(currFeat, b_mag, b_orient, cell_size, nblock, bin_num);

			for(int i = 0; i < feat_dim; i++)
			{
				HogFeature.at<float>(i,(k - 1) * x_step_num + j) = currFeat.at<float>(i,1);
			}
		    
		}
	}

	cout<<HogFeature<<endl;
	
}



Mat BingHogFeature(Mat currFeat, Mat b_mag, Mat b_orient, int cell_size, int nblock, int bin_num)
{
	Mat blockfeat = Mat::zeros(bin_num * nblock * nblock, 1, CV_32FC1);
	
	for(int n = 1; n <= nblock; n++)
	{
		for(int m = 1; m <= nblock; m++)
		{
			int x_off = (m - 1) * cell_size + 1;
			int y_off = (n -1 )* cell_size + 1;

			Mat c_mag = b_mag(Rect(x_off, y_off, cell_size, cell_size));
			Mat c_orient = b_orient(Rect(x_off, y_off, cell_size, cell_size));

			Mat c_feat = Mat::zeros(bin_num, 1, CV_32FC1);
			for(int i = 1; i <= bin_num; i++)
			{
				for( int s = 0; s < cell_size; s++)
					for(int w = 0; w < cell_size; w++)
					{
						if(c_orient.at<float>(s, w)== i)
					     c_feat.at<float>(i, 1) = c_feat.at<float>(i, 1) + c_mag.at<float>(s, w);
					}
						
			}



			int count = (n - 1) * nblock + m;

			for(int p = 0; p < count; p++)
			   {
				   blockfeat.at<float>(p, 1) = c_feat.at<float>(p, 1);
		     	}

		}
	}

	   int sum = eps;
	   for(int i = 0; i < bin_num * nblock * nblock; i++)
	   {
		   sum = blockfeat.at<float>(i,1) * blockfeat.at<float>(i,1);
	   }

	   for(int j = 0; j < bin_num * nblock * nblock; j++)
	   {
		   blockfeat.at<float>(j,1) = blockfeat.at<float>(j,1)/sum;
	   }


	   return blockfeat;

}


Mat atan2Mat(Mat a, Mat b)
{
    int rows = a.rows;
    int cols = a.cols;
	Mat result(rows, cols, CV_32FC1);
    for(int i=0; i<rows; i++)
    {
		
        float* ptra = ( float*)(a.data+i*a.step);
        float* ptrb = ( float*)(b.data+i*b.step);
        float* ptrout = ( float*)(result.data+i*result.step);
        for(int j=0; j<cols; j++)
        {
            *ptrout = atan2(*ptra,*ptrb);
            ptra++;
            ptrb++;
            ptrout++;
        }
    }
    return result;
}


Mat sqrt2Mat(Mat a, Mat b)
{
    int rows = a.rows;
    int cols = a.cols;
	Mat result(rows, cols, CV_32FC1);
    for(int i=0; i<rows; i++)
    {
		for(int j=0; j<cols; j++)
		{
			result.at<float>(i,j) = a.at<float>(i,j) * a.at<float>(i,j) + b.at<float>(i,j) * b.at<float>(i,j);
			sqrt((double)result.at<float>(i,j));
		}
	}
	   
    return result;
}