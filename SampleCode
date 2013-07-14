/*
HEY!!!!!! READ THIS BEFORE YOU LOOK AT THE CODE
Please take this code for what it is: a work in progress. It still has tons of debugging to be done, needs to be formatted,
more throughly commented, etc. This is far, far from a final; I have better sense than to include header stuff in the .cpp file and
the littany of other best-practices that i've violated in this file.

This is more just to serve as proof that I can learn to work with other peoples libraries pretty quickly (ive only used opencv 
for a few weeks)
*/
#include "StdAfx.h"
#include "afx.h"

#include <opencv/cv.h>      // include it to used Main OpenCV functions.
#include <opencv/highgui.h> //include it to use GUI functions.
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>


#include <time.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <dirent.h>
#include <sstream>
#include <string>
#include <Windows.h>
#include <math.h>
#include <stdlib.h>


#define M_PI 3.14159265358979323846
using namespace std;
using namespace cv;

struct  CINEFILEHEADER
{
	WORD type;
	WORD headerSize;
	WORD Compression;
	WORD version;
	LONG firstMovieImage;
	DWORD totalImageCount;	
	LONG firstImageNo;
	DWORD imageCount;
	DWORD offImageHeader;
	DWORD offSetup;
	DWORD offImageOffsets;
	long long triggerTime;
};

struct scoredRotatedRect
{
	double score;
	double timeTaken;
	RotatedRect ellipse;
	vector<Point> contour;

};


	CINEFILEHEADER m_cineFileHead;
	BITMAPINFOHEADER m_bitmapHead;
	CFile m_hFile;
	CFile hfile;

scoredRotatedRect fit_pupil(Mat src,int pupil_threshold,bool displayImage,bool printTiming,Mat *srcImage,vector<int> Offsets );
/* Takes an RGB subPicture of our picture that contains the pupil, a pupil threshold, 2 debugging booleans, 
a pointer to an image to optionally write to, and 2 offsets describing the location of the subPicture within
the larger picture
Process is: OLD! Fix scoring metric
convert RGB subpicture to grayscale and threshold image (lines:  )
find contours within subPicture (lines: )
for each contour, score as follows:
	fit and ellipse to the contour, refer to its height and width as height, width
	score=absolute value(height-width)/(min(height,width) (Heuristic to choose most circular)
	score=score*(# of thresholded pixels in containing rectangle)/(area of ellipse) (heuristic to choose an ellipse that has an area equivalent to number of thresholded pixels in contour)
	TODO: change this to other scoring method
return scoredRotatedRect with best score
*/
vector<int> locate_subPictures(Mat src,Mat* leftEye,Mat* rightEye,Mat* triangle,vector<scoredRotatedRect*> scoredTrianglePoints, int thresh_level_triangle,bool displayImages);
/* Takes an RGB image (src), pointers to subPictures of the left eye, right eye, and triangle which we will fill, 
a pointer to a set of scoredRotatedRects which we will fill with our guesses at the triangles ellipses, the threshold level for the triangle,
and a threshold level for the triangle
Process is: OLD! Fix scoring metric
Initialize variables we will be using (lines: )
convert our picture to rgb and threshold it (lines: )
floodFill borders of threshold to get rid of background (lines: )
Attempt to find the 3 triangle points as follows:
	for each contour:
		fit ellipse to contour. If ellipse is too large/too small, continue
		else, add it to the contours we consider
	for each contour we consider:
		score= ((# of thresholded pixels within ellipse)/(area of fitted ellipse ))-1
		TODO ^^ fix this scoring
		Take 3 contours with score closest to 0, these are your triangle ellipses

Based on three ellipse points and their contours, find largest/smallest x and y contained in the contours of our triangles ellipse
create buffers for largest and smallest x and y values by adding/subtracting 10% of the difference between largest and smallest x (or y) values
check to make sure those points are not outside our picture, fix accordingly (lines: )
These buffered points are now the estimated height and width of our triangle
Create subpictures of leftEye, rightEye, and triangle based on these values
return appropriate offsets for these pictures in offsetss
*/


bool get_Cine_Handle(CString fName);
/* gets a handle to retrieve images from */

void get_Next_Frame(int m_nDisplayedImage,IplImage* m_pImage);
/* retrieves frames from our video handle */

vector<scoredRotatedRect> locate_blob(vector<scoredRotatedRect> scoredPriors,Mat src,Mat src_gray,int threshold_level_pupil,int threshold_level_triangle);
/* locate_blob:
takes priors (x,y, height, width of previous ellipse), a src to draw on, a src_gray to take subpictures of, and threshold levels,  and produces a new prediction for the position of the given ellipses. 
Processes the triangle points and the pupils all at once.
This function is repeated in locate_single_blob and single_blob_handle, but is seperated here so we can test locate_single_blob_flood
TODO: Multithread the processesing of each ellipse
Refer to locate_single_blob and single_blob_handle for summary of the process*/

scoredRotatedRect locate_single_blob_flood(scoredRotatedRect scoredPrior,Mat threshold_output_copy);
/* locate_single_blob_flood:
takes a single prior (x,y,height, width of previous ellipse) and a thresholded subpicture and produces a new prediction of the ellipse
Process is:
obtain offsets (lines:   )
floodfill our (guess) at the pupil location (line:  )
get rid of all border material (lines: )
reset value of pupil (lines: )
find borders(contours) of cleaned up image (line: )
score each contour  and keep best(lines : )
return best contour/scoredRotatedRect
*/
vector<scoredRotatedRect> single_blob_handle(vector<scoredRotatedRect> Priors,Mat src,Mat src_gray,int threshold_level_pupil,int threshold_level_triangle,bool isFlood);
/*single_blob_handle:
The testing handle for locate_single_blob_flood and locate_single_blob
process is:
initialize timing/data structures (lines: )
for each prior (ellipse):
	-create subPicture for that prior
	-check subPicture size
	-threshold the subpicture and pass it to the appropriate locating function
	-takes note of time taken and stores that in the scoredRotatedRect variable
return new predictions
*/

scoredRotatedRect locate_single_blob(scoredRotatedRect Prior,Mat threshold_output_copy);
/*locate_single_blob:
takes a single prior (x,y,height, width of previous ellipse) and a thresholded subpicture and produces a new prediction of the ellipse
Process is:
obtain offsets (lines:   )
find borders(contours) of cleaned up image (line: )
score each contour and keep best (lines: )
return best contour/scoredRotatedRect
*/

scoredRotatedRect score_contour(Mat threshold_output_copy_preserve,vector<Point> current_contour,RotatedRect curEllipse,RotatedRect Prior);
/*
score_contour:
takes a thresholded image of the ellipse, its corresponding contour and the ellipse fit of that contour, and a prior of the previous rotatedrect
and return a score in [0,1] reflecting the likelihood that this is our ellipse
Process is:
-create mask of current ellipse and perform bitwise and with thresholded image
-count number of nonzero pixels in that image
-Score=(non zero pixel count of bitwise and image)/(non zero pixel count of thresholded image)
PositionScore = 1/(euclidean distance between center of Prior and center of curEllipse)
if euclidean distance > 20:
	Score=Score*(PositionScore^3) (Prevents shifting to far away ellipses)
return score
*/
int _tmain(int argc, _TCHAR* argv[])
{
  struct dirent *entry;
  DIR *pDIR;
  std::size_t pngFound;
  Mat threshold_output;
  Mat src; Mat src_gray;
  CvMoments moments;
  vector<scoredRotatedRect> trianglePoints(3);
  vector<scoredRotatedRect> pupils(2);
  //std::string dirpath = "c:/pupils/frames/";
  std::string dirpath = "c:/pupils/test_images/full_frame/";
/*  if (pDIR=opendir("c:/pupils/test_images/full_frame")){ //c:/pupils/frames")){ 
	  while (entry=readdir(pDIR)){
		  pngFound=std::string(entry->d_name).find(".bmp"); This code performs the process for a directory of images
		  if (pngFound!=std::string::npos){
			  cout<<std::string(dirpath)+std::string(entry->d_name)<<endl;//  c:/pupils/frames/")+std::string(entry->d_name)<<endl;
			  std::string imagePath=std::string(dirpath)+std::string(entry->d_name);
			  char * pImgPath=new char[imagePath.size()+1];
			  std::copy(imagePath.begin(),imagePath.end(),pImgPath);
			  pImgPath[imagePath.size()]='\0';
			  src=imread(pImgPath);
			  //src=imread("c:/opencv/fb111107.ag1.bmp");
			  cout << pImgPath <<endl;
			  waitKey(0);
			  delete[] pImgPath;
			  //blur( src_gray, src_gray, Size(20,20) );//... results are meh..
			  //medianBlur(src_gray,src_gray,5); median blur... results are meh
			 // Mat morphElement=getStructuringElement(CV_SHAPE_ELLIPSE,Size(21,21),Point(10,10));
			 // morphologyEx(src_gray,src_gray,MORPH_OPEN,morphElement);
			  Mat* LeftEye=new Mat;
			  Mat* RightEye=new Mat;
			  Mat* Triangle=new Mat;
			  vector<int> Offsets(6);
			  Offsets=locate_subPictures(src,LeftEye,RightEye,Triangle,false);
			 // cout<<"Offset"<<Offsets[0]<<","<<Offsets[1]<<","<<Offsets[2]<<","<<Offsets[3]<<","<<Offsets[4]<<","<<Offsets[5]<<endl;
			  vector<int> LeftEyeOffsets(2);
			  vector<int> RightEyeOffsets(2);
			  vector<int> TriangleOffsets(2);
			  LeftEyeOffsets[0]=Offsets[0];
			  LeftEyeOffsets[1]=Offsets[1];
			  RightEyeOffsets[0]=Offsets[2];
			  RightEyeOffsets[1]=Offsets[3];
			  TriangleOffsets[0]=Offsets[4];
			  TriangleOffsets[1]=Offsets[5];
			  pupils[0]=fit_pupil(*LeftEye,30,true,true,&src,LeftEyeOffsets);
			  pupils[1]=fit_pupil(*RightEye,30,true,true,&src,RightEyeOffsets);
			  trianglePoints=fit_triangle(*Triangle,0,100,true,true,&src,TriangleOffsets);
		  }
	  }
  }*/

  /* This code performs the process for an .avi image
  IplImage* frame;
  IplImage* frame1;
  CvCapture* capture=cvCaptureFromFile("c:/pupils/cines/cf120913.ra1_FlashCine25.avi");
  for (int i=0;i<10;i++){
	  frame=cvQueryFrame(capture);
  }
  for (int i=0;i<100;i++)
  {
	  frame=cvQueryFrame(capture);
	  cout<<"here"<<endl;
	  src=Mat(frame);
	  cout<<"here"<<endl;
	   Mat* LeftEye=new Mat;
	   Mat* RightEye=new Mat;
	   Mat* Triangle=new Mat;
	   vector<int> Offsets(6);
	   vector<RotatedRect> TrianglePoints(3);
	   startTime=GetTickCount();
	   Offsets=locate_subPictures(src,LeftEye,RightEye,Triangle,TrianglePoints,true);
	   cout<<"timeTaken:"<<GetTickCount()-startTime<<endl;
	   cout<<"done locating"<<endl;
	   // cout<<"Offset"<<Offsets[0]<<","<<Offsets[1]<<","<<Offsets[2]<<","<<Offsets[3]<<","<<Offsets[4]<<","<<Offsets[5]<<endl;
	   vector<int> LeftEyeOffsets(2);
	   vector<int> RightEyeOffsets(2);
	   vector<int> TriangleOffsets(2);
	   LeftEyeOffsets[0]=Offsets[0];
	   LeftEyeOffsets[1]=Offsets[1];
	   RightEyeOffsets[0]=Offsets[2];
	   RightEyeOffsets[1]=Offsets[3];
	   TriangleOffsets[0]=Offsets[4];
	   TriangleOffsets[1]=Offsets[5];
	   cout<<"fitting pupils"<<endl;
	   	  startTime=GetTickCount();
	   pupils[0]=fit_pupil_1(*LeftEye,30,true,true,&src,LeftEyeOffsets);
	   pupils[1]=fit_pupil_1(*RightEye,30,true,true,&src,RightEyeOffsets);
	   cout<<"timeTaken:"<<GetTickCount()-startTime<<endl;
	   namedWindow("src");
	   imshow("src",src);
	   cvWaitKey(1);
	 //  cvDestroyWindow("src");
	  // trianglePoints=fit_triangle(*Triangle,0,100,true,true,&src,TriangleOffsets);
  }
  */
   /*
  // My method for processing a cine 
  IplImage * currentFrame;
   //CString fName("c:/pupils/cines/cf111005.da1_FlashCine56.cine");
  CString fName("c:/pupils/cines/cf111102.jb1_FlashCine8.cine");
//   CString fName("c:/pupils/cines/cf111123.ag1_FlashCine8.cine");
  currentFrame=cvCreateImage(cvSize(1280,800),8,1);
  bool recomputePriors=true;
  vector<RotatedRect> Priors(5);
   if (get_Cine_Handle(fName)){
	   	   vector<int> Offsets(6);
		   namedWindow("src");
  for (int i=1;i<10000;i++){
	  get_Next_Frame(i,currentFrame);
	  Mat src_flipped(currentFrame);
	  Mat src(Size(src_flipped.size()),CV_8UC1);
	  flip(src_flipped,src,0);
	  Mat src_fake_color(Size(src_flipped.size()),CV_8UC3);
	  cvtColor(src,src_fake_color,CV_GRAY2BGR);
	  Mat* LeftEye=new Mat;
	   Mat* RightEye=new Mat;
	   Mat* Triangle=new Mat;
	   vector<RotatedRect*> TrianglePoints;
	   TrianglePoints.clear();
	   TrianglePoints.push_back(new RotatedRect());
	   TrianglePoints.push_back(new RotatedRect());
	   TrianglePoints.push_back(new RotatedRect());
	   startTime=GetTickCount();
	   Offsets=locate_subPictures(src_fake_color,LeftEye,RightEye,Triangle,TrianglePoints,50,true);
	   vector<int> LeftEyeOffsets(2);
	   vector<int> RightEyeOffsets(2);
	   vector<int> TriangleOffsets(2);
	   LeftEyeOffsets[0]=Offsets[0];
	   LeftEyeOffsets[1]=Offsets[1];
	   RightEyeOffsets[0]=Offsets[2];
	   RightEyeOffsets[1]=Offsets[3];
	   TriangleOffsets[0]=Offsets[4];
	   TriangleOffsets[1]=Offsets[5];
	   pupils[0]=fit_pupil(*LeftEye,20,true,true,&src_fake_color,LeftEyeOffsets);
	   pupils[1]=fit_pupil(*RightEye,20,true,true,&src_fake_color,RightEyeOffsets);//These come back with pupils relative to the x and y offsets
	   pupils[0].center.x=pupils[0].center.x+LeftEyeOffsets[0];
	   pupils[0].center.y=pupils[0].center.y+LeftEyeOffsets[1];
	   pupils[1].center.x=pupils[1].center.x+RightEyeOffsets[0];
	   pupils[1].center.y=pupils[1].center.y+RightEyeOffsets[1];
	   Priors[0]=pupils[0];
	   Priors[1]=pupils[1];
	   Priors[2]=*TrianglePoints[0];
	   Priors[3]=*TrianglePoints[1];
	   Priors[4]=*TrianglePoints[2];
	   imshow("src",src_fake_color);
	   cout<<"waitKeying..."<<endl;
	   waitKey(1);
  }
  } //*/
 // /*

/* The following is the regular, no race starting point
 //CString fName("c:/pupils/cines/cf111005.da1_FlashCine56.cine");
 // CString fName("c:/pupils/cines/cf111102.jb1_FlashCine8.cine");
   CString fName("c:/pupils/cines/cf111123.ag1_FlashCine8.cine");
  IplImage * currentFrame;
  currentFrame=cvCreateImage(cvSize(1280,800),8,1);
  int pupil_threshold=15;
  int triangle_threshold=30;
  bool recomputePriors=true;
  vector<RotatedRect> Priors(5);
   if (get_Cine_Handle(fName)){
	   	   vector<int> Offsets(6);
		     for (int i=500;i<10000;i=i+1){
		   if (recomputePriors){
get_Next_Frame(i,currentFrame);
	  Mat src_flipped(currentFrame);
	  Mat src(Size(src_flipped.size()),CV_8UC1);
	  flip(src_flipped,src,0);
	  Mat src_fake_color(Size(src_flipped.size()),CV_8UC3);
	  cvtColor(src,src_fake_color,CV_GRAY2BGR);
	  Mat* LeftEye=new Mat;
	   Mat* RightEye=new Mat;
	   Mat* Triangle=new Mat;
	   vector<RotatedRect*> TrianglePoints;
	   TrianglePoints.clear();
	   TrianglePoints.push_back(new RotatedRect());
	   TrianglePoints.push_back(new RotatedRect());
	   TrianglePoints.push_back(new RotatedRect());
	   startTime=GetTickCount();
	   Offsets=locate_subPictures(src_fake_color,LeftEye,RightEye,Triangle,TrianglePoints,triangle_threshold,true);
	   vector<int> LeftEyeOffsets(2);
	   vector<int> RightEyeOffsets(2);
	   vector<int> TriangleOffsets(2);
	   LeftEyeOffsets[0]=Offsets[0];
	   LeftEyeOffsets[1]=Offsets[1];
	   RightEyeOffsets[0]=Offsets[2];
	   RightEyeOffsets[1]=Offsets[3];
	   TriangleOffsets[0]=Offsets[4];
	   TriangleOffsets[1]=Offsets[5];
	   pupils[0]=fit_pupil(*LeftEye,pupil_threshold,true,true,&src_fake_color,LeftEyeOffsets);
	   pupils[1]=fit_pupil(*RightEye,pupil_threshold,true,true,&src_fake_color,RightEyeOffsets);//These come back with pupils relative to the x and y offsets
	   pupils[0].center.x=pupils[0].center.x+LeftEyeOffsets[0];
	   pupils[0].center.y=pupils[0].center.y+LeftEyeOffsets[1];
	   pupils[1].center.x=pupils[1].center.x+RightEyeOffsets[0];
	   pupils[1].center.y=pupils[1].center.y+RightEyeOffsets[1];
	   Priors[0]=pupils[0];
	   Priors[1]=pupils[1];
	   Priors[2]=*TrianglePoints[0];
	   Priors[3]=*TrianglePoints[1];
	   Priors[4]=*TrianglePoints[2];
	   LeftEye->release();
	   RightEye->release();
	   Triangle->release();
	   namedWindow("blobs");
	   imshow("blobs",src_fake_color);
	   waitKey(1);
	   recomputePriors=false;
		   }
		   else
		   {
	  get_Next_Frame(i,currentFrame);
	  Mat src_flipped(currentFrame);
	  Mat src(Size(src_flipped.size()),CV_8UC1);
	  flip(src_flipped,src,0);
	  Mat src_fake_color(Size(src_flipped.size()),CV_8UC3);
	  cvtColor(src,src_fake_color,CV_GRAY2BGR);
	  //Priors=locate_blob(Priors,src_fake_color,src,pupil_threshold,triangle_threshold);
	  Priors=single_blob_handle(Priors,src_fake_color,src,pupil_threshold,triangle_threshold,false);
	  cout<<i<<endl;
	  for (int j=0;j<5;j++){
		  if (Priors[j].size.height>50 || Priors[j].size.width>50 ||Priors[j].size.width<10|| Priors[j].size.height<10){
			  cout<<"recomputing"<<endl;
			  recomputePriors=true;
			  break;
		  }
	  }
  }
  }
  }//*/
///* The following is to write comparison pictures
LARGE_INTEGER Frequency;
QueryPerformanceFrequency(&Frequency);
 CString fName("c:/pupils/cines/cf111005.da1_FlashCine56.cine");
 // CString fName("c:/pupils/cines/cf111102.jb1_FlashCine8.cine");
 //  CString fName("c:/pupils/cines/cf111123.ag1_FlashCine8.cine");
  IplImage * currentFrame;
  currentFrame=cvCreateImage(cvSize(1280,800),8,1);
  int pupil_threshold=15;
  int triangle_threshold=30;
  bool recomputePriors=true;
  vector<scoredRotatedRect> Priors(15);
  bool firstRun=true;
  bool reassignFirst=false;
  bool reassignSecond=false;

 // CvVideoWriter *outputVideo=cvCreateVideoWriter("sampleTrack.avi", CV_FOURCC('M', 'J', 'P', 'G'), 60, cv::Size(1280,800), 0);
 VideoWriter outputVideo("sampleTrack2.avi",CV_FOURCC('M','J','P','G'),60.0,Size(1280,800),true);
outputVideo.open("sampleTrack2.avi",CV_FOURCC('M','J','P','G'),60.0,Size(1280,800),true);
if (outputVideo.isOpened()){
	cout<<"it worked"<<endl;
}
else{
	cout<<"it didn't work"<<endl;
}

 /*if (!outputVideo==NULL){
	  cout<<"it worked"<<endl;
  }
  else{
	  cout<<"it didn't work"<<endl;
  }*/
   if (get_Cine_Handle(fName)){
	   	   vector<int> Offsets(6);
		     for (int i=1100;i<1100+10*60;i=i+1){
				 cout<<"on frame: "<<i<<endl;
get_Next_Frame(i,currentFrame);
vector<scoredRotatedRect> curPriors(5);
	  Mat src_flipped(currentFrame);
	  Mat src(Size(src_flipped.size()),CV_8UC1);
	  flip(src_flipped,src,0);
	  Mat src_fake_color(Size(src_flipped.size()),CV_8UC3);
	  Mat src_fake_color_copy;
	  cvtColor(src,src_fake_color,CV_GRAY2BGR);
	  src_fake_color.copyTo(src_fake_color_copy);
	  	  Mat* LeftEye=new Mat;
	   Mat* RightEye=new Mat;
	   Mat* Triangle=new Mat;
	   vector<scoredRotatedRect*> TrianglePoints;
	   TrianglePoints.clear();
	   TrianglePoints.push_back(new scoredRotatedRect());
	   TrianglePoints.push_back(new scoredRotatedRect());
	   TrianglePoints.push_back(new scoredRotatedRect());
	   LARGE_INTEGER startTime;
	   QueryPerformanceCounter(&startTime);
	   Offsets=locate_subPictures(src_fake_color,LeftEye,RightEye,Triangle,TrianglePoints,triangle_threshold,true);
	   vector<int> LeftEyeOffsets(2);
	   vector<int> RightEyeOffsets(2);
	   vector<int> TriangleOffsets(2);
	   LeftEyeOffsets[0]=Offsets[0];
	   LeftEyeOffsets[1]=Offsets[1];
	   RightEyeOffsets[0]=Offsets[2];
	   RightEyeOffsets[1]=Offsets[3];
	   TriangleOffsets[0]=Offsets[4];
	   TriangleOffsets[1]=Offsets[5];
	   pupils[0]=fit_pupil(*LeftEye,pupil_threshold,true,true,&src_fake_color,LeftEyeOffsets);
	   pupils[1]=fit_pupil(*RightEye,pupil_threshold,true,true,&src_fake_color,RightEyeOffsets);//These come back with pupils relative to the x and y offsets
	   LARGE_INTEGER endTime;
	   QueryPerformanceCounter(&endTime);
	   pupils[0].ellipse.center.x=pupils[0].ellipse.center.x+LeftEyeOffsets[0];
	   pupils[0].ellipse.center.y=pupils[0].ellipse.center.y+LeftEyeOffsets[1];
	   pupils[1].ellipse.center.x=pupils[1].ellipse.center.x+RightEyeOffsets[0];
	   pupils[1].ellipse.center.y=pupils[1].ellipse.center.y+RightEyeOffsets[1];
	   Priors[10]=pupils[0];
	   Priors[11]=pupils[1];
	   Priors[12]=*TrianglePoints[0];
	   Priors[13]=*TrianglePoints[1];
	   Priors[14]=*TrianglePoints[2];
	   Mat leftEye_Comparison;
	   Mat rightEye_Comparison;
	   char buffer[32];
	   double freqD=(double)Frequency.QuadPart/1000;
	   double delta=(double)((endTime.QuadPart-startTime.QuadPart)/freqD);
	   ltoa((long)(delta),buffer,10);
	   putText(src_fake_color,buffer,cvPoint(LeftEyeOffsets[0],LeftEyeOffsets[1]+20),FONT_HERSHEY_COMPLEX_SMALL,0.8,cvScalar(0,0,250),1,CV_AA);
	  putText(src_fake_color,"no priors",cvPoint(LeftEyeOffsets[0]+50,LeftEyeOffsets[1]+20),FONT_HERSHEY_COMPLEX_SMALL,0.8,cvScalar(0,0,250),1,CV_AA);
	   LeftEye->copyTo(leftEye_Comparison);
	   RightEye->copyTo(rightEye_Comparison);

	   if (firstRun){
 for (int j=0;j<2;j=j+1){
		  for (int q=0;q<5;q++){
			  Priors[j*5+q]=curPriors[q];
		  }
 }
 firstRun=false;
	   }
LARGE_INTEGER endTimeFlood;
LARGE_INTEGER endTimeScore;
LARGE_INTEGER startTimeFlood;
LARGE_INTEGER startTimeScore;
	   QueryPerformanceCounter(&endTime);
	   	  for (int j=0;j<2;j=j+1){
		  for (int q=0;q<5;q++){
			  curPriors[q]=Priors[j*5+q];
		  }
		  bool isFlood=(j==1);
		  if (isFlood){
		  QueryPerformanceCounter(&startTimeFlood);
		  }
		  else{
			  QueryPerformanceCounter(&startTimeScore);
		  }
	  curPriors=single_blob_handle(curPriors,src_fake_color,src,pupil_threshold,triangle_threshold,isFlood);  
	  		  if (isFlood){
		  QueryPerformanceCounter(&endTimeFlood);
		  }
		  else{
			  QueryPerformanceCounter(&endTimeScore);
		  }
	  for (int q=0;q<5;q++){
			 Priors[j*5+q]=curPriors[q];
		  }
	  }

	   for (int j=0;j<2;j++){
		   src_fake_color_copy.copyTo(src_fake_color);
		   Mat blankImg=Mat::zeros(src_fake_color_copy.size(),CV_8UC3);
		   for (int q=0;q<5;q++){
			   ellipse( src_fake_color, Priors[5*j+q].ellipse, Scalar(0,0,255), 1, 8 );
			   vector<vector<Point>> contours;
			   contours.push_back(Priors[5*j+q].contour);
			   if (q<2){
			   ellipse(blankImg,Priors[5*j+q].ellipse,Scalar(0,0,255),-1,8);
			   }
			   else{
				ellipse(blankImg,Priors[5*j+q].ellipse,Scalar(0,0,255),1,8);
			   }
			   drawContours( blankImg, contours, 0, Scalar(200,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
		   }
		   Point ellipseOne=Point(Priors[2].ellipse.center.x,Priors[2].ellipse.center.y);
		   Point ellipseTwo=Point(Priors[3].ellipse.center.x,Priors[3].ellipse.center.y);
		   Point ellipseThree=Point(Priors[4].ellipse.center.x,Priors[4].ellipse.center.y);
		   line(blankImg,ellipseOne,ellipseTwo,Scalar(0,255,0));
		   line(blankImg,ellipseOne,ellipseThree,Scalar(0,255,0));
		   line(blankImg,ellipseThree,ellipseTwo,Scalar(0,255,0));
		   if (j==0){
			   cout<<"writing frame"<<endl;
			   IplImage* currentFrame=cvCloneImage(&(IplImage)blankImg);
			   namedWindow("ThisOne");
			   imshow("ThisOne",src_fake_color);
			   if (Priors[1].score>1 && Priors[0].score>1){
				   putText(blankImg,"blink detected",cvPoint(640,400),FONT_HERSHEY_COMPLEX_SMALL,0.8,cvScalar(250,0,0),1,CV_AA);
			   }
			 //  int result=cvWriteFrame(outputVideo,currentFrame);
			//   cout<<result<<endl;
			   outputVideo.write(blankImg);
		   }
		   namedWindow("currentAttempt");
		   imshow("currentAttempt",blankImg);
	   if (j==0){
					   double freqD=(double)Frequency.QuadPart/1000;
	   double delta=(double)((endTimeScore.QuadPart-startTimeScore.QuadPart)/freqD);
	   stringstream ss1(stringstream::in | stringstream::out);
	   ss1<<"time:";
	   ss1<<Priors[1].timeTaken;
	   ss1<<"ms";
	   stringstream ss (stringstream::in | stringstream::out);
	   ss<<"score:";
	   if (Priors[1].score<=1){
	   ss<<(Priors[1].score);
	   }
	   else{
		   ss<<0;
	   }
	   putText(src_fake_color,ss.str(),cvPoint(LeftEyeOffsets[0],LeftEyeOffsets[1]+35),FONT_HERSHEY_COMPLEX_SMALL,0.8,cvScalar(0,0,250),1,CV_AA);
	   putText(src_fake_color,ss1.str(),cvPoint(LeftEyeOffsets[0],LeftEyeOffsets[1]+20),FONT_HERSHEY_COMPLEX_SMALL,0.8,cvScalar(0,0,250),1,CV_AA);
	  putText(src_fake_color,"scoring",cvPoint(LeftEyeOffsets[0]+130,LeftEyeOffsets[1]+20),FONT_HERSHEY_COMPLEX_SMALL,0.8,cvScalar(0,0,250),1,CV_AA);
	   }
	   else{
						   double freqD=(double)Frequency.QuadPart/1000;
	   double delta=(double)((endTimeFlood.QuadPart-startTimeFlood.QuadPart)/freqD);
	   stringstream ss1(stringstream::in | stringstream::out);
	   ss1<<"time:";
	   ss1<<Priors[8].timeTaken;
	   ss1<<" ms";
	   stringstream ss (stringstream::in | stringstream::out);
	   ss<<"score:";
	   //if (Priors[6].score<=1){
	   ss<<(Priors[6].score);
	   //}
	   //else{
		//   ss<<0;
	   //}
	   putText(src_fake_color,ss.str(),cvPoint(LeftEyeOffsets[0],LeftEyeOffsets[1]+35),FONT_HERSHEY_COMPLEX_SMALL,0.8,cvScalar(0,0,250),1,CV_AA);
	   putText(src_fake_color,ss1.str(),cvPoint(LeftEyeOffsets[0],LeftEyeOffsets[1]+20),FONT_HERSHEY_COMPLEX_SMALL,0.8,cvScalar(0,0,250),1,CV_AA);
	  putText(src_fake_color,"flood",cvPoint(LeftEyeOffsets[0]+90,LeftEyeOffsets[1]+20),FONT_HERSHEY_COMPLEX_SMALL,0.8,cvScalar(0,0,250),1,CV_AA);
	   }

		   hconcat(leftEye_Comparison,*LeftEye,leftEye_Comparison);
		   hconcat(rightEye_Comparison,*LeftEye,rightEye_Comparison);
	   }
	   namedWindow("leftEye_Comparison");
	   imshow("leftEye_Comparison",leftEye_Comparison);
	   waitKey(1);
	   for (int q=0;q<2;q++){
	     for (int j=0;j<5;j++){
		  if (Priors[5*q+j].ellipse.size.height>50 || Priors[5*q+j].ellipse.size.width>50 ||Priors[5*q+j].ellipse.size.width<10|| Priors[5*q+j].ellipse.size.height<10){
			  cout<<"recomputing"<<endl;
			    for (int n=0;n<5;n++){
					Priors[5*q+n]=Priors[10+n];
				}
			  break;
		  }
	  }
	   }

	   LeftEye->release();
	   RightEye->release();
	   Triangle->release();



  }
  }
  cout<<"releasing"<<endl;
//  cvReleaseVideoWriter(&outputVideo);
/*

// CString fName("c:/pupils/cines/cf111005.da1_FlashCine56.cine");
//  CString fName("c:/pupils/cines/cf111102.jb1_FlashCine8.cine");
   CString fName("c:/pupils/cines/cf111123.ag1_FlashCine8.cine");
  IplImage * currentFrame;
  currentFrame=cvCreateImage(cvSize(1280,800),8,1);
  int pupil_threshold=10;
  int triangle_threshold=30;
  bool recomputePriors=true;
  bool recompute_Priors_flood=true;
  bool recompute_Priors_score=true;
  vector<RotatedRect> Priors_flood(5);
  vector<RotatedRect> Priors_score(5);
   if (get_Cine_Handle(fName)){
	   	   vector<int> Offsets(6);
		     for (int i=500;i<10000;i=i+1){
		   if (recomputePriors){
get_Next_Frame(i,currentFrame);
	  Mat src_flipped(currentFrame);
	  Mat src(Size(src_flipped.size()),CV_8UC1);
	  flip(src_flipped,src,0);
	  Mat src_fake_color(Size(src_flipped.size()),CV_8UC3);
	  cvtColor(src,src_fake_color,CV_GRAY2BGR);
	  Mat* LeftEye=new Mat;
	   Mat* RightEye=new Mat;
	   Mat* Triangle=new Mat;
	   vector<RotatedRect*> TrianglePoints;
	   TrianglePoints.clear();
	   TrianglePoints.push_back(new RotatedRect());
	   TrianglePoints.push_back(new RotatedRect());
	   TrianglePoints.push_back(new RotatedRect());
	   startTime=GetTickCount();
	   Offsets=locate_subPictures(src_fake_color,LeftEye,RightEye,Triangle,TrianglePoints,triangle_threshold,true);
	   vector<int> LeftEyeOffsets(2);
	   vector<int> RightEyeOffsets(2);
	   vector<int> TriangleOffsets(2);
	   LeftEyeOffsets[0]=Offsets[0];
	   LeftEyeOffsets[1]=Offsets[1];
	   RightEyeOffsets[0]=Offsets[2];
	   RightEyeOffsets[1]=Offsets[3];
	   TriangleOffsets[0]=Offsets[4];
	   TriangleOffsets[1]=Offsets[5];
	   pupils[0]=fit_pupil(*LeftEye,pupil_threshold,true,true,&src_fake_color,LeftEyeOffsets);
	   pupils[1]=fit_pupil(*RightEye,pupil_threshold,true,true,&src_fake_color,RightEyeOffsets);//These come back with pupils relative to the x and y offsets
	   pupils[0].center.x=pupils[0].center.x+LeftEyeOffsets[0];
	   pupils[0].center.y=pupils[0].center.y+LeftEyeOffsets[1];
	   pupils[1].center.x=pupils[1].center.x+RightEyeOffsets[0];
	   pupils[1].center.y=pupils[1].center.y+RightEyeOffsets[1];
	   if (recompute_Priors_flood){
	   Priors_flood[0]=pupils[0];
	   Priors_flood[1]=pupils[1];
	   Priors_flood[2]=*TrianglePoints[0];
	   Priors_flood[3]=*TrianglePoints[1];
	   Priors_flood[4]=*TrianglePoints[2];
	   LeftEye->release();
	   RightEye->release();
	   Triangle->release();
	   recompute_Priors_flood=false;
	   }
	   else if (recompute_Priors_score){
	   Priors_score[0]=pupils[0];
	   Priors_score[1]=pupils[1];
	   Priors_score[2]=*TrianglePoints[0];
	   Priors_score[3]=*TrianglePoints[1];
	   Priors_score[4]=*TrianglePoints[2];
	   LeftEye->release();
	   RightEye->release();
	   Triangle->release();
	   recompute_Priors_score=false;
	   }
	   namedWindow("blobs");
	   imshow("blobs",src_fake_color);
	   waitKey(1);
	   recomputePriors=false;
		   }
		   else
		   {
	  get_Next_Frame(i,currentFrame);
	  Mat src_flipped(currentFrame);
	  Mat src(Size(src_flipped.size()),CV_8UC1);
	  flip(src_flipped,src,0);
	  Mat src_fake_color(Size(src_flipped.size()),CV_8UC3);
	  cvtColor(src,src_fake_color,CV_GRAY2BGR);
	  //Priors=locate_blob(Priors,src_fake_color,src,pupil_threshold,triangle_threshold);
	  DWORD startTime=GetTickCount();
	  Priors_flood=single_blob_handle(Priors_flood,src_fake_color,src,pupil_threshold,triangle_threshold,true);
	  DWORD endTime=GetTickCount();
	  Priors_score=single_blob_handle(Priors_score,src_fake_color,src,pupil_threshold,triangle_threshold,false);
	  cout<<"flood:"<<endTime-startTime<<"||| score:"<<GetTickCount()-endTime<<endl;
	  cout<<i<<endl;
	  for (int j=0;j<5;j++){
		  if (Priors_flood[j].size.height>50 || Priors_flood[j].size.width>50 ||Priors_flood[j].size.width<10|| Priors_flood[j].size.height<10){
			  cout<<"recomputing for flood"<<endl;
			  recomputePriors=true;
			  recompute_Priors_flood=true;
			  break;
		  }
	  }
	  	  for (int j=0;j<5;j++){
		  if (Priors_score[j].size.height>50 || Priors_score[j].size.width>50 ||Priors_score[j].size.width<10|| Priors_score[j].size.height<10){
			  cout<<"recomputing for score"<<endl;
			  recomputePriors=true;
			  recompute_Priors_score=true;
			  break;
		  }
	  }
  }
  }
  }
  //*/


  return(0);
}


vector<scoredRotatedRect> locate_blob(vector<scoredRotatedRect> scoredPriors,Mat src,Mat src_gray,int threshold_level_pupil,int threshold_level_triangle){
  vector<vector<Point> > contours;
  vector<RotatedRect> Priors(scoredPriors.size());
  for (int i=0;i<scoredPriors.size();i++){
	  Priors[i]=scoredPriors[i].ellipse;
  }
  // cvtColor( src, src_gray, CV_BGR2GRAY );
  //threshold( src_gray, threshold_output_pupil, threshold_level_pupil, 255, THRESH_BINARY_INV );
 //  threshold( src_gray, threshold_output_triangle, threshold_level_triangle, 255, THRESH_BINARY_INV );
  	vector<vector<Point> > contours_final(Priors.size());
		vector<RotatedRect> Ellipses(Priors.size());
				vector<double> scores(Priors.size());
  for (int i=0;i<Priors.size();i++){
	  	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	    RotatedRect Prior=Priors[i];
  int x_lower=(int)(Prior.center.x-Prior.size.width)>0?(int)(Prior.center.x-Prior.size.width):0;
	int y_lower=(int)(Prior.center.y-Prior.size.height)>0?(int)(Prior.center.y-Prior.size.height):0;
	int x=(int)Prior.center.x;
	int y=(int)Prior.center.y;
	  Mat threshold_output_copy=Mat();
	  Mat threshold_output_copy_preserve=Mat();
	//   cout<<"about to compose"<<endl;
	  if (i<2){
	Mat output_pupil=Mat();
	 Mat output_pupil_copy=Mat();
	 // threshold_output_pupil(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(threshold_output_copy);
	 //threshold_output_pupil(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(threshold_output_copy_preserve);
		 int subPicture_width=x_lower+(int)2*Prior.size.width<src_gray.cols?(int)2*Prior.size.width:src_gray.cols-x_lower;
	 int subPicture_height=y_lower+(int)2*Prior.size.height<src_gray.rows?(int)2*Prior.size.height:src_gray.rows-y_lower;
	 //	 cout<<"x_lower:"<<x_lower<<endl;
	// cout<<"y_lower:"<<y_lower<<endl;
	// cout<<"x:"<<Prior.center.x<<endl;
	// cout<<"y:"<<Prior.center.y<<endl;
	// cout<<"height:"<<subPicture_height<<endl;
	// cout<<"width:"<<subPicture_width<<endl;
	 src_gray(cv::Rect(x_lower,y_lower,subPicture_width,subPicture_height)).copyTo(output_pupil);
	 threshold(output_pupil,threshold_output_copy,threshold_level_pupil,255,THRESH_BINARY_INV);
	 threshold_output_copy.copyTo(threshold_output_copy_preserve);
	  }
	  else{
	Mat output_triangle=Mat();
	 Mat output_triangle_copy=Mat();
     // threshold_output_triangle(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(threshold_output_copy);
	//        threshold_output_triangle(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(threshold_output_copy_preserve);
		 int subPicture_width=x_lower+(int)2*Prior.size.width<src_gray.cols?(int)2*Prior.size.width:src_gray.cols-x_lower;
	 int subPicture_height=y_lower+(int)2*Prior.size.height<src_gray.rows?(int)2*Prior.size.height:src_gray.rows-y_lower;
	// cout<<"x_lower:"<<x_lower<<endl;
	// cout<<"y_lower:"<<y_lower<<endl;
	// cout<<"height:"<<subPicture_height<<endl;
	// cout<<"width:"<<subPicture_width<<endl;
	 src_gray(cv::Rect(x_lower,y_lower,subPicture_width,subPicture_height)).copyTo(output_triangle);
	 threshold(output_triangle,threshold_output_copy,threshold_level_triangle,255,THRESH_BINARY_INV);
	 threshold_output_copy.copyTo(threshold_output_copy_preserve);
	  }
	//  cout<<"done composing"<<endl;
	  if (threshold_output_copy.size().height==0 && threshold_output_copy.size().width==0){
		  Ellipses[i]=Priors[i];
		  Ellipses[i].size.height=0;
		  Ellipses[i].size.width=0;
		  contours_final[i].push_back(Point(0,0));
		  continue;
	  }
	//  	  	  namedWindow("test1");
	//  imshow("test1",threshold_output_copy);
	//  waitKey(0);
	int xOffset=0;
	int yOffset=0;
	int counter=0;
/*	while ((int)threshold_output_copy.at<uchar>((int)(Prior.size.height)+yOffset,(int)(Prior.size.width)+xOffset)==0){
		cout<<":"<<(int)threshold_output_copy.at<uchar>((int)(Prior.size.height)+yOffset,(int)(Prior.size.width)+xOffset)<<":"<<endl;
		xOffset=rand()%10;
		yOffset=rand()%10;
		 //threshold_output_copy.at<float>((int)(Prior.size.width)+xOffset+x_lower,(int)(Prior.size.height)+yOffset+y_lower)=200;
		if (i==0){
 		  namedWindow("blobsCheck");
  imshow("blobsCheck",threshold_output_copy);
  waitKey(1);}
	}
	cout<<":"<<(int)threshold_output_copy.at<uchar>((int)(Prior.size.height)+yOffset,(int)(Prior.size.width)+xOffset)<<":"<<endl;
	  //floodFill(threshold_output_copy,Point((int)(Prior.size.width),(int)(Prior.size.height)),Scalar(100));
	  //floodFill(threshold_output_copy,Point((int)(Prior.size.width)+4,(int)(Prior.size.height)+4),Scalar(100));
	 if (i==0){
  		  namedWindow("blobsCheck2");
  imshow("blobsCheck2",threshold_output_copy);
 }
	  floodFill(threshold_output_copy,Point((int)(Prior.size.width)+xOffset,(int)(Prior.size.height)+yOffset),Scalar(100));
	  	 if (i==0){
  		  namedWindow("blobsCheck3");
  imshow("blobsCheck3",threshold_output_copy);
 }
	floodFill(threshold_output_copy,Point(0,0),Scalar(255));
	floodFill(threshold_output_copy,Point(0,0),Scalar(0));
		 if (i==0){
  		  namedWindow("blobsCheck4");
  imshow("blobsCheck4",threshold_output_copy);
 }
	floodFill(threshold_output_copy,Point(0,0),Scalar(255));
	floodFill(threshold_output_copy,Point(0,0),Scalar(0));
		 if (i==0){
  		  namedWindow("blobsCheck5");
  imshow("blobsCheck5",threshold_output_copy);
 }
	floodFill(threshold_output_copy,Point((int)(Prior.size.width)+xOffset,(int)(Prior.size.height)+yOffset),Scalar(255));*/
	//floodFill(threshold_output_copy,Point((int)(Prior.size.width),(int)(Prior.size.height)),Scalar(255));
		//floodFill(threshold_output_copy,Point((int)(Prior.size.width)+4,(int)(Prior.size.height)+4),Scalar(255));
/* src.at<Vec3b>((int)(Prior.size.height)+yOffset+y_lower,(int)(Prior.size.width)+xOffset+x_lower)[0]=200;
 src.at<Vec3b>((int)(Prior.size.height)+yOffset+y_lower,(int)(Prior.size.width)+xOffset+x_lower)[1]=0;
 src.at<Vec3b>((int)(Prior.size.height)+yOffset+y_lower,(int)(Prior.size.width)+xOffset+x_lower)[2]=0;*/
// if (i==0){
  	//	  namedWindow("blobsCheck1");
 // imshow("blobsCheck1",threshold_output_copy);
// }
 //cout<<"finding contours"<<endl;
		findContours( threshold_output_copy, contours, hierarchy, CV_RETR_LIST,CV_CHAIN_APPROX_NONE, Point(x_lower, y_lower) );
	//contours_final[i]=contours[0];//Could be error prone
	/*int largestContourIndex=0;
	int largestContourSize=0;
	if (contours.size()>0){
	//cout<<"choosing contours"<<endl;
	for (int j=0;j<contours.size();j++){
		if (contours[j].size()>largestContourSize && contours[j].size()<200){
			largestContourIndex=j;
			largestContourSize=contours[j].size();
		}
	}
		contours_final[i]=contours[largestContourIndex];
	Ellipses[i]=fitEllipse( Mat(contours_final[i]) );
	if (!largestContourSize==0){
		contours_final[i]=contours[largestContourIndex];
	Ellipses[i]=fitEllipse( Mat(contours_final[i]) );
	}
	else{
	Ellipses[i]=Priors[i];//Super dangerous
	Ellipses[i].size.height=0;
	Ellipses[i].size.width=0;
	}
  }*/

		/*Scoring metric 1, Dorions */ 
		// /*
	//	cout<<"about to do scoring"<<endl;
		double bestScore=1000;
		for (int j=0;j<contours.size();j++){
			if (contours[j].size()>200 || contours[j].size()<75){
				continue;
			}
			RotatedRect curEllipse=fitEllipse(Mat(contours[j]));
			if (curEllipse.size.height<20 || curEllipse.size.height>50 || curEllipse.size.width<20 || curEllipse.size.width>50){
				continue;
			}
			Mat mask=Mat();
			Mat countPixel=Mat();
			mask=Mat::zeros(threshold_output_copy.size(),CV_8UC1);
		vector<Point> current_contour=contours.at(j);
		//const Point* elementPoints[1]={&current_contour[0]};
		int numberOfPoints=(int)current_contour.size();
		//fillPoly(mask,elementPoints,&numberOfPoints,1,Scalar(1));
		countPixel=Mat(threshold_output_copy.size(),CV_8UC1);
	//			namedWindow("ellipse1");
		//imshow("ellipse1",mask);
		curEllipse.center.x=curEllipse.center.x-x_lower;
		curEllipse.center.y=curEllipse.center.y-y_lower;
		ellipse( mask, curEllipse, Scalar(255), -1, 8 );
		curEllipse.center.x=curEllipse.center.x+x_lower;
		curEllipse.center.y=curEllipse.center.y+y_lower;
		bitwise_and(threshold_output_copy_preserve,mask,countPixel);
		double curScore=countNonZero(countPixel)/(M_PI*curEllipse.size.height*curEllipse.size.width);
		//double sizeScore=(M_PI*curEllipse.size.height*curEllipse.size.width)/(M_PI*Priors[i].size.height*Priors[i].size.width);
		//curScore=curScore*pow(sizeScore,10);
		double posScore=1/sqrt(pow(Priors[i].center.x-curEllipse.center.x,2)+pow(Priors[i].center.y-curEllipse.center.y,2));
	//	cout<<Priors[i].center.x<<"-"<<curEllipse.center.x<<","<<Priors[i].center.y<<"-"<<curEllipse.center.y<<":"<<posScore<<endl;
		if (posScore<0.05){
		curScore=curScore*pow(posScore,3);
		}
	//	 cout<<"curScore"<<curScore<<endl;
	//	namedWindow("countPixel");
		//imshow("countPixel",countPixel);
		mask.release();
		countPixel.release();
		threshold_output_copy.release();
		threshold_output_copy_preserve.release();
				//namedWindow("countPixelContour");
		//imshow("countPixelContour",countPixel);
		//waitKey(100);
		if (abs(curScore-1)<bestScore){
			bestScore=abs(curScore-1);
			contours_final[i]=contours[j];
	//		cout<<"contours size:"<<contours[j].size()<<endl;
			Ellipses[i]=curEllipse;
			scores[i]=100/(bestScore);
		}
		}
		if (bestScore==1000){
			Ellipses[i]=Priors[i];
			Ellipses[i].size.height=0;
			Ellipses[i].size.width=0;
			contours_final[i].push_back(Point(0,0));
			scores[i]=100/(bestScore);
		}

  } //*/
 // cout<<"about to draw"<<endl;
 for (int i=0;i<Priors.size();i++){
	// cout<<"scores:"<<scores[i]<<endl;
	 if (!contours_final[i].empty()){
	  		drawContours( src, contours_final, i, Scalar(200,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
	 }
			if ((scores[i])<255){
				ellipse( src, Ellipses[i], Scalar(0,0,(int)(scores[i])), 1, 8 );
			}
			else{
			ellipse( src, Ellipses[i], Scalar(0,0,255), 1, 8 );
			}
  
 }
 //cout<<"done drawing"<<endl;
  namedWindow("blobs");
  imshow("blobs",src);
  waitKey(1);
  vector<scoredRotatedRect> scoredEllipses(scoredPriors.size());
  for (int i=0;i<scoredPriors.size();i++){
	  scoredEllipses[i].ellipse=Ellipses[i];
	  scoredEllipses[i].contour=contours_final[i];
	  scoredEllipses[i].score=scores[i];
  }
	return scoredEllipses;

}

scoredRotatedRect score_contour(Mat threshold_output_copy_preserve,vector<Point> current_contour,RotatedRect curEllipse,RotatedRect Prior){
				Mat mask=Mat();
			Mat countPixel=Mat();
			scoredRotatedRect curScored;
			mask=Mat::zeros(threshold_output_copy_preserve.size(),CV_8UC1);
		int numberOfPoints=(int)current_contour.size();
			    int x_lower=(int)(curEllipse.center.x-curEllipse.size.width)>0?(int)(curEllipse.center.x-curEllipse.size.width):0;
	int y_lower=(int)(curEllipse.center.y-curEllipse.size.height)>0?(int)(curEllipse.center.y-curEllipse.size.height):0;
		//fillPoly(mask,elementPoints,&numberOfPoints,1,Scalar(1));
		countPixel=Mat(threshold_output_copy_preserve.size(),CV_8UC1);
	//			namedWindow("ellipse1");
		//imshow("ellipse1",mask);
		curEllipse.center.x=curEllipse.center.x-x_lower;
		curEllipse.center.y=curEllipse.center.y-y_lower;
		ellipse( mask, curEllipse, Scalar(255), -1, 8 );
		curEllipse.center.x=curEllipse.center.x+x_lower;
		curEllipse.center.y=curEllipse.center.y+y_lower;
		bitwise_and(threshold_output_copy_preserve,mask,countPixel);
		//double curScore=countNonZero(countPixel)/(M_PI*curEllipse.size.height*curEllipse.size.width);
		double curScore=((double)countNonZero(countPixel))/((double)(countNonZero(mask)));
		//double sizeScore=(M_PI*curEllipse.size.height*curEllipse.size.width)/(M_PI*Priors[i].size.height*Priors[i].size.width);
		//curScore=curScore*pow(sizeScore,10);
		double posScore=1/sqrt(pow(Prior.center.x-curEllipse.center.x,2)+pow(Prior.center.y-curEllipse.center.y,2));
	//	cout<<Priors[i].center.x<<"-"<<curEllipse.center.x<<","<<Priors[i].center.y<<"-"<<curEllipse.center.y<<":"<<posScore<<endl;
		if (posScore<0.05){
		curScore=curScore*pow(posScore,3);
		}
		/*cout<<"------------------"<<endl;
		curEllipse.center.x=curEllipse.center.x-x_lower;
		curEllipse.center.y=curEllipse.center.y-y_lower;
		ellipse(countPixel,curEllipse,Scalar(100),1,8);
		curEllipse.center.x=curEllipse.center.x+x_lower;
		curEllipse.center.y=curEllipse.center.y+y_lower;
		cout<<countNonZero(countPixel)<<endl;
		cout<<(M_PI*curEllipse.size.height*curEllipse.size.width)<<endl;
		cout<<curScore<<endl;
		cout<<"------------------"<<endl;
	//	 cout<<"curScore"<<curScore<<endl;
		namedWindow("countPixel");
		imshow("countPixel",countPixel);*/
		mask.release();
		countPixel.release();
		curScored.ellipse=curEllipse;
		curScored.contour=current_contour;
		//curScored.score=curScore-1;
		curScore=floorf(curScore*1000 + 0.5)/1000;//Round to nearest
		curScored.score=curScore;
		return curScored;
}

vector<scoredRotatedRect> single_blob_handle(vector<scoredRotatedRect> Priors,Mat src,Mat src_gray,int threshold_level_pupil,int threshold_level_triangle,bool isFlood){
	 vector<vector<Point> > contours;
  	vector<vector<Point> > contours_final(Priors.size());
		vector<scoredRotatedRect> Ellipses(Priors.size());
				vector<double> scores(Priors.size());
				LARGE_INTEGER Frequency;
				LARGE_INTEGER startTime;
				LARGE_INTEGER endTime;
QueryPerformanceFrequency(&Frequency);
  for (int i=0;i<Priors.size();i++){
	    RotatedRect PriorEllipse=Priors[i].ellipse;
	  Mat threshold_output_copy=Mat();
	  Mat threshold_output_copy_preserve=Mat();
	    int x_lower=(int)(PriorEllipse.center.x-PriorEllipse.size.width)>0?(int)(PriorEllipse.center.x-PriorEllipse.size.width):0;
	int y_lower=(int)(PriorEllipse.center.y-PriorEllipse.size.height)>0?(int)(PriorEllipse.center.y-PriorEllipse.size.height):0;
	int x=(int)PriorEllipse.center.x;
	int y=(int)PriorEllipse.center.y;
		 Mat src_subPicture=Mat();
	  if (i<2){
	Mat output_pupil=Mat();
	 Mat output_pupil_copy=Mat();
		 int subPicture_width=x_lower+(int)2*PriorEllipse.size.width<src_gray.cols?(int)2*PriorEllipse.size.width:src_gray.cols-x_lower;
	 int subPicture_height=y_lower+(int)2*PriorEllipse.size.height<src_gray.rows?(int)2*PriorEllipse.size.height:src_gray.rows-y_lower;
	 src_gray(cv::Rect(x_lower,y_lower,subPicture_width,subPicture_height)).copyTo(output_pupil);
	 output_pupil.copyTo(src_subPicture);
	 threshold(output_pupil,threshold_output_copy,threshold_level_pupil,255,THRESH_BINARY_INV);
	  }
	  else{
	Mat output_triangle=Mat();
	 Mat output_triangle_copy=Mat();
		 int subPicture_width=x_lower+(int)2*PriorEllipse.size.width<src_gray.cols?(int)2*PriorEllipse.size.width:src_gray.cols-x_lower;
	 int subPicture_height=y_lower+(int)2*PriorEllipse.size.height<src_gray.rows?(int)2*PriorEllipse.size.height:src_gray.rows-y_lower;
	 src_gray(cv::Rect(x_lower,y_lower,subPicture_width,subPicture_height)).copyTo(output_triangle);
	 output_triangle.copyTo(src_subPicture);
	 threshold(output_triangle,threshold_output_copy,threshold_level_triangle,255,THRESH_BINARY_INV);
	  }
	  if (threshold_output_copy.size().height==0 && threshold_output_copy.size().width==0){
		  Ellipses[i]=Priors[i];
		  Ellipses[i].ellipse.size.height=0;
		  Ellipses[i].ellipse.size.width=0;
		  contours_final[i].push_back(Point(0,0));
		  continue;
	  }
	  if (isFlood){

		  	   QueryPerformanceCounter(&startTime);
	  Ellipses[i]=locate_single_blob_flood(Priors[i],threshold_output_copy);
	  QueryPerformanceCounter(&endTime);
	  double freqD=(double)Frequency.QuadPart/1000;
	   double delta=(double)((endTime.QuadPart-startTime.QuadPart)/freqD);
	   Ellipses[i].timeTaken=floorf(delta*100+0.5)/100;
 }
	  else{
		  	   QueryPerformanceCounter(&startTime);
		  Ellipses[i]=locate_single_blob(Priors[i],threshold_output_copy);
		  	  QueryPerformanceCounter(&endTime);
	  double freqD=(double)Frequency.QuadPart/1000;
	   double delta=(double)((endTime.QuadPart-startTime.QuadPart)/freqD);
	  Ellipses[i].timeTaken=floorf(delta*100+0.5)/100;
		//  Ellipses[i]=locate_single_blob_edge(Priors[i],src_subPicture);
	  }
  }
	return Ellipses;

}

scoredRotatedRect locate_single_blob(scoredRotatedRect scoredPrior,Mat threshold_output_copy){
		int xOffset=0;
	int yOffset=0;
	int counter=0;
	RotatedRect Prior=scoredPrior.ellipse;
	Mat threshold_output_copy_preserve;
	  	vector<vector<Point> > contours;
		vector<Point> contourFinal;
		RotatedRect Ellipse;
	vector<Vec4i> hierarchy;
	  int x_lower=(int)(Prior.center.x-Prior.size.width)>0?(int)(Prior.center.x-Prior.size.width):0;
	int y_lower=(int)(Prior.center.y-Prior.size.height)>0?(int)(Prior.center.y-Prior.size.height):0;
	int x=(int)Prior.center.x;
	int y=(int)Prior.center.y;
	threshold_output_copy.copyTo(threshold_output_copy_preserve);
		findContours( threshold_output_copy, contours, hierarchy, CV_RETR_LIST,CV_CHAIN_APPROX_NONE, Point(x_lower, y_lower) );
		scoredRotatedRect bestRect;
		bestRect.score=0;
		for (int j=0;j<contours.size();j++){
			if (contours[j].size()>200 || contours[j].size()<75){
				continue;
			}
			RotatedRect curEllipse=fitEllipse(Mat(contours[j]));
			if (curEllipse.size.height<20 || curEllipse.size.height>50 || curEllipse.size.width<20 || curEllipse.size.width>50){
				continue;
			}
			vector<Point> current_contour=contours.at(j);
		scoredRotatedRect curRotatedRect=score_contour(threshold_output_copy_preserve,current_contour,curEllipse,Prior);
		double curScore=curRotatedRect.score;
		threshold_output_copy.release();
		threshold_output_copy_preserve.release();
		if (abs(curScore)>bestRect.score){
			bestRect=curRotatedRect;
		}
		}
		if (bestRect.score==0){
			bestRect.ellipse=Prior;
			bestRect.ellipse.size.height=0;
			bestRect.ellipse.size.width=0;
			bestRect.contour.push_back(Point(0,0));
		}
		return bestRect;
}

scoredRotatedRect locate_single_blob_flood(scoredRotatedRect scoredPrior,Mat threshold_output_copy){
		int xOffset=0;
	int yOffset=0;
	int counter=0;
	RotatedRect Prior=scoredPrior.ellipse;
	scoredRotatedRect bestRect;
	Mat threshold_output_copy_preserve;
	  	vector<vector<Point> > contours;
		vector<Point> contourFinal;
		RotatedRect Ellipse;
	vector<Vec4i> hierarchy;
	  int x_lower=(int)(Prior.center.x-Prior.size.width)>0?(int)(Prior.center.x-Prior.size.width):0;
	int y_lower=(int)(Prior.center.y-Prior.size.height)>0?(int)(Prior.center.y-Prior.size.height):0;
	int x=(int)Prior.center.x;
	int y=(int)Prior.center.y;
	threshold_output_copy.copyTo(threshold_output_copy_preserve);
	floodFill(threshold_output_copy,Point((int)(Prior.size.width),(int)(Prior.size.height)),Scalar(100));
	floodFill(threshold_output_copy,Point(0,0),Scalar(0));
	floodFill(threshold_output_copy,Point(0,0),Scalar(255));
	floodFill(threshold_output_copy,Point(0,0),Scalar(0));
	floodFill(threshold_output_copy,Point((int)(Prior.size.width),(int)(Prior.size.height)),Scalar(255));
	findContours( threshold_output_copy, contours, hierarchy, CV_RETR_LIST,CV_CHAIN_APPROX_NONE, Point(x_lower, y_lower) );

		double bestScore=0;
		for (int j=0;j<contours.size();j++){
			if (contours[j].size()>200 || contours[j].size()<75){
				continue;
			}
			scoredRotatedRect curRotatedRect=score_contour(threshold_output_copy_preserve,contours[j],fitEllipse(Mat(contours[j])),Prior);
			/*if (contours[j].size()>bestScore){
			bestRect.contour=contours[j];
			//contourFinal=contours[j];
			RotatedRect curEllipse=fitEllipse(Mat(contours[j]));
			bestRect.ellipse=curEllipse;
			bestRect.score=contours[j].size();
			//Ellipse=curEllipse;
			bestScore=contours[j].size();
			}*/
			if (curRotatedRect.score>bestScore){
				bestScore=curRotatedRect.score;
				bestRect=curRotatedRect;
			}
		}
		if (bestScore==0){
			Ellipse=Prior;
			Ellipse.size.height=0;
			Ellipse.size.width=0;
			bestRect.contour.push_back(Point(0,0));
			bestRect.ellipse=Ellipse;
			bestRect.score=0;
			//contourFinal.push_back(Point(0,0));
		}
		return bestRect;
}

vector<RotatedRect> locate_blob_flood(vector<scoredRotatedRect> scoredPriors,Mat src,Mat src_gray,int threshold_level_pupil,int threshold_level_triangle){
	Mat threshold_output_pupil;
	Mat threshold_output_triangle;
  vector<vector<Point> > contours;
  vector<RotatedRect> Priors(scoredPriors.size());
  for (int i=0;i<scoredPriors.size();i++){
	  Priors[i]=scoredPriors[i].ellipse;
  }
  // cvtColor( src, src_gray, CV_BGR2GRAY );
 // cout<<"in locate blobs"<<endl;
  threshold( src_gray, threshold_output_pupil, threshold_level_pupil, 255, THRESH_BINARY_INV );
   threshold( src_gray, threshold_output_triangle, threshold_level_triangle, 255, THRESH_BINARY_INV );
  	vector<vector<Point> > contours_final(Priors.size());
		vector<RotatedRect> Ellipses(Priors.size());
  for (int i=0;i<Priors.size();i++){
	  	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	    RotatedRect Prior=Priors[i];
  int x_lower=(int)(Prior.center.x-Prior.size.width);
	int y_lower=(int)(Prior.center.y-Prior.size.height);
	int x=(int)Prior.center.x;
	int y=(int)Prior.center.y;
	  Mat threshold_output_copy;
	//  cout<<"composing images"<<endl;
		  if (i<2){
	 // threshold_output_pupil(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(threshold_output_copy);
	 //threshold_output_pupil(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(threshold_output_copy_preserve);
	 Mat output_pupil;

	 src_gray(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(output_pupil);
	 threshold(output_pupil,threshold_output_copy,threshold_level_pupil,255,THRESH_BINARY_INV);
	// threshold_output_copy.copyTo(threshold_output_copy_preserve);
	  }
	  else{
     // threshold_output_triangle(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(threshold_output_copy);
	//        threshold_output_triangle(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(threshold_output_copy_preserve);
	 	 Mat output_triangle;
	 src_gray(cv::Rect(x_lower,y_lower,(int)(2*Prior.size.width),(int)(2*Prior.size.height))).copyTo(output_triangle);
	 threshold(output_triangle,threshold_output_copy,threshold_level_triangle,255,THRESH_BINARY_INV);
	// threshold_output_copy.copyTo(threshold_output_copy_preserve);
	  }
//	  cout<<"done composing images"<<endl;
//	  cout<<threshold_output_copy.size()<<endl;
	  if (threshold_output_copy.size().height==0 && threshold_output_copy.size().width==0){
		  Ellipses[i]=Priors[i];
		  Ellipses[i].size.height=0;
		  Ellipses[i].size.width=0;
		  contours_final[i].push_back(Point(0,0));
		  continue;
	  }
	//if (i==0){
	//cout<<"+++++"<<endl;
	//cout<<(int)threshold_output_copy.at<uchar>(Prior.size.width,Prior.size.height)<<endl;
	//cout<<Prior.size.width<<","<<Prior.size.height<<endl;
	//cout<<x_lower<<","<<y_lower<<endl;
	//}
	int xOffset=0;
	int yOffset=0;
	int counter=0;
	while ((int)threshold_output_copy.at<uchar>((int)(Prior.size.height)+yOffset,(int)(Prior.size.width)+xOffset)==0){
//		cout<<":"<<(int)threshold_output_copy.at<uchar>((int)(Prior.size.height)+yOffset,(int)(Prior.size.width)+xOffset)<<":"<<endl;
		xOffset=xOffset+1;
		if (xOffset>(0.5*((int)Prior.size.height))){
			xOffset=-0.5*((int)Prior.size.height);
			yOffset=yOffset+1;
			if (yOffset>(0.5*((int)Prior.size.width))){
				//throw error...
				yOffset=numeric_limits<int>::max();
				xOffset=numeric_limits<int>::max();
				break;
			}
		}
		//xOffset=rand()%(0.5*((int)Prior.size.height)-1)-((int)Prior.size.height);
		//yOffset=rand()%(0.5*((int)Prior.size.width)-1)-((int)Prior.size.width);
		//counter=counter+1;
		 //threshold_output_copy.at<float>((int)(Prior.size.width)+xOffset+x_lower,(int)(Prior.size.height)+yOffset+y_lower)=200;
	//	if (i==0 || counter>50){
		//string windowName="blobsCheck"+to_string(i);
 		//  namedWindow(windowName);
  //imshow(windowName,threshold_output_copy);
  waitKey(1);//}
	}
		/*while ((int)threshold_output_copy.at<uchar>((int)(Prior.size.height)+yOffset,(int)(Prior.size.width)+xOffset)==0){
		cout<<":"<<(int)threshold_output_copy.at<uchar>((int)(Prior.size.height)+yOffset,(int)(Prior.size.width)+xOffset)<<":"<<endl;
		xOffset=rand()%10;
		yOffset=rand()%10;
		counter=counter+1;
		 //threshold_output_copy.at<float>((int)(Prior.size.width)+xOffset+x_lower,(int)(Prior.size.height)+yOffset+y_lower)=200;
	//	if (i==0 || counter>50){
 		  namedWindow("blobsCheck");
  imshow("blobsCheck",threshold_output_copy);
  waitKey(1);//}
	}*/
//	cout<<yOffset<<endl;
//	cout<<xOffset<<endl;
	if (!(yOffset==numeric_limits<int>::max() && xOffset==numeric_limits<int>::max())){
//	cout<<"in locate blobs"<<endl;
//	cout<<":"<<(int)threshold_output_copy.at<uchar>((int)(Prior.size.height)+yOffset,(int)(Prior.size.width)+xOffset)<<":"<<endl;
//	cout<<"in locate blobs"<<endl;
	  //floodFill(threshold_output_copy,Point((int)(Prior.size.width),(int)(Prior.size.height)),Scalar(100));
	  //floodFill(threshold_output_copy,Point((int)(Prior.size.width)+4,(int)(Prior.size.height)+4),Scalar(100));
	// if (i==0 || counter>50){
  		//  namedWindow("blobsCheck2");
 // imshow("blobsCheck2",threshold_output_copy);
 //}
	  floodFill(threshold_output_copy,Point((int)(Prior.size.width)+xOffset,(int)(Prior.size.height)+yOffset),Scalar(100));
	//  	 if (i==0 || counter>50){
  	//	  namedWindow("blobsCheck3");
 // imshow("blobsCheck3",threshold_output_copy);
 //}
	floodFill(threshold_output_copy,Point(0,0),Scalar(255));
	floodFill(threshold_output_copy,Point(0,0),Scalar(0));
	//	 if (i==0 || counter>50){
  	//	  namedWindow("blobsCheck4");
 // imshow("blobsCheck4",threshold_output_copy);
 //}
	floodFill(threshold_output_copy,Point(0,0),Scalar(255));
	floodFill(threshold_output_copy,Point(0,0),Scalar(0));
	//	 if (i==0 || counter>50){
  	//	  namedWindow("blobsCheck5");
  //imshow("blobsCheck5",threshold_output_copy);
 //}
	floodFill(threshold_output_copy,Point((int)(Prior.size.width)+xOffset,(int)(Prior.size.height)+yOffset),Scalar(255));
	//floodFill(threshold_output_copy,Point((int)(Prior.size.width),(int)(Prior.size.height)),Scalar(255));
		//floodFill(threshold_output_copy,Point((int)(Prior.size.width)+4,(int)(Prior.size.height)+4),Scalar(255));
	//cout<<"about to color point"<<endl;
 src.at<Vec3b>((int)(Prior.size.height)+yOffset+y_lower,(int)(Prior.size.width)+xOffset+x_lower)[0]=200;
 src.at<Vec3b>((int)(Prior.size.height)+yOffset+y_lower,(int)(Prior.size.width)+xOffset+x_lower)[1]=0;
 src.at<Vec3b>((int)(Prior.size.height)+yOffset+y_lower,(int)(Prior.size.width)+xOffset+x_lower)[2]=0;
// cout<<"done coloring"<<endl;
 //if (i==0 || counter>50){
  	//	  namedWindow("blobsCheck1");
  imshow("blobsCheck1",threshold_output_copy);
 //}
  waitKey(1);
 //cout<<"finding contours"<<endl;
		findContours( threshold_output_copy, contours, hierarchy, CV_RETR_LIST,CV_CHAIN_APPROX_NONE, Point(x_lower, y_lower) );
	//contours_final[i]=contours[0];//Could be error prone
	int largestContourIndex=0;
	int largestContourSize=0;
	if (contours.size()>0){
	//cout<<"choosing contours"<<endl;
	for (int j=0;j<contours.size();j++){
		if (contours[j].size()>largestContourSize){
			largestContourIndex=j;
			largestContourSize=contours[j].size();
		}
	}
	contours_final[i]=contours[largestContourIndex];
	Ellipses[i]=fitEllipse( Mat(contours_final[i]) );
	if (!largestContourSize==0){
		contours_final[i]=contours[largestContourIndex];
	Ellipses[i]=fitEllipse( Mat(contours_final[i]) );
	}
	else{
	Ellipses[i]=Priors[i];//Super dangerous
	}
  }
  }
		else{
		Ellipses[i]=Priors[i];
		Ellipses[i].size.height=0;
		Ellipses[i].size.width=0;
		contours_final[i].push_back(Point(0,0));
	}
	}
//  cout<<"about to draw"<<endl;
 for (int i=0;i<Priors.size();i++){
	 if (!contours_final[i].empty()){
	  		drawContours( src, contours_final, i, Scalar(200,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
	 }
			ellipse( src, Ellipses[i], Scalar(0,0,200), 1, 8 );
  
 }
// cout<<"done drawing"<<endl;
  namedWindow("blobs");
  imshow("blobs",src);
  waitKey(1);
  vector<scoredRotatedRect> scoredEllipses(scoredPriors.size());
  for (int i=0;i<scoredEllipses.size();i++){
	  scoredEllipses[i].ellipse=Ellipses[i];
	  scoredEllipses[i].contour=contours_final[i];
	  scoredEllipses[i].score=contours_final[i].size();
  }
	return Ellipses;

}

scoredRotatedRect fit_pupil(Mat src,int pupil_threshold,bool displayImage,bool printTiming,Mat *srcImage,vector<int> Offsets )
{
  Mat threshold_output;
  Mat src_gray;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  DWORD startTime;
  scoredRotatedRect bestRect;
  if (printTiming)
  {
  startTime=GetTickCount();
  }
  /// Detect edges using Threshold
  cvtColor( src, src_gray, CV_BGR2GRAY );
  threshold( src_gray, threshold_output, pupil_threshold, 255, THRESH_BINARY );
  //cout<<"done thresholding..."<<endl;
 //threshold( src, threshold_output, pupil_threshold, 255, THRESH_BINARY );
 
  //threshold( src, threshold_output, pupil_threshold, 255, THRESH_BINARY );

  // optional blurring... adds extra time
  //Mat threshold_output_occlude;
  //Mat morphElement=getStructuringElement(CV_SHAPE_ELLIPSE,Size(21,21),Point(10,10));
  //morphologyEx(threshold_output,threshold_output_occlude,MORPH_OPEN,morphElement);

  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  vector <double> ellipseErrors(contours.size());
  vector <double> contourAreas(contours.size());
  vector <double> pixelArea(contours.size());
  vector <double> ellipseAreas(contours.size());
  vector<RotatedRect> minEllipse( contours.size() );
  vector<vector<Point>> pupilContour(1);
  RotatedRect pupil;
  //cout<<"pupil:"<<pupil.size.width<<endl;
  double pupilError=10;
	for( int i = 0; i < contours.size(); i++ )
	{
		if( contours[i].size() > 25   )
			{minEllipse[i] = fitEllipse( Mat(contours[i]) );
			ellipseAreas[i]=M_PI*minEllipse[i].size.height*minEllipse[i].size.width;
			//contourAreas[i]=contourArea(Mat(contours[i]));
			//ellipseErrors[i]=abs(contourAreas[i]-ellipseAreas[i])/ellipseAreas[i];
			ellipseErrors[i]=abs(minEllipse[i].size.height-minEllipse[i].size.width)/min(minEllipse[i].size.width,minEllipse[i].size.height );
			int contour_Least_x=src.cols;
		int contour_Least_y=src.rows;
		int contour_Max_x=0;
		int contour_Max_y=0;
		vector<Point> current_contour=contours[i];
		for (int j=0;j<current_contour.size();j++){
			Point curPoint=current_contour[j]; //This is dangerous, it finds the point within the minimally containing rectangle, not within the contour
			contour_Least_x=curPoint.x<contour_Least_x? curPoint.x:contour_Least_x;
			contour_Max_x=curPoint.x>contour_Max_x?curPoint.x:contour_Max_x;
			contour_Least_y=curPoint.y<contour_Least_y?curPoint.y:contour_Least_y;
			contour_Max_y=curPoint.y>contour_Max_y? curPoint.y:contour_Max_y;
		}
		int contour_height=contour_Max_y-contour_Least_y;
		int contour_width=contour_Max_x-contour_Least_x;
		Mat current_contour_subImage=threshold_output(cv::Rect(contour_Least_x,contour_Least_y,contour_width,contour_height));
		int pixelCount=countNonZero(current_contour_subImage);
		ellipseErrors[i]=ellipseErrors[i]*((ellipseAreas[i])/pixelCount);
			//cout<<"ellipseErrors:"<<ellipseErrors[i]<<endl;
			if (ellipseErrors[i]<pupilError){
				pupilError=ellipseErrors[i];
				pupil=minEllipse[i];
				pupilContour[0]=contours[i];
			}
		
		}
	}
  /// Draw contours + rotated rects + ellipses
//	cout<<"pupil:"<<pupil.size.width<<endl;
	//cout<<contours.size<<endl;
  if (displayImage && pupil.size.width>0 && pupil.size.height>0 && contours.size()>0){//if we are displaying image and not blinking
       Scalar color = Scalar( 0, 0,255 );
	   Scalar color1 = Scalar(0,255,0);
       ellipse( src, pupil, color, 1, 8 );
	          drawContours( src, pupilContour, 0, color1, 1, 8, vector<Vec4i>(), 0, Point() );
  /// Show in a window
  Mat threshold_display;
  //cout<<"Offsets:"<<Offsets[0]<<","<<Offsets[1]<<endl;
  //cout<<"Offsets:"<<src.cols<<","<<src.rows<<endl;
  //cout<<"Adjusted Offsets:"<<Offsets[0]+src.cols<<","<<Offsets[1]+src.rows<<endl;
  Mat srcImageSubMatrix=srcImage->colRange(Offsets[0],Offsets[0]+src.cols).rowRange(Offsets[1],Offsets[1]+src.rows);
  src.copyTo(srcImageSubMatrix);
  threshold( src_gray, threshold_display, pupil_threshold, 255, THRESH_BINARY );
  //namedWindow( "ellipses", CV_WINDOW_AUTOSIZE );
  //imshow( "ellipses", *srcImage );
  //namedWindow("Threshold Contours",CV_WINDOW_AUTOSIZE);
  //imshow("Threshold Contours",threshold_output);
 // namedWindow("Threshold",CV_WINDOW_AUTOSIZE);
 // imshow("Threshold",threshold_display);
 // waitKey(0);
  }
  if (printTiming)
  {DWORD timeTaken=(GetTickCount()-startTime);
			  cout<<timeTaken<<endl;
  }
  //cout<<"done finding pupil"<<endl;
  bestRect.ellipse=pupil;
  bestRect.contour=pupilContour[0];
  bestRect.score=pupilError;
  return bestRect;
}


vector<int> locate_subPictures(Mat src,Mat* leftEye,Mat* rightEye,Mat* triangle,vector<scoredRotatedRect*> scoredTrianglePoints, int thresh_level_triangle,bool displayImages)//Assumes triangle has grayscale value > 250
{
//	cout<<"locating subPictures"<<endl;
	Mat src_gray;
	Mat threshold_output;
	Mat threshold_output_preserve;
	Mat threshold_output_upper;
	Mat threshold_output_lower;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<RotatedRect*> trianglePoints;
   trianglePoints.push_back(new RotatedRect());
   trianglePoints.push_back(new RotatedRect());
   trianglePoints.push_back(new RotatedRect());
	//threshold(src_gray,threshold_output_upper,100,255,THRESH_BINARY);
	//threshold(src_gray,threshold_output_lower,200,255,THRESH_BINARY_INV);
	//namedWindow("Threshold_upper",CV_WINDOW_AUTOSIZE);
	//imshow("Threshold_upper",threshold_output_upper);
	//namedWindow("Threshold_lower",CV_WINDOW_AUTOSIZE);
	//imshow("Threshold_lower",threshold_output_lower);
	//bitwise_and(threshold_output_upper,threshold_output_lower,threshold_output);
	//namedWindow("Threshold",CV_WINDOW_AUTOSIZE);
	//imshow("Threshold",threshold_output);
	//cvWaitKey(0);

	cvtColor( src, src_gray, CV_BGR2GRAY );
	//threshold( src_gray, threshold_output, 100, 255, THRESH_BINARY );
	threshold( src_gray, threshold_output, thresh_level_triangle, 255, THRESH_BINARY_INV );
	threshold_output.copyTo(threshold_output_preserve);
	DWORD startTime=GetTickCount();
	floodFill(threshold_output,Point(0,0),Scalar(0));
	floodFill(threshold_output,Point(0,src.rows-1),Scalar(0));
	floodFill(threshold_output,Point(src.cols-1,src.rows-1),Scalar(0));
	floodFill(threshold_output,Point(src.cols-1,0),Scalar(0));
//	namedWindow("Threshold",CV_WINDOW_AUTOSIZE);
//	imshow("Threshold",threshold_output);
//	cout<<"done showing thresholding"<<endl;
	findContours( threshold_output, contours, hierarchy, CV_RETR_LIST,CV_CHAIN_APPROX_NONE, Point(0, 0) );
	cout<<"locating 1: "<<GetTickCount()-startTime<<endl;
	int least_X=src.cols-1;
	int least_Y=src.rows-1;
	int greatest_X=0;
	int greatest_Y=0;
	int least_X_Buffer=0;
	int least_Y_Buffer=0;
	int greatest_X_Buffer=src.cols-1;
	int greatest_Y_Buffer=src.rows-1;
	int centroidCounter=0;
	vector<RotatedRect> PossibleTrianglePoints;
	vector<vector<Point>> PossibleTriangleContours;
	vector<vector<Point>> triangleContours(3);
	startTime=GetTickCount();
	for (int i=0;i<contours.size();i++)
	{
		if (contours[i].size()<200 && contours[i].size()>75)
		{
			RotatedRect curEllipse=fitEllipse(Mat(contours[i]));
			if (curEllipse.size.height<20 || curEllipse.size.height>50 || curEllipse.size.width<20 || curEllipse.size.width>50){
				continue;
			}
			PossibleTrianglePoints.push_back(curEllipse);
			PossibleTriangleContours.push_back(contours[i]);
		}
		else{
			//drawContours( src, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
			continue;
		}
		if (displayImages)
		{
	//	drawContours( src, contours, i, Scalar(255,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
		}
	}
	cout<<"locating 2: "<<GetTickCount()-startTime<<endl;
	//namedWindow("contours");
	//imshow("contours",src);
	//cvWaitKey(0);
	vector<double> triangleScores(3);
	triangleScores[0]=100;
	triangleScores[1]=100;
	triangleScores[2]=100;
	for (int i=0;i<PossibleTrianglePoints.size();i++){
		//double curScore=abs(PossibleTrianglePoints[i].size.height-PossibleTrianglePoints[i].size.width)/min(PossibleTrianglePoints[i].size.width,PossibleTrianglePoints[i].size.height );
		int contour_Least_x=src.cols-1;
		int contour_Least_y=src.rows-1;
		int contour_Max_x=0;
		int contour_Max_y=0;
		vector<Point> current_contour=PossibleTriangleContours[i];
		if (current_contour.size()<50 || current_contour.size()>250){
			continue;
		}
		for (int j=0;j<current_contour.size();j++){
			Point curPoint=current_contour[j]; //This is dangerous, it finds the point within the minimally containing rectangle, not within the contour
			contour_Least_x=curPoint.x<contour_Least_x? curPoint.x:contour_Least_x;
			contour_Max_x=curPoint.x>contour_Max_x?curPoint.x:contour_Max_x;
			contour_Least_y=curPoint.y<contour_Least_y?curPoint.y:contour_Least_y;
			contour_Max_y=curPoint.y>contour_Max_y? curPoint.y:contour_Max_y;
		}
		int contour_height=contour_Max_y-contour_Least_y;
		int contour_width=contour_Max_x-contour_Least_x;
		//cout<<contour_Least_x<<","<<contour_Least_y<<","<<contour_width<<","<<contour_height<<endl;
		Mat current_contour_subImage=threshold_output_preserve(cv::Rect(contour_Least_x,contour_Least_y,contour_width,contour_height));
		/*int pixelCount=countNonZero(current_contour_subImage);
		curScore=curScore*((M_PI*PossibleTrianglePoints[i].size.height*PossibleTrianglePoints[i].size.width)/pixelCount);*/
		Mat mask=Mat::zeros(current_contour_subImage.size(),CV_8UC1);
		Mat countPixel=Mat(current_contour_subImage.size(),CV_8UC1);
		RotatedRect curEllipse=fitEllipse(Mat(current_contour));
		curEllipse.center.x=curEllipse.center.x-contour_Least_x;
		curEllipse.center.y=curEllipse.center.y-contour_Least_y;
		ellipse( mask, curEllipse, Scalar(255), -1, 8 );
		curEllipse.center.x=curEllipse.center.x+contour_Least_x;
		curEllipse.center.y=curEllipse.center.y+contour_Least_y;
		bitwise_and(current_contour_subImage,mask,countPixel);
	//					namedWindow("ellipse1");
//		imshow("ellipse1",mask);
//		namedWindow("subImage");
//		imshow("subImage",current_contour_subImage);
		//waitKey(0);
		double curScore=abs((countNonZero(countPixel)/(M_PI*curEllipse.size.height*curEllipse.size.width))-1);


		if (curScore<triangleScores[0]){
			triangleScores[2]=triangleScores[1];
			triangleScores[1]=triangleScores[0];
			triangleScores[0]=curScore;
			triangleContours[2]=triangleContours[1];
			triangleContours[1]=triangleContours[0];
			triangleContours[0]=PossibleTriangleContours[i];
//			cout<<"assigning"<<endl;
			*trianglePoints[2]=*trianglePoints[1];
			*trianglePoints[1]=*trianglePoints[0];
			*trianglePoints[0]=PossibleTrianglePoints[i];
//			cout<<"done assigning"<<endl;

		}
		else if (curScore<triangleScores[1])
		{
			triangleScores[2]=triangleScores[1];
			triangleScores[1]=curScore;
			triangleContours[2]=triangleContours[1];
			triangleContours[1]=PossibleTriangleContours[i];
			*trianglePoints[2]=*trianglePoints[1];
			*trianglePoints[1]=PossibleTrianglePoints[i];
		}
		else if (curScore<triangleScores[2])
		{
			triangleScores[2]=curScore;
			triangleContours[2]=PossibleTriangleContours[i];
			*trianglePoints[2]=PossibleTrianglePoints[i];
		}
	}
	for (int i=0;i<3;i++){
		drawContours( src, triangleContours, i, Scalar(200,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
		ellipse( src, *trianglePoints[i], Scalar(0,0,200), 1, 8 );
	}
	for (int i=0;i<3;i++)
	{
		vector<Point> currentContour=triangleContours[i];
		for (int j=0;j<currentContour.size();j++)
		{
			Point curPoint=currentContour[j]; /*can optimize this... already computed previously*/
			least_X=curPoint.x<least_X? curPoint.x:least_X;
			greatest_X=curPoint.x>greatest_X?curPoint.x:greatest_X;
			least_Y=curPoint.y<least_Y?curPoint.y:least_Y;
			greatest_Y=curPoint.y>greatest_Y? curPoint.y:greatest_Y;
		}
	}
	cout<<"locating 3: "<<GetTickCount()-startTime<<endl;
	//namedWindow("contours");
	//imshow("contours",src);
	//cvWaitKey(0);
	int height=greatest_Y-least_Y;
	int width=greatest_X-least_X;
	if ((int)greatest_X+0.1*width<src.cols){greatest_X_Buffer=(int)(greatest_X+0.1*width);}
	if ((int)greatest_Y+0.1*height<src.rows){greatest_Y_Buffer=(int)(greatest_Y+0.1*height);} 
	if ((int)least_X-0.1*width>0){least_X_Buffer=(int)(least_X-0.1*width);}
	if ((int)least_Y-0.1*height>0){least_Y_Buffer=(int)(least_Y-0.1*height);}
//	cout<<"Least_Y_Buffer "<<least_Y_Buffer<<endl;
//	cout<<"greatest_Y_Buffer"<<greatest_Y_Buffer<<endl;
	int height_Buffer=greatest_Y_Buffer-least_Y_Buffer;
	int width_Buffer=greatest_X_Buffer-least_X_Buffer;
	least_X_Buffer=least_X_Buffer+width_Buffer<src.cols-1?least_X_Buffer:src.cols-1-width_Buffer;
	least_Y_Buffer=least_Y_Buffer+height_Buffer<src.rows-1?least_Y_Buffer:src.rows-1-height_Buffer;
/*	cout<<"Least_Y_Buffer"<<least_Y_Buffer<<endl;
	cout<<"greatest_Y_Buffer"<<greatest_Y_Buffer<<endl;
	cout<<"greatest_Y"<<greatest_Y<<endl;
	cout<<"greatest_X"<<greatest_X<<endl;
	cout<<"least_Y"<<least_Y<<endl;
	cout<<"least_X"<<least_X<<endl;
	cout<<"rows:"<<src.rows<<endl;
	cout<<"cols:"<<src.cols<<endl;
	cout<<"width_Buffer"<<width_Buffer<<endl;
	cout<<"height_Buffer"<<height_Buffer<<endl;*/
	//int Lower_Y_Eye=greatest_Y-(int)(0.66*(height));
//	Lower_Y_Eye=Lower_Y_Eye>src.rows-height-1? src.rows-height-1 : Lower_Y_Eye;
	int Lower_Y_Eye=greatest_Y-(int)(0.66*height)>=0 && greatest_Y-(int)(0.66*height)<=src.rows-1?greatest_Y-(int)(0.66*height):0;
	//int Upper_Y_Eye=greatest_Y+(int)(0.34*(height));
	//Upper_Y_Eye=Upper_Y_Eye>src.rows-1? src.rows-1 : Upper_Y_Eye;
	int Upper_Y_Eye=greatest_Y+(int)(0.34*height)<=src.rows-1 && greatest_Y+(int)(0.34*height)>=0?greatest_Y+(int)(0.34*height):src.rows-1;
	int Left_Left_X=least_X-(int)(0.75*(width))>=0 && least_X-(int)(0.75*(width))<= src.cols-1?least_X-(int)(0.75*(width)):0;
	//Left_Left_X=Left_Left_X>src.cols-width-1? src.cols-width-1 : Left_Left_X;
	//int Left_Right_X=least_X+(int)(0.25*(width));
	//Left_Right_X=Left_Right_X> src.cols-1? src.cols-1 : Left_Right_X;
	int Left_Right_X=least_X+(int)(0.25*width)<=src.cols-1 && least_X+(int)(0.25*width)>=0?least_X+(int)(0.25*width):src.cols-1;
	//int Right_Left_X=greatest_X-(int)(.25*(width));
	//Right_Left_X=Right_Left_X>src.cols-width-1? src.cols-width-1: Right_Left_X;
	int Right_Left_X=greatest_X-(int)(0.25*width)>=0 && greatest_X-(int)(0.25*width)<=src.cols-1?greatest_X-(int)(0.25*width):0;
	//int Right_Right_X=greatest_X+(int)(0.75*(width));
	int Right_Right_X=greatest_X+(int)(0.75*width)<=src.cols-1 && greatest_X+(int)(0.75*width)>=0?greatest_X+(int)(0.75*width):src.cols-1;
	//Right_Right_X=Right_Right_X>src.rows-1? src.rows-1 : Right_Right_X;
	vector<int> Offsets(6);
	Offsets[0]=Left_Left_X;
	Offsets[1]=Lower_Y_Eye;
	Offsets[2]=Right_Left_X;
	Offsets[3]=Lower_Y_Eye;
	Offsets[4]=least_X_Buffer;
	Offsets[5]=least_Y_Buffer;
	
	/*cout <<"rows"<<src.rows<<" cols"<<src.cols<<endl;
	cout<<"height:"<<height<<" width:"<<width<<endl;
	cout<<"Left_Left_X"<<Left_Left_X<<" ,Right_Left_X"<<Right_Left_X<<endl;
	cout<<"Lower_Y_Eye"<<Upper_Y_Eye<<endl;
	cout<<"triangleScore:"<<triangleScores[0]<<", "<<triangleScores[1]<<", "<<triangleScores[2]<<endl;
	cout<<PossibleTriangleContours.size()<<endl;*/
	*leftEye=Mat();
	*rightEye=Mat();
	*triangle=Mat(); //Release the memory of previous images
	//*leftEye=src(cv::Rect(Left_Left_X,Lower_Y_Eye,width,height));
	//*rightEye=src(cv::Rect(Right_Left_X,Lower_Y_Eye,width,height));
//	cout<<"Left_Left_X:"<<Left_Left_X<<",Left_Right_X:"<<Left_Right_X<<endl;
//	cout<<"Right_Left_X:"<<Right_Left_X<<",Right_Right_X:"<<Right_Right_X<<endl;
	*leftEye=src(cv::Rect(Left_Left_X,Lower_Y_Eye,Left_Right_X-Left_Left_X,Upper_Y_Eye-Lower_Y_Eye));
	*rightEye=src(cv::Rect(Right_Left_X,Lower_Y_Eye,Right_Right_X-Right_Left_X,Upper_Y_Eye-Lower_Y_Eye));
	//cout<<height<<endl;
	//cout<<greatest_Y_Buffer-least_Y_Buffer<<endl;
/*	cout<<"doing triangle now"<<endl;
	cout<<"Least_X_Buffer"<<least_X_Buffer<<endl;
	cout<<"Least_Y_Buffer"<<least_Y_Buffer<<endl;
	cout<<"height_Buffer"<<height_Buffer<<endl;
	cout<<"width_Buffer"<<width_Buffer<<endl;*/

	*triangle=src(cv::Rect(least_X_Buffer,least_Y_Buffer,width_Buffer,height_Buffer));
	//cout<<"done with triangle now"<<endl;
	if (displayImages){
	imshow("leftEye",*leftEye);
	imshow("rightEye",*rightEye);
	imshow("triangle",*triangle);
	//cvWaitKey(0);
	  }
	/*	cout<<"--------------"<<endl;
	cout<<trianglePoints[0]->size.height<<endl;
	cout<<trianglePoints[1]->size.height<<endl;
	cout<<trianglePoints[2]->size.height<<endl;
	cout<<"--------------"<<endl;
	cout<<"done locating"<<endl;*/
	for (int i=0;i<trianglePoints.size();i++){
		scoredTrianglePoints[i]->ellipse=*trianglePoints[i];
		scoredTrianglePoints[i]->score=triangleScores[i];
		scoredTrianglePoints[i]->contour=triangleContours[i];
	}
	return Offsets;
}










bool get_Cine_Handle(CString fName){
	if(hfile.Open(fName, CFile::modeRead | CFile::osRandomAccess | CFile::shareDenyWrite)){
		cout<<"worked"<<endl;
	}
	else{
		cout<<"didnt work"<<endl;
		return false;
	}
	int bytesRead = hfile.Read(&m_cineFileHead, sizeof(CINEFILEHEADER));
	if(m_cineFileHead.type != 0x4943 || bytesRead != sizeof(CINEFILEHEADER)){
		cout<<"error 1"<<endl;
		return false;
	}
	hfile.Seek(m_cineFileHead.offImageHeader, CFile::begin);
	bytesRead = hfile.Read(&m_bitmapHead, sizeof(BITMAPINFOHEADER));
	if(bytesRead != sizeof(BITMAPINFOHEADER)){
		cout<<"error 2"<<endl;
		return false;
	}	
	m_hFile.Open(fName, CFile::modeRead | CFile::osRandomAccess | CFile::shareDenyWrite);
	return true;
}

void get_Next_Frame(int m_nDisplayedImage,IplImage* m_pImage){
	DWORD offsetToSel = 8 * m_nDisplayedImage;//The image you want?
	m_hFile.Seek(m_cineFileHead.offImageOffsets + offsetToSel, CFile::begin);
	long long imagPos;
	m_hFile.Read(&imagPos, 8);
	imagPos += 8;
	m_hFile.Seek(imagPos, CFile::begin);
	m_hFile.Read(m_pImage->imageData, m_bitmapHead.biSizeImage);
}
