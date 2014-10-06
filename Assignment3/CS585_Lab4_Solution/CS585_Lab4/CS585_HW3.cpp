/**
	CS585_Lab4.cpp
	@author: Ajjen Joshi
	@version: 1.0 9/25/2014
 
	CS585 Image and Video Computing Fall 2014
	Lab 4
	--------------
	This program introduces the following concepts:
 a) Understanding and applying basic morphological operations on images
 b) Finding and labeling blobs in a binary image
	--------------
 */


#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <time.h>

using namespace cv;
using namespace std;

/**
	Function that detects blobs from a binary image
	@param binaryImg The source binary image (binary image contains pixels labeled 0 or 1 (not 255))
	@param blobs Vector to store all blobs, each of which is stored as a vector of 2D x-y coordinates
 */
void convertToBinary (Mat &src, Mat &dst);
void findBiggestContour (Mat &, vector<vector<Point>> &, vector<Vec4i> &, int & , int & );
int calculateEulerNumber(Mat &);
double getOrientation (vector<Point>);
double calculateCircularity (Mat & src, vector<Point> & contour);
void thinningIteration(Mat& im, int iter);
void thinning(Mat& im);
double* part1(Mat & img, bool);
void part3(Mat & src);
void part4(vector<Mat> & images);
void part5(vector<Mat> & images);
void correlation(Mat & src, Mat & templ, Mat & dest, Point & max, float & maxIntensity);

int main(int argc, char **argv)
{
    
    // load in all the relevent image files
    vector<Mat> tumorSlides;
    tumorSlides.push_back(imread("bcancer1.png"));
    tumorSlides.push_back(imread("bcancer2.png"));
    tumorSlides.push_back(imread("bcancer3.png"));
    tumorSlides.push_back(imread("bcancer4.png"));
    
    vector<Mat> tissueFolds;
    tissueFolds.push_back(imread("bcancer2.2.png"));
    tissueFolds.push_back(imread("bcancer3.1.png"));
    tissueFolds.push_back(imread("bcancer3.2.png"));
    tissueFolds.push_back(imread("bcancer3.3.png"));
    
    // prepare timing things
    clock_t t;
    
    // easily select between different parts
    int part = 4; // choose between 1 and 5, 1 and 2 are the same
    
    // run the actual parts
    if (part==1 || part==2){
        t = clock();
        part1(tumorSlides[0],  true);
        t = clock() - t;
        cout << endl << "TIMING: " << t << endl;
    }
    else if (part==3){
        t = clock();
        part3(tissueFolds[2]);
        t = clock() - t;
        cout << endl << "TIMING: " << t << endl;
    }
    else if (part==4){
        t = clock();
        part4(tumorSlides);
        t = clock() - t;
        cout << endl << "TIMING: " << t << endl;
    }
    else if (part==5){
        t = clock();
        part5(tumorSlides);
        t = clock() - t;
        cout << endl << "TIMING: " << t << endl;
    }
    
    //namedWindow("Original");
    //imshow("Original", tumorSlides[2]);	// note img_bw is inverted here, as original processing made the forground white
    
    waitKey();
    return 0;
}


double * part1(Mat & img, bool showWindow = true){
    Mat im_gray;
    Mat binary;
    
    cvtColor(img.clone(),im_gray,CV_RGB2GRAY);
    
    //// invert the image
    bitwise_not ( im_gray, im_gray );
    convertToBinary(im_gray.clone(), binary);
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    int maxsize = 0;
    int maxind = 0;
    
    findBiggestContour (binary, contours, hierarchy, maxsize, maxind);
    Mat contour_output = Mat::zeros( binary.size(), CV_8UC3 );
    
    // draw filled in largest object in black
    drawContours(contour_output, contours, maxind, 	Scalar::all(255), CV_FILLED, 8, hierarchy);
    
    // get information about the object:
    int area = maxsize;
    double orientation = getOrientation(contours[maxind]); // angle that the axis of least inertia makes with x-axis
    double circularity = calculateCircularity (contour_output, contours[maxind]) ; // Emin/Emax
    double perimeter = arcLength(contours[maxind],1); // perimeter of largest contour
    double euler = calculateEulerNumber(binary);// euler number, which in the way we are doing it now should always be 1 (one blob, no holes)
    
    
    if (showWindow){
        
        cout << endl;
        cout << "Area: " << area << " pixels" << endl;
        cout << "Orientation: " << orientation << " degrees" << endl;
        cout << "Circularity: " << circularity << " units" << endl;
        cout << "Length of Perimeter: " << perimeter << " pixels" << endl;
        cout << "Compactness: " << perimeter*perimeter/area << " units" << endl;
        cout << "Euler number: " << euler << endl;
        
        //// create windows
        namedWindow("Thresh");
        namedWindow("Contour");
        
        moveWindow("Thresh", 30,190);
        moveWindow("Contour", 760,190);
        bitwise_not(binary,binary);
        imshow("Thresh",binary);	// note img_bw is inverted here, as original processing made the forground white
        bitwise_not(contour_output,contour_output);
        imshow("Contour",contour_output);
    }
    
    double * result = new double[5];
    result[0] = area;
    result[1] = orientation;
    result[2] = circularity;
    result[3] = perimeter;
    result[4] = euler;
    return result;
}



void part3(Mat & src){
    
    Mat dst;
    Mat channel[3];
    split(src, channel);
    blur( channel[0], dst, Size(4,4) );
    
    double average = 0;
    for(int x = 0; x < src.rows; x ++ ) {
        for(int y = 0; y < src.cols; y ++ ) {
            average += src.at<Vec3b>(x,y)[0];
        }
    }
    
    average /= (src.rows * src.cols);
    
    threshold(dst, dst, average,  0, THRESH_TOZERO_INV);
    
    
    int erosion_size = 4;
    int dilation_size = 4;
    
    
    Mat element = getStructuringElement( MORPH_RECT,
                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                        Point( erosion_size, erosion_size ) );
    
    //perform erosions and dilations
    erode(dst, dst, element, Point(-1, -1), 2);
    dilate(dst, dst, element);
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    findContours( dst.clone(), contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    
    dst = Mat::zeros( src.size(), CV_8UC1 );
    
    for(int i = 0; i < contours.size(); i++) {
        RotatedRect rect = minAreaRect( Mat(contours[i] ));
        Size size = rect.size;
        
        if ((contourArea(contours[i]) > 10000) && (size.height > (3 * size.width) || size.width > (3 * size.height)))  {
            drawContours(dst, contours, i, 	Scalar::all(255), CV_FILLED, 8, hierarchy);
        }
    }
    
    
    Mat img = dst.clone();
    
    thinning(img);
    
    namedWindow("Hi");
    imshow("Hi", img);
    imwrite("abc.png", img);	// note img_bw is inverted here, as original processing made the forground white
    
}


void part4(vector<Mat> & images) {
    
    // read in the "lost" image
    Mat missingID = imread("bcancer4.png");
    namedWindow("X-ID");
    imshow("X-ID",missingID);
    
    vector<double *> results;
    for (int i = 0; i < images.size(); i++ ) {
        results.push_back(part1(images[i], false));
    }
    
    double * missStats = new double[5];
    missStats = part1(missingID,false);
    
    double difference = 10; // measurd in %, for each category. this defualt also serves as the threshold for not counting any matches,
    // meaning at most cumulative 50% across all categories can be different for it to count as a possible match
    double index = -1;
    
    // algorithm: for each caegory, excluding orientation, add the % change to total match
    
    // print out the parameters/ comparisons with missing ID
    for (int i = 0; i < results.size(); i++ ) {
        double tmpMatch = 0;
        cout << "matching index: " << i << ": " << endl;
        for (int j=0; j< 5; j++){
            
            double tmp = abs((results[i][j] - missStats[j])/missStats[j]);
            tmpMatch += tmp;
            cout << tmpMatch << "/" << tmp << " :: " ;
        }
        cout << endl;
        if (tmpMatch < difference){
            difference = tmpMatch;
            index = i;
        }
    }
    cout << endl << "Best matching index: " << index << ", difference added: " << difference << endl;
    
    // show the one the algorithm thinks matches:
    namedWindow("Best Match for unknown ID");
    imshow("Best Match for unknown ID", images[index]);
}


void part5(vector<Mat> & images){
    
    Mat missingID = imread("xID_03.png");
    namedWindow("X-ID");
    imshow("X-ID",missingID);
    
    Mat IDbounded = missingID.clone();
    
    // now run the tempalte maching for each of the images in the "database", the array of images brought in
    Point p;
    float maxIntensity = 0; // setting this to values above zero act as a threshold for preventing false positive matches if none actually exist
    float temp = 0;
    int index=-1;
    
    Mat d;
    for (int q=0; q<images.size();q++){
        temp = -1;
        if (images[q].cols >= IDbounded.cols && images[q].rows >= IDbounded.rows){
            
            Mat dest;
            p = Point(-1,-1);
            matchTemplate( images[q], IDbounded, dest, CV_TM_CCOEFF_NORMED);
            
            // get the highest intensity, the location in the result image that has the best match
            for (int x=0; x<dest.cols; x++){
                for (int y=0; y<dest.rows; y++){
                    float loc = dest.at<float>(y, x);
                    if (loc > temp) {
                        temp = loc;
                    }
                }
            }
            
        }
        // check if the best match for this image is better than any previous
        cout << "Best match for image index " << q << ": " << temp << endl;
        if (temp > maxIntensity){
            maxIntensity = temp;
            index = q;
        }
    }
    
    // show the resulting best matched image, or not
    if (index != -1){
        cout << "Best match: " << maxIntensity << ", index image: " << index << endl;
        namedWindow("Best Match");
        imshow("Best Match",images[index]);
        
    }
    else{
        cout << "No image matches the broken template" << endl;
        Mat z = Mat::zeros(100,100,CV_8UC1); // just show black, since there was no match at all
        namedWindow("Best Match");
        imshow("Best Match",images[index]);
    }
    
}


int calculateEulerNumber(Mat &src)
{
    int holes = 0;
    int bodies = 0;
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    findContours( src.clone(), contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    bodies = contours.size();
    
    findContours( src.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );
    
    holes = 0;
    for(int idx=0; idx<hierarchy.size(); ++idx)
    {
        if(hierarchy[idx][3] != -1){
            holes ++;
        }
    }
    
    //imshow("Original", src);
    
    return bodies - holes;
}

void convertToBinary (Mat &src, Mat &dst) {
    
    blur( src, src, Size(5,5) );
    int erosion_size = 1;
    int dilation_size = 1;
    Mat element = getStructuringElement( MORPH_RECT,
                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                        Point( erosion_size, erosion_size ) );
    
    ////perform erosions and dilations and  blur
    blur( src, src, Size(5,5) );
    erode(src, src, element);
    dilate( src, src, element );
    
    //// prior to invert
    threshold(src, dst, 40, 255, THRESH_BINARY);
    
}

void findBiggestContour (Mat & src, vector<vector<Point>> & contours, vector<Vec4i> & hierarchy, int & maxsize, int & maxind ){
    
    findContours(src.clone(), contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    for(unsigned int i = 0; i < contours.size(); i++ )
    {
        // Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
        double area = contourArea(contours[i]);
        if (area > maxsize) {
            maxsize = (int) area;
            maxind = i;
        }
    }
}


double getOrientation (vector<Point> contour) {
    Moments mo = moments(contour,false);
    double xc = mo.m10/mo.m00;
    double yc = mo.m01/mo.m00;
    double a = mo.m20/mo.m00 - xc*xc;
    double b = 2*(mo.m11/mo.m00 - xc*yc);
    double c = mo.m02/mo.m00 - yc*yc;
    return atan2(b,(a-c))/2 * (180 / 3.1415926); // output in degrees
}

double calculateCircularity (Mat & src, vector<Point> & contour) {
    
    double a = 0, b = 0, c = 0;
    
    Moments m = moments(contour, true);
    Point centroid(m.m10/m.m00, m.m01/m.m00);
    
    for(int x = 0; x < src.rows; x ++ ) {
        for(int y = 0; y < src.cols; y ++ ) {
            if (src.at<Vec3b>(x,y)[0] != 0){
                a += (x - centroid.y) * (x - centroid.y);
                b += (x - centroid.y) * (y - centroid.x);
                c += (y - centroid.x) * (y - centroid.x);
            }
        }
    }
    b *= 2;
    double first_term = ((a - c)/2) * ((a - c) / sqrt((a - c) * (a - c) + b * b));
    double double_term = (b/2) * (b / sqrt((a - c) * (a - c) + b * b));
    
    double emin = (a + c) /2 - first_term - double_term;
    double emax = (a + c) /2 + first_term + double_term;
    
    return emin / emax;
    
}


//http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/
void thinningIteration(Mat& im, int iter)
{
    Mat marker = Mat::zeros(im.size(), CV_8UC1);
    
    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);
            
            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
            (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
            (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
            (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
            
            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }
    
    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinning(Mat& im)
{
    im /= 255;
    
    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;
    
    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (countNonZero(diff) > 0);
    
    im *= 255;
}

// This function gets the brightest, best correlating position after template matching
void correlation(Mat & src, Mat & templ, Mat & dest, Point & max, float & maxIntensity) {
    // use the built in template matching function
    matchTemplate( src, templ, dest, CV_TM_CCOEFF_NORMED);
    
    
    // get the brightest pixel, representing the best matched location
    maxIntensity = 0;
    float temp = 0;
    
    for(int x = 0; x < dest.cols; x ++ ) {
        for(int y = 0; y < dest.rows; y ++ ) {
            temp = dest.at<float>(y, x);
            if (temp > maxIntensity && temp > 0.6) {
                maxIntensity = temp;
                max = Point(x, y);
            }
        }
    }
    
}



