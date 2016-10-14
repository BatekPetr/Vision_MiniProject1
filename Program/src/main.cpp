/*
 *  ex3_template.cpp
 *  Exercise 3 - The Frequency domain and filtering
 *
 *  Created by Stefan-Daniel Suvei on 19/09/16.
 *  Copyright 2016 SDU Robotics. All rights reserved.
 *
 */


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "image4_1.h"
#include "image4_2.h"
#include "helpFunctions.h"

/*
void run(const std::string& filename, bool highpass);
//void paddZeros(cv::Mat_<float>& img, cv::Mat_<float>& padded);
cv::Mat histEqualization(const cv::Mat &image);
//void dftshift(cv::Mat_<float>& magnitude);
//void notchRejectFilter(cv::Mat_<cv::Vec2f> filt, int cutOff, int order, int uk, int vk);
void bandRejectFilter(cv::Mat_<cv::Vec2f> filt, int cutOff, int width, int order);
//void plot(std::string name, cv::Mat& img, int width, int height);
//cv::Mat_<float> normalizeAndPlot(std::string name, cv::Mat_<float>& img, bool plot = false, int width = 0, int height = 0);
void calcAndPlotHist(std::string name, cv::Mat& img, cv::MatND& hist, int histSize, float range[]);
void plotMagSpecGraph(cv::Mat& img);
//cv::Mat_<float> plotMgSpec(std::string name, cv::Mat_<cv::Vec2f> img_dft, cv::Mat_<float> imgs[]);
void rotate(cv::Mat& src, double angle, cv::Mat& dst);
//void rot90(cv::Mat &matImage, int rotflag);
//void on_mouse( int e, int x, int y, int d, void *ptr);
*/


int main(int argc, char **argv) {
    // Test picture
    //run("../../images/lena.bmp", false);

    // MiniProject
    run4_1("../../images/Image4_1.png");
    //run4_2("../../images/Image4_2.png");

    return 0;
}

void run(const std::string& filename, bool highpass) {

    // A gray image
    cv::Mat_<uchar> img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    // Get dimensions of original picture
    int imgRows = img.rows;
    int imgCols = img.cols;

    // Find suitable uniform region in the image to identify noise
    //lower-left corner
    cv::Point lowLeft = cv::Point(900, 1750);
    cv::Point upRight = cv::Point(1520, 1550);
    int noiseRectWidth = upRight.x - lowLeft.x;
    int noiseRectHeight = lowLeft.y - upRight.y;
    // Draw rectangle into original image
    cv::rectangle(img, lowLeft + cv::Point(-1, 1), upRight + cv::Point(1, -1), cv::Scalar(0, 0, 0), 1, 4, 0);

    // Cut Noise square from the image
    cv::Mat tmp = img(cv::Rect(lowLeft,upRight));
    cv::Mat noise;
    tmp.copyTo(noise);
    // Plot the Noise Square
    //plot("Noise Square", noise, noiseRectWidth, noiseRectHeight);


    // Plot Input image resized to fit the screen
    plot("Input Image", img, 484, 1158);


/*
    // Analysis for Salt and Pepper and Gaussian noise

    //Calculate and plot histogram
    cv::MatND hist;
    /// Establish the number of bins
    int histSize = 128;
    /// Set the ranges
    float range[] = { 0.0, 256.0 } ;
    calcAndPlotHist("Histogram of Original Image", img, hist, histSize, range);

    // Compute Histogram of noise
    cv::MatND histNoise;
    calcAndPlotHist("Histogram of Noise Square", noise, histNoise, histSize, range);
    float percentageOfPepperNoise = histNoise.at<float>(0)/(noiseRectWidth*noiseRectHeight);
    float percentageOfSaltNoise = histNoise.at<float>(histSize-1)/(noiseRectWidth*noiseRectHeight);
    std::cout << "Percentage of Pepper noise: " << percentageOfPepperNoise * 100 << "\n";
    std::cout << "Percentage of Salt noise: " << percentageOfSaltNoise * 100 << "\n";
*/



/*
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;


    // Histogram equalization
    // Using of openCV equalizeHist
    cv::Mat_<uchar> imgEqualizedMine = histEqualization(img);
    plot("Equalized Input Image by custom function", imgEqualizedMine,762, 1158);
    cv::MatND histEqualizedMine;
    calcAndPlotHist("Histogram of Equalized Image by custom function", imgEqualizedMine, histEqualizedMine, histSize, range);

    cv::Mat imgEqualized;
    equalizeHist(img, imgEqualized);
    plot("Equalized Input image", imgEqualized, 762, 1158);
    cv::MatND histEqualized;
    calcAndPlotHist("Histogram of Equalized Image", imgEqualized, histEqualized, histSize, range);
*/

    /*
    //Space filtering
    //Apply median filter
    cv::Mat_<uchar> imgMedFilt;
    medianBlur(img,imgMedFilt,7);
    plot("Application of Median filter", imgMedFilt, 762, 1158);
    calcAndPlotHist("Histogram of Image filtered by Median filter", imgMedFilt, hist, histSize, range);
    */

    /*
    cv::Mat imgMedEqualized;
    equalizeHist(imgMedFilt, imgMedEqualized);
    plot("Equalized Median Filtered Image", imgMedEqualized, 762, 1158);
    cv::MatND histMedEqualized;
    calcAndPlotHist("Histogram of Equalized Median Filtered Image", imgMedEqualized, histMedEqualized, histSize, range);
    */

    /*
    //Apply Median filter one more time
    medianBlur(imgMedFilt,imgMedFilt,7);
    plot("Application of Median filter 2nd Time", imgMedFilt, 762, 1158);
    calcAndPlotHist("Histogram of Image filtered by Median filter for 2 times", imgMedFilt, hist, histSize, range);
*/


/*
    //Apply box filter = just averaging kernel points
    cv::Mat_<uchar> imgBoxFilt;
    boxFilter(img, imgBoxFilt, img.depth(), cv::Size(5,5), cv::Point(-1,-1));
    plot("Application of Box filter", imgBoxFilt, 762, 1158);
    calcAndPlotHist("Histogram of Image filtered by Box filter", imgBoxFilt, hist, histSize, range);
*/



}



cv::Mat histEqualization(const cv::Mat &image)
{
    // compute histogram with 256 bins
    cv::MatND hist_temp;
    /// Establish the number of bins
    int histSize = 256;
    /// Set the ranges
    float range[] = { 0.0, 255.0 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    /// Compute the histograms:
    cv::calcHist( &image, 1, 0, cv::Mat(), hist_temp, 1, &histSize, &histRange, uniform, accumulate );


    // create lookup table
    cv::Mat lookup(1, 256, CV_8UC1);
    double sum;
    // build lookup table
    for (int i=0; i<256; i++)
    {
        sum = 0;
        for (int j = 0; j <= i; j++)
        {
            sum += hist_temp.at<float>(j);
            //std::cout << hist.at<float>(j) << "\n";  // Control Message for histogram of pixel intensities
        }
        uchar s = 255*sum/(image.cols*image.rows);
        lookup.at<uchar>(i)= round(s);
    }
    //std::cout << sum << "\n"; // Control message -> Total number of pixels in image => should be equal to image.cols*image.rows

    cv::Mat_<uchar> result;
    cv::LUT(image,lookup, result); //apply the lookup table
    return result;
}


void calcAndPlotHist(std::string name, cv::Mat &img, cv::MatND &hist, int histSize, float range[])
{
    // Computes histogram and plots histogram
    /*
     * img adress of input image
     * hist adress of output histogram object - unnormalized histogram
     * histSize - number of bins of histogram
     * range - range of pixel intensities in the image
     */

    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    /// Compute the histograms:
    cv::calcHist( &img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    // Histogram width = 4*histSize -> good for plotting bins with spaces in between
    int hist_w = 4*histSize; int hist_h = 400;
    int bin_w = cvRound( (double) (hist_w/2)/histSize );
    // Number of pixel intensities which will be in one bin in histogram
    int bin_w_data = range[1]/histSize;

    // Create White BackGround for black historgram
    cv::Mat histGraph( hist_h, hist_w, CV_8UC3, cv::Scalar( 255, 255, 255) );

    /// Normalize the result to [ 0, histImage.rows ]
    cv::Mat normHist;
    cv::normalize(hist, normHist, 0, (histGraph.rows), cv::NORM_MINMAX, -1, cv::Mat() );

    // Draw Histogram Graph
    int bin_plot_w = 2;
    for( int i = 0; i < histSize; i++ )
    {
        line(histGraph, cv::Point((bin_w * 2*i), (hist_h)),
             cv::Point((bin_w * 2*i), hist_h - cvRound(normHist.at<float>(i))),
             cv::Scalar(255, 0, 0), bin_plot_w, 8, 0);
    }

    // Create Frame for Axes ---------------------------------------------------------------
    // Padding Values
    int leftPadd = 15;
    int rightPadd = 15;
    int bottomPadd = 15;
    int topPadd = 15;

    // Width and Height of Frame
    int frame_w = leftPadd + hist_w + rightPadd;
    int frame_h = bottomPadd + hist_h + topPadd;

    // Create Frame image
    cv::Mat frame( frame_h, frame_w, CV_8UC3, cv::Scalar( 240, 240, 240) );

    // Insert histogram Graph into Frame
    histGraph.copyTo(frame(cv::Rect( leftPadd, bottomPadd, histGraph.cols, histGraph.rows)));

    // Draw Axes ----------------------------------------------------------------------------
    //Draw X - Axes
    line(frame, cv::Point(0, frame_h - bottomPadd + 1), cv::Point(frame_w, frame_h - bottomPadd + 1), cv::Scalar(0, 0, 0), 1, 8, 0);
    //Draw Y - Axes
    line(frame, cv::Point(leftPadd-1, topPadd), cv::Point(leftPadd-1, frame_h), cv::Scalar(0, 0, 0), 1, 8, 0);

    // Put Scale to the X Axes
    int maxX = 255;
    for (int i = 0; i <= maxX; i += 1)
    {
        // Draw small scale line
        line(frame, cv::Point((leftPadd + bin_plot_w*i*bin_w) , (frame_h - bottomPadd)), cv::Point((leftPadd + bin_plot_w*i*bin_w) , (frame_h - bottomPadd * 3/4)), cv::Scalar(0, 0, 0), 1, 8, 0);
        if (i%5 == 0)
        {
            //Draw Large scale line
            line(frame, cv::Point((leftPadd + bin_plot_w*i*bin_w) , (frame_h - bottomPadd)), cv::Point((leftPadd + bin_plot_w*i*bin_w) , (frame_h - bottomPadd * 1/2)), cv::Scalar(0, 0, 0), 1, 8, 0);
        }
        if (i%10 == 0)
        {
            //Draw Large scale line
            line(frame, cv::Point((leftPadd + bin_plot_w*i*bin_w) , (frame_h - bottomPadd)), cv::Point((leftPadd + bin_plot_w*i*bin_w) , frame_h), cv::Scalar(0, 0, 0), 1, 8, 0);
        }
    }
    // Put Scale to the Y Axes
    // size of Y axes segments
    int segmentY = hist_h/10;
    for (int i = 0; i <= hist_h; i += segmentY)
    {
        //Draw Large scale line
        line(frame, cv::Point(0 , frame_h - bottomPadd - i), cv::Point( leftPadd , frame_h - bottomPadd - i ), cv::Scalar(0, 0, 0), 1, 8, 0);
    }

    // Create Frame for graph ---------------------------------------------------------------
    // Padding Values
    int paddFrLeft = 60;
    int paddFrRight = 30;
    int paddFrBottom = 60;
    int paddFrTop = 60;

    int histImg_h = paddFrBottom + frame_h + paddFrTop;
    int histImg_w = paddFrLeft + frame_w + paddFrRight;

    // Create Frame image
    cv::Mat histImg( histImg_h, histImg_w , CV_8UC3, cv::Scalar( 240, 240, 240) );

    // Insert histogram with frame into Histogram Image
    frame.copyTo(histImg(cv::Rect( paddFrLeft, paddFrBottom, frame.cols, frame.rows)));

    // Create stream for designing axes labels
    std::ostringstream convert;

    // put scale labels on X axes
    for (int i = 0; i <= maxX; i += 10*bin_w_data)
    {
        // Clear stream Variable
        convert.str(std::string());
        //Convert bin number to string
        convert << i;
        //Put label points to axes
        cv::putText(histImg, convert.str(), cv::Point((bin_w * 2/bin_w_data * i + paddFrLeft + leftPadd - 10), (histImg_h - paddFrBottom * 3/4)), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 0, 0), 1, 8, 0);
    }

    // Put scale labels on Y axes -----------------------------------------------------------------------------
    // Find out extremes in histgram
    double mxY;
    double mnY;
    cv::minMaxLoc(hist, &mnY, &mxY, 0, 0);
    // convert extremes to integers
    int maxY = (int) mxY;
    int minY = (int) mnY;

    int j = 1;
    int exp = 0;
    while (maxY/j > 100)
    {
        j *= 10;
        exp++;
    }

    // size of one Y axes segment
    int segmentYsize = maxY / 10;
    for (int i = 0; i<=10; i++)
    {
        // Clear stream Variable
        convert.str(std::string());
        //Convert segment number to string
        convert << (i * segmentYsize) / j;
        //Put label points to axes
        cv::putText(histImg, convert.str(), cv::Point(paddFrLeft - 20, (histImg_h - paddFrBottom - 10 - (i * hist_h / 10))), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 0, 0), 1, 8, 0);
    }
    // Clear stream Variable
    convert.str(std::string());
    convert << "x 10^" << exp;
    cv::putText(histImg, convert.str(), cv::Point(paddFrLeft - 20, (histImg_h - paddFrBottom - 30 - hist_h)), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 0, 0), 1, 8, 0);

    // Put X axes label
    cv::putText(histImg, "Pixel Intensities", cv::Point(histImg_w / 2 - 30 , histImg_h - paddFrBottom/3), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 0, 0), 1, 8, 0);

    // Put name of The plot
    cv::putText(histImg, name, cv::Point(histImg_w / 3 - name.length() * 5/6 , paddFrTop * 2/3), CV_FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, 8, 0);

    // Insert vertical Y axes label using temp Mat Object
    cv::Mat textImg(histImg_w, histImg_h, CV_8UC3, cv::Scalar( 0, 0, 0) );
    putText(textImg, "No. Of Pixels", cv::Point(histImg_h/2 - 30, paddFrBottom/3), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
    // Rotate Temp object
    rot90(textImg,2);
    //cv::imshow("Test", textImg); // control print
    // Add temporary image to the Histogram Image
    histImg -= textImg;

    /// Display
    cv::namedWindow(name, CV_WINDOW_AUTOSIZE );
    imshow(name, histImg );
    //cv::resizeWindow(name, 512, 256);
}

void plotMagSpecGraph(cv::Mat& specGraph)
{
    int graphRows = specGraph.rows;
    int graphCols = specGraph.cols;

    // Create Frame for Axes ---------------------------------------------------------------
    // Padding Values
    int leftPadd = 15;
    int rightPadd = 15;
    int bottomPadd = 15;
    int topPadd = 15;

    // Width and Height of Frame
    int frame_w = leftPadd + graphCols + rightPadd;
    int frame_h = bottomPadd + graphRows + topPadd;

    // Create Frame image
    cv::Mat frame( frame_h, frame_w, CV_32FC1,240 );

    // Insert histogram Graph into Frame
    specGraph.copyTo(frame(cv::Rect( leftPadd, bottomPadd, specGraph.cols, specGraph.rows)));

    // Draw Axes ----------------------------------------------------------------------------
    //Draw X - Axes
    line(frame, cv::Point(0, frame_h - bottomPadd), cv::Point(frame_w, frame_h - bottomPadd), cv::Scalar(0, 0, 0), 1, 8, 0);
    //Draw Y - Axes
    line(frame, cv::Point(leftPadd-1, topPadd), cv::Point(leftPadd-1, frame_h), cv::Scalar(0, 0, 0), 1, 8, 0);

    /// Display
    cv::namedWindow("Magnitude Spectrum Graph", CV_WINDOW_AUTOSIZE );
    imshow("Magnitude Spectrum Graph", frame );
}

/*

void dftshift(cv::Mat_<float>& magnitude) {
    const int cx = magnitude.cols/2;
    const int cy = magnitude.rows/2;

    cv::Mat_<float> tmp;
    cv::Mat_<float> topLeft(magnitude, cv::Rect(0, 0, cx, cy));
    cv::Mat_<float> topRight(magnitude, cv::Rect(cx, 0, cx, cy));
    cv::Mat_<float> bottomLeft(magnitude, cv::Rect(0, cy, cx, cy));
    cv::Mat_<float> bottomRight(magnitude, cv::Rect(cx, cy, cx, cy));

    topLeft.copyTo(tmp);
    bottomRight.copyTo(topLeft);
    tmp.copyTo(bottomRight);

    topRight.copyTo(tmp);
    bottomLeft.copyTo(topRight);
    tmp.copyTo(bottomLeft);
}
*/





