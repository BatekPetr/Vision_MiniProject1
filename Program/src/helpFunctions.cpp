//
// Created by petr on 12.10.16.
//


#include "helpFunctions.h"

// ##################### GENERAL Functions #############################################################################

cv::Mat_<float> normalizeAndPlot(std::string name, const cv::Mat_<float>& img, bool plot, int width, int height)
{
    // function normalizes Mat_<float> object intensities to the interval [0, 1]
    // Don't want to change original image -> created new Mat object
    cv::Mat_<float> normalized;
    // Normalize float image
    // For floating point images its neccessary to normalize pixel intensities to the interval [0,1] before plotting
    cv::normalize(img, normalized, 0.0, 1.0, CV_MINMAX);

    if (plot)
    {
        cv::namedWindow(name, CV_WINDOW_NORMAL);
        cv::imshow(name, normalized);
        cv::resizeWindow(name, width, height);
    }

    return normalized;
}

void plot(std::string name, cv::Mat& img, int width, int height)
{
    // function for plotting Mat_<uchar> images
    cv::namedWindow(name, CV_WINDOW_NORMAL);
    cv::imshow(name, img);
    cv::resizeWindow(name, width, height);
}

void rot90(cv::Mat &matImage, int rotflag)
{
    //1=CW, 2=CCW, 3=180
    if (rotflag == 1){
        transpose(matImage, matImage);
        flip(matImage, matImage,1); //transpose+flip(1)=CW
    } else if (rotflag == 2) {
        transpose(matImage, matImage);
        flip(matImage, matImage,0); //transpose+flip(0)=CCW
    } else if (rotflag ==3){
        flip(matImage, matImage,-1);    //flip(-1)=180
    } else if (rotflag != 0){ //if not 0,1,2,3:
        std::cout  << "Unknown rotation flag(" << rotflag << ")" << std::endl;
    }
}

void on_mouse( int e, int x, int y, int d, void* ptr)
{
    if (e == CV_EVENT_LBUTTONDOWN)
    {
        cv::Mat_<float>* image= (cv::Mat_<float>* ) ptr;

        float max = 0;
        int maxU = x-100, maxV = y-100;
        float intensity;
        for (int u = x-100; u <= x + 100; u++)
        {
            for (int v = y-100; v <= y + 100; v++)
            {
                intensity = (float)image->at<float>(v,u);
                if (intensity > max)
                {
                    max = intensity;
                    maxU = u;
                    maxV = v;
                }

            }
        }
        std::cout << "Maximum intensity of value " << max << " found at point (u, v) = " << maxU << " , " << maxV << std::endl;
    }
}

// ################### FREQUENCY Filtering #############################################################################
// Padding before Fourier Transform
void paddZeros(const cv::Mat_<float>& img, cv::Mat_<float>& padded)
{
    //Compute no. of pixels needed for padding in each dimension
    //Use getOptimalDFTSize(A+B-1). See G&W page 251,252 and 263 and dft tutorial. (Typicly A+B-1 ~ 2A is used)
    int rows = cv::getOptimalDFTSize(2*img.rows);
    int cols = cv::getOptimalDFTSize(2*img.cols);

    // Pad the image with borders using copyMakeBorders.
    // on the border add zero pixels
    // write new image to location of padded Mat object - output pointer of this function
    copyMakeBorder(img, padded, 0, rows - img.rows, 0, cols - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}

cv::Mat_<float> plotMgSpec(std::string name, cv::Mat_<cv::Vec2f> img_dft, cv::Mat_<float> imgs[])
{
    // Split img_dft, you can save result into imgs
    cv::split(img_dft, imgs);                   // imgs[0] = Re(DFT(img_dft), planes[1] = Im(DFT(img_dft))

    // Compute magnitude/phase (e.g. cartToPolar), use as input imgs
    cv::Mat_<float> magnitude, phase;
    cv::cartToPolar(imgs[0], imgs[1], magnitude, phase);// imgs[0] = magnitude

    // Define Logarithm of magnitude and Output image for HPF
    cv::Mat_<float> magnitudel;
    // Add 1 to all pixels in order to avoid negative logarithms
    magnitudel = magnitude + 1.0f;
    // Take logarithm of modified magnitude (log()), save result into magnitudel
    cv::log(magnitudel, magnitudel);

    cv::Mat_<float> normalizedMagSpec = normalizeAndPlot(name, magnitudel, true, 308, 480);
    return normalizedMagSpec;

}


