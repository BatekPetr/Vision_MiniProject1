//
// Created by petr on 12.10.16.
//

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "image4_1.h"
#include "helpFunctions.h"

void run4_1(std::string filename)
{
    // A gray image as a float Mat
    cv::Mat_<uchar> img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    // Get dimensions of original picture
    int imgRows = img.rows;
    int imgCols = img.cols;

    // Plot Input image resized to fit the screen
    plot("Input Image", img, 484, 1158);

// FREQUENCY DOMAIN FILTERING  #########################################################################################
// Analysis for parasitic frequencies ----------------------------------------------------------------------------------

    cv::Mat_<float> imgNormalized = normalizeAndPlot("Input Image", img, true, 471, 1158); //if the image is loaded as float
    // Create container for padded image
    cv::Mat_<float> padded;
    //Pad the image with borders - wrapping function
    paddZeros( img, padded);
    // Plot Padded image
    cv::Mat_<float> normPadded = normalizeAndPlot("Padded Image", padded, true, 471, 1158);
    cv::imwrite("../../images/padded4_1.png", normPadded * 255);

    //Copy the gray image into the first channel of a new 2-channel image of type Mat_<Vec2f>, e.g. using merge(), save it in img_dft
    //The second channel should be all zeros.
    cv::Mat_<float> imgs[] = {padded.clone(), cv::Mat_<float>(padded.rows, padded.cols, 0.0f)};
    //Compute FT of padded image
    // Merge Re and Im component from imgs into Mat_<Vec2f>, e.g. using merge(), save it in img_dft
    cv::Mat_<cv::Vec2f> img_dft;

    // Merge 2 images from imgs[] vector into img_dft, which is Vec2f format
    cv::merge(imgs, 2, img_dft);         // Add to the expanded another plane with zeros - imaginary part of intensities values

    // Compute DFT using img_dft as input
    cv::dft(img_dft, img_dft);            // this way the result may fit in the source matrix

    // Shift Quadrants of DFT
    dftshift(img_dft);

    // Plot Magnitude Spectrum and set Callback function
    cv::Mat_<float> normalizedMagSpec;
    normalizedMagSpec = plotMgSpec("Magnitude Spectrum", img_dft, imgs);
    cv::imwrite("../../images/mgSpec4_1.png", normalizedMagSpec * 255);
    //std::cout << "Mag. Spec Size (u,v) = " << normalizedMagSpec.cols << " x " << normalizedMagSpec.rows << std::endl;
    //std::cout << "Position of origin of Mg. Spec. (u,v) = " << normalizedMagSpec.cols/2 << " x " << normalizedMagSpec.rows/2 << std::endl;
    // Set Mouse callback function to extract coordinates of specified point in the image
    setMouseCallback("Magnitude Spectrum", on_mouse, &normalizedMagSpec);

// Design of FILTER for IMAGE 4_1 --------------------------------------------------------------------------------------

    // Design Notch filters
    cv::Mat_<cv::Vec2f> notchRF1 = cv::Mat_<cv::Vec2f> (padded.rows, padded.cols, 0.0f);
    notchRejectFilter(notchRF1, 100, 2, 2143, 1771);
    cv::Mat_<cv::Vec2f> notchRF2 = cv::Mat_<cv::Vec2f> (padded.rows, padded.cols, 0.0f);
    notchRejectFilter(notchRF2, 100, 2, 1333, 2200);
    // Final Notch filter is composed from two others
    cv::Mat_<cv::Vec2f> notchRF;
    cv::mulSpectrums(notchRF1, notchRF2, notchRF, cv::DFT_ROWS);
    cv::Mat_<float> normNotchRF = plotMgSpec("Magnitude Spectrum of notch filter", notchRF, imgs);
    cv::imwrite("../../images/notchRF4_1.png", normNotchRF * 255);
    cv::Mat filter = notchRF;

// Filtering and IDFT for IMAGE 4_1 ------------------------------------------------------------------------------------
    // Filtration of amp. spectrum
    // Do the filtration in frequency domain - multiplication
    cv::mulSpectrums(filter, img_dft, img_dft, cv::DFT_ROWS);
    cv::Mat_<float> normImg_dft = plotMgSpec("Filtered Magnitude Spectrum", img_dft, imgs);
    cv::imwrite("../../images/FilteredSpectrum4_1.png", normImg_dft * 255);

    // Reconstruct image using inverse DFT and cropp the padding
    cv::Mat_<float> output;
    // Before recunstruction, it is neccesary to shift Mg Spec back to original(mathematical) position
    dftshift(img_dft);
    cv::dft(img_dft, output, cv::DFT_INVERSE| cv::DFT_REAL_OUTPUT);
    cv::Mat_<float> croppedOutput(output,cv::Rect(0,0,imgCols,imgRows));

    croppedOutput = normalizeAndPlot("Output", croppedOutput, true, 471, 1158);
    cv::imwrite("../../images/OutputImage4_1.png", croppedOutput * 255);

    // Wait key has to be HERE in order to resolve pointer to the image in function inmouse !!!!
    cv::waitKey(0);
}


void notchRejectFilter(cv::Mat_<cv::Vec2f> filt, int cutOff, int order, int uk, int vk)
{
    // Create temporary filter which is mirrored by origin of 1st HF
    int filtRows = filt.rows;
    int filtCols = filt.cols;
    int centerRows = filtRows / 2;
    int centerCols = filtCols / 2;
    float D;
    float Dtemp;

    //
    uk = uk - centerCols;
    vk = vk - centerRows;
    std::cout << uk << std::endl;
    std::cout << vk << std::endl;

    for (int u = 0; u < filtCols; u++) {
        for (int v = 0; v < filtRows; v++) {
            D = pow((pow(u - centerCols - uk, 2) + pow(v - centerRows - vk, 2)), 0.5);
            Dtemp = pow((pow(u - centerCols + uk, 2) + pow(v - centerRows + vk, 2)), 0.5);
            filt(v,u)[0] = (float) (1.0 / (1.0 + pow(cutOff/D, 2 * order))) * (1.0 / (1.0 + pow(cutOff/Dtemp, 2 * order)));
            filt(v,u)[1] = 0;
        }
    }
}


template<class ImgT>
void dftshift(ImgT& img) {
    const int cx = img.cols/2;
    const int cy = img.rows/2;

    ImgT tmp;
    ImgT topLeft(img, cv::Rect(0, 0, cx, cy));
    ImgT topRight(img, cv::Rect(cx, 0, cx, cy));
    ImgT bottomLeft(img, cv::Rect(0, cy, cx, cy));
    ImgT bottomRight(img, cv::Rect(cx, cy, cx, cy));

    topLeft.copyTo(tmp);
    bottomRight.copyTo(topLeft);
    tmp.copyTo(bottomRight);

    topRight.copyTo(tmp);
    bottomLeft.copyTo(topRight);
    tmp.copyTo(bottomLeft);
}




