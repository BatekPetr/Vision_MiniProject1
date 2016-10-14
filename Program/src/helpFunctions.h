//
// Created by petr on 12.10.16.
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#ifndef MINIPROJECT1_HELPFUNCTIONS_H
#define MINIPROJECT1_HELPFUNCTIONS_H

    // Function Prototypes
    // General used functions
    cv::Mat_<float> normalizeAndPlot(std::string name, const cv::Mat_<float>& img, bool plot, int width, int height);
    void plot(std::string name, cv::Mat& img, int width, int height);
    void rot90(cv::Mat &matImage, int rotflag);
void on_mouse( int e, int x, int y, int d, void* ptr);

    // Frequency filtering
    void paddZeros(const cv::Mat_<float>& img, cv::Mat_<float>& padded);
    cv::Mat_<float> plotMgSpec(std::string name, cv::Mat_<cv::Vec2f> img_dft, cv::Mat_<float> imgs[]);

#endif //MINIPROJECT1_HELPFUNCTIONS_H
