//
// Created by petr on 12.10.16.
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#ifndef MINIPROJECT1_IMAGE4_2_H
#define MINIPROJECT1_IMAGE4_2_H

void run4_2(std::string filename);
void bandRejectFilter(cv::Mat_<cv::Vec2f> filt, int cutOff, int width, int order);

template<class ImgT>
void dftshift(ImgT& img);

#endif //MINIPROJECT1_IMAGE4_2_H
