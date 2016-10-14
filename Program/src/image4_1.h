//
// Created by petr on 12.10.16.
//

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#ifndef MINIPROJECT1_IMAGE4_1_H
#define MINIPROJECT1_IMAGE4_1_H

// function prototypes
void run4_1(std::string filename);
void notchRejectFilter(cv::Mat_<cv::Vec2f> filt, int cutOff, int order, int uk, int vk);

template<class ImgT>
void dftshift(ImgT& img);

#endif //MINIPROJECT1_IMAGE4_1_H
