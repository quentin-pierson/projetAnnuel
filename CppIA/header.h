//
// Created by quentin pierson on 09/05/2021.
//
#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::Matrix2f;
using Eigen::Matrix3f;
using Eigen::Map;
using Eigen::VectorXf;
using Eigen::Vector3f;
using Eigen::RowVectorXf;
using Eigen::ArrayXf;

#ifndef CPPIA_HEADER_H
#define CPPIA_HEADER_H
#endif //CPPIA_HEADER_H

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

