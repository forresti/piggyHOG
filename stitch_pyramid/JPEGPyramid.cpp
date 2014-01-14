//--------------------------------------------------------------------------------------------------
// Implementation of the paper "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012.
//
// Copyright (c) 2012 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLD (the Fast Fourier Linear Detector)
//
// FFLD is free software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License version 3 as published by the Free Software Foundation.
//
// FFLD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along with FFLD. If not, see
// <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#include "JPEGPyramid.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdio.h>

//#define DISABLE_HOG_BLOCKS //if defined: just compute hog cells; don't normalize into hog blocks

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Eigen;
using namespace FFLD;
using namespace std;

JPEGPyramid::JPEGPyramid() : padx_(0), pady_(0), interval_(0)
{
}

JPEGPyramid::JPEGPyramid(int padx, int pady, int interval, const vector<Level> & levels) : padx_(0),
pady_(0), interval_(0)
{
	if ((padx < 1) || (pady < 1) || (interval < 1))
		return;
	
	padx_ = padx;
	pady_ = pady;
	interval_ = interval;
	levels_ = levels;
}

JPEGPyramid::JPEGPyramid(const JPEGImage & image, int padx, int pady, int interval) : padx_(0),
pady_(0), interval_(0)
{
	if (image.empty() || (padx < 1) || (pady < 1) || (interval < 1))
		return;
	
	// Copmute the number of scales such that the smallest size of the last level is 5
	const int maxScale = ceil(log(min(image.width(), image.height()) / 40.0) / log(2.0)) * interval;
	
	// Cannot compute the pyramid on images too small
	if (maxScale < interval)
		return;
	
	padx_ = padx;
	pady_ = pady;
	interval_ = interval;
	levels_.resize(maxScale + 1);

    vector<double> scales(maxScale+1);
	
	int i;
#pragma omp parallel for private(i)
    //i=0;
	for (i = 0; i < interval; ++i) 
    {
		double scale = pow(2.0, static_cast<double>(-i) / interval);
		JPEGImage scaled = image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5);

#if 0		
		// First octave at twice the image resolution
#ifndef FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES
		Hog(scaled, levels_[i], padx, pady, 4);
	
#if 1
		// Second octave at the original resolution
		if (i + interval <= maxScale)
			Hog(scaled, levels_[i + interval], padx, pady, 8);
		
		// Remaining octaves
		for (int j = 2; i + j * interval <= maxScale; ++j) {
			scale *= 0.5;
			scaled = image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5);
			Hog(scaled, levels_[i + j * interval], padx, pady, 8);
		}
#endif
#else
		Hog(scaled.scanLine(0), scaled.width(), scaled.height(), scaled.depth(), levels_[i], 4);

		// Second octave at the original resolution
		if (i + interval <= maxScale)
			Hog(scaled.scanLine(0), scaled.width(), scaled.height(), scaled.depth(),
				levels_[i + interval], 8);
        //scales[i + interval] = scale;

		// Remaining octaves
		for (int j = 2; i + j * interval <= maxScale; ++j) {
			scale *= 0.5;
			scaled = image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5);
			Hog(scaled.scanLine(0), scaled.width(), scaled.height(), scaled.depth(),
				levels_[i + j * interval], 8);
           //scales[i + j*interval] = scale;
		}
#endif
#endif
    }

	// Add padding
#ifdef FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES
	for (int i = 0; i <= maxScale; ++i) {
		Level tmp = Level::Constant(levels_[i].rows() + (pady + 1) * 2,
									levels_[i].cols() + (padx + 1) * 2, Cell::Zero());
		
		// Set the last feature to 1
		for (int y = 0; y < tmp.rows(); ++y)
			for (int x = 0; x < tmp.cols(); ++x)
				tmp(y, x)(31) = 1;
		tmp.block(pady + 1, padx + 1, levels_[i].rows(), levels_[i].cols()) = levels_[i];
		
		levels_[i].swap(tmp);
	}
#endif
}

int JPEGPyramid::padx() const
{
	return padx_;
}

int JPEGPyramid::pady() const
{
	return pady_;
}

int JPEGPyramid::interval() const
{
	return interval_;
}

const vector<JPEGPyramid::Level> & JPEGPyramid::levels() const
{
	return levels_;
}

bool JPEGPyramid::empty() const
{
	return levels().empty();
}

#if 0
Map<JPEGPyramid::Matrix, Aligned> JPEGPyramid::Convert(Level & level)
{
	return Map<Matrix, Aligned>(level.data()->data(), level.rows(),
											  level.cols() * NbFeatures);
}

Map<const JPEGPyramid::Matrix, Aligned> JPEGPyramid::Convert(const Level & level)
{
	return Map<const Matrix, Aligned>(level.data()->data(), level.rows(),
													level.cols() * NbFeatures);
}

FFLD::JPEGPyramid::Level JPEGPyramid::Flip(const JPEGPyramid::Level & filter)
{
	// Symmetric features
	const int symmetry[NbFeatures] = {
		9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
		18, 26, 25, 24, 23, 22, 21, 20, 19, // Contrast-insensitive
		28, 27, 30, 29, // Texture
		31 // Truncation
	};
	
	// Symmetric filter
	JPEGPyramid::Level result(filter.rows(), filter.cols());
	
	for (int y = 0; y < filter.rows(); ++y)
		for (int x = 0; x < filter.cols(); ++x)
			for (int i = 0; i < NbFeatures; ++i)
				result(y, x)(i) = filter(y, filter.cols() - 1 - x)(symmetry[i]);
	
	return result;
}
#endif

