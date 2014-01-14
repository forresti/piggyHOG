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

#include "Patchwork.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <set>

using namespace Eigen;
using namespace FFLD;
using namespace std;

int Patchwork::MaxRows_(0);
int Patchwork::MaxCols_(0);
int Patchwork::HalfCols_(0);

Patchwork::Patchwork() : padx_(0), pady_(0), interval_(0)
{
}

Patchwork::Patchwork(const JPEGPyramid & pyramid) : padx_(pyramid.padx()), pady_(pyramid.pady()),
interval_(pyramid.interval())
{
	// Remove the padding from the bottom/right sides since convolutions with Fourier wrap around
	const int nbLevels = pyramid.levels().size();
	
	rectangles_.resize(nbLevels);
	
	for (int i = 0; i < nbLevels; ++i) {
		rectangles_[i].first.setWidth(pyramid.levels()[i].cols() - padx_);
		rectangles_[i].first.setHeight(pyramid.levels()[i].rows() - pady_);
	}
	
	// Build the patchwork planes
	const int nbPlanes = BLF(rectangles_);
	
	// Constructs an empty patchwork in case of error
	if (nbPlanes <= 0)
		return;
	
	planes_.resize(nbPlanes);
	
	for (int i = 0; i < nbPlanes; ++i) {
		planes_[i] = Plane::Constant(MaxRows_, HalfCols_, Cell::Zero());
		
		Map<JPEGPyramid::Level, Aligned>
			plane(reinterpret_cast<JPEGPyramid::Cell *>(planes_[i].data()), MaxRows_, HalfCols_ * 2);
		
		// Set the last feature to 1
		for (int y = 0; y < MaxRows_; ++y)
			for (int x = 0; x < MaxCols_; ++x)
				plane(y, x)(JPEGPyramid::NbFeatures - 1) = 1.0f;
	}
	
	// Recopy the pyramid levels into the planes
	for (int i = 0; i < nbLevels; ++i) {
		Map<JPEGPyramid::Level, Aligned>
			plane(reinterpret_cast<JPEGPyramid::Cell *>(planes_[rectangles_[i].second].data()),
				  MaxRows_, HalfCols_ * 2);
		
		plane.block(rectangles_[i].first.y(), rectangles_[i].first.x(),
					rectangles_[i].first.height(), rectangles_[i].first.width()) =
			pyramid.levels()[i].topLeftCorner(rectangles_[i].first.height(),
											  rectangles_[i].first.width());
	}
	
}

int Patchwork::padx() const
{
	return padx_;
}

int Patchwork::pady() const
{
	return pady_;
}

int Patchwork::interval() const
{
	return interval_;
}

bool Patchwork::empty() const
{
	return planes_.empty();
}

bool Patchwork::Init(int maxRows, int maxCols)
{
	// It is an error if maxRows or maxCols are too small
	if ((maxRows < 2) || (maxCols < 2))
		return false;
	
	// Temporary matrices
	JPEGPyramid::Matrix tmp(maxRows * JPEGPyramid::NbFeatures, maxCols + 2);
	
	int dims[2] = {maxRows, maxCols};
}

int Patchwork::MaxRows()
{
	return MaxRows_;
}

int Patchwork::MaxCols()
{
	return MaxCols_;
}

void Patchwork::TransformFilter(const JPEGPyramid::Level & filter, Filter & result)
{
	// Early return if no filter given or if Init was not called or if the filter is too large
	if (!filter.size() || !MaxRows_ || (filter.rows() > MaxRows_) || (filter.cols() > MaxCols_)) {
		result = Filter();
		return;
	}
	
	// Recopy the filter into a plane
	result.first = Plane::Constant(MaxRows_, HalfCols_, Cell::Zero());
	result.second = pair<int, int>(filter.rows(), filter.cols());
	
	Map<JPEGPyramid::Level, Aligned> plane(reinterpret_cast<JPEGPyramid::Cell *>(result.first.data()),
										  MaxRows_, HalfCols_ * 2);
	
	for (int y = 0; y < filter.rows(); ++y)
		for (int x = 0; x < filter.cols(); ++x)
			plane((MaxRows_ - y) % MaxRows_, (MaxCols_ - x) % MaxCols_) = filter(y, x) /
																		  (MaxRows_ * MaxCols_);
}

namespace FFLD
{
namespace detail
{
// Order rectangles by decreasing area.
class AreaComparator
{
public:
	AreaComparator(const vector<pair<Rectangle, int> > & rectangles) :
	rectangles_(rectangles)
	{
	}
	
	/// Returns whether rectangle @p a comes before @p b.
	bool operator()(int a, int b) const
	{
		const int areaA = rectangles_[a].first.area();
		const int areaB = rectangles_[b].first.area();
		
		return (areaA > areaB) || ((areaA == areaB) && (rectangles_[a].first.height() >
														rectangles_[b].first.height()));
	}
	
private:
	const vector<pair<Rectangle, int> > & rectangles_;
};

// Order free gaps (rectangles) by position and then by size
struct PositionComparator
{
	// Returns whether rectangle @p a comes before @p b
	bool operator()(const Rectangle & a, const Rectangle & b) const
	{
		return (a.y() < b.y()) ||
			   ((a.y() == b.y()) &&
				((a.x() < b.x()) ||
				 ((a.x() == b.x()) &&
				  ((a.height() > b.height()) ||
				   ((a.height() == b.height()) && (a.width() > b.width()))))));
	}
};
}
}

int Patchwork::BLF(vector<pair<Rectangle, int> > & rectangles)
{
	// Order the rectangles by decreasing area. If a rectangle is bigger than MaxRows x MaxCols
	// return -1
	vector<int> ordering(rectangles.size());
	
	for (int i = 0; i < rectangles.size(); ++i) {
		if ((rectangles[i].first.width() > MaxCols_) || (rectangles[i].first.height() > MaxRows_))
			return -1;
		
		ordering[i] = i;
	}
	
	sort(ordering.begin(), ordering.end(), detail::AreaComparator(rectangles));
	
	// Index of the plane containing each rectangle
	for (int i = 0; i < rectangles.size(); ++i)
		rectangles[i].second = -1;
	
	vector<set<Rectangle, detail::PositionComparator> > gaps;
	
	// Insert each rectangle in the first gap big enough
	for (int i = 0; i < rectangles.size(); ++i) {
		pair<Rectangle, int> & rect = rectangles[ordering[i]];
		
		// Find the first gap big enough
		set<Rectangle, detail::PositionComparator>::iterator g;
		
		for (int i = 0; (rect.second == -1) && (i < gaps.size()); ++i) {
			for (g = gaps[i].begin(); g != gaps[i].end(); ++g) {
				if ((g->width() >= rect.first.width()) && (g->height() >= rect.first.height())) {
					rect.second = i;
					break;
				}
			}
		}
		
		// If no gap big enough was found, add a new plane
		if (rect.second == -1) {
			set<Rectangle, detail::PositionComparator> plane;
			plane.insert(Rectangle(MaxCols_, MaxRows_)); // The whole plane is free
			gaps.push_back(plane);
			g = gaps.back().begin();
			rect.second = gaps.size() - 1;
		}
		
		// Insert the rectangle in the gap
		rect.first.setX(g->x());
		rect.first.setY(g->y());
		
		// Remove all the intersecting gaps, and add newly created gaps
		for (g = gaps[rect.second].begin(); g != gaps[rect.second].end();) {
			if (!((rect.first.right() < g->left()) || (rect.first.bottom() < g->top()) ||
				  (rect.first.left() > g->right()) || (rect.first.top() > g->bottom()))) {
				// Add a gap to the left of the new rectangle if possible
				if (g->x() < rect.first.x())
					gaps[rect.second].insert(Rectangle(g->x(), g->y(), rect.first.x() - g->x(),
													   g->height()));
				
				// Add a gap on top of the new rectangle if possible
				if (g->y() < rect.first.y())
					gaps[rect.second].insert(Rectangle(g->x(), g->y(), g->width(),
													   rect.first.y() - g->y()));
				
				// Add a gap to the right of the new rectangle if possible
				if (g->right() > rect.first.right())
					gaps[rect.second].insert(Rectangle(rect.first.right() + 1, g->y(),
													   g->right() - rect.first.right(),
													   g->height()));
				
				// Add a gap below the new rectangle if possible
				if (g->bottom() > rect.first.bottom())
					gaps[rect.second].insert(Rectangle(g->x(), rect.first.bottom() + 1, g->width(),
													   g->bottom() - rect.first.bottom()));
				
				// Remove the intersecting gap
				gaps[rect.second].erase(g++);
			}
			else {
				++g;
			}
		}
	}
	
	return gaps.size();
}
