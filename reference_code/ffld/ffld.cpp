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

#include "SimpleOpt.h"

#include "HOGPyramid.h"
#include "JPEGImage.h" 
//#include "Intersector.h"
//#include "Mixture.h"
//#include "Scene.h" //for Object.h, which is needed for "PASCAL object type." TODO: remove soon.

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#ifndef _WIN32
#include <sys/time.h>

timeval Start, Stop;

inline void start()
{
	gettimeofday(&Start, 0);
}

inline int stop()
{
	gettimeofday(&Stop, 0);
	
	timeval duration;
	timersub(&Stop, &Start, &duration);
	
	return duration.tv_sec * 1000 + (duration.tv_usec + 500) / 1000;
}
#else
#include <time.h>
#include <windows.h>

ULARGE_INTEGER Start, Stop;

inline void start()
{
	GetSystemTimeAsFileTime((FILETIME *)&Start);
}

inline int stop()
{
	GetSystemTimeAsFileTime((FILETIME *)&Stop);
	Stop.QuadPart -= Start.QuadPart;
	return (Stop.QuadPart + 5000) / 10000;
}
#endif

using namespace FFLD;
using namespace std;

#if 0
struct Detection : public FFLD::Rectangle
{
	HOGPyramid::Scalar score;
	int l;
	int x;
	int y;
	
	Detection() : score(0), l(0), x(0), y(0)
	{
	}
	
	Detection(HOGPyramid::Scalar score, int l, int x, int y, FFLD::Rectangle bndbox) :
	FFLD::Rectangle(bndbox), score(score), l(l), x(x), y(y)
	{
	}
	
	bool operator<(const Detection & detection) const
	{
		return score > detection.score;
	}
};
#endif

// SimpleOpt array of valid options
enum
{
	OPT_HELP, OPT_MODEL, OPT_NAME, OPT_RESULTS, OPT_IMAGES, OPT_NB_NEG, OPT_PADDING, OPT_INTERVAL,
	OPT_THRESHOLD, OPT_OVERLAP
};

CSimpleOpt::SOption SOptions[] =
{
	{ OPT_HELP, "-h", SO_NONE },
	{ OPT_HELP, "--help", SO_NONE },
	//{ OPT_MODEL, "-m", SO_REQ_SEP },
	//{ OPT_MODEL, "--model", SO_REQ_SEP },
	//{ OPT_NAME, "-n", SO_REQ_SEP },
	//{ OPT_NAME, "--name", SO_REQ_SEP },
	//{ OPT_RESULTS, "-r", SO_REQ_SEP },
	//{ OPT_RESULTS, "--results", SO_REQ_SEP },
	//{ OPT_IMAGES, "-i", SO_REQ_SEP },
	//{ OPT_IMAGES, "--images", SO_REQ_SEP },
	//{ OPT_NB_NEG, "-z", SO_REQ_SEP },
	//{ OPT_NB_NEG, "--nb-negatives", SO_REQ_SEP },
	{ OPT_PADDING, "-p", SO_REQ_SEP },
	{ OPT_PADDING, "--padding", SO_REQ_SEP },
	{ OPT_INTERVAL, "-e", SO_REQ_SEP },
	{ OPT_INTERVAL, "--interval", SO_REQ_SEP },
	//{ OPT_THRESHOLD, "-t", SO_REQ_SEP },
	//{ OPT_THRESHOLD, "--threshold", SO_REQ_SEP },
	//{ OPT_OVERLAP, "-v", SO_REQ_SEP },
	//{ OPT_OVERLAP, "--overlap", SO_REQ_SEP },
	SO_END_OF_OPTIONS
};

void showUsage()
{
	cout << "Usage: test [options] image.jpg, or\n       test [options] image_set.txt\n\n"
			"Options:\n"
			"  -h,--help               Display this information\n"
			//"  -m,--model <file>       Read the input model from <file> (default \"model.txt\")\n"
			//"  -n,--name <arg>         Name of the object to detect (default \"person\")\n"
			//"  -r,--results <file>     Write the detection results to <file> (default none)\n"
			//"  -i,--images <folder>    Draw the detections to <folder> (default none)\n"
			//"  -z,--nb-negatives <arg> Maximum number of negative images to consider (default all)\n"
			"  -p,--padding <arg>      Amount of zero padding in HOG cells (default 12)\n"
			"  -e,--interval <arg>     Number of levels per octave in the HOG pyramid (default 10)\n"
			//"  -t,--threshold <arg>    Minimum detection threshold (default -10)\n"
			//"  -v,--overlap <arg>      Minimum overlap in non maxima suppression (default 0.5)"
		 << endl;
}

// Test a mixture model (compute a ROC curve)
int main(int argc, char * argv[])
{
	// Default parameters
	string model("model.txt");
	//Object::Name name = Object::PERSON;
	string results;
	string images;
	int nbNegativeScenes = -1;
	int padding = 12;
	int interval = 10;
	double threshold =-10.0;
	double overlap = 0.5;
	
	// Parse the parameters
	CSimpleOpt args(argc, argv, SOptions);

	while (args.Next()) {
		if (args.LastError() == SO_SUCCESS) {
			if (args.OptionId() == OPT_HELP) {
				showUsage();
				return 0;
			}
#if 0
			else if (args.OptionId() == OPT_MODEL) {
				model = args.OptionArg();
			}
			else if (args.OptionId() == OPT_NAME) {
				string arg = args.OptionArg();
				transform(arg.begin(), arg.end(), arg.begin(), static_cast<int (*)(int)>(tolower));
				const string Names[20] =
				{
					"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
					"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
					"sheep", "sofa", "train", "tvmonitor"
				};
				
				const string * iter = find(Names, Names + 20, arg);
				
				// Error checking
				if (iter == Names + 20) {
					showUsage();
					cerr << "\nInvalid name arg " << args.OptionArg() << endl;
					return -1;
				}
				else {
					name = static_cast<Object::Name>(iter - Names);
				}
			}
			else if (args.OptionId() == OPT_RESULTS) {
				results = args.OptionArg();
			}
#endif
			else if (args.OptionId() == OPT_IMAGES) {
				images = args.OptionArg();
			}

#if 0
			else if (args.OptionId() == OPT_NB_NEG) {
				nbNegativeScenes = atoi(args.OptionArg());
				
				// Error checking
				if (nbNegativeScenes < 0) {
					showUsage();
					cerr << "\nInvalid nb-negatives arg " << args.OptionArg() << endl;
					return -1;
				}
			}
#endif
			else if (args.OptionId() == OPT_PADDING) {
				padding = atoi(args.OptionArg());
				
				// Error checking
				if (padding <= 1) {
					showUsage();
					cerr << "\nInvalid padding arg " << args.OptionArg() << endl;
					return -1;
				}
			}

			else if (args.OptionId() == OPT_INTERVAL) {
				interval = atoi(args.OptionArg());
				
				// Error checking
				if (interval <= 0) {
					showUsage();
					cerr << "\nInvalid interval arg " << args.OptionArg() << endl;
					return -1;
				}
			}
			else if (args.OptionId() == OPT_THRESHOLD) {
				threshold = atof(args.OptionArg());
			}
			else if (args.OptionId() == OPT_OVERLAP) {
				overlap = atof(args.OptionArg());
				
				// Error checking
				if ((overlap <= 0.0) || (overlap >= 1.0)) {
					showUsage();
					cerr << "\nInvalid overlap arg " << args.OptionArg() << endl;
					return -1;
				}
			}
		}
		else {
			showUsage();
			cerr << "\nUnknown option " << args.OptionText() << endl;
			return -1;
		}
	}
	
	if (!args.FileCount()) {
		showUsage();
		cerr << "\nNo image/dataset provided" << endl;
		return -1;
	}
	else if (args.FileCount() > 1) {
		showUsage();
		cerr << "\nMore than one image/dataset provided" << endl;
		return -1;
	}
	
	// Try to open the mixture
#if 0
	ifstream in(model.c_str(), ios::binary);
	
	if (!in.is_open()) {
		showUsage();
		cerr << "\nInvalid model file " << model << endl;
		return -1;
	}
	
	Mixture mixture;
	in >> mixture;
	
	if (mixture.empty()) {
		showUsage();
		cerr << "\nInvalid model file " << model << endl;
		return -1;
	}
#endif
	// The image/dataset
	const string file(args.File(0));
	
	const size_t lastDot = file.find_last_of('.');
	
	if ((lastDot == string::npos) ||
		((file.substr(lastDot) != ".jpg") && (file.substr(lastDot) != ".txt"))) {
		showUsage();
		cerr << "\nInvalid file " << file << ", should be .jpg or .txt" << endl;
		return -1;
	}
	
	// Try to open the results
#if 0
	ofstream out;
	
	if (!results.empty()) {
		out.open(results.c_str(), ios::binary);
		
		if (!out.is_open()) {
			showUsage();
			cerr << "\nInvalid results file " << results << endl;
			return -1;
		}
	}
#endif	

	// Try to load the image
	if (file.substr(lastDot) == ".jpg") {
		JPEGImage image(file);
		
		if (image.empty()) {
			showUsage();
			cerr << "\nInvalid image " << file << endl;
			return -1;
		}
		
		// Compute the HOG features
		start();
		
		HOGPyramid pyramid(image, padding, padding, interval);
		
		if (pyramid.empty()) {
			showUsage();
			cerr << "\nInvalid image " << file << endl;
			return -1;
		}
		
		cout << "Computed HOG features in " << stop() << " ms" << endl;
    }
    else{
        cout << "need to input a JPG image" << endl;
    }

	return EXIT_SUCCESS;
}
