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
#include "JPEGPyramid.h"
#include "JPEGImage.h" 
#include "Patchwork.h"
#include "common/helpers.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace FFLD;
using namespace std;

// SimpleOpt array of valid options
enum
{
    OPT_HELP, OPT_PADDING, OPT_INTERVAL
};

CSimpleOpt::SOption SOptions[] =
{
	{ OPT_HELP, "-h", SO_NONE },
	{ OPT_HELP, "--help", SO_NONE },
	{ OPT_PADDING, "-p", SO_REQ_SEP },
	{ OPT_PADDING, "--padding", SO_REQ_SEP },
	{ OPT_INTERVAL, "-e", SO_REQ_SEP },
	{ OPT_INTERVAL, "--interval", SO_REQ_SEP },
	SO_END_OF_OPTIONS
};

void showUsage(){
	cout << "Usage: test [options] image.jpg, or\n       test [options] image_set.txt\n\n"
			"Options:\n"
			"  -h,--help               Display this information\n"
    		"  -p,--padding <arg>      Amount of zero padding in HOG cells (default 12)\n"
			"  -e,--interval <arg>     Number of levels per octave in the HOG pyramid (default 10)\n"
		 << endl;
}

// Parse command line parameters
//   put the appropriate values in (padding, interval, file) based on cmd-line args
void parseArgs(int &padding, int &interval, string &file, int argc, char * argv[]){
	CSimpleOpt args(argc, argv, SOptions);

	while (args.Next()) {
		if (args.LastError() == SO_SUCCESS) {
			if (args.OptionId() == OPT_HELP) {
				showUsage();
				exit(0);
			}
			else if (args.OptionId() == OPT_PADDING) {
				padding = atoi(args.OptionArg());
				
				// Error checking
				if (padding <= 1) {
					showUsage();
					cerr << "\nInvalid padding arg " << args.OptionArg() << endl;
					exit(1);
				}
			}
			else if (args.OptionId() == OPT_INTERVAL) {
				interval = atoi(args.OptionArg());
				
				// Error checking
				if (interval <= 0) {
					showUsage();
					cerr << "\nInvalid interval arg " << args.OptionArg() << endl;
					exit(1);
				}
			}
		}
		else {
			showUsage();
			cerr << "\nUnknown option " << args.OptionText() << endl;
			exit(1);
		}
	}
	if (!args.FileCount()) {
		showUsage();
		cerr << "\nNo image/dataset provided" << endl;
		exit(1);
	}
	else if (args.FileCount() > 1) {
		showUsage();
		cerr << "\nMore than one image/dataset provided" << endl;
		exit(1);
	}

	// The image/dataset
    file = args.File(0);
	const size_t lastDot = file.find_last_of('.');
	if ((lastDot == string::npos) ||
		((file.substr(lastDot) != ".jpg") && (file.substr(lastDot) != ".txt"))) {
		showUsage();
		cerr << "\nInvalid file " << file << ", should be .jpg or .txt" << endl;
		exit(1);
	}

	// Try to load the image
	if (file.substr(lastDot) != ".jpg") {
        cout << "need to input a JPG image" << endl;
        exit(1);
    }
}

void printHogSizes(JPEGPyramid pyramid);
void writePyraToCsv(JPEGPyramid pyramid);
void writePyraToJPG(JPEGPyramid pyramid);
void writePatchworkToJPG(Patchwork patchwork);

// Test a mixture model (compute a ROC curve)
int main(int argc, char * argv[]){
	// Default parameters
    string file;
	int padding = 12;
	int interval = 10;

    //parseArgs params are passed by reference, so they get updated here
    parseArgs(padding, interval, file, argc, argv); //update parameters with any command-line inputs

    printf("    padding = %d \n", padding);
    printf("    interval = %d \n", interval);
    printf("    file = %s \n", file.c_str());
 
	JPEGImage image(file);
    if (image.empty()) {
        showUsage();
        cerr << "\nInvalid image " << file << endl;
        return -1;
    }   
    // Compute the downsample+stitch

    double start_downsample = read_timer();    

    int padx = 11; //ignoring cmd-line padding arg.
    int pady = 6; //to match voc5 dims
    //JPEGPyramid pyramid(image, padding, padding, interval);
    JPEGPyramid pyramid(image, padx, pady, interval); 

    if (pyramid.empty()) {
        showUsage();
        cerr << "\nInvalid image " << file << endl;
        return -1;
    }
    
    double time_downsample = read_timer() - start_downsample;
    cout << "  Multiscale downsampling in " << time_downsample << " ms" << endl;


    double start_stitch = read_timer();

    Patchwork::Init((pyramid.levels()[0].height() - pady + 15) & ~15,
                    (pyramid.levels()[0].width() - padx + 15) & ~15); //TODO: add err checking 
    const Patchwork patchwork(pyramid); //TODO: patchwork.planes_[i] = a JPEGImage

    double time_stitch = read_timer() - start_stitch;
    cout << "  Stitched scales in " << time_stitch << " ms" << endl;

    printHogSizes(pyramid);
    //writePyraToCsv(pyramid);
    writePyraToJPG(pyramid);
    writePatchworkToJPG(patchwork);

   	return EXIT_SUCCESS;
}

void printHogSizes(JPEGPyramid pyramid){
    int nlevels = pyramid.levels().size();

    for(int level = 0; level < nlevels; level++){ 
        //const float* raw_hog = pyramid.levels()[level].data()->data();
        int width = pyramid.levels()[level].width();
        int height = pyramid.levels()[level].height();
        int depth = pyramid.NbChannels;
        printf("level %d: width=%d, height=%d, depth=%d \n", level, width, height, depth);
    }
}

// nRows = 32
// nCols = width*height
void writePyraToCsv(JPEGPyramid pyramid){
    int nlevels = pyramid.levels().size();

    for(int level = 0; level < nlevels; level++){
        //printf("writing to CSV: level %d \n", level);
        //const float* raw_hog = pyramid.levels()[level].data()->data();        
        const uint8_t* raw_hog = pyramid.levels()[level].bits(); 
        int width = pyramid.levels()[level].width();
        int height = pyramid.levels()[level].height();
        int depth = pyramid.NbChannels;
        ostringstream fname;
        fname << "../ffld_results/level" << level << ".csv"; //TODO: get orig img name into the CSV name.
    
        //writeCsv_3d_Hog_Float(raw_hog, width, height, depth, fname.str()); //FIXME: make a version of this function that takes uint8_t data
    }
}

//assumes NbChannels == 3
void writePyraToJPG(JPEGPyramid pyramid){
    int nlevels = pyramid.levels().size();

    for(int level = 0; level < nlevels; level++){
        ostringstream fname;
        fname << "../pyra_results/level" << level << ".jpg"; //TODO: get orig img name into the JPEG name.
        //cout << fname.str() << endl;

        pyramid.levels()[level].save(fname.str());
    }
}

void writePatchworkToJPG(Patchwork patchwork){
    int nlevels = patchwork.planes_.size();

    for(int level = 0; level < nlevels; level++){
        ostringstream fname;
        fname << "../stitched_results/level" << level << ".jpg"; //TODO: get orig img name into the JPEG name.
        //cout << fname.str() << endl;

        patchwork.planes_[level].save(fname.str());
    }
}
