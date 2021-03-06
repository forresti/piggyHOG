
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
    OPT_HELP, OPT_PADDING, OPT_INTERVAL, OPT_OUTPUT_STITCHED_DIR
};

CSimpleOpt::SOption SOptions[] =
{
	{ OPT_HELP, "-h", SO_NONE },
	{ OPT_HELP, "--help", SO_NONE },
	{ OPT_PADDING, "-p", SO_REQ_SEP },
	{ OPT_PADDING, "--padding", SO_REQ_SEP },
	{ OPT_INTERVAL, "-e", SO_REQ_SEP },
	{ OPT_INTERVAL, "--interval", SO_REQ_SEP },
    { OPT_OUTPUT_STITCHED_DIR, "--output-stitched-dir", SO_REQ_SEP },
	SO_END_OF_OPTIONS
};

void showUsage(){
	cout << "Usage: test [options] image.jpg, or\n       test [options] image_set.txt\n\n"
			"Options:\n"
			"  -h,--help               Display this information\n"
    		"  -p,--padding <arg>      Amount of zero padding in JPEG images (default 8)\n"
			"  -e,--interval <arg>     Number of levels per octave in the JPEG pyramid (default 10)\n"
            "  --output-stitched-dir <arg>   Where to save stitched pyramids (default ../stitched_results)\n"
		 << endl;
}

// Parse command line parameters
//   put the appropriate values in (padding, interval, file) based on cmd-line args
void parseArgs(int &padding, int &interval, string &file, string &output_stitched_dir, int argc, char * argv[]){
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
            else if (args.OptionId() == OPT_OUTPUT_STITCHED_DIR) {
                output_stitched_dir = args.OptionArg();	
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

//e.g. file = ../../images_640x480/carsgraz_001.image.jpg
void TEST_file_parsing(string file){
    size_t lastSlash = file.find_last_of("/\\");
    size_t lastDot = file.find_last_of('.');
    cout << "    file.substr(lastDot) = " << file.substr(lastDot) << endl; // .jpg
    cout << "    file.substr(lastSlash) = " << file.substr(lastSlash) << endl; // /carsgraz_001.image.jpg
    cout << "    file.substr(lastSlash, lastDot) = " << file.substr(lastSlash, lastDot-lastSlash) << endl; // /carsgraz_001.image
}

//e.g. file = ../../images_640x480/carsgraz_001.image.jpg
string parse_base_filename(string file){
    size_t lastSlash = file.find_last_of("/\\");
    size_t lastDot = file.find_last_of('.');

    string base_filename = file.substr(lastSlash, lastDot-lastSlash); // /carsgraz_001.image
    return base_filename;
}

void printScaleSizes(JPEGPyramid pyramid);
void writePyraToJPG(JPEGPyramid pyramid);
void writePatchworkToJPG(Patchwork patchwork, string output_stitched_dir, string base_filename);

// Test a mixture model (compute a ROC curve)
int main(int argc, char * argv[]){
	// Default parameters
    string file;
    string output_stitched_dir = "../stitched_results";
	int padding = 8;
	int interval = 10;

    //parseArgs params are passed by reference, so they get updated here
    parseArgs(padding, interval, file, output_stitched_dir, argc, argv); //update parameters with any command-line inputs
    string base_filename = parse_base_filename(file);

    printf("    padding = %d \n", padding);
    printf("    interval = %d \n", interval);
    printf("    file = %s \n", file.c_str());
    printf("    base_filename = %s \n", base_filename.c_str());
    printf("    output_stitched_dir = %s \n", output_stitched_dir.c_str());
 
	JPEGImage image(file);
    if (image.empty()) {
        showUsage();
        cerr << "\nInvalid image " << file << endl;
        return -1;
    }

    //image = image.resize(image.width()*4, image.height()*4); //UPSAMPLE so that Caffe's 16x downsampling looks like 4x downsampling
    image = image.resize(image.width()*2, image.height()*2); //UPSAMPLE so that Caffe's 16x downsampling looks like 8x downsampling

  // Compute the downsample+stitch
    double start_downsample = read_timer();    

    JPEGPyramid pyramid(image, padding, padding, interval); //DOWNSAMPLE with (padx == pady == padding)

    if (pyramid.empty()) {
        showUsage();
        cerr << "\nInvalid image " << file << endl;
        return -1;
    }
    
    double time_downsample = read_timer() - start_downsample;
    cout << "  Multiscale downsampling in " << time_downsample << " ms" << endl;

    double start_stitch = read_timer();

    int planeWidth = (pyramid.levels()[0].width() + 15) & ~15; //TODO: don't subtract padx, pady? 
    int planeHeight = (pyramid.levels()[0].height() + 15) & ~15; 
    planeWidth = max(planeWidth, planeHeight);  //SQUARE planes for Caffe convnet
    planeHeight = max(planeWidth, planeHeight);

    Patchwork::Init(planeHeight, planeWidth); 
    const Patchwork patchwork(pyramid); //STITCH

    double time_stitch = read_timer() - start_stitch;
    cout << "  Stitched scales in " << time_stitch << " ms" << endl;

    //printScaleSizes(pyramid);
    //writePyraToJPG(pyramid);
    writePatchworkToJPG(patchwork, output_stitched_dir, base_filename); //outputs to output_stitched_dir/base_filename_[planeID].jpg

   	return EXIT_SUCCESS;
}

void printScaleSizes(JPEGPyramid pyramid){
    int nlevels = pyramid.levels().size();

    for(int level = 0; level < nlevels; level++){ 
        int width = pyramid.levels()[level].width();
        int height = pyramid.levels()[level].height();
        int depth = pyramid.NbChannels;
        printf("        level %d: width=%d, height=%d, depth=%d \n", level, width, height, depth);
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

void writePatchworkToJPG(Patchwork patchwork, string output_stitched_dir, string base_filename){
    int nplanes = patchwork.planes_.size();

    for(int planeID = 0; planeID < nplanes; planeID++){
        ostringstream fname;
        fname << output_stitched_dir << "/" << base_filename << "_plane" << planeID << ".jpg";
        //cout << fname.str() << endl;

        patchwork.planes_[planeID].save(fname.str());
    }
}

