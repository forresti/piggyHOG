#ifndef __GRADIENT_MEX_H__
#define __GRADIENT_MEX_H__

void gradMag(float *I, float *M, float *O, int h, int w, int d, bool full);

void fhog(float *M, float *O, float *H, int h, int w, int binSize,
          int nOrients, int softBin, float clip);


#endif

