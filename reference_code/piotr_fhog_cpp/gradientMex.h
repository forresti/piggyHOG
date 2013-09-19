#ifndef __GRADIENT_MEX_H__
#define __GRADIENT_MEX_H__

//mGradMag() -> gradMag()
void gradMag(float *I, float *M, float *O, int h, int w, int d, bool full);

//mGradHist() -> fhog()
void fhog(float *M, float *O, float *H, int h, int w, int binSize,
          int nOrients, int softBin, float clip);


#endif

