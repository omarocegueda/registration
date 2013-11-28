/*# -*- coding: utf-8 -*-
Created on Fri Sep 20 19:08:47 2013

@author: khayyam
*/
void integrateTensorFieldProductsCPP(double *q, int *dims, double *diff, double *A, double *b);
void quantizeImageCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist);
void quantizePositiveImageCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist);
void quantizeVolumeCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist);
void quantizePositiveVolumeCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist);
void computeImageClassStatsCPP(double *v, int *dims, int numLabels, int *labels, double *means, double *variances);
void computeMaskedImageClassStatsCPP(int *mask, double *v, int *dims, int numLabels, int *labels, double *means, double *variances);
void computeVolumeClassStatsCPP(double *v, int *dims, int numLabels, int *labels, double *means, double *variances);
void computeMaskedVolumeClassStatsCPP(int *mask, double *v, int *dims, int numLabels, int *labels, double *means, double *variances);
void integrateWeightedTensorFieldProductsCPP(double *q, int *dims, double *diff, int numLabels, int *labels, double *weights, double *Aw, double *bw);
void integrateMaskedWeightedTensorFieldProductsCPP(int *mask, double *q, int *dims, double *diff, int numLabels, int *labels, double *weights, double *Aw, double *bw);

double iterateDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *previousDisplacement, double *displacementField, double *residual);
double iterateDisplacementField3DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *previousDisplacement, double *displacementField, double *residual);

void computeMaskedVolumeClassStatsProbsCPP(int *mask, double *img, int *dims, int nclasses, double *probs, double *means, double *variances);
void integrateMaskedWeightedTensorFieldProductsProbsCPP(int *mask, double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw);
double iterateMaskedDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *mask, int *dims, double lambdaParam, double *previousDisplacement, double *displacementField, double *residual);

int invertVectorField(double *d, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *invd, double *stats);
int invertVectorFieldFixedPoint(double *d, int nrows, int ncols, int maxIter, double tolerance, double *invd, double *stats);
int composeVectorFields(double *d1, double *d2, int nrows, int ncols, double *comp, double *stats);
int vectorFieldExponential(double *v, int nrows, int ncols, double *expv, double *invexpv);

int readDoubleBuffer(char *fname, int nDoubles, double *buffer);
int writeDoubleBuffer(double *buffer, int nDoubles, char *fname);
void createInvertibleDisplacementField(int nrows, int ncols, double b, double m, double *dField);
int invertVectorFieldYan(double *forward, int nrows, int ncols, int maxloop, double tolerance, double *inv);

void countSupportingDataPerPixel(double *forward, int nrows, int ncols, int *counts);
int vectorFieldAdjointInterpolation(double *d1, double *d2, int nrows, int ncols, double *sol);
int vectorFieldInterpolation(double *d1, double *d2, int nrows, int ncols, double *comp);
int invertVectorField_TV_L2(double *forward, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *inv);

void consecutiveLabelMap(int *v, int n, int *out);
int composeVectorFields3D(double *d1, double *d2, int nslices, int nrows, int ncols, double *comp);
int vectorFieldExponential3D(double *v, int nslices, int nrows, int ncols, double *expv, double *invexpv);