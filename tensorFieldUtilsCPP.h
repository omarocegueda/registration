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

double computeDemonsStep2D(double *deltaField, double *gradientField, int *dims, double maxStepSize, double scale, double *demonsStep);
double computeDemonsStep3D(double *deltaField, double *gradientField, int *dims, double maxStepSize, double scale, double *demonsStep);

double iterateDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField, double *residual);
double computeEnergySSD2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField);
double iterateDisplacementField3DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField, double *residual);
double computeEnergySSD3DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField);
double iterateResidualDisplacementFieldSSD2D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField);
int computeResidualDisplacementFieldSSD2D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField, double *residual);
double iterateResidualDisplacementFieldSSD3D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField);
int computeResidualDisplacementFieldSSD3D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField, double *residual);

void computeMaskedVolumeClassStatsProbsCPP(int *mask, double *img, int *dims, int nclasses, double *probs, double *means, double *variances);
void integrateMaskedWeightedTensorFieldProductsProbsCPP(int *mask, double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw);
double iterateMaskedDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *mask, int *dims, double lambdaParam, double *displacementField, double *residual);

int invertVectorField(double *d, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *invd, double *stats);
int invertVectorFieldFixedPoint(double *d, int nr1, int nc1, int nr2, int nc2, int maxIter, double tolerance, double *invd, double *start, double *stats);
int invertVectorFieldFixedPoint3D(double *d, int ns1, int nr1, int nc1, int ns2, int nr2, int nc2, int maxIter, double tolerance, double *invd, double *start, double *stats);
int composeVectorFields(double *d1, int nr1, int nc1, double *d2, int nr2, int nc2, double *comp, double *stats);
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
int composeVectorFields3D(double *d1, int ns1, int nr1, int nc1, double *d2, int ns2, int nr2, int nc2, double *comp, double *stats);
int vectorFieldExponential3D(double *v, int nslices, int nrows, int ncols, double *expv, double *invexpv);

int upsampleDisplacementField(double *d1, int nrows, int ncols, double *up, int nr, int nc);
int upsampleDisplacementField3D(double *d1, int ns, int nr, int nc, double *up, int nslices, int nrows, int ncols);
int accumulateUpsampleDisplacementField3D(double *d1, int nslices, int nrows, int ncols, double *current, int ns, int nr, int nc);

int downsampleDisplacementField(double *d1, int nr, int nc, double *down);
int downsampleScalarField(double *d1, int nr, int nc, double *down);
int downsampleDisplacementField3D(double *d1, int ns, int nr, int nc, double *down);
int downsampleScalarField3D(double *d1, int ns, int nr, int nc, double *down);

int warpImageAffine(double *img, int nrImg, int ncImg, double *affine, double *warped, int nrRef, int ncRef);
int warpImage(double *img, int nrImg, int ncImg, double *d1, int nrows, int ncols, double *affinePre, double *affinePost, double *warped);
int warpImageNN(double *img, int nrImg, int ncImg, double *d1, int nrows, int ncols, double *affinePre, double *affinePost, double *warped);
int warpDiscreteImageNNAffine(int *img, int nrImg, int ncImg, double *affine, int *warped, int nrRef, int ncRef);
int warpDiscreteImageNN(int *img, int nrImg, int ncImg, double *d1, int nrows, int ncols, double *affinePre, double *affinePost, int *warped);

int warpVolumeAffine(double *volume, int nsVol, int nrVol, int ncVol, double *affine, double *warped, int nsRef, int nrRef, int ncRef);
int multVectorFieldByAffine3D(double *displacement, int nslices, int nrows, int ncols, double *affine);
int warpVolume(double *volume, int nsVol, int nrVol, int ncVol, double *d1, int nslices, int nrows, int ncols, double *affinePre, double *affinePost, double *warped);
int warpVolumeNN(double *volume, int nsVol, int nrVol, int ncVol, double *d1, int nslices, int nrows, int ncols, double *affinePre, double *affinePost, double *warped);
int warpDiscreteVolumeNNAffine(int *volume, int nsVol, int nrVol, int ncVol, double *affine, int *warped, int nsRef, int nrRef, int ncRef);
int warpDiscreteVolumeNN(int *volume, int nsVol, int nrVol, int ncVol, double *d1, int nslices, int nrows, int ncols, double *affinePre, double *affinePost, int *warped);
int invertVectorField3D(double *forward, int nslices, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *inv, double *stats);
int prependAffineToDisplacementField2D(double *d1, int nrows, int ncols, double *affine);
int prependAffineToDisplacementField3D(double *d1, int nslices, int nrows, int ncols, double *affine);
int appendAffineToDisplacementField2D(double *d1, int nrows, int ncols, double *affine);
int appendAffineToDisplacementField3D(double *d1, int nslices, int nrows, int ncols, double *affine);

void getVotingSegmentation(int *votes, int nslices, int nrows, int ncols, int nvotes, int *seg);
int getDisplacementRange(double *d, int nslices, int nrows, int ncols, double *affine, double *minVal, double *maxVal);
int computeJacard(int *A, int *B, int nslices, int nrows, int ncols, double *jacard, int nlabels);
int precomputeCCFactors3D(double *I, double *J, int ns, int nr, int nc, int radius, double *factors);
int computeCCForwardStep3D(double *gradFixed, double *gradMoving, int ns, int nr, int nc, double *factors, double *out);
int computeCCBackwardStep3D(double *gradFixed, double *gradMoving, int ns, int nr, int nc, double *factors, double *out);
