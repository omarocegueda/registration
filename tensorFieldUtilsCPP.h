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