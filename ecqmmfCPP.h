#ifndef ECQMMF_H
#define ECQMMF_H
int updateConstantModels(double *img, double *probs, int nrows, int ncols, int nclasses, double *means, double *variances);
int iterateMarginalsAt(int row, int col, double *likelihood, double *probs, int nrows, int ncols, int nclasses, double lambda, double mu, double *N, double *D);
int iterateMarginals(double *likelihood, double *probs, int nrows, int ncols, int nclasses, double lambda, double mu, double *N, double *D);
int computeNegLogLikelihoodConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances, double *likelihood);
int initializeConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances);
int initializeMaximumLikelihoodProbs(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs);
int initializeNormalizedLikelihood(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs);
int getImageModes(double *probs, int nrows, int ncols, int nclasses, double *means, double *modes);
#endif