#ifndef ECQMMF_H
#define ECQMMF_H
//#define ESTIMATE_VARIANCES
const int NEIGH_SIZE=4;
const int dRow[]={-1, 0, 1, 0};
const int dCol[]={0, 1, 0, -1};
#define EPSILON 1e-9
double updateConstantModels(double *img, double *probs, int nrows, int ncols, int nclasses, double *means, double *variances);
int updateVariances(double *img, double *probs, int nrows, int ncols, int nclasses, double *means, double *variances);
double iterateMarginalsAt(int row, int col, double *likelihood, double *probs, int nrows, int ncols, int nclasses, double lambda, double mu, double *N, double *D, double *prev);
double iterateMarginals(double *likelihood, double *probs, int nrows, int ncols, int nclasses, double lambda, double mu, double *N, double *D, double *prev);
int computeNegLogLikelihoodConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances, double *likelihood);
int initializeConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances);
int initializeMaximumLikelihoodProbs(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs);
int initializeNormalizedLikelihood(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs);
int getImageModes(double *probs, int nrows, int ncols, int nclasses, double *means, double *modes);
#endif