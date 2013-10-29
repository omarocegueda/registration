#ifndef ECQMMF_REGCPP_H
#define ECQMMF_REGCPP_H
#define EPSILON 1e-9
int updateRegistrationConstantModels(double *fixed, double *moving, double *probs, int nrows, int ncols, int nclasses, 
                                    double *meansFixed, double *meansMoving, double *variancesFixed, double *variancesMoving, double *tbuffer);
void computeMaskedWeightedImageClassStats(int *mask, double *img, double *probs, int *dims, int *labels, double *means, double *variances, double *tbuffer);
void integrateRegistrationProbabilisticWeightedTensorFieldProductsCPP(double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw);
int computeRegistrationNegLogLikelihoodConstantModels(double *fixed, double *moving, int nrows, int ncols, int nclasses, 
                                    double *meansFixed, double *meansMoving, double *negLogLikelihood);
int initializeCoupledConstantModels(double *probsFixed, double *probsMoving, int *dims, double *meansMoving);
double optimizeECQMMFDisplacementField2DCPP(double *deltaField, double *gradientField, double *probs, int *dims, double lambda2, double *displacementField, double *residual, int maxIter, double tolerance);
#endif
