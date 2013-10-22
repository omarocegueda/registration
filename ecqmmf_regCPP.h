#ifndef ECQMMF_REGCPP_H
#define ECQMMF_REGCPP_H
#define EPSILON 1e-9
int updateRegistrationConstantModels(double *fixed, double *moving, double *probs, int nrows, int ncols, int nclasses, 
                                    double *meansFixed, double *meansMoving, double *variancesFixed, double *variancesMoving, double *tbuffer);
void integrateRegistrationProbabilisticWeightedTensorFieldProductsCPP(double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw);
int computeRegistrationNegLogLikelihoodConstantModels(double *fixed, double *moving, double *probs, int nrows, int ncols, int nclasses, 
                                    double *meansFixed, double *meansMoving, double *variancesFixed, double *variancesMoving, double *negLogLikelihood);
#endif
