#include <string.h>
#include "bitsCPP.h"
#include "ecqmmf_regCPP.h"
/*
    Updates the means and variances of the fixed and moving images by maximizing
    the ECQMMF energy, leaving the transformation fixed
*/
int updateRegistrationConstantModels(double *fixed, double *moving, double *probs, int nrows, int ncols, int nclasses, 
                                    double *meansFixed, double *meansMoving, double *variancesFixed, double *variancesMoving, double *tbuffer){
    int nsites=nrows*ncols;
    int tableSize=nclasses*nclasses;
    memset(tbuffer, 0, sizeof(double)*nclasses*nclasses);
    memset(meansFixed, 0, sizeof(double)*nclasses);
    memset(meansMoving, 0, sizeof(double)*nclasses);
    memset(variancesFixed, 0, sizeof(double)*nclasses);
    memset(variancesMoving, 0, sizeof(double)*nclasses);
    double *p=probs;
    for(int x=0;x<nsites;++x, p+=tableSize){
        double *q=tbuffer;
        for(int i=0;i<nclasses;++i, p+=nclasses, q+=nclasses){
            for(int j=0;j<nclasses;++j){
                double p2=p[j]*p[j];
                meansFixed[i]+=fixed[x]*p2;
                meansMoving[j]+=moving[x]*p2;
                q[j]+=p2;
            }
        }
    }
    p=probs;
    for(int x=0;x<nsites;++x, p+=tableSize){
        double *q=tbuffer;
        for(int i=0;i<nclasses;++i, p+=nclasses, q+=nclasses){
            for(int j=0;j<nclasses;++j){
                double p2=p[j]*p[j];
                double diff=(fixed[x]-meansFixed[i]);
                variancesFixed[i]+=diff*diff*p2;
                diff=(moving[x]-meansMoving[j]);
                variancesMoving[j]+=diff*diff*p2;
            }
        }
    }
    //--------compute left statistics----------
    double *q=tbuffer;
    for(int i=0;i<nclasses;++i, q+=nclasses){
        double sj=0;
        for(int j=0;j<nclasses;++j){
            sj+=q[j];
        }
        meansFixed[i]/=sj;
        variancesFixed[i]/=sj;
    }
    //--------compute right statistics--------
    for(int j=0;j<nclasses;++j){
        q=tbuffer+j;
        double si=0;
        for(int i=0;i<nclasses;++i, q+=nclasses){
            si+=q[i];
        }
        meansMoving[j]/=si;
        variancesMoving[j]/=si;
    }
    
    return 0;
}

/*
    Computes the linear system whose solution is the optimal estimator of the
    registration parameters (here we assume a linear model)
*/
void integrateRegistrationProbabilisticWeightedTensorFieldProductsCPP(double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw){
    int k=dims[2];
    int nsites=dims[0]*dims[1];
    double *AA=new double[k*k*nclasses];
    double *bb=new double[k*nclasses];
    memset(AA, 0, sizeof(double)*k*k*nclasses);
    memset(bb, 0, sizeof(double)*k*nclasses);
    double *qq=q;
    double *pp=probs;
    int tableSize=nclasses*nclasses;
    for(int pos=0;pos<nsites;++pos, qq+=k, pp+=tableSize){
        for(int idx=0;idx<nclasses;++idx){
            double marginal=0;
            //---compute the 'idx'-th "marginal prob" at site 'pos'--
            double *p=pp+idx;
            for(int i=0;i<nclasses;++i,p+=nclasses){
                marginal+=(*p)*(*p);
            }
            //-----------------------------------------------------
            double *A=&AA[k*k*idx];
            double *b=&bb[k*idx];
            for(int i=0;i<k;++i){
                b[i]+=marginal*qq[i]*diff[pos];
                for(int j=0;j<k;++j){
                    A[k*i+j]+=marginal*qq[i]*qq[j];
                }
            }
        }
    }
    memset(Aw, 0, sizeof(double)*k*k);
    memset(bw, 0, sizeof(double)*k);
    double *A=AA;
    double *b=bb;
    for(int c=0;c<nclasses;++c, A+=(k*k), b+=k){
        for(int i=0;i<k;++i){
            bw[i]+=weights[c]*b[i];
            for(int j=0;j<k;++j){
                Aw[i*k+j]+=weights[c]*A[i*k+j];
            }
        }
    }
    delete[] AA;
    delete[] bb;
}

/*
    Computes the negative log-likelihood (a divergence measure) of the k^2 
    registration models
*/
int computeRegistrationNegLogLikelihoodConstantModels(double *fixed, double *moving, double *probs, int nrows, int ncols, int nclasses, 
                                    double *meansFixed, double *meansMoving, double *variancesFixed, double *variancesMoving, double *negLogLikelihood){
    int nsites=nrows*ncols;
    int tableSize=nclasses*nclasses;
    double *v=negLogLikelihood;
    for(int x=0;x<nsites;++x, v+=tableSize){
        int k=0;
        for(int i=0;i<nclasses;++i){
            for(int j=0;j<nclasses;++j,++k){
                if(isInfinite(meansFixed[i])||isInfinite(meansMoving[j])){
                    v[k]=INF64;
                    continue;
                }
                double diffFixed=meansFixed[i]-fixed[x];
                double diffMoving=meansMoving[j]-moving[x];
                double fixedTerm, movingTerm;
                if(variancesFixed[i]<EPSILON){
                    if((diffFixed*diffFixed)<EPSILON){
                        fixedTerm=0;
                    }else{
                        fixedTerm=INF64;
                    }
                }else{
                    fixedTerm=(diffFixed*diffFixed)/(2.0*variancesFixed[i]);
                }      
                if(variancesMoving[j]<EPSILON){
                    if((diffMoving*diffMoving)<EPSILON){
                        movingTerm=0;
                    }else{
                        movingTerm=INF64;
                    }
                }else{
                    movingTerm=(diffMoving*diffMoving)/(2.0*variancesMoving[j]);
                }
                if(isInfinite(fixedTerm)||isInfinite(movingTerm)){
                    v[k]=INF64;
                }else{
                    v[k]=fixedTerm+movingTerm;
                }
            }
        }
    }
    return 0;
}
