#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "bitsCPP.h"
#include "ecqmmf_regCPP.h"
#include "hungarian.h"

void solve2DSymmetricPositiveDefiniteSystem(double *A, double *y, double *x, double *residual){
    double den=(A[0]*A[2]-A[1]*A[1]);
    x[1]=(A[0]*y[1]-A[1]*y[0])/den;
    x[0]=(y[0]-A[1]*x[1])/A[0];
    if(residual!=NULL){
        double r0=A[0]*x[0]+A[1]*x[1] - y[0];
        double r1=A[1]*x[0]+A[2]*x[1] - y[1];
        *residual=sqrt(r0*r0+r1*r1);
    }
}


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


void computeMaskedImageClassStatsCPP(int *mask, double *img, double *probs, int *dims, int *labels, double *means, double *variances, double *tbuffer){
    int numPixels=dims[0]*dims[1];
    int numLabels=dims[2];    
    memset(means, 0, sizeof(double)*numLabels);
    memset(variances, 0, sizeof(double)*numLabels);
    memset(tbuffer, 0, sizeof(double)*numLabels);
    double *p=probs;
    for(int i=0;i<numPixels;++i, p+=numLabels)if(mask[i]!=0){
        for(int k=0;k<numLabels;++k){
            double p2=p[k]*p[k];
            means[k]+=img[i]*p2;
            tbuffer[k]+=p2;
        }
    }
    for(int i=0;i<numLabels;++i){
        if(tbuffer[i]>EPSILON){
            means[i]/=tbuffer[i];
        }else{
            means[i]=INF64;
        }
    }
    p=probs;
    for(int i=0;i<numPixels;++i, p+=numLabels)if(mask[i]!=0){
        for(int k=0;k<numLabels;++k){
            double p2=p[k]*p[k];
            double diff=(img[i]-means[k])*p2;
            variances[k]+=diff*diff;
        }
    }
    for(int i=0;i<numLabels;++i){
        if(tbuffer[i]>EPSILON){
            variances[i]/=tbuffer[i];
        }else{
            variances[i]=INF64;
        }
    }
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
/*int computeRegistrationNegLogLikelihoodConstantModels_SQUARE(double *fixed, double *moving, double *probs, int nrows, int ncols, int nclasses, 
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
*/

int computeRegistrationNegLogLikelihoodConstantModels(double *fixed, double *moving, int nrows, int ncols, int nclasses, 
                                    double *meansFixed, double *meansMoving, double *negLogLikelihood){
    int nsites=nrows*ncols;
    double *v=negLogLikelihood;    
    for(int i=0;i<nsites;++i, v+=nclasses){
        for(int k=0;k<nclasses;++k){
            if(isInfinite(meansFixed[k])||isInfinite(meansMoving[k])){
                v[k]=INF64;
                continue;
            }
            double diffFixed=meansFixed[k]-fixed[i];
            double diffMoving=meansMoving[k]-moving[i];
#ifndef ESTIMATE_VARIANCES
            v[k]=(diffFixed*diffFixed)+(diffMoving*diffMoving);
#else
            if(variances[k]<EPSILON){
                if((diff*diff)<EPSILON){
                    v[k]=0;
                }else{
                    v[k]=INF64;
                }
            }else{
                v[k]=(diff*diff)/(2.0*variances[k]);
            }      
#endif
        }
    }
    return 0;
}

double iterateECQMMFDisplacementField2DCPP(double *deltaField, double *gradientField, double *probs, int *dims, double lambda2, double *displacementField, double *residual){
    const static int numNeighbors=4;
    const static int dRow[]={-1, 0, 1,  0, -1, 1,  1, -1};
    const static int dCol[]={ 0, 1, 0, -1,  1, 1, -1, -1};
    int nrows=dims[0];
    int ncols=dims[1];
    int nclasses=dims[2];
    double *d=displacementField;
    double *g=gradientField;
    double y[2];
    double A[3];
    int pos=0;
    double maxDisplacement=0;
    double *delta=deltaField;
    double *p=probs;
    for(int r=0;r<nrows;++r){
        for(int c=0;c<ncols;++c, ++pos, d+=2, g+=2, delta+=nclasses, probs+=nclasses){
            double expectedDelta=0;
            double sump2=0;
            for(int k=0;k<nclasses;++k){
                double p2=p[k]*p[k];
                sump2+=p2;
                expectedDelta+=delta[k]*p2;
            }
            int nn=0;
            y[0]=y[1]=0;
            for(int k=0;k<numNeighbors;++k){
                int dr=r+dRow[k];
                if((dr<0) || (dr>=nrows)){
                    continue;
                }
                int dc=c+dCol[k];
                if((dc<0) || (dc>=ncols)){
                    continue;
                }
                ++nn;
                double *dneigh=&displacementField[2*(dr*ncols + dc)];
                y[0]+=dneigh[0];
                y[1]+=dneigh[1];
            }
            y[0]=(expectedDelta*g[0]) + lambda2*y[0];
            y[1]=(expectedDelta*g[1]) + lambda2*y[1];
            A[0]=sump2*g[0]*g[0] + lambda2*nn;
            A[1]=sump2*g[0]*g[1];
            A[2]=sump2*g[1]*g[1] + lambda2*nn;
            double xx=d[0];
            double yy=d[1];
            solve2DSymmetricPositiveDefiniteSystem(A,y,d, &residual[pos]);
            xx-=d[0];
            yy-=d[1];
            double opt=xx*xx+yy*yy;
            if(maxDisplacement<opt){
                maxDisplacement=opt;
            }//if
        }//cols
    }//rows
    return sqrt(maxDisplacement);
}

double optimizeECQMMFDisplacementField2DCPP(double *deltaField, double *gradientField, double *probs, int *dims, double lambda2, double *displacementField, double *residual, int maxIter, double tolerance){
    double maxDisplacement=0;
    for(int iter=0;iter<maxIter;++iter){
        double maxDisplacement=iterateECQMMFDisplacementField2DCPP(deltaField, gradientField, probs, dims, lambda2, displacementField, residual);
        if(maxDisplacement<tolerance){
            break;
        }
    }
    return maxDisplacement;
}


int initializeCoupledConstantModels(double *probsFixed, double *probsMoving, int *dims, double *meansMoving){
    int nrows=dims[0];
    int ncols=dims[1];
    int nclasses=dims[2];
    double *joint=new double[nclasses*nclasses];
    memset(joint, 0, sizeof(double)*nclasses*nclasses);
    double *p=probsFixed;
    double *q=probsMoving;
    for(int r=0;r<nrows;++r){
        for(int c=0;c<ncols;++c, p+=nclasses, q+=nclasses){
            double *J=joint;
            for(int i=0;i<nclasses;++i){
                for(int j=0;j<nclasses;++j,++J){
                    J[0]+=p[i]*q[j];
                }
            }
        }
    }
    Hungarian H(nclasses);
    double *J=joint;
    for(int i=0;i<nclasses;++i){
        for(int j=0;j<nclasses;++j, ++J){
            H.setCost(i,j,1-*J);
        }
    }
    H.solve();
    int *assignment=new int[nclasses];
    H.getAssignment(assignment);
    memcpy(joint, meansMoving, sizeof(double)*nclasses);//recycle the joint buffer
    for(int i=0;i<nclasses;++i){
        meansMoving[i]=joint[assignment[i]];
    }
    delete[] assignment;
    delete[] joint;
    return 0;
}
