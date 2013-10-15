#include <string.h>
#include <math.h>
#include "bitsCPP.h"
#include "ecqmmfCPP.h"
//-------------------------------------------------------------------
int updateConstantModels(double *img, double *probs, int nrows, int ncols, int nclasses, double *means, double *variances){
    int nsites=nrows*ncols;
    double *sump2=new double[nclasses];
    double *p=probs;
    memset(means, 0, sizeof(double)*nclasses);
    memset(variances, 0, sizeof(double)*nclasses);
    memset(sump2, 0, sizeof(double)*nclasses);
    for(int i=0;i<nsites;++i, p+=nclasses){
        for(int k=0;k<nclasses;++k){
            double p2=p[k]*p[k];
            means[k]+=p2*img[i];
            sump2[k]+=p2;
        }
    }
    for(int k=0;k<nclasses;++k){
        if(sump2[k]<EPSILON){
            means[k]=-1;
        }else{
            means[k]/=sump2[k];
        }
    }
#ifdef ESTIMATE_VARIANCES    
    p=probs;
    for(int i=0;i<nsites;++i, p+=nclasses){
        for(int k=0;k<nclasses;++k){
            double p2=p[k]*p[k];
            double diff=(img[i]-means[k]);
            variances[k]+=p2*diff*diff;
        }
    }
    for(int k=0;k<nclasses;++k){
        if(sump2[k]<EPSILON){
            variances[k]=0;
        }else{
            variances[k]/=sump2[k];
        }
    }
#endif    
    delete[] sump2;
    return 0;
}

int iterateMarginalsAt(int row, int col, double *negLogLikelihood, double *probs, int nrows, int ncols, 
                        int nclasses, double lambda, double mu, double *N, double *D){
    double *v=&negLogLikelihood[nclasses*(row*ncols+col)];
    for(int k=0;k<nclasses;++k){
        if(v[k]<EPSILON){
            double *p=&probs[nclasses*(row*ncols+col)];
            memset(p, 0, sizeof(double)*nclasses);
            p[k]=1;
            return 0;
        }
        double num=0;
        int cnt=0;
        for(int n=0;n<NEIGH_SIZE;++n){
            int rr=row+dRow[n];
            int cc=col+dCol[n];
            if((rr<0)||(cc<0)||(rr>=nrows)||(cc>=ncols)){
                continue;
            }
            double *p=&probs[nclasses*(rr*ncols+cc)];
            num+=p[k];
            ++cnt;
        }
        N[k]=lambda*num;
        if(isInfinite(v[k])){
            D[k]=INF64;
        }else{
            D[k]=v[k] - mu + lambda*cnt;
        }
    }
    double sNum=0, sDen=0;
    for(int k=0;k<nclasses;k++){
        if(!isInfinite(D[k])){
            sNum+=N[k]/D[k];
            sDen+=1.0/D[k];
        }
    }
    double *p=&probs[nclasses*(row*ncols+col)];
    double ss=0;
    for(int k=0;k<nclasses;k++){
        if(isInfinite(D[k])){
            p[k]=0;
            continue;
        }
        p[k]=(1-sNum)/(D[k]*sDen) + N[k]/D[k];
        if(p[k]<0){
            p[k]=0;
        }            
        if(p[k]>1){
            p[k]=1;
        }
        ss+=p[k];
    }
    for(int k=0;k<nclasses;k++){
        p[k]/=ss;
    }
    return 0;
}

int iterateMarginals(double *likelihood, double *probs, int nrows, int ncols, 
                        int nclasses, double lambda, double mu, double *N, double *D){
    int nsites=nrows*ncols;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j){
            iterateMarginalsAt(i,j,likelihood, probs, nrows, ncols, nclasses, lambda, mu, N, D);   
        }
    }
    return 0;
}

int computeNegLogLikelihoodConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances, double *likelihood){
    int nsites=nrows*ncols;
    double *v=likelihood;    
    for(int i=0;i<nsites;++i, v+=nclasses){
        for(int k=0;k<nclasses;++k){
            if(isInfinite(means[k])){
                v[k]=INF64;
                continue;
            }
            double diff=means[k]-img[i];
#ifndef ESTIMATE_VARIANCES
            v[k]=(diff*diff);            
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

int initializeConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances){
    double maxVal=img[0];
    double minVal=img[0];
    int nsites=nrows*ncols;
    for(int i=0;i<nsites;++i){
        if(img[i]<minVal){
            minVal=img[i];
        }
        if(img[i]>maxVal){
            maxVal=img[i];
        }
    }
    double delta=(maxVal-minVal)/nclasses;
    if(delta<EPSILON){//dynamic range too small
        means[0]=minVal;
        variances[0]=0;
        for(int k=1;k<nclasses;++k){
            means[k]=INF64;
            variances[k]=0;
        }
        return 0;
    }
    //--compute means and variances--
    delta+=EPSILON;//ensure the maximum quantization value is less than nclasses
    int *cnt=new int[nclasses];
    memset(cnt,0, sizeof(int)*nclasses);
    memset(means,0, sizeof(double)*nclasses);
    memset(variances,0, sizeof(double)*nclasses);
    for(int i=0;i<nsites;++i){
        int closest=(int)((img[i]-minVal)/delta);
        means[closest]+=img[i];
        variances[closest]+=img[i]*img[i];
        cnt[closest]++;
    }
    for(int i=0;i<nclasses;++i){
        if(cnt[i]<1){
            means[i]=INF64;
            variances[i]=0;
            continue;
        }
        means[i]/=cnt[i];
#ifndef ESTIMATE_VARIANCES
        variances[i]=1;
#else
        variances[i]=variances[i]/cnt[i] - means[i]*means[i];
#endif
    }

    delete[] cnt;
    return 0;
}

int initializeMaximumLikelihoodProbs(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs){
    int nsites=nrows*ncols;
    double *v=negLogLikelihood;
    double *p=probs;
    memset(p, 0, sizeof(double)*nsites*nclasses);
    for(int i=0;i<nsites;++i, v+=nclasses, p+=nclasses){
        int sel=0;
        for(int k=1;k<nclasses;++k){
            if(v[k]<v[sel]){
                sel=k;
            }
        }
        p[sel]=1;
    }
    return 0;
}

int initializeNormalizedLikelihood(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs){
    int nsites=nrows*ncols;
    double *v=negLogLikelihood;
    double *p=probs;
    for(int i=0;i<nsites;++i, v+=nclasses, p+=nclasses){
        double sum=0;
        for(int k=0;k<nclasses;++k){
            if(isInfinite(v[k])){
                p[k]=0;
            }else{
                p[k]=exp(-v[k]);
            }            
            sum+=p[k];
        }
        for(int k=0;k<nclasses;++k){
            p[k]/=sum;
        }
    }
    return 0;
}

int getImageModes(double *probs, int nrows, int ncols, int nclasses, double *means, double *modes){
    int nsites=nrows*ncols;
    double *p=probs;
    for(int i=0;i<nsites;++i, p+=nclasses){
        int sel=0;
        for(int k=1;k<nclasses;++k){
            if(p[sel]<p[k]){
                sel=k;
            }
        }
        modes[i]=means[sel];
    }
    return 0;
}
