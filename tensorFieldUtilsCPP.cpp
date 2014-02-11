/*# -*- coding: utf-8 -*-
Created on Fri Sep 20 19:03:32 2013

@author: khayyam
*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <set>
#include <map>
#include "bitsCPP.h"
#include "derivatives.h"
#include "tensorFieldUtilsCPP.h"
#include "macros.h"
using namespace std;

void integrateTensorFieldProductsCPP(double *q, int *dims, double *diff, double *A, double *b){
    int k=dims[3];
    int nvox=dims[0]*dims[1]*dims[2];
    memset(A, 0, sizeof(double)*k*k);
    memset(b, 0, sizeof(double)*k);
    double *qq=q;
    for(int pos=0;pos<nvox;++pos, qq+=k){
        for(int i=0;i<k;++i){
            b[i]+=qq[i]*diff[pos];
            for(int j=i;j<k;++j){
                A[k*i+j]+=qq[i]*qq[j];
            }
        }
    }
    for(int i=1;i<k;++i){
        for(int j=0;j<i;++j){
            A[k*i+j]=A[k*j+i];
        }
    }
}

void quantizeImageCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist){
    if(numLevels<1){
        return;
    }
    memset(hist, 0, sizeof(int)*numLevels);
    memset(levels, 0, sizeof(double)*numLevels);
    double minVal=v[0];
    double maxVal=v[0];
    int npix=dims[0]*dims[1];
    for(int i=0;i<npix;++i){
        if(v[i]<minVal){
            minVal=v[i];
        }else if(v[i]>maxVal){
            maxVal=v[i];
        }
    }
    const double epsilon=1e-8;
    double delta=(maxVal-minVal+epsilon)/numLevels;
    if((numLevels<2)||(delta<epsilon)){
        memset(out, 0, sizeof(double)*npix);
        levels[0]=0.5*(minVal+maxVal);
        hist[0]=npix;
        return;
    }
    levels[0]=delta*0.5;
    for(int i=1;i<numLevels;++i){
        levels[i]=levels[i-1]+delta;
    }
    for(int i=0;i<npix;++i){
        int l=(v[i]-minVal)/delta;
        out[i]=l;
        hist[l]++;
    }
}

void quantizePositiveImageCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist){
    --numLevels;//zero is one of the levels
    if(numLevels<1){
        return;
    }
    memset(hist, 0, sizeof(int)*(numLevels+1));
    memset(levels, 0, sizeof(double)*(numLevels+1));
    double minVal=-1;
    double maxVal=-1;
    int npix=dims[0]*dims[1];
    for(int i=0;i<npix;++i)if(v[i]>0){
        if((minVal<0) || (v[i]<minVal)){
            minVal=v[i];
        }
        if(v[i]>maxVal){
            maxVal=v[i];
        }
    }
    const double epsilon=1e-8;
    double delta=(maxVal-minVal+epsilon)/numLevels;
    if((numLevels<2)||(delta<epsilon)){
        for(int i=0;i<npix;++i){
            if(v[i]>0){
                out[i]=1;
            }else{
                out[i]=0;
                ++hist[0];
            }
        }
        levels[0]=0;
        levels[1]=0.5*(minVal+maxVal);
        hist[1]=npix-hist[0];
        return;
    }
    levels[0]=0;
    levels[1]=delta*0.5;
    for(int i=2;i<=numLevels;++i){
        levels[i]=levels[i-1]+delta;
    }
    for(int i=0;i<npix;++i){
        if(v[i]>0){
            int l=(v[i]-minVal)/delta;
            out[i]=l+1;
            hist[l+1]++;
        }else{
            out[i]=0;
            hist[0]++;
        }
    }
}



void quantizeVolumeCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist){
    if(numLevels<1){
        return;
    }
    memset(hist, 0, sizeof(int)*numLevels);
    memset(levels, 0, sizeof(double)*numLevels);
    double minVal=v[0];
    double maxVal=v[0];
    int nvox=dims[0]*dims[1]*dims[2];
    for(int i=0;i<nvox;++i){
        if(v[i]<minVal){
            minVal=v[i];
        }else if(v[i]>maxVal){
            maxVal=v[i];
        }
    }
    const double epsilon=1e-8;
    double delta=(maxVal-minVal+epsilon)/numLevels;
    if((numLevels<2)||(delta<epsilon)){
        memset(out, 0, sizeof(double)*nvox);
        levels[0]=0.5*(minVal+maxVal);
        hist[0]=nvox;
        return;
    }
    levels[0]=delta*0.5;
    for(int i=1;i<numLevels;++i){
        levels[i]=levels[i-1]+delta;
    }
    for(int i=0;i<nvox;++i){
        int l=(v[i]-minVal)/delta;
        out[i]=l;
        hist[l]++;
    }
}

void quantizePositiveVolumeCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist){
    --numLevels;//zero is one of the levels
    if(numLevels<1){
        return;
    }
    memset(hist, 0, sizeof(int)*(numLevels+1));
    memset(levels, 0, sizeof(double)*(numLevels+1));
    double minVal=-1;
    double maxVal=-1;
    int nvox=dims[0]*dims[1]*dims[2];
    for(int i=0;i<nvox;++i)if(v[i]>0){
        if((minVal<0) || (v[i]<minVal)){
            minVal=v[i];
        }
        if(v[i]>maxVal){
            maxVal=v[i];
        }
    }
    const double epsilon=1e-8;
    double delta=(maxVal-minVal+epsilon)/numLevels;
    if((numLevels<2)||(delta<epsilon)){
        for(int i=0;i<nvox;++i){
            if(v[i]>0){
                out[i]=1;
            }else{
                out[i]=0;
                ++hist[0];
            }
        }
        levels[0]=0;
        levels[1]=0.5*(minVal+maxVal);
        hist[1]=nvox-hist[0];
        return;
    }
    levels[0]=0;
    levels[1]=delta*0.5;
    for(int i=2;i<=numLevels;++i){
        levels[i]=levels[i-1]+delta;
    }
    for(int i=0;i<nvox;++i){
        if(v[i]>0){
            int l=(v[i]-minVal)/delta;
            out[i]=l+1;
            hist[l+1]++;
        }else{
            out[i]=0;
            hist[0]++;
        }
    }
}

void computeImageClassStatsCPP(double *v, int *dims, int numLabels, int *labels, double *means, double *variances){
    memset(means, 0, sizeof(double)*numLabels);
    memset(variances, 0, sizeof(double)*numLabels);
    int *counts=new int[numLabels];
    memset(counts, 0, sizeof(int)*numLabels);
    int numPixels=dims[0]*dims[1];
    for(int i=0;i<numPixels;++i){
        means[labels[i]]+=v[i];
        variances[labels[i]]+=v[i]*v[i];
        counts[labels[i]]++;
    }
    for(int i=0;i<numLabels;++i){
        if(counts[i]>0){
            means[i]/=counts[i];
        }
        if(counts[i]>1){
            variances[i]=variances[i]/counts[i]-means[i]*means[i];
        }else{
            variances[i]=INF64;
        }
    }
    delete[] counts;
}

void computeMaskedImageClassStatsCPP(int *mask, double *v, int *dims, int numLabels, int *labels, double *means, double *variances){
    memset(means, 0, sizeof(double)*numLabels);
    memset(variances, 0, sizeof(double)*numLabels);
    int *counts=new int[numLabels];
    memset(counts, 0, sizeof(int)*numLabels);
    int numPixels=dims[0]*dims[1];
    for(int i=0;i<numPixels;++i)if(mask[i]!=0){
        means[labels[i]]+=v[i];
        variances[labels[i]]+=v[i]*v[i];
        counts[labels[i]]++;
    }
    for(int i=0;i<numLabels;++i){
        if(counts[i]>0){
            means[i]/=counts[i];
        }
        if(counts[i]>1){
            variances[i]=variances[i]/counts[i]-means[i]*means[i];
        }else{
            variances[i]=INF64;
        }
    }
    delete[] counts;
}

void computeVolumeClassStatsCPP(double *v, int *dims, int numLabels, int *labels, double *means, double *variances){
    memset(means, 0, sizeof(double)*numLabels);
    memset(variances, 0, sizeof(double)*numLabels);
    int *counts=new int[numLabels];
    memset(counts, 0, sizeof(int)*numLabels);
    int numVoxels=dims[0]*dims[1]*dims[2];
    for(int i=0;i<numVoxels;++i){
        means[labels[i]]+=v[i];
        variances[labels[i]]+=v[i]*v[i];
        counts[labels[i]]++;
    }
    for(int i=0;i<numLabels;++i){
        if(counts[i]>0){
            means[i]/=counts[i];
        }
        if(counts[i]>1){
            variances[i]=variances[i]/counts[i]-means[i]*means[i];
        }else{
            variances[i]=INF64;
        }
    }
    delete[] counts;
}

void computeMaskedVolumeClassStatsCPP(int *mask, double *v, int *dims, int numLabels, int *labels, double *means, double *variances){
    memset(means, 0, sizeof(double)*numLabels);
    memset(variances, 0, sizeof(double)*numLabels);
    int *counts=new int[numLabels];
    memset(counts, 0, sizeof(int)*numLabels);
    int numVoxels=dims[0]*dims[1]*dims[2];
    for(int i=0;i<numVoxels;++i)if(mask[i]!=0){
        means[labels[i]]+=v[i];
        variances[labels[i]]+=v[i]*v[i];
        counts[labels[i]]++;
    }
    for(int i=0;i<numLabels;++i){
        if(counts[i]>0){
            means[i]/=counts[i];
        }
        if(counts[i]>1){
            variances[i]=variances[i]/counts[i]-means[i]*means[i];
        }else{
            variances[i]=INF64;
        }
    }
    delete[] counts;
}


void integrateWeightedTensorFieldProductsCPP(double *q, int *dims, double *diff, int numLabels, int *labels, double *weights, double *Aw, double *bw){
    int k=dims[3];
    int nvox=dims[0]*dims[1]*dims[2];
    double *AA=new double[k*k*numLabels];
    double *bb=new double[k*numLabels];
    memset(AA, 0, sizeof(double)*k*k*numLabels);
    memset(bb, 0, sizeof(double)*k*numLabels);
    double *qq=q;
    for(int pos=0;pos<nvox;++pos, qq+=k){
        int idx=labels[pos];
        double *A=&AA[k*k*idx];
        double *b=&bb[k*idx];
        for(int i=0;i<k;++i){
            b[i]+=qq[i]*diff[pos];
            for(int j=0;j<k;++j){
                A[k*i+j]+=qq[i]*qq[j];
            }
        }
    }
    memset(Aw, 0, sizeof(double)*k*k);
    memset(bw, 0, sizeof(double)*k);
    double *A=AA;
    double *b=bb;
    for(int c=0;c<numLabels;++c, A+=(k*k), b+=k){
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

void integrateMaskedWeightedTensorFieldProductsCPP(int *mask, double *q, int *dims, double *diff, int numLabels, int *labels, double *weights, double *Aw, double *bw){
    int k=dims[3];
    int nvox=dims[0]*dims[1]*dims[2];
    double *AA=new double[k*k*numLabels];
    double *bb=new double[k*numLabels];
    memset(AA, 0, sizeof(double)*k*k*numLabels);
    memset(bb, 0, sizeof(double)*k*numLabels);
    double *qq=q;
    for(int pos=0;pos<nvox;++pos, qq+=k)if(mask[pos]!=0){
        int idx=labels[pos];
        double *A=&AA[k*k*idx];
        double *b=&bb[k*idx];
        for(int i=0;i<k;++i){
            b[i]+=qq[i]*diff[pos];
            for(int j=0;j<k;++j){
                A[k*i+j]+=qq[i]*qq[j];
            }
        }
    }
    memset(Aw, 0, sizeof(double)*k*k);
    memset(bw, 0, sizeof(double)*k);
    double *A=AA;
    double *b=bb;
    for(int c=0;c<numLabels;++c, A+=(k*k), b+=k){
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

void solve3DSymmetricPositiveDefiniteSystem(double *A, double *y, double *x, double *residual){
    double a=A[0], b=A[1], c=A[2];
    double d=(a*A[3]-b*b)/a;
    double e=(a*A[4]-b*c)/a;
    double f=(a*A[5]-c*c)/a - (e*e*a)/(a*A[3]-b*b);
    double y0=y[0];
    double y1=(y[1]*a-y0*b)/a;
    double y2=(y[2]*a-A[2]*y0)/a - (e*(y[1]*a-b*y0))/(a*A[3]-b*b);
    x[2]=y2/f;
    x[1]=(y1-e*x[2])/d;
    x[0]=(y0-b*x[1]-c*x[2])/a;
    if(residual!=NULL){
        double r0=A[0]*x[0]+A[1]*x[1]+A[2]*x[2] - y[0];
        double r1=A[1]*x[0]+A[3]*x[1]+A[4]*x[2] - y[1];
        double r2=A[2]*x[0]+A[4]*x[1]+A[5]*x[2] - y[2];
        *residual=sqrt(r0*r0+r1*r1+r2*r2);
    }
}

double computeDemonsStep2D(double *deltaField, double *gradientField, int *dims, double maxStepSize, double scale, double *demonsStep){
    int nrows=dims[0];
    int ncols=dims[1];
    double *g=gradientField;
    double *res=demonsStep;
    int pos=0;
    double maxDisplacement=0;
    memset(demonsStep, 0, sizeof(double)*2*nrows*ncols);
    for(int r=0;r<nrows;++r){
        for(int c=0;c<ncols;++c, ++pos, g+=2, res+=2){
            double nrm2=g[0]*g[0]+g[1]*g[1];
            double delta=deltaField[pos];
            double den=(nrm2 + delta*delta);
            double factor=0;
            if(den!=0){
                factor=delta/(nrm2 + delta*delta);
            }
            res[0]=factor*g[0];
            res[1]=factor*g[1];
            nrm2=res[0]*res[0]+res[1]*res[1];
            if(maxDisplacement<nrm2){
                maxDisplacement=nrm2;
            }
        }//cols
    }//rows
    maxDisplacement=sqrt(maxDisplacement);
    double factor=maxStepSize/maxDisplacement;
    for(int p=2*nrows*ncols-1;p>=0;--p){
        demonsStep[p]*=factor;
    }
    return maxDisplacement;
}

double computeDemonsStep3D(double *deltaField, double *gradientField, int *dims, double maxStepSize, double scale, double *demonsStep){
    int nslices=dims[0];
    int nrows=dims[1];
    int ncols=dims[2];
    double *g=gradientField;
    double *res=demonsStep;
    int pos=0;
    double maxDisplacement=0;
    memset(demonsStep, 0, sizeof(double)*3*nslices*nrows*ncols);
    for(int s=0;s<nslices;++s){
        for(int r=0;r<nrows;++r){
            for(int c=0;c<ncols;++c, ++pos, g+=3, res+=3){
                double nrm2=g[0]*g[0]+g[1]*g[1]+g[2]*g[2];
                double delta=deltaField[pos];
                double den=(nrm2 + delta*delta);
                double factor=0;
                if(den!=0){
                    factor=delta/(nrm2 + delta*delta);
                }
                res[0]=factor*g[0];
                res[1]=factor*g[1];
                res[2]=factor*g[2];
                nrm2=res[0]*res[0]+res[1]*res[1]+res[2]*res[2];
                if(maxDisplacement<nrm2){
                    maxDisplacement=nrm2;
                }
            }//cols
        }//rows
    }//slices
    maxDisplacement=sqrt(maxDisplacement);
    double factor=maxStepSize/maxDisplacement;
    for(int p=3*nslices*nrows*ncols-1;p>=0;--p){
        demonsStep[p]*=factor;
    }
    return maxDisplacement;
}

double iterateDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField, double *residual){
    const static int numNeighbors=4;
    const static int dRow[]={-1, 0, 1,  0, -1, 1,  1, -1};
    const static int dCol[]={ 0, 1, 0, -1,  1, 1, -1, -1};
    int nrows=dims[0];
    int ncols=dims[1];
    double *d=displacementField;
    double *g=gradientField;
    double y[2];
    double A[3];
    int pos=0;
    double maxDisplacement=0;
    for(int r=0;r<nrows;++r){
        for(int c=0;c<ncols;++c, ++pos, d+=2, g+=2){
            double delta=deltaField[pos];
            double sigma=(sigmaField!=NULL)?sigmaField[pos]:1;
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
            if(isInfinite(sigma)){
                double xx=d[0];
                double yy=d[1];
                d[0]=y[0]/nn;
                d[1]=y[1]/nn;
                xx-=d[0];
                yy-=d[1];
                double opt=xx*xx+yy*yy;
                if(maxDisplacement<opt){
                    maxDisplacement=opt;
                }
                if(residual!=NULL){
                    residual[pos]=0;
                }
            }else if (sigma<1e-9){
                    double nrm2=g[0]*g[0]+g[1]*g[1];
                    if(nrm2<1e-9){
                        d[0]=d[1]=0;
                    }else{
                        d[0]=(g[0]*delta)/nrm2;
                        d[1]=(g[1]*delta)/nrm2;
                    }
            }else{
                y[0]=(delta*g[0]) + sigma*lambdaParam*y[0];
                y[1]=(delta*g[1]) + sigma*lambdaParam*y[1];
                A[0]=g[0]*g[0] + sigma*lambdaParam*(nn);
                A[1]=g[0]*g[1];
                A[2]=g[1]*g[1] + sigma*lambdaParam*(nn);
                double xx=d[0];
                double yy=d[1];
                if(residual!=NULL){
                    solve2DSymmetricPositiveDefiniteSystem(A,y,d, &residual[pos]);
                }else{
                    solve2DSymmetricPositiveDefiniteSystem(A,y,d, NULL);
                }
                xx-=d[0];
                yy-=d[1];
                double opt=xx*xx+yy*yy;
                if(maxDisplacement<opt){
                    maxDisplacement=opt;
                }
            }//if
        }//cols
    }//rows
    return sqrt(maxDisplacement);
}

double iterateResidualDisplacementFieldSSD2D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField){
    const static int numNeighbors=4;
    const static int dRow[]={-1, 0, 1,  0, -1, 1,  1, -1};
    const static int dCol[]={ 0, 1, 0, -1,  1, 1, -1, -1};
    double commonRes[2]={0,0};
    double *b=target;
    int offsetb=2;
    if(b==NULL){
        offsetb=0;
        b=commonRes;
    }
    int nrows=dims[0];
    int ncols=dims[1];
    double *d=displacementField;
    double *g=gradientField;
    double y[2];
    double A[3];
    int pos=0;
    double maxDisplacement=0;
    for(int r=0;r<nrows;++r){
        for(int c=0;c<ncols;++c, ++pos, d+=2, g+=2, b+=offsetb){
            double delta=deltaField[pos];
            double sigma=(sigmaField!=NULL)?sigmaField[pos]:1;
            if(target==NULL){
                b[0]=delta*g[0];
                b[1]=delta*g[1];
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
            if(isInfinite(sigma)){
                double xx=d[0];
                double yy=d[1];
                d[0]=y[0]/nn;
                d[1]=y[1]/nn;
                xx-=d[0];
                yy-=d[1];
                double opt=xx*xx+yy*yy;
                if(maxDisplacement<opt){
                    maxDisplacement=opt;
                }
            }else if (sigma==0){
                    double nrm2=g[0]*g[0]+g[1]*g[1];
                    if(nrm2==0){
                        d[0]=d[1]=0;
                    }else{
                        d[0]=(b[0])/nrm2;
                        d[1]=(b[1])/nrm2;
                    }
            }else{
                y[0]=b[0] + sigma*lambdaParam*y[0];
                y[1]=b[1] + sigma*lambdaParam*y[1];
                A[0]=g[0]*g[0] + sigma*lambdaParam*(nn);
                A[1]=g[0]*g[1];
                A[2]=g[1]*g[1] + sigma*lambdaParam*(nn);
                double xx=d[0];
                double yy=d[1];
                solve2DSymmetricPositiveDefiniteSystem(A,y,d, NULL);
                xx-=d[0];
                yy-=d[1];
                double opt=xx*xx+yy*yy;
                if(maxDisplacement<opt){
                    maxDisplacement=opt;
                }
            }//if
        }//cols
    }//rows
    return sqrt(maxDisplacement);
}

int computeResidualDisplacementFieldSSD2D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField, double *residual){
    const static int numNeighbors=4;
    const static int dRow[]={-1, 0, 1,  0, -1, 1,  1, -1};
    const static int dCol[]={ 0, 1, 0, -1,  1, 1, -1, -1};
    double commonRes[2]={0,0};
    double *b=target;
    double *res=residual;
    int offsetRes=2;
    if(b==NULL){
        offsetRes=0;
        b=commonRes;
    }
    int nrows=dims[0];
    int ncols=dims[1];
    double *d=displacementField;
    double *g=gradientField;
    double y[2];
    int pos=0;
    for(int r=0;r<nrows;++r){
        for(int c=0;c<ncols;++c, ++pos, d+=2, g+=2, res+=2, b+=offsetRes){
            double delta=deltaField[pos];
            double sigma=(sigmaField!=NULL)?sigmaField[pos]:1;
            if(target==NULL){
                b[0]=delta*g[0];
                b[1]=delta*g[1];
            }
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
                double *dneigh=&displacementField[2*(dr*ncols + dc)];
                y[0]+=d[0]-dneigh[0];
                y[1]+=d[1]-dneigh[1];
            }
            if(isInfinite(sigma)){
                res[0]=-lambdaParam*y[0];
                res[1]=-lambdaParam*y[1];
            }else{
                double dotP=g[0]*d[0]+g[1]*d[1];
                res[0]=b[0]-(g[0]*dotP+sigma*lambdaParam*y[0]);
                res[1]=b[1]-(g[1]*dotP+sigma*lambdaParam*y[1]);
            }//if
        }//cols
    }//rows
    return 0;
}

double computeEnergySSD2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField){
    int nrows=dims[0];
    int ncols=dims[1];
    double *d=displacementField;
    double *g=gradientField;
    int pos=0;
    double energy=0;
    for(int r=0;r<nrows;++r){
        for(int c=0;c<ncols;++c, ++pos, d+=2, g+=2){
            double delta=deltaField[pos];
            /*double sigma=(sigmaField!=NULL)?sigmaField[pos]:1;
            double dotp=d[0]*g[0]+d[1]*g[1];
            double localEnergy=0;
            if(r>0){
                double *nd=&displacementField[2*((r-1)*ncols+c)];
                double dst=(nd[0]-d[0])*(nd[0]-d[0])+(nd[1]-d[1])*(nd[1]-d[1]);
                localEnergy+=dst;
            }
            if(c>0){
                double *nd=&displacementField[2*(r*ncols+c-1)];
                double dst=(nd[0]-d[0])*(nd[0]-d[0])+(nd[1]-d[1])*(nd[1]-d[1]);
                localEnergy+=dst;
            }
            localEnergy=0.5*lambdaParam*localEnergy;
            if((!isInfinite(sigma)) && (sigma>0)){
                localEnergy+=0.5*(delta-dotp)*(delta-dotp)/sigma;
            }*/
            double localEnergy=delta*delta;
            energy+=localEnergy;
        }//cols
    }//rows
    return energy;
}

double computeEnergySSD3DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField){
    int nslices=dims[0];
    int nrows=dims[1];
    int ncols=dims[2];
    int sliceSize=nrows*ncols;
    double *d=displacementField;
    double *g=gradientField;
    int pos=0;
    double energy=0;
    for(int s=0;s<nslices;++s){
        for(int r=0;r<nrows;++r){
            for(int c=0;c<ncols;++c, ++pos, d+=3, g+=3){
                double delta=deltaField[pos];
                /*double sigma=(sigmaField!=NULL)?sigmaField[pos]:1;
                double dotp=d[0]*g[0]+d[1]*g[1]+d[2]*g[2];
                double localEnergy=0;
                if(s>0){
                    double *nd=&displacementField[3*((s-1)*sliceSize+r*ncols+c)];
                    double dst=(nd[0]-d[0])*(nd[0]-d[0])+(nd[1]-d[1])*(nd[1]-d[1])+(nd[2]-d[2])*(nd[2]-d[2]);
                    localEnergy+=dst;
                }
                if(r>0){
                    double *nd=&displacementField[3*(s*sliceSize+(r-1)*ncols+c)];
                    double dst=(nd[0]-d[0])*(nd[0]-d[0])+(nd[1]-d[1])*(nd[1]-d[1])+(nd[2]-d[2])*(nd[2]-d[2]);
                    localEnergy+=dst;
                }
                if(c>0){
                    double *nd=&displacementField[3*(s*sliceSize+r*ncols+c-1)];
                    double dst=(nd[0]-d[0])*(nd[0]-d[0])+(nd[1]-d[1])*(nd[1]-d[1])+(nd[2]-d[2])*(nd[2]-d[2]);
                    localEnergy+=dst;
                }
                localEnergy=0.5*lambdaParam*localEnergy;
                if((!isInfinite(sigma)) && (sigma>0)){
                    localEnergy+=0.5*(delta-dotp)*(delta-dotp)/sigma;
                }*/
                double localEnergy=delta*delta;
                energy+=localEnergy;
            }//cols
        }//rows
    }
    return energy;
}

double iterateMaskedDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *mask, int *dims, double lambdaParam, double *displacementField, double *residual){
    const static int numNeighbors=4;
    const static int dRow[]={-1, 0, 1,  0, -1, 1,  1, -1};
    const static int dCol[]={ 0, 1, 0, -1,  1, 1, -1, -1};
    int nrows=dims[0];
    int ncols=dims[1];
    double *d=displacementField;
    double *g=gradientField;
    double y[2];
    double A[3];
    int pos=0;
    double maxDisplacement=0;
    int *mm=mask;
    for(int r=0;r<nrows;++r){
        for(int c=0;c<ncols;++c, ++pos, d+=2, g+=2, ++mm){
            if(!*mm){
                d[0]=d[1]=0;
                continue;
            }
            double delta=deltaField[pos];
            double sigma=sigmaField[pos];
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
                if(mask[dr*ncols+dc]==0){
                    continue;
                }
                ++nn;
                double *dneigh=&displacementField[2*(dr*ncols + dc)];
                y[0]+=dneigh[0];
                y[1]+=dneigh[1];
            }
            if(nn<2){
                d[0]=d[1]=0;
                continue;
            }
            if(isInfinite(sigma)){
                double xx=d[0];
                double yy=d[1];
                d[0]=y[0]/nn;
                d[1]=y[1]/nn;
                xx-=d[0];
                yy-=d[1];
                double opt=xx*xx+yy*yy;
                if(maxDisplacement<opt){
                    maxDisplacement=opt;
                }
                residual[pos]=0;
            }else{
                y[0]=(delta*g[0]) + sigma*lambdaParam*y[0];
                y[1]=(delta*g[1]) + sigma*lambdaParam*y[1];
                A[0]=g[0]*g[0] + sigma*lambdaParam*(nn);
                A[1]=g[0]*g[1];
                A[2]=g[1]*g[1] + sigma*lambdaParam*(nn);
                double xx=d[0];
                double yy=d[1];
                solve2DSymmetricPositiveDefiniteSystem(A,y,d, &residual[pos]);
                xx-=d[0];
                yy-=d[1];
                double opt=xx*xx+yy*yy;
                if(maxDisplacement<opt){
                    maxDisplacement=opt;
                }
            }//if
        }//cols
    }//rows
    return sqrt(maxDisplacement);
}


double iterateDisplacementField3DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField, double *residual){
    const static int numNeighbors=6;
    const static int dSlice[numNeighbors]={-1,  0, 0, 0,  0, 1};
    const static int dRow[numNeighbors]  ={ 0, -1, 0, 1,  0, 0};
    const static int dCol[numNeighbors]  ={ 0,  0, 1, 0, -1, 0};
    int nslices=dims[0];
    int nrows=dims[1];
    int ncols=dims[2];
    double *d=displacementField;
    double *g=gradientField;
    int sliceSize=ncols*nrows;
    double y[3];
    double A[6];
    int pos=0;
    double maxDisplacement=0;
    for(int s=0;s<nslices;++s){
        for(int r=0;r<nrows;++r){
            for(int c=0;c<ncols;++c, ++pos, d+=3, g+=3){
                double delta=deltaField[pos];
                double sigma=sigmaField[pos];
                int nn=0;
                y[0]=y[1]=y[2]=0;
                for(int k=0;k<numNeighbors;++k){
                    int ds=s+dSlice[k];
                    if((ds<0) || (ds>=nslices)){
                        continue;
                    }
                    int dr=r+dRow[k];
                    if((dr<0) || (dr>=nrows)){
                        continue;
                    }
                    int dc=c+dCol[k];
                    if((dc<0) || (dc>=ncols)){
                        continue;
                    }
                    ++nn;
                    double *dneigh=&displacementField[3*(ds*sliceSize + dr*ncols + dc)];
                    y[0]+=dneigh[0];
                    y[1]+=dneigh[1];
                    y[2]+=dneigh[2];
                }
                if(isInfinite(sigma)){
                    double xx=d[0];
                    double yy=d[1];
                    double zz=d[2];
                    d[0]=y[0]/nn;
                    d[1]=y[1]/nn;
                    d[2]=y[2]/nn;
                    xx-=d[0];
                    yy-=d[1];
                    zz-=d[2];
                    double opt=xx*xx+yy*yy+zz*zz;
                    if(maxDisplacement<opt){
                        maxDisplacement=opt;
                    }
                    if(residual!=NULL){
                        residual[pos]=0;
                    }
                    
                }else if (sigma==0){
                    double nrm2=g[0]*g[0]+g[1]*g[1]+g[2]*g[2];
                    if(nrm2==0){
                        d[0]=d[1]=d[2]=0;
                    }else{
                        d[0]=(g[0]*delta)/nrm2;
                        d[1]=(g[1]*delta)/nrm2;
                        d[2]=(g[2]*delta)/nrm2;
                    }
                }else{
                    y[0]=(delta*g[0]) + sigma*lambdaParam*y[0];
                    y[1]=(delta*g[1]) + sigma*lambdaParam*y[1];
                    y[2]=(delta*g[2]) + sigma*lambdaParam*y[2];
                    A[0]=g[0]*g[0] + sigma*lambdaParam*nn;
                    A[1]=g[0]*g[1];
                    A[2]=g[0]*g[2];
                    A[3]=g[1]*g[1] + sigma*lambdaParam*nn;
                    A[4]=g[1]*g[2];
                    A[5]=g[2]*g[2] + sigma*lambdaParam*nn;
                    double xx=d[0];
                    double yy=d[1];
                    double zz=d[2];
                    if(residual!=NULL){
                        solve3DSymmetricPositiveDefiniteSystem(A,y,d, &residual[pos]);
                    }else{
                        solve3DSymmetricPositiveDefiniteSystem(A,y,d, NULL);
                    }
                    xx-=d[0];
                    yy-=d[1];
                    zz-=d[2];
                    double opt=xx*xx+yy*yy+zz*zz;
                    if(maxDisplacement<opt){
                        maxDisplacement=opt;
                    }
                }
            }//cols
        }//rows
    }//slices
    return sqrt(maxDisplacement);
}

double iterateResidualDisplacementFieldSSD3D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField){
    const static int numNeighbors=6;
    const static int dSlice[numNeighbors]={-1,  0, 0, 0,  0, 1};
    const static int dRow[numNeighbors]  ={ 0, -1, 0, 1,  0, 0};
    const static int dCol[numNeighbors]  ={ 0,  0, 1, 0, -1, 0};
    double commonRes[3]={0,0,0};
    double *b=target;
    int offsetb=3;
    if(b==NULL){
        offsetb=0;
        b=commonRes;
    }
    int nslices=dims[0];
    int nrows=dims[1];
    int ncols=dims[2];
    int sliceSize=ncols*nrows;
    double *d=displacementField;
    double *g=gradientField;
    double y[3];
    double A[6];
    int pos=0;
    double maxDisplacement=0;
    for(int s=0;s<nslices;++s){
        for(int r=0;r<nrows;++r){
            for(int c=0;c<ncols;++c, ++pos, d+=3, g+=3, b+=offsetb){
                double delta=deltaField[pos];
                double sigma=(sigmaField!=NULL)?sigmaField[pos]:1;
                if(target==NULL){
                    b[0]=delta*g[0];
                    b[1]=delta*g[1];
                    b[2]=delta*g[2];
                }
                int nn=0;
                y[0]=y[1]=y[2]=0;
                for(int k=0;k<numNeighbors;++k){
                    int ds=s+dSlice[k];
                    if((ds<0) || (ds>=nslices)){
                        continue;
                    }
                    int dr=r+dRow[k];
                    if((dr<0) || (dr>=nrows)){
                        continue;
                    }
                    int dc=c+dCol[k];
                    if((dc<0) || (dc>=ncols)){
                        continue;
                    }
                    ++nn;
                    double *dneigh=&displacementField[3*(ds*sliceSize+dr*ncols + dc)];
                    y[0]+=dneigh[0];
                    y[1]+=dneigh[1];
                    y[2]+=dneigh[2];
                }
                if(isInfinite(sigma)){
                    double xx=d[0];
                    double yy=d[1];
                    double zz=d[2];
                    d[0]=y[0]/nn;
                    d[1]=y[1]/nn;
                    d[2]=y[2]/nn;
                    xx-=d[0];
                    yy-=d[1];
                    zz-=d[2];
                    double opt=xx*xx+yy*yy+zz*zz;
                    if(maxDisplacement<opt){
                        maxDisplacement=opt;
                    }
                }else if (sigma==0){
                        double nrm2=g[0]*g[0]+g[1]*g[1]+g[2]*g[2];
                        if(nrm2==0){
                            d[0]=d[1]=d[2]=0;
                        }else{
                            d[0]=(b[0])/nrm2;
                            d[1]=(b[1])/nrm2;
                            d[2]=(b[2])/nrm2;
                        }
                }else{
                    y[0]=b[0] + sigma*lambdaParam*y[0];
                    y[1]=b[1] + sigma*lambdaParam*y[1];
                    y[2]=b[2] + sigma*lambdaParam*y[2];
                    A[0]=g[0]*g[0] + sigma*lambdaParam*nn;
                    A[1]=g[0]*g[1];
                    A[2]=g[0]*g[2];
                    A[3]=g[1]*g[1] + sigma*lambdaParam*nn;
                    A[4]=g[1]*g[2];
                    A[5]=g[2]*g[2] + sigma*lambdaParam*nn;
                    double xx=d[0];
                    double yy=d[1];
                    double zz=d[2];
                    solve3DSymmetricPositiveDefiniteSystem(A,y,d, NULL);
                    xx-=d[0];
                    yy-=d[1];
                    zz-=d[2];
                    double opt=xx*xx+yy*yy+zz*zz;
                    if(maxDisplacement<opt){
                        maxDisplacement=opt;
                    }
                }//if
            }//cols
        }//rows
    }//slices
    return sqrt(maxDisplacement);
}

int computeResidualDisplacementFieldSSD3D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField, double *residual){
    const static int numNeighbors=6;
    const static int dSlice[numNeighbors]={-1,  0, 0, 0,  0, 1};
    const static int dRow[numNeighbors]  ={ 0, -1, 0, 1,  0, 0};
    const static int dCol[numNeighbors]  ={ 0,  0, 1, 0, -1, 0};
    double commonRes[3]={0,0,0};
    double *b=target;
    double *res=residual;
    int offsetRes=3;
    if(b==NULL){
        offsetRes=0;
        b=commonRes;
    }
    int nslices=dims[0];
    int nrows=dims[1];
    int ncols=dims[2];
    int sliceSize=ncols*nrows;
    double *d=displacementField;
    double *g=gradientField;
    double y[3];
    int pos=0;
    for(int s=0;s<nslices;++s){
        for(int r=0;r<nrows;++r){
            for(int c=0;c<ncols;++c, ++pos, d+=3, g+=3, res+=3, b+=offsetRes){
                double delta=deltaField[pos];
                double sigma=(sigmaField!=NULL)?sigmaField[pos]:1;
                if(target==NULL){
                    b[0]=delta*g[0];
                    b[1]=delta*g[1];
                    b[2]=delta*g[2];
                }
                y[0]=y[1]=y[2]=0;
                for(int k=0;k<numNeighbors;++k){
                    int ds=s+dSlice[k];
                    if((ds<0) || (ds>=nslices)){
                        continue;
                    }
                    int dr=r+dRow[k];
                    if((dr<0) || (dr>=nrows)){
                        continue;
                    }
                    int dc=c+dCol[k];
                    if((dc<0) || (dc>=ncols)){
                        continue;
                    }
                    double *dneigh=&displacementField[3*(ds*sliceSize+dr*ncols + dc)];
                    y[0]+=d[0]-dneigh[0];
                    y[1]+=d[1]-dneigh[1];
                    y[2]+=d[2]-dneigh[2];
                }
                if(isInfinite(sigma)){
                    res[0]=-lambdaParam*y[0];
                    res[1]=-lambdaParam*y[1];
                    res[2]=-lambdaParam*y[2];
                }else{
                    double dotP=g[0]*d[0]+g[1]*d[1]+g[2]*d[2];
                    res[0]=b[0]-(g[0]*dotP+sigma*lambdaParam*y[0]);
                    res[1]=b[1]-(g[1]*dotP+sigma*lambdaParam*y[1]);
                    res[2]=b[2]-(g[2]*dotP+sigma*lambdaParam*y[2]);
                }//if
            }//cols
        }//rows
    }
    return 0;
}



void computeMaskedVolumeClassStatsProbsCPP(int *mask, double *img, int *dims, int nclasses, double *probs, double *means, double *variances){
    const double EPSILON=1e-9;
    memset(means, 0, sizeof(double)*nclasses);
    memset(variances, 0, sizeof(double)*nclasses);
    double *sums=new double[nclasses];
    memset(sums, 0, sizeof(double)*nclasses);
    int nsites=dims[0]*dims[1];
    //---means---
    double *p=probs;
    for(int i=0;i<nsites;++i, p+=nclasses)if(mask[i]!=0){
        for(int k=0;k<nclasses;++k){
            double p2=p[k]*p[k];
            means[k]+=img[i]*p2;
            sums[k]+=p2;
        }
    }
    for(int k=0;k<nclasses;++k){
        if(sums[k]>EPSILON){
            means[k]/=sums[k];
        }
    }
    //---variances---
    p=probs;
    for(int i=0;i<nsites;++i, p+=nclasses)if(mask[i]!=0){
        for(int k=0;k<nclasses;++k){
            double p2=p[k]*p[k];
            variances[k]+=(img[i]-means[k])*(img[i]-means[k])*p2;
        }
    }
    for(int k=0;k<nclasses;++k){
        if(sums[k]>EPSILON){
            variances[k]/=sums[k];
        }else{
            variances[k]=INF64;
        }
    }
    delete[] sums;
}


void integrateMaskedWeightedTensorFieldProductsProbsCPP(int *mask, double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw){
    int k=dims[2];
    int nsites=dims[0]*dims[1];
    double *AA=new double[k*k*nclasses];
    double *bb=new double[k*nclasses];
    double *sums=new double[nclasses];
    memset(AA, 0, sizeof(double)*k*k*nclasses);
    memset(bb, 0, sizeof(double)*k*nclasses);
    memset(sums, 0, sizeof(double)*nclasses);
    double *qq=q;
    double *p=probs;
    //--------TO-DO: how to handle the p^2's ? --
    for(int pos=0;pos<nsites;++pos, qq+=k, p+=nclasses)if(mask[pos]!=0){
        for(int idx=0;idx<nclasses;++idx){
            double *A=&AA[k*k*idx];
            double *b=&bb[k*idx];
            double p2=p[idx]*p[idx];
            sums[idx]+=p2;
            for(int i=0;i<k;++i){
                b[i]+=p2*qq[i]*diff[pos];
                for(int j=0;j<k;++j){
                    A[k*i+j]+=p2*qq[i]*qq[j];
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
    delete[] sums;
}

double computeInverseEnergy_deprecated(double *d, double *invd, int nrows, int ncols, double lambda){
    double stats[3];
    double *residual=new double[2*nrows*ncols];
    composeVectorFields(d, nrows, ncols,invd, nrows, ncols, residual, stats);
    double energy=0;
    for(int i=0;i<nrows-1;++i){
        for(int j=0;j<ncols-1;++j){
            double d00=invd[2*(i*ncols+j)] - invd[2*((i+1)*ncols+j)];
            double d01=invd[2*(i*ncols+j)+1] - invd[2*((i+1)*ncols+j)+1];
            double d10=invd[2*(i*ncols+j)] - invd[2*(i*ncols+j+1)];
            double d11=invd[2*(i*ncols+j)+1] - invd[2*(i*ncols+j+1)+1];
            energy+=d00*d00+d01*d01+d10*d10+d11*d11;
        }    
    }
    energy*=lambda;
    double *r=residual;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, r+=2){
            energy+=r[0]+r[0]+r[1]*r[1];
        }    
    }
    delete[] residual;
    return energy;
    
}

double computeInverseEnergy(double *d, int nr1, int nc1, double *invd, int nr2, int nc2, double lambda){
    double stats[3];
    double *residual=new double[2*nr1*nc1];
    composeVectorFields(d, nr1, nc1, invd, nr2, nc2, residual, stats);
    double energy=0;
    for(int i=0;i<nr2-1;++i){
        for(int j=0;j<nc2-1;++j){
            double d00=invd[2*(i*nc2+j)] - invd[2*((i+1)*nc2+j)];
            double d01=invd[2*(i*nc2+j)+1] - invd[2*((i+1)*nc2+j)+1];
            double d10=invd[2*(i*nc2+j)] - invd[2*(i*nc2+j+1)];
            double d11=invd[2*(i*nc2+j)+1] - invd[2*(i*nc2+j+1)+1];
            energy+=d00*d00+d01*d01+d10*d10+d11*d11;
        }    
    }
    energy*=lambda;
    double *r=residual;
    for(int i=0;i<nr1;++i){
        for(int j=0;j<nc1;++j, r+=2){
            energy+=r[0]+r[0]+r[1]*r[1];
        }    
    }
    delete[] residual;
    return energy;
    
}

inline bool checkIsFinite(double *error, int nrows, int ncols, int r, int c){
    if((r<0)||(c<0)||(r>=nrows)||(c>=ncols)){
        return false;
    }
    if(isInfinite(error[r*ncols+c])){
        return false;
    }
    return true;
}

void findNearesFinite(double *error, int nrows, int ncols, int r, int c, int &rr, int &cc){
    int ring=1;
    while(true){
        rr=r-ring-1;
        cc=c-1;
        for(int k=0;k<=ring;++k){//top-right side of the diamond
            ++rr;
            ++cc;
            if(checkIsFinite(error, nrows, ncols, rr, cc)){
                return;
            }
        }
        for(int k=0;k<ring;++k){//bottom-right side of the diamond
            ++rr;
            --cc;
            if(checkIsFinite(error, nrows, ncols, rr, cc)){
                return;
            }
        }
        for(int k=0;k<ring;++k){//bottom-left side of the diamond
            --rr;
            --cc;
            if(checkIsFinite(error, nrows, ncols, rr, cc)){
                return;
            }
        }
        for(int k=0;k<ring-1;++k){//top-left side of the diamond
            --rr;
            ++cc;
            if(checkIsFinite(error, nrows, ncols, rr, cc)){
                return;
            }
        }
        ++ring;
    }

}

void interpolateDisplacementFieldAt(double *forward, int nrows, int ncols, double pr, double pc, double &fr, double &fc){
    int i=floor(pr);
    int j=floor(pc);
    if((i<0) || (j<0) || (i>=nrows) || (j>=ncols)){
        fr=0;
        fc=0;
        return;
    }
    double alphac=pr-i;
    double betac=pc-j;
    double alpha=1.0-alphac;
    double beta=1.0-betac;
    double *ff=&forward[2*(i*ncols+j)];
    fr=alpha*beta*ff[0];
    fc=alpha*beta*ff[1];
    ++j;
    if(j<ncols){
        ff=&forward[2*(i*ncols+j)];
        fr+=alpha*betac*ff[0];
        fc+=alpha*betac*ff[1];
    }
    ++i;
    if((i<nrows)&&(j<ncols)){
        ff=&forward[2*(i*ncols+j)];
        fr+=alphac*betac*ff[0];
        fc+=alphac*betac*ff[1];
    }
    --j;
    if(i<nrows){
        ff=&forward[2*(i*ncols+j)];
        fr+=alphac*beta*ff[0];
        fc+=alphac*beta*ff[1];
    }
}


int initializeNearestNeighborInverseField(double *forward, int nrows, int ncols, double *inv, double *error){
    const static int nRow[]={ 0, 0, 1, 1};
    const static int nCol[]={ 0, 1, 0, 1};
    bool needsDelete=false;
    if(error==NULL){
        needsDelete=true;
        error=new double[nrows*ncols];
    }
    // Step 0: initialize delta, MAXLOOP and inverse computing error
    for(int i=nrows*ncols-1;i>=0;--i){
        error[i]=INF64;
    }
    //Step 1a: Map points from p in R, assign e(q) for points q in S which are immediately adjacent to f(p) if assignment reduces e(q)
    double *f=forward;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, f+=2){//p=(i,j) in R
            double dii=i+f[0];
            double djj=j+f[1];//(dii, djj) = f(p)
            if((dii<0) || (djj<0) || (dii>nrows-1)||(djj>ncols-1)){//no one is affected
                continue;
            }
            //find the top left index and the interpolation coefficients
            int ii=floor(dii);
            int jj=floor(djj);
            //assign all grid points immediately adjacent to f(p)
            for(int k=0;k<4;++k){
                int iii=ii+nRow[k];
                int jjj=jj+nCol[k];//(iii, jjj)=q is a surrounding point
                if((iii<0)||(jjj<0)||(iii>=nrows)||(jjj>=ncols)){
                    continue;//the point is outside the lattice
                }
                double dr=dii-iii;
                double dc=djj-jjj;//(dr,dc) = f(p) - q
                double opt=sqrt(dr*dr+dc*dc);//||q-f(p)||
                if(opt<error[iii*ncols+jjj]){//if(||q-f(p)||^2 < e(q))
                    double *dq=&inv[2*(iii*ncols+jjj)];
                    dq[0]=i-iii;
                    dq[1]=j-jjj;//g(q)=p  <<==>>  q+inv[q] = p <<==>> inv[q]=p-q
                    error[iii*ncols+jjj]=opt;
                }
            }
        }
    }
    // Step 1b: map unmapped points in S via nearest neighbor
    double *dq=inv;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j,dq+=2){//q=(i,j)
            if(!isInfinite(error[i*ncols+j])){
                continue;
            }
            //find nearest neighbor q’ in S with finite e(q’)
            int ii,jj;
            findNearesFinite(error, nrows, ncols, i, j, ii, jj);//(ii, jj)=q'
            double *dqprime=&inv[2*(ii*ncols+jj)];
            dq[0]=ii+dqprime[0]-i;
            dq[1]=jj+dqprime[1]-j;//g(q)=g(q') <<==>> q+inv[q] = q'+inv[q']  <<==>>  inv[q]=q'+inv[q']-q
            double fr,fc;
            interpolateDisplacementFieldAt(forward, nrows, ncols, ii+dqprime[0], jj+dqprime[1], fr, fc);//(ii+dqprime[0], jj+dqprime[1])+(fr,fc) = f(g(q')) 
            double dr=ii+dqprime[0]+fr-i;
            double dc=jj+dqprime[1]+fc-j;//(dr, dc)=f(g(q'))-q
            error[i*ncols+j]=sqrt(dr*dr+dc*dc);
        }
    }
    if(needsDelete){
        delete[] error;
    }
    return 0;
}


/*int initializeNearestNeighborInverseField3D(double *forward, int nslices, int nrows, int ncols, double *inv, double *error){
    const static int nSlice[]={0, 0, 0, 0, 1, 1, 1, 1};    
    const static int nRow[]  ={0, 0, 1, 1, 0, 0, 1, 1};
    const static int nCol[]  ={0, 1, 0, 1, 0, 1, 0, 1};
    int sliceSize=nrows*ncols;
    bool needsDelete=false;
    if(error==NULL){
        needsDelete=true;
        error=new double[nslices*nrows*ncols];
    }
    // Step 0: initialize delta, MAXLOOP and inverse computing error
    for(int i=nslices*nrows*ncols-1;i>=0;--i){
        error[i]=INF64;
    }
    //Step 1a: Map points from p in R, assign e(q) for points q in S which are immediately adjacent to f(p) if assignment reduces e(q)
    double *f=forward;
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, f+=3){//p=(k,i,j) in R
                double dkk=k+f[0];  
                double dii=i+f[1];
                double djj=j+f[2];//(dkk, dii, djj) = f(p)
                if((dkk<0) || (dii<0) || (djj<0) || (dkk>nslices-1) || (dii>nrows-1)||(djj>ncols-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int kk=floor(dkk);
                int ii=floor(dii);
                int jj=floor(djj);
                //assign all grid points immediately adjacent to f(p)
                for(int idx=0;idx<8;++idx){
                    int kkk=kk+nSlice[idx];
                    int iii=ii+nRow[idx];
                    int jjj=jj+nCol[idx];//(kkk, iii, jjj)=q is a surrounding point
                    if((kkk<0)||(iii<0)||(jjj<0)||(kkk>=nslices)||(iii>=nrows)||(jjj>=ncols)){
                        continue;//the point is outside the lattice
                    }
                    double ds=dkk-kkk;
                    double dr=dii-iii;
                    double dc=djj-jjj;//(ds, dr,dc) = f(p) - q
                    double opt=sqrt(ds*ds+dr*dr+dc*dc);//||q-f(p)||
                    if(opt<error[kkk*sliceSize+iii*ncols+jjj]){//if(||q-f(p)||^2 < e(q))
                        double *dq=&inv[3*(kkk*sliceSize+iii*ncols+jjj)];
                        dq[0]=k-kkk;
                        dq[1]=i-iii;
                        dq[2]=j-jjj;//g(q)=p  <<==>>  q+inv[q] = p <<==>> inv[q]=p-q
                        error[kkk*sliceSize+iii*ncols+jjj]=opt;
                    }
                }
            }
        }
    }
    // Step 1b: map unmapped points in S via nearest neighbor
    double *dq=inv;
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j,dq+=3){//q=(k,i,j)
                if(!isInfinite(error[k*sliceSize+i*ncols+j])){
                    continue;
                }
                //find nearest neighbor q’ in S with finite e(q’)
                int kk,ii,jj;
not implemented:findNearesFinite3D(error, nslices, nrows, ncols, k, i, j, kk, ii, jj);//(kk, ii, jj)=q'
                double *dqprime=&inv[3*(kk*sliceSize+ii*ncols+jj)];
                dq[0]=kk+dqprime[0]-k;
                dq[1]=ii+dqprime[1]-i;
                dq[2]=jj+dqprime[2]-j;//g(q)=g(q') <<==>> q+inv[q] = q'+inv[q']  <<==>>  inv[q]=q'+inv[q']-q
                double fs, fr,fc;
not implemented:interpolateDisplacementField3DAt(forward, nslices, nrows, ncols, kk+dqprime[0], ii+dqprime[1], jj+dqprime[2], fs, fr, fc);//(ii+dqprime[0], jj+dqprime[1])+(fr,fc) = f(g(q')) 
                double ds=kk+dqprime[0]+fs-k;
                double dr=ii+dqprime[1]+fr-i;
                double dc=jj+dqprime[2]+fc-j;//(dr, dc)=f(g(q'))-q
                error[k*sliceSize+i*ncols+j]=sqrt(ds*ds+dr*dr+dc*dc);
            }
        }
    }
    if(needsDelete){
        delete[] error;
    }
    return 0;
}

*/
int invertVectorField(double *forward, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *inv, double *stats){
    const static int numNeighbors=4;
    const static int dRow[]={-1, 0, 1,  0, -1, 1,  1, -1};
    const static int dCol[]={ 0, 1, 0, -1,  1, 1, -1, -1};
    const static int nRow[]={ 0, 0, 1, 1};
    const static int nCol[]={ 0, 1, 0, 1};
    double gamma[4];
    double gamma2[4];
    double *temp=new double[nrows*ncols*2];
    double *residual=new double[nrows*ncols*2];
    double *denom=new double[nrows*ncols];
    //memset(inv, 0, sizeof(double)*nrows*ncols*2);
    initializeNearestNeighborInverseField(forward, nrows, ncols, inv, NULL);
    double maxChange=tolerance+1;
    int iter;
    double substats[3];
    for(iter=0;(tolerance*tolerance<maxChange)&&(iter<maxIter);++iter){
        memset(temp, 0, sizeof(double)*nrows*ncols*2);
        memset(denom, 0, sizeof(double)*nrows*ncols);
        //---interpolate the current approximation and accumulate with the forward field---
        composeVectorFields(forward, nrows, ncols, inv, nrows, ncols, residual, substats);
        double *r=residual;
        double *f=forward;
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j,r+=2, f+=2){
                //--accumulate all regularization terms--
                for(int k=0;k<numNeighbors;++k){
                    int ii=i+dRow[k];
                    if((ii<0)||(ii>=nrows)){
                        continue;
                    }
                    int jj=j+dCol[k];
                    if((jj<0)||(jj>=ncols)){
                        continue;
                    }
                    denom[i*ncols+j]+=lambdaParam;
                    temp[2*(i*ncols+j)]+=lambdaParam*inv[2*(ii*ncols+jj)];
                    temp[2*(i*ncols+j)+1]+=lambdaParam*inv[2*(ii*ncols+jj)+1];
                }
                //find the variables that are affected by this input point, and their interpolation coefficients
                double dii=i+f[0];
                double djj=j+f[1];
                //find the top left index and the interpolation coefficients
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (ii>=nrows)||(jj>=ncols)){//no one is affected
                    continue;
                }
                double calpha=dii-ii;
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                //---finally accumulate the affected terms---
                gamma[0]=alpha*beta;//top left
                gamma[1]=alpha*cbeta;//top right
                gamma[2]=calpha*beta;//bottom left
                gamma[3]=calpha*cbeta;//bottom right
                for(int k=0;k<4;++k){
                    int iii=ii+nRow[k];
                    int jjj=jj+nCol[k];
                    if((iii<0)||(jjj<0)||(iii>=nrows)||(jjj>=ncols)){
                        continue;
                    }
                    gamma2[k]=gamma[k]*gamma[k];
                    temp[2*(iii*ncols+jjj)]+=inv[2*(iii*ncols+jjj)]*gamma2[k] - r[0]*gamma[k];
                    temp[2*(iii*ncols+jjj)+1]+=inv[2*(iii*ncols+jjj)+1]*gamma2[k] - r[1]*gamma[k];
                    denom[iii*ncols+jjj]+=gamma2[k];
                }//for k
            }//for j
        }//for i
        //now execute the Jacobi iteration
        double *id=inv;
        double *tmp=temp;
        double *den=denom;
        maxChange=0;
        double theta=0;
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, id+=2, tmp+=2, den++){
                tmp[0]/=(*den);
                tmp[1]/=(*den);
                double nrm=(tmp[0]-id[0])*(tmp[0]-id[0])+(tmp[1]-id[1])*(tmp[1]-id[1]);
                if(maxChange<nrm){
                    maxChange=nrm;
                }
                id[0]=theta*id[0]+(1-theta)*tmp[0];
                id[1]=theta*id[1]+(1-theta)*tmp[1];
            }
        }
    }//for iter
    //composeVectorFields(forward, inv, nrows, ncols, residual, substats);
    delete[] temp;
    delete[] denom;
    delete[] residual;
    //stats[0]=sqrt(maxChange);
    stats[0]=substats[1];
    stats[1]=iter;
    return 0;
}



int invertVectorField3D(double *forward, int nslices, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *inv, double *stats){
    const static int numNeighbors=6;
    const static int dSlice[]   ={ 0, 0, 0, 0, 1, -1};
    const static int dRow[]     ={-1, 0, 1,  0, 0, 0};
    const static int dCol[]     ={ 0, 1, 0, -1, 0, 0};
    const static int nSlice[]   ={0, 0, 0, 0, 1, 1, 1, 1};
    const static int nRow[]     ={0, 0, 1, 1, 0, 0, 1, 1};
    const static int nCol[]     ={0, 1, 0, 1, 0, 1, 0, 1};
    double gamma[8];
    double gamma2[8];
    double substats[3];
    int nsites=nslices*nrows*ncols;
    int sliceSize=nrows*ncols;
    double *temp=new double[nsites*3];
    double *residual=new double[nsites*3];
    double *denom=new double[nsites];
    memset(inv, 0, sizeof(double)*nsites*3);
    double maxChange=tolerance+1;
    int iter;
    for(iter=0;(tolerance*tolerance<maxChange)&&(iter<maxIter);++iter){
        memset(temp, 0, sizeof(double)*nsites*3);
        memset(denom, 0, sizeof(double)*nsites);
        //---interpolate the current approximation and accumulate with the forward field---
        composeVectorFields3D(forward, nslices, nrows, ncols, inv, nslices, nrows, ncols, residual,substats);
        double *r=residual;
        double *f=forward;
        for(int k=0;k<nslices;++k){
            for(int i=0;i<nrows;++i){
                for(int j=0;j<ncols;++j,r+=3, f+=3){
                    //--accumulate all regularization terms--
                    for(int q=0;q<numNeighbors;++q){
                        int kk=k+dSlice[q];
                        if((kk<0)||(kk>=nslices)){
                            continue;
                        }
                        int ii=i+dRow[q];
                        if((ii<0)||(ii>=nrows)){
                            continue;
                        }
                        int jj=j+dCol[q];
                        if((jj<0)||(jj>=ncols)){
                            continue;
                        }
                        denom[k*sliceSize+i*ncols+j]+=lambdaParam;
                        temp[3*(k*sliceSize+i*ncols+j)]+=lambdaParam*inv[3*(kk*sliceSize+ii*ncols+jj)];
                        temp[3*(k*sliceSize+i*ncols+j)+1]+=lambdaParam*inv[3*(kk*sliceSize+ii*ncols+jj)+1];
                        temp[3*(k*sliceSize+i*ncols+j)+2]+=lambdaParam*inv[3*(kk*sliceSize+ii*ncols+jj)+2];
                    }
                    //find the variables that are affected by this input point, and their interpolation coefficients
                    double dkk=k+f[0];
                    double dii=i+f[1];
                    double djj=j+f[2];
                    if((dii<0) || (djj<0) || (dkk<0) || (dii>nrows-1)||(djj>ncols-1)||(dkk>nslices-1)){//no one is affected
                        continue;
                    }
                    //find the top left index and the interpolation coefficients
                    int kk=floor(dkk);
                    int ii=floor(dii);
                    int jj=floor(djj);
                    if((ii<0) || (jj<0) || (kk<0) || (ii>=nrows)||(jj>=ncols)||(kk>=nslices)){//no one is affected
                        continue;
                    }
                    double cgamma=dkk-kk;
                    double calpha=dii-ii;
                    double cbeta=djj-jj;
                    double alpha=1-calpha;
                    double beta=1-cbeta;
                    double gammaScalar=1-cgamma;
                    //---finally accumulate the affected terms---
                    gamma[0]=alpha*beta*gammaScalar;//top left same slice
                    gamma[1]=alpha*cbeta*gammaScalar;//top right same slice
                    gamma[2]=calpha*beta*gammaScalar;//bottom left same slice
                    gamma[3]=calpha*cbeta*gammaScalar;//bottom right same slice
                    gamma[4]=alpha*beta*cgamma;//top left next slice
                    gamma[5]=alpha*cbeta*cgamma;//top right next slice
                    gamma[6]=calpha*beta*cgamma;//bottom left next slice
                    gamma[7]=calpha*cbeta*cgamma;//bottom right next slice
                    for(int q=0;q<8;++q){
                        int iii=ii+nRow[q];
                        int jjj=jj+nCol[q];
                        int kkk=kk+nSlice[q];
                        if((iii<0)||(jjj<0)||(kkk<0)||(iii>=nrows)||(jjj>=ncols)||(kkk>=nslices)){
                            continue;
                        }
                        gamma2[q]=gamma[q]*gamma[q];
                        temp[3*(kkk*sliceSize+iii*ncols+jjj)]+=inv[3*(kkk*sliceSize+iii*ncols+jjj)]*gamma2[q] - r[0]*gamma[q];
                        temp[3*(kkk*sliceSize+iii*ncols+jjj)+1]+=inv[3*(kkk*sliceSize+iii*ncols+jjj)+1]*gamma2[q] - r[1]*gamma[q];
                        temp[3*(kkk*sliceSize+iii*ncols+jjj)+2]+=inv[3*(kkk*sliceSize+iii*ncols+jjj)+2]*gamma2[q] - r[2]*gamma[q];
                        denom[kkk*sliceSize+iii*ncols+jjj]+=gamma2[q];
                    }//for q
                }//for j
            }//for i
        }//for k
        //now execute the Jacobi iteration
        double *id=inv;
        double *tmp=temp;
        double *den=denom;
        maxChange=0;
        for(int k=0;k<nslices;++k){
            for(int i=0;i<nrows;++i){
                for(int j=0;j<ncols;++j, id+=3, tmp+=3, den++){
                    tmp[0]/=(*den);
                    tmp[1]/=(*den);
                    tmp[2]/=(*den);
                    double nrm=(tmp[0]-id[0])*(tmp[0]-id[0])+(tmp[1]-id[1])*(tmp[1]-id[1])+(tmp[2]-id[2])*(tmp[2]-id[2]);
                    if(maxChange<nrm){
                        maxChange=nrm;
                    }
                    id[0]=tmp[0];
                    id[1]=tmp[1];
                    id[2]=tmp[2];
                }
            }
        }
    }//for iter
    delete[] temp;
    delete[] denom;
    delete[] residual;
    stats[0]=sqrt(maxChange);
    stats[1]=iter;
    return 0;
}
//==================================================================================================
//==================================================================================================
//=====================================Yan's algorithm begins=======================================
//==================================================================================================
//==================================================================================================
//==================================================================================================

int invertVectorFieldYan(double *forward, int nrows, int ncols, int maxloop, double tolerance, double *inv){
    const int numNeighbors=4;
    const static int dRow[]={-1, 0, 1,  0, -1, 1,  1, -1};
    const static int dCol[]={ 0, 1, 0, -1,  1, 1, -1, -1};
    const static int nRow[]={ 0, 0, 1, 1};
    const static int nCol[]={ 0, 1, 0, 1};
    // Step 0: initialize delta, MAXLOOP and inverse computing error
    double *error=new double[nrows*ncols];
    for(int i=nrows*ncols-1;i>=0;--i){
        error[i]=INF64;
    }
    //Step 1a: Map points from p in R, assign e(q) for points q in S which are immediately adjacent to f(p) if assignment reduces e(q)
    double *f=forward;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, f+=2){//p=(i,j) in R
            double dii=i+f[0];
            double djj=j+f[1];//(dii, djj) = f(p)
            //find the top left index and the interpolation coefficients
            int ii=floor(dii);
            int jj=floor(djj);
            if((ii<0) || (jj<0) || (ii>=nrows)||(jj>=ncols)){//no one is affected
                continue;
            }
            //assign all grid points immediately adjacent to f(p)
            for(int k=0;k<4;++k){
                int iii=ii+nRow[k];
                int jjj=jj+nCol[k];//(iii, jjj)=q is a surrounding point
                if((iii<0)||(jjj<0)||(iii>=nrows)||(jjj>=ncols)){
                    continue;//the point is outside the lattice
                }
                double dr=dii-iii;
                double dc=djj-jjj;//(dr,dc) = f(p) - q
                double opt=sqrt(dr*dr+dc*dc);//||q-f(p)||
                if(opt<error[iii*ncols+jjj]){//if(||q-f(p)||^2 < e(q))
                    double *dq=&inv[2*(iii*ncols+jjj)];
                    dq[0]=i-iii;
                    dq[1]=j-jjj;//g(q)=p  <<==>>  q+inv[q] = p <<==>> inv[q]=p-q
                    error[iii*ncols+jjj]=opt;
                }
            }
        }
    }
    // Step 1b: map unmapped points in S via nearest neighbor
    double *dq=inv;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j,dq+=2){//q=(i,j)
            if(!isInfinite(error[i*ncols+j])){
                continue;
            }
            //find nearest neighbor q’ in S with finite e(q’)
            int ii,jj;
            findNearesFinite(error, nrows, ncols, i, j, ii, jj);//(ii, jj)=q'
            double *dqprime=&inv[2*(ii*ncols+jj)];
            dq[0]=ii+dqprime[0]-i;
            dq[1]=jj+dqprime[1]-j;//g(q)=g(q') <<==>> q+inv[q] = q'+inv[q']  <<==>>  inv[q]=q'+inv[q']-q
            double fr,fc;
            interpolateDisplacementFieldAt(forward, nrows, ncols, ii+dqprime[0], jj+dqprime[1], fr, fc);//(ii+dqprime[0], jj+dqprime[1])+(fr,fc) = f(g(q')) 
            double dr=ii+dqprime[0]+fr-i;
            double dc=jj+dqprime[1]+fc-j;//(dr, dc)=f(g(q'))-q
            error[i*ncols+j]=sqrt(dr*dr+dc*dc);
        }
    }
    // Step 2a: Search around points with e(q) > tolerance in S
    double *e=error;
    dq=inv;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j,++e, dq+=2){//q=(i,j) in S
            double stepSize=*e;
            double cr=i+dq[0];
            double cc=j+dq[1];//(cr,cc) = g(q)
            int iter=0;
            while((*e > tolerance)&&(iter<maxloop)){
                for(int signRow=-1;signRow<=1;signRow+=2){
                    for(int signCol=-1;signCol<=1;signCol+=2){
                        double vr=cr+signRow*stepSize*0.5;
                        double vc=cc+signCol*stepSize*0.5;// p'=(vr,vc) (corner of he box centered at g(q))
                        double fvr, fvc;
                        interpolateDisplacementFieldAt(forward, nrows, ncols, vr, vc, fvr, fvc);//f(p') = (vr+fvr, vc+fvc)
                        double dr=i-vr-fvr;
                        double dc=j-vc-fvc;//q-f(p') = (i,j) - (vr+fvr, vc+fvc)
                        double opt=sqrt(dr*dr+dc*dc);
                        if(opt<*e){
                            dq[0]=vr-i;
                            dq[1]=vc-j;//g(q)=p'  <<==>> q+inv[q] = p'  <<==>> inv[q] = p'-q
                            *e=opt;
                        }
                    }
                }
                ++iter;
                stepSize*=0.5;
            }
        }
    }
    // Step 2b: Systematic search around active grid points (adjacent points with low errors) instead of around the points themselves
    bool finished=false;
    int iter=0;
    while((!finished) && (iter<maxloop)){
        ++iter;
        e=error;
        dq=inv;
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j,++e, dq+=2){//q=(i,j)
                if(*e>=tolerance){
                    continue;
                }
                //now error[q]<tolerance
                for(int k=0;k<numNeighbors;++k){//look for adjacent vertices
                    int ii=i+dRow[k];
                    int jj=j+dCol[k];
                    if((ii<0)||(jj<0)||(i>=nrows)||(j>=ncols)){
                        continue;//q' is outside of the lattice
                    }
                    if(error[ii*ncols+jj]<=tolerance){//q'= (ii,jj)
                        continue;
                    }
                    //now error[q']>tolerance
                    double stepSize=sqrt(2.0);
                    for(int innerIter=0;innerIter<maxloop;++innerIter){
                        double cr=i+dq[0];
                        double cc=j+dq[1];//g(q)=(cr,cc)
                        for(int signRow=-1;signRow<=1;signRow+=2){
                            for(int signCol=-1;signCol<=1;signCol+=2){
                                double vr=cr+signRow*stepSize*0.5;
                                double vc=cc+signCol*stepSize*0.5;//(vr,vc)=p' (corner of the box )
                                double fvr, fvc;
                                interpolateDisplacementFieldAt(forward, nrows, ncols, vr, vc, fvr, fvc);//f(p') = (vr+fvr, vc+fvc)
                                //find the lattice points q'' surrounding f(p')=(fvr, fvc)
                                //find the top left index and the interpolation coefficients
                                int ii=floor(vr+fvr);
                                int jj=floor(vc+fvc);
                                //assign all grid points immediately adjacent to f(p)
                                for(int k=0;k<4;++k){
                                    int iii=ii+nRow[k];
                                    int jjj=jj+nCol[k];//(iii, jjj) is a corner q'' surrounding f(p')
                                    if((iii<0)||(jjj<0)||(iii>=nrows)||(jjj>=ncols)){
                                        continue;//the corner does not exist
                                    }
                                    double dr=iii-(vr+fvr);
                                    double dc=jjj-(vc+fvc);//(dr, dc) = q''-f(p')
                                    double opt=sqrt(dr*dr+dc*dc);//||q''-f(p')||^2
                                    if(opt<error[iii*ncols+jjj]){
                                        double *dqbiprime=&inv[2*(iii*ncols+jjj)];
                                        dqbiprime[0]=vr-iii;
                                        dqbiprime[1]=vc-jjj;//g(q'')=p' <<==>> q''+inv[q''] = p'  <<==>> inv[q'']=p'-q''
                                        error[iii*ncols+jjj]=opt;
                                    }
                                }//for each q'' surrounding f(p')
                            }
                        }//for each vertex p' of the box G(g(q))
                    }//while innerIter<maxloop
                    
                }//foreach point q' adjacent to q such that e(q')>tolerance
            }
        }//foreach grid point q in S
    }
    delete[] error;
    return 0;
}


//==================================================================================================
//==================================================================================================
//=====================================Yan's algorithm ends=========================================
//==================================================================================================
//==================================================================================================
//==================================================================================================


/*
    Computes comp(x)=d2(x+d1(x))+d1(x) (i.e. applies first d1, then add d2 to the result)
    deprecated bacause it assumed both transformations were endomorphisms
*/
int composeVectorFields_deprecated(double *d1, double *d2, int nrows, int ncols, double *comp, double *stats){
    double *dx=d1;
    double *res=comp;
    double maxNorm=0;
    double meanNorm=0;
    double stdNorm=0;
    int cnt=0;
    memset(comp, 0, sizeof(double)*nrows*ncols*2); 
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, dx+=2, res+=2){
            double dii=i+dx[0];
            double djj=j+dx[1];
            if((dii<0)||(nrows-1<dii)||(djj<0)||(ncols-1<djj)){
                continue;
            }
            int ii=floor(dii);
            int jj=floor(djj);
            if((ii<0) || (nrows<=ii)||(jj<0)||(ncols<=jj) ){
                continue;
            }
            double calpha=dii-ii;//by definition these factors are nonnegative
            double cbeta=djj-jj;
            double alpha=1-calpha;
            double beta=1-cbeta;
            //---top-left
            res[0]=dx[0];
            res[1]=dx[1];
            double *z=&d2[2*(ii*ncols+jj)];
            res[0]+=alpha*beta*z[0];
            res[1]+=alpha*beta*z[1];
            //---top-right
            ++jj;
            if(jj<ncols){
                z=&d2[2*(ii*ncols+jj)];
                res[0]+=alpha*cbeta*z[0];
                res[1]+=alpha*cbeta*z[1];
            }
            //---bottom-right
            ++ii;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                z=&d2[2*(ii*ncols+jj)];
                res[0]+=calpha*cbeta*z[0];
                res[1]+=calpha*cbeta*z[1];
            }
            //---bottom-left
            --jj;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                z=&d2[2*(ii*ncols+jj)];
                res[0]+=calpha*beta*z[0];
                res[1]+=calpha*beta*z[1];
            }
            //consider only displacements that land inside the image
            if((i+dx[0]>=0 && i+dx[0]<=nrows-1) && (j+dx[1]>=0 && j+dx[1]<=ncols-1)){
                double nn=res[0]*res[0]+res[1]*res[1];
                if(maxNorm<nn){
                    maxNorm=nn;
                }
                meanNorm+=nn;
                stdNorm+=nn*nn;
                ++cnt;
            }
        }
    }
    meanNorm/=cnt;
    stats[0]=sqrt(maxNorm);
    stats[1]=sqrt(meanNorm);
    stats[2]=sqrt(stdNorm/cnt - meanNorm*meanNorm);
    return 0;
}

/*
    Computes comp(x)=d2(x+d1(x))+d1(x) (i.e. applies first d1, then add d2 to the result)
*/
int composeVectorFields(double *d1, int nr1, int nc1, double *d2, int nr2, int nc2, double *comp, double *stats){
    double *dx=d1;
    double *res=comp;
    double maxNorm=0;
    double meanNorm=0;
    double stdNorm=0;
    int cnt=0;
    memset(comp, 0, sizeof(double)*nr1*nc1*2); 
    for(int i=0;i<nr1;++i){
        for(int j=0;j<nc1;++j, dx+=2, res+=2){
            double dii=i+dx[0];
            double djj=j+dx[1];
            if((dii<0)||(nr2-1<dii)||(djj<0)||(nc2-1<djj)){
                continue;
            }
            int ii=floor(dii);
            int jj=floor(djj);
            if((ii<0) || (nr2<=ii)||(jj<0)||(nc2<=jj) ){
                continue;
            }
            double calpha=dii-ii;//by definition these factors are nonnegative
            double cbeta=djj-jj;
            double alpha=1-calpha;
            double beta=1-cbeta;
            //---top-left
            res[0]=dx[0];
            res[1]=dx[1];
            double *z=&d2[2*(ii*nc2+jj)];
            res[0]+=alpha*beta*z[0];
            res[1]+=alpha*beta*z[1];
            //---top-right
            ++jj;
            if(jj<nc2){
                z=&d2[2*(ii*nc2+jj)];
                res[0]+=alpha*cbeta*z[0];
                res[1]+=alpha*cbeta*z[1];
            }
            //---bottom-right
            ++ii;
            if((ii>=0)&&(jj>=0)&&(ii<nr2)&&(jj<nc2)){
                z=&d2[2*(ii*nc2+jj)];
                res[0]+=calpha*cbeta*z[0];
                res[1]+=calpha*cbeta*z[1];
            }
            //---bottom-left
            --jj;
            if((ii>=0)&&(jj>=0)&&(ii<nr2)&&(jj<nc2)){
                z=&d2[2*(ii*nc2+jj)];
                res[0]+=calpha*beta*z[0];
                res[1]+=calpha*beta*z[1];
            }
            //consider only displacements that land inside the image
            if((dii>=0 && dii<=nr2-1) && (djj>=0 && djj<=nc2-1)){
                double nn=res[0]*res[0]+res[1]*res[1];
                if(maxNorm<nn){
                    maxNorm=nn;
                }
                meanNorm+=nn;
                stdNorm+=nn*nn;
                ++cnt;
            }
        }
    }
    meanNorm/=cnt;
    stats[0]=sqrt(maxNorm);
    stats[1]=sqrt(meanNorm);
    stats[2]=sqrt(stdNorm/cnt - meanNorm*meanNorm);
    return 0;
}

/*
    Computes comp(x)=d2(x+d1(x))+d1(x) (i.e. applies first d1, then d2 to the result)
    this version assumed both domains were the same
*/
int composeVectorFields3D_deprecated(double *d1, double *d2, int nslices, int nrows, int ncols, double *comp, double *stats){
    int sliceSize=nrows*ncols;
    double *dx=d1;
    double *res=comp;
    double maxNorm=0;
    double meanNorm=0;
    double stdNorm=0;
    int cnt=0;
    memset(comp, 0, sizeof(double)*nslices*sliceSize*3); 
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=3, res+=3){
                double dkk=k+dx[0];
                double dii=i+dx[1];
                double djj=j+dx[2];
                if((dii<0)||(djj<0)||(dkk<0)||(dii>nrows-1)||(djj>ncols-1)||(dkk>nslices-1)){
                    continue;
                }
                //---top-left
                int kk=floor(dkk);
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0)||(jj<0)||(kk<0)||(ii>=nrows)||(jj>=ncols)||(kk>=nslices)){
                    continue;
                }
                double cgamma=dkk-kk;
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                double gamma=1-cgamma;
                res[0]=dx[0];
                res[1]=dx[1];
                res[2]=dx[2];
                double *z=&d2[3*(kk*sliceSize+ii*ncols+jj)];
                res[0]+=alpha*beta*gamma*z[0];
                res[1]+=alpha*beta*gamma*z[1];
                res[2]+=alpha*beta*gamma*z[2];
                //---top-right
                ++jj;
                if(jj<ncols){
                    z=&d2[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=alpha*cbeta*gamma*z[0];
                    res[1]+=alpha*cbeta*gamma*z[1];
                    res[2]+=alpha*cbeta*gamma*z[2];
                }
                //---bottom-right
                ++ii;
                if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                    z=&d2[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=calpha*cbeta*gamma*z[0];
                    res[1]+=calpha*cbeta*gamma*z[1];
                    res[2]+=calpha*cbeta*gamma*z[2];
                }
                //---bottom-left
                --jj;
                if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                    z=&d2[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=calpha*beta*gamma*z[0];
                    res[1]+=calpha*beta*gamma*z[1];
                    res[2]+=calpha*beta*gamma*z[2];
                }
                ++kk;
                if(kk<nslices){
                    --ii;
                    z=&d2[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=alpha*beta*cgamma*z[0];
                    res[1]+=alpha*beta*cgamma*z[1];
                    res[2]+=alpha*beta*cgamma*z[2];
                    ++jj;
                    if(jj<ncols){
                        z=&d2[3*(kk*sliceSize+ii*ncols+jj)];
                        res[0]+=alpha*cbeta*cgamma*z[0];
                        res[1]+=alpha*cbeta*cgamma*z[1];
                        res[2]+=alpha*cbeta*cgamma*z[2];
                    }
                    //---bottom-right
                    ++ii;
                    if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                        z=&d2[3*(kk*sliceSize+ii*ncols+jj)];
                        res[0]+=calpha*cbeta*cgamma*z[0];
                        res[1]+=calpha*cbeta*cgamma*z[1];
                        res[2]+=calpha*cbeta*cgamma*z[2];
                    }
                    //---bottom-left
                    --jj;
                    if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                        z=&d2[3*(kk*sliceSize+ii*ncols+jj)];
                        res[0]+=calpha*beta*cgamma*z[0];
                        res[1]+=calpha*beta*cgamma*z[1];
                        res[2]+=calpha*beta*cgamma*z[2];
                    }
                }
                if((k+dx[0]>=0 && k+dx[0]<=nslices-1) && (i+dx[1]>=0 && i+dx[1]<=nrows-1) && (j+dx[2]>=0 && j+dx[2]<=ncols-1)){
                    double nn=res[0]*res[0]+res[1]*res[1]+res[2]*res[2];
                    if(maxNorm<nn){
                        maxNorm=nn;
                    }
                    meanNorm+=nn;
                    stdNorm+=nn*nn;
                    ++cnt;
                }
            }
        }
    }
    meanNorm/=cnt;
    stats[0]=sqrt(maxNorm);
    stats[1]=sqrt(meanNorm);
    stats[2]=sqrt(stdNorm/cnt - meanNorm*meanNorm);
    return 0;
}


/*
    Computes comp(x)=d2(x+d1(x))+d1(x) (i.e. applies first d1, then d2 to the result)
*/
int composeVectorFields3D(double *d1, int ns1, int nr1, int nc1, double *d2, int ns2, int nr2, int nc2, double *comp, double *stats){
    int sliceSizeD2=nr2*nc2;
    double *dx=d1;
    double *res=comp;
    double maxNorm=0;
    double meanNorm=0;
    double stdNorm=0;
    int cnt=0;
    memset(comp, 0, sizeof(double)*ns1*nr1*nc1*3); 
    for(int k=0;k<ns1;++k){
        for(int i=0;i<nr1;++i){
            for(int j=0;j<nc1;++j, dx+=3, res+=3){
                double dkk=k+dx[0];
                double dii=i+dx[1];
                double djj=j+dx[2];
                if((dii<0)||(djj<0)||(dkk<0)||(dii>nr2-1)||(djj>nc2-1)||(dkk>ns2-1)){
                    continue;
                }
                //---top-left
                int kk=floor(dkk);
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0)||(jj<0)||(kk<0)||(ii>=nr2)||(jj>=nc2)||(kk>=ns2)){
                    continue;
                }
                double cgamma=dkk-kk;
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                double gamma=1-cgamma;
                res[0]=dx[0];
                res[1]=dx[1];
                res[2]=dx[2];
                double *z=&d2[3*(kk*sliceSizeD2+ii*nc2+jj)];
                res[0]+=alpha*beta*gamma*z[0];
                res[1]+=alpha*beta*gamma*z[1];
                res[2]+=alpha*beta*gamma*z[2];
                //---top-right
                ++jj;
                if(jj<nc2){
                    z=&d2[3*(kk*sliceSizeD2+ii*nc2+jj)];
                    res[0]+=alpha*cbeta*gamma*z[0];
                    res[1]+=alpha*cbeta*gamma*z[1];
                    res[2]+=alpha*cbeta*gamma*z[2];
                }
                //---bottom-right
                ++ii;
                if((ii>=0)&&(jj>=0)&&(ii<nr2)&&(jj<nc2)){
                    z=&d2[3*(kk*sliceSizeD2+ii*nc2+jj)];
                    res[0]+=calpha*cbeta*gamma*z[0];
                    res[1]+=calpha*cbeta*gamma*z[1];
                    res[2]+=calpha*cbeta*gamma*z[2];
                }
                //---bottom-left
                --jj;
                if((ii>=0)&&(jj>=0)&&(ii<nr2)&&(jj<nc2)){
                    z=&d2[3*(kk*sliceSizeD2+ii*nc2+jj)];
                    res[0]+=calpha*beta*gamma*z[0];
                    res[1]+=calpha*beta*gamma*z[1];
                    res[2]+=calpha*beta*gamma*z[2];
                }
                ++kk;
                if(kk<ns2){
                    --ii;
                    z=&d2[3*(kk*sliceSizeD2+ii*nc2+jj)];
                    res[0]+=alpha*beta*cgamma*z[0];
                    res[1]+=alpha*beta*cgamma*z[1];
                    res[2]+=alpha*beta*cgamma*z[2];
                    ++jj;
                    if(jj<nc2){
                        z=&d2[3*(kk*sliceSizeD2+ii*nc2+jj)];
                        res[0]+=alpha*cbeta*cgamma*z[0];
                        res[1]+=alpha*cbeta*cgamma*z[1];
                        res[2]+=alpha*cbeta*cgamma*z[2];
                    }
                    //---bottom-right
                    ++ii;
                    if((ii>=0)&&(jj>=0)&&(ii<nr2)&&(jj<nc2)){
                        z=&d2[3*(kk*sliceSizeD2+ii*nc2+jj)];
                        res[0]+=calpha*cbeta*cgamma*z[0];
                        res[1]+=calpha*cbeta*cgamma*z[1];
                        res[2]+=calpha*cbeta*cgamma*z[2];
                    }
                    //---bottom-left
                    --jj;
                    if((ii>=0)&&(jj>=0)&&(ii<nr2)&&(jj<nc2)){
                        z=&d2[3*(kk*sliceSizeD2+ii*nc2+jj)];
                        res[0]+=calpha*beta*cgamma*z[0];
                        res[1]+=calpha*beta*cgamma*z[1];
                        res[2]+=calpha*beta*cgamma*z[2];
                    }
                }
                if((dkk>=0 && dkk<=ns2-1) && (dii>=0 && dii<=nr2-1) && (djj>=0 && djj<=nc2-1)){
                    double nn=res[0]*res[0]+res[1]*res[1]+res[2]*res[2];
                    if(maxNorm<nn){
                        maxNorm=nn;
                    }
                    meanNorm+=nn;
                    stdNorm+=nn*nn;
                    ++cnt;
                }
            }
        }
    }
    meanNorm/=cnt;
    stats[0]=sqrt(maxNorm);
    stats[1]=sqrt(meanNorm);
    stats[2]=sqrt(stdNorm/cnt - meanNorm*meanNorm);
    return 0;
}




int upsampleDisplacementField(double *d1, int nrows, int ncols, double *up, int nr, int nc){
    double *res=up;
    memset(up, 0, sizeof(double)*nr*nc*2);
        for(int i=0;i<nr;++i){
            for(int j=0;j<nc;++j, res+=2){
                double dii=(i&1)?0.5*i:i/2;
                double djj=(j&1)?0.5*j:j/2;
                if((dii<0) || (djj<0) || (dii>nrows-1)||(djj>ncols-1)){//no one is affected
                    continue;
                }
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (ii>=nrows)||(jj>=ncols)){//no one is affected
                    continue;
                }
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                //---top-left
                double *z=&d1[2*(ii*ncols+jj)];
                res[0]+=alpha*beta*z[0];
                res[1]+=alpha*beta*z[1];
                //---top-right
                ++jj;
                if(jj<ncols){
                    z=&d1[2*(ii*ncols+jj)];
                    res[0]+=alpha*cbeta*z[0];
                    res[1]+=alpha*cbeta*z[1];
                }
                //---bottom-right
                ++ii;
                if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                    z=&d1[2*(ii*ncols+jj)];
                    res[0]+=calpha*cbeta*z[0];
                    res[1]+=calpha*cbeta*z[1];
                }
                //---bottom-left
                --jj;
                if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                    z=&d1[2*(ii*ncols+jj)];
                    res[0]+=calpha*beta*z[0];
                    res[1]+=calpha*beta*z[1];
                }
            }
        }
    return 0;
}

int downsampleDisplacementField(double *d1, int nr, int nc, double *down){
    int nnr=(nr+1)/2;
    int nnc=(nc+1)/2;
    unsigned char *cnt=new unsigned char[nnr*nnc];
    memset(cnt, 0, sizeof(unsigned char)*nnr*nnc);
    double *d=d1;
    memset(down, 0, sizeof(double)*nnr*nnc*2);
    for(int i=0;i<nr;++i){
        for(int j=0;j<nc;++j,d+=2){
            int ii=i/2;
            int jj=j/2;
            down[2*(ii*nnc+jj)]+=d[0];
            down[2*(ii*nnc+jj)+1]+=d[1];
            cnt[ii*nnc+jj]++;
        }
    }
    d-=2;
    for(int p=nnr*nnc-1;p>=0;--p, d-=2){
        if(cnt[p]>0){
            d[0]/=cnt[p];
            d[1]/=cnt[p];
        }
    }
    delete[] cnt;
    return 0;
}

int downsampleScalarField(double *d1, int nr, int nc, double *down){
    int nnr=(nr+1)/2;
    int nnc=(nc+1)/2;
    unsigned char *cnt=new unsigned char[nnr*nnc];
    memset(cnt, 0, sizeof(unsigned char)*nnr*nnc);
    double *d=d1;
    memset(down, 0, sizeof(double)*nnr*nnc);
    for(int i=0;i<nr;++i){
        for(int j=0;j<nc;++j,d++){
            int ii=i/2;
            int jj=j/2;
            down[ii*nnc+jj]+=d[0];
            cnt[ii*nnc+jj]++;
        }
    }
    for(int p=nnr*nnc-1;p>=0;--p){
        if(cnt[p]>0){
            down[p]/=cnt[p];
        }
    }
    delete[] cnt;
    return 0;
}


int downsampleDisplacementField3D(double *d1, int ns, int nr, int nc, double *down){
    int nns=(ns+1)/2;
    int nnr=(nr+1)/2;
    int nnc=(nc+1)/2;
    int sliceSize=nnr*nnc;
    unsigned char *cnt=new unsigned char[nns*sliceSize];
    memset(cnt, 0, sizeof(unsigned char)*nns*sliceSize);
    double *d=d1;
    memset(down, 0, sizeof(double)*nns*nnr*nnc*3);
    for(int k=0;k<ns;++k){
        for(int i=0;i<nr;++i){
            for(int j=0;j<nc;++j,d+=3){
                int kk=k/2;
                int ii=i/2;
                int jj=j/2;
                down[3*(kk*sliceSize+ii*nnc+jj)]+=d[0];
                down[3*(kk*sliceSize+ii*nnc+jj)+1]+=d[1];
                down[3*(kk*sliceSize+ii*nnc+jj)+2]+=d[2];
                cnt[kk*sliceSize+ii*nnc+jj]++;
            }
        }
    }
    d-=3;
    for(int p=sliceSize*nns-1;p>=0;--p, d-=3){
        if(cnt[p]>0){
            d[0]/=cnt[p];
            d[1]/=cnt[p];
            d[2]/=cnt[p];
        }
    }
    delete[] cnt;
    return 0;
}

int downsampleScalarField3D(double *d1, int ns, int nr, int nc, double *down){
    int nns=(ns+1)/2;
    int nnr=(nr+1)/2;
    int nnc=(nc+1)/2;
    int sliceSize=nnr*nnc;
    unsigned char *cnt=new unsigned char[nns*sliceSize];
    memset(cnt, 0, sizeof(unsigned char)*nns*sliceSize);
    double *d=d1;
    memset(down, 0, sizeof(double)*nns*nnr*nnc);
    for(int k=0;k<ns;++k){
        for(int i=0;i<nr;++i){
            for(int j=0;j<nc;++j,d++){
                int kk=k/2;
                int ii=i/2;
                int jj=j/2;
                down[kk*sliceSize+ii*nnc+jj]+=d[0];
                cnt[kk*sliceSize+ii*nnc+jj]++;
            }
        }
    }
    for(int p=sliceSize*nns-1;p>=0;--p){
        if(cnt[p]>0){
            down[p]/=cnt[p];
        }
    }
    delete[] cnt;
    return 0;
}

/*
    It assumes that up was already allocated: (nslices)x(nrows)x(ncols)x3
    nslices, nrows, ncols are the dimensions of the upsampled field
*/
int upsampleDisplacementField3D(double *d1, int nslices, int nrows, int ncols, double *up, int ns, int nr, int nc){
    int sliceSize=nrows*ncols;
    double dx[3];
    double *res=up;
    memset(up, 0, sizeof(double)*ns*nr*nc*3);
    for(int k=0;k<ns;++k){
        for(int i=0;i<nr;++i){
            for(int j=0;j<nc;++j, res+=3){
                dx[0]=(k&1)?0.5*k:k/2;
                dx[1]=(i&1)?0.5*i:i/2;
                dx[2]=(j&1)?0.5*j:j/2;
                if((dx[0]<0) || (dx[1]<0) || (dx[2]<0) || (dx[1]>nrows-1)||(dx[2]>ncols-1)||(dx[0]>nslices-1)){//no one is affected
                    continue;
                }
                int kk=floor(dx[0]);
                int ii=floor(dx[1]);
                int jj=floor(dx[2]);
                if((kk<0) || (ii<0) || (jj<0) || (ii>=nrows)||(jj>=ncols)||(kk>=nslices)){//no one is affected
                    continue;
                }
                double cgamma=dx[0]-kk;
                double calpha=dx[1]-ii;//by definition these factors are nonnegative
                double cbeta=dx[2]-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                double gamma=1-cgamma;
                //---top-left
                double *z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                res[0]+=alpha*beta*gamma*z[0];
                res[1]+=alpha*beta*gamma*z[1];
                res[2]+=alpha*beta*gamma*z[2];
                //---top-right
                ++jj;
                if(jj<ncols){
                    z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=alpha*cbeta*gamma*z[0];
                    res[1]+=alpha*cbeta*gamma*z[1];
                    res[2]+=alpha*cbeta*gamma*z[2];
                }
                //---bottom-right
                ++ii;
                if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                    z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=calpha*cbeta*gamma*z[0];
                    res[1]+=calpha*cbeta*gamma*z[1];
                    res[2]+=calpha*cbeta*gamma*z[2];
                }
                //---bottom-left
                --jj;
                if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                    z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=calpha*beta*gamma*z[0];
                    res[1]+=calpha*beta*gamma*z[1];
                    res[2]+=calpha*beta*gamma*z[2];
                }
                ++kk;
                if(kk<nslices){
                    --ii;
                    z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=alpha*beta*cgamma*z[0];
                    res[1]+=alpha*beta*cgamma*z[1];
                    res[2]+=alpha*beta*cgamma*z[2];
                    ++jj;
                    if(jj<ncols){
                        z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                        res[0]+=alpha*cbeta*cgamma*z[0];
                        res[1]+=alpha*cbeta*cgamma*z[1];
                        res[2]+=alpha*cbeta*cgamma*z[2];
                    }
                    //---bottom-right
                    ++ii;
                    if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                        z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                        res[0]+=calpha*cbeta*cgamma*z[0];
                        res[1]+=calpha*cbeta*cgamma*z[1];
                        res[2]+=calpha*cbeta*cgamma*z[2];
                    }
                    //---bottom-left
                    --jj;
                    if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                        z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                        res[0]+=calpha*beta*cgamma*z[0];
                        res[1]+=calpha*beta*cgamma*z[1];
                        res[2]+=calpha*beta*cgamma*z[2];
                    }
                }
            }
        }
    }
    return 0;
}


int accumulateUpsampleDisplacementField3D(double *d1, int nslices, int nrows, int ncols, double *current, int ns, int nr, int nc){
    int sliceSize=nrows*ncols;
    double dx[3];
    double *res=current;
    for(int k=0;k<ns;++k){
        for(int i=0;i<nr;++i){
            for(int j=0;j<nc;++j, res+=3){
                dx[0]=(k&1)?0.5*k:k/2;
                dx[1]=(i&1)?0.5*i:i/2;
                dx[2]=(j&1)?0.5*j:j/2;
                if((dx[0]<0) || (dx[1]<0) || (dx[2]<0) || (dx[1]>nrows-1)||(dx[2]>ncols-1)||(dx[0]>nslices-1)){//no one is affected
                    continue;
                }
                int kk=floor(dx[0]);
                int ii=floor(dx[1]);
                int jj=floor(dx[2]);
                if((kk<0) || (ii<0) || (jj<0) || (ii>=nrows)||(jj>=ncols)||(kk>=nslices)){//no one is affected
                    continue;
                }
                double cgamma=dx[0]-kk;
                double calpha=dx[1]-ii;//by definition these factors are nonnegative
                double cbeta=dx[2]-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                double gamma=1-cgamma;
                //---top-left
                double *z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                res[0]+=alpha*beta*gamma*z[0];
                res[1]+=alpha*beta*gamma*z[1];
                res[2]+=alpha*beta*gamma*z[2];
                //---top-right
                ++jj;
                if(jj<ncols){
                    z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=alpha*cbeta*gamma*z[0];
                    res[1]+=alpha*cbeta*gamma*z[1];
                    res[2]+=alpha*cbeta*gamma*z[2];
                }
                //---bottom-right
                ++ii;
                if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                    z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=calpha*cbeta*gamma*z[0];
                    res[1]+=calpha*cbeta*gamma*z[1];
                    res[2]+=calpha*cbeta*gamma*z[2];
                }
                //---bottom-left
                --jj;
                if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                    z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=calpha*beta*gamma*z[0];
                    res[1]+=calpha*beta*gamma*z[1];
                    res[2]+=calpha*beta*gamma*z[2];
                }
                ++kk;
                if(kk<nslices){
                    --ii;
                    z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                    res[0]+=alpha*beta*cgamma*z[0];
                    res[1]+=alpha*beta*cgamma*z[1];
                    res[2]+=alpha*beta*cgamma*z[2];
                    ++jj;
                    if(jj<ncols){
                        z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                        res[0]+=alpha*cbeta*cgamma*z[0];
                        res[1]+=alpha*cbeta*cgamma*z[1];
                        res[2]+=alpha*cbeta*cgamma*z[2];
                    }
                    //---bottom-right
                    ++ii;
                    if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                        z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                        res[0]+=calpha*cbeta*cgamma*z[0];
                        res[1]+=calpha*cbeta*cgamma*z[1];
                        res[2]+=calpha*cbeta*cgamma*z[2];
                    }
                    //---bottom-left
                    --jj;
                    if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                        z=&d1[3*(kk*sliceSize+ii*ncols+jj)];
                        res[0]+=calpha*beta*cgamma*z[0];
                        res[1]+=calpha*beta*cgamma*z[1];
                        res[2]+=calpha*beta*cgamma*z[2];
                    }
                }
            }
        }
    }
    return 0;
}
#define APPLY_AFFINE_2D_X0(x0,x1,affine) (affine[0]*(x0) + affine[1]*(x1) + affine[2])
#define APPLY_AFFINE_2D_X1(x0,x1,affine) (affine[3]*(x0) + affine[4]*(x1) + affine[5])
int warpImageAffine(double *img, int nrImg, int ncImg, double *affine, double *warped, int nrRef, int ncRef){
    double *res=warped;
    memset(warped, 0, sizeof(double)*nrRef*ncRef);
        for(int i=0;i<nrRef;++i){
            for(int j=0;j<ncRef;++j, ++res){
                double dii=APPLY_AFFINE_2D_X0(i,j,affine);
                double djj=APPLY_AFFINE_2D_X1(i,j,affine);
                if((dii<0) || (djj<0) || (dii>nrImg-1)||(djj>ncImg-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (ii>=nrImg)||(jj>=ncImg)){//no one is affected
                    continue;
                }
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                //---top-left
                (*res)=0;
                double *z=&img[ii*ncImg+jj];
                (*res)+=alpha*beta*(*z);
                //---top-right
                ++jj;
                if(jj<ncImg){
                    z=&img[ii*ncImg+jj];
                    (*res)+=alpha*cbeta*(*z);
                }
                //---bottom-right
                ++ii;
                if((ii>=0)&&(jj>=0)&&(ii<nrImg)&&(jj<ncImg)){
                    z=&img[ii*ncImg+jj];
                    (*res)+=calpha*cbeta*(*z);
                }
                //---bottom-left
                --jj;
                if((ii>=0)&&(jj>=0)&&(ii<nrImg)&&(jj<ncImg)){
                    z=&img[ii*ncImg+jj];
                    (*res)+=calpha*beta*(*z);
                }
            }
        }
    return 0;
}

int warpImage(double *img, int nrImg, int ncImg, double *d1, int nrows, int ncols, double *affinePre, double *affinePost, double *warped){
    double zero[2]={0.0, 0.0};
    double *dx=d1;
    int offset=2;
    if(d1==NULL){
        dx=zero;
        offset=0;
    }
    double *res=warped;
    memset(warped, 0, sizeof(double)*nrows*ncols); 
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=offset, ++res){
                double dii,djj;
                if(affinePre!=NULL){
                    dii=APPLY_AFFINE_2D_X0(i,j,affinePre)+dx[0];
                    djj=APPLY_AFFINE_2D_X1(i,j,affinePre)+dx[1];
                }else{
                    dii=i+dx[0];
                    djj=j+dx[1];
                }
                if(affinePost!=NULL){
                    double temp=APPLY_AFFINE_2D_X0(dii,djj,affinePost);
                    djj=APPLY_AFFINE_2D_X1(dii,djj,affinePost);
                    dii=temp;
                }
                if((dii<0) || (djj<0) || (dii>nrImg-1)||(djj>ncImg-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (ii>=nrImg)||(jj>=ncImg)){//no one is affected
                    continue;
                }
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                //---top-left
                (*res)=0;
                double *z=&img[ii*ncImg+jj];
                (*res)+=alpha*beta*(*z);
                //---top-right
                ++jj;
                if(jj<ncImg){
                    z=&img[ii*ncImg+jj];
                    (*res)+=alpha*cbeta*(*z);
                }
                //---bottom-right
                ++ii;
                if((ii>=0)&&(jj>=0)&&(ii<nrImg)&&(jj<ncImg)){
                    z=&img[ii*ncImg+jj];
                    (*res)+=calpha*cbeta*(*z);
                }
                //---bottom-left
                --jj;
                if((ii>=0)&&(jj>=0)&&(ii<nrImg)&&(jj<ncImg)){
                    z=&img[ii*ncImg+jj];
                    (*res)+=calpha*beta*(*z);
                }
            }
        }
    return 0;
}

int warpImageNN(double *img, int nrImg, int ncImg, double *d1, int nrows, int ncols, double *affinePre, double *affinePost, double *warped){
    double zero[2]={0.0, 0.0};
    double *dx=d1;
    int offset=2;
    if(d1==NULL){
        offset=0;
        dx=zero;
    }
    double *res=warped;
    memset(warped, 0, sizeof(double)*nrows*ncols); 
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=offset, ++res){
                double dii,djj;
                if(affinePre!=NULL){
                    dii=APPLY_AFFINE_2D_X0(i,j,affinePre)+dx[0];
                    djj=APPLY_AFFINE_2D_X1(i,j,affinePre)+dx[1];
                }else{
                    dii=i+dx[0];
                    djj=j+dx[1];
                }
                if(affinePost!=NULL){
                    double temp=APPLY_AFFINE_2D_X0(dii,djj,affinePost);
                    djj=APPLY_AFFINE_2D_X1(dii,djj,affinePost);
                    dii=temp;
                }
                if((dii<0) || (djj<0) || (dii>nrImg-1)||(djj>ncImg-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (ii>=nrImg)||(jj>=ncImg)){//no one is affected
                    continue;
                }
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                if(alpha<calpha){
                    ++ii;
                }
                if(beta<cbeta){
                    ++jj;
                }
                if((ii<0) || (jj<0) || (ii>=nrImg)||(jj>=ncImg)){//no one is affected
                    continue;
                }else{
                    (*res)=img[ii*ncImg + jj];
                }
            }
        }
    return 0;
}

int warpDiscreteImageNNAffine(int *img, int nrImg, int ncImg, double *affine, int *warped, int nrRef, int ncRef){
    int *res=warped;
    memset(warped, 0, sizeof(int)*nrRef*ncRef);
        for(int i=0;i<nrRef;++i){
            for(int j=0;j<ncRef;++j, ++res){
                double dii,djj;
                if(affine!=NULL){
                    dii=APPLY_AFFINE_2D_X0(i,j,affine);
                    djj=APPLY_AFFINE_2D_X1(i,j,affine);
                }else{
                    dii=i;
                    djj=j;
                }
                if((dii<0) || (djj<0) || (dii>nrImg-1)||(djj>ncImg-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (ii>=nrImg)||(jj>=ncImg)){//no one is affected
                    continue;
                }
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                if(alpha<calpha){
                    ++ii;
                }
                if(beta<cbeta){
                    ++jj;
                }
                if((ii<0) || (jj<0) || (ii>=nrImg)||(jj>=ncImg)){//no one is affected
                    continue;
                }else{
                    (*res)=img[ii*ncImg + jj];
                }
            }
        }
    return 0;
}


/*
    Warp volume using Nearest Neighbor interpolation
*/
int warpDiscreteImageNN(int *img, int nrImg, int ncImg, double *d1, int nrows, int ncols, double *affinePre, double *affinePost, int *warped){
    double zero[2]={0.0, 0.0};
    double *dx=d1;
    int offset=2;
    if(d1==NULL){
        dx=zero;
        offset=0;
    }
    int *res=warped;
    memset(warped, 0, sizeof(int)*nrows*ncols); 
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=offset, ++res){
                double dii,djj;
                if(affinePre!=NULL){
                    dii=APPLY_AFFINE_2D_X0(i,j,affinePre)+dx[0];
                    djj=APPLY_AFFINE_2D_X1(i,j,affinePre)+dx[1];
                }else{
                    dii=i+dx[0];
                    djj=j+dx[1];
                }
                if(affinePost!=NULL){
                    double temp=APPLY_AFFINE_2D_X0(dii,djj,affinePost);
                    djj=APPLY_AFFINE_2D_X1(dii,djj,affinePost);
                    dii=temp;
                }
                if((dii<0) || (djj<0) || (dii>nrImg-1)||(djj>ncImg-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (ii>=nrImg)||(jj>=ncImg)){//no one is affected
                    continue;
                }
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                if(alpha<calpha){
                    ++ii;
                }
                if(beta<cbeta){
                    ++jj;
                }
                if((ii<0) || (jj<0) || (ii>=nrImg)||(jj>=ncImg)){//no one is affected
                    continue;
                }else{
                    (*res)=img[ii*ncImg + jj];
                }
            }
        }
    return 0;
}

#define APPLY_AFFINE_X0(x0,x1,x2,affine) (affine[0]*(x0) + affine[1]*(x1) + affine[2]*(x2) + affine[3])
#define APPLY_AFFINE_X1(x0,x1,x2,affine) (affine[4]*(x0) + affine[5]*(x1) + affine[6]*(x2) + affine[7])
#define APPLY_AFFINE_X2(x0,x1,x2,affine) (affine[8]*(x0) + affine[9]*(x1) + affine[10]*(x2) + affine[11])

int multVectorFieldByAffine3D(double *displacement, int nslices, int nrows, int ncols, double *affine){
    double *d=displacement;
    for(int s=0;s<nslices;++s){
        for(int r=0;r<nrows;++r){
            for(int c=0;c<ncols;++c, d+=3){
                double d0=APPLY_AFFINE_X0(d[0], d[1], d[2], affine)-affine[3];
                double d1=APPLY_AFFINE_X1(d[0], d[1], d[2], affine)-affine[7];
                double d2=APPLY_AFFINE_X2(d[0], d[1], d[2], affine)-affine[11];
                d[0]=d0;
                d[1]=d1;
                d[2]=d2;
            }
        }
    }
    return 0;
}

int warpVolumeAffine(double *volume, int nsVol, int nrVol, int ncVol, double *affine, double *warped, int nsRef, int nrRef, int ncRef){
    int sliceSizeVol=nrVol*ncVol;
    double *res=warped;
    memset(warped, 0, sizeof(double)*nsRef*nrRef*ncRef);
    for(int k=0;k<nsRef;++k){
        for(int i=0;i<nrRef;++i){
            for(int j=0;j<ncRef;++j, ++res){
                double dkk=APPLY_AFFINE_X0(k,i,j,affine);
                double dii=APPLY_AFFINE_X1(k,i,j,affine);
                double djj=APPLY_AFFINE_X2(k,i,j,affine);
                if((dii<0) || (djj<0) || (dkk<0) || (dii>nrVol-1)||(djj>ncVol-1)||(dkk>nsVol-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int kk=floor(dkk);
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (kk<0) || (ii>=nrVol)||(jj>=ncVol)||(kk>=nsVol)){//no one is affected
                    continue;
                }
                double cgamma=dkk-kk;
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                double gamma=1-cgamma;
                //---top-left
                (*res)=0;
                double *z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                (*res)+=alpha*beta*gamma*(*z);
                //---top-right
                ++jj;
                if(jj<ncVol){
                    z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                    (*res)+=alpha*cbeta*gamma*(*z);
                }
                //---bottom-right
                ++ii;
                if((ii>=0)&&(jj>=0)&&(ii<nrVol)&&(jj<ncVol)){
                    z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                    (*res)+=calpha*cbeta*gamma*(*z);
                }
                //---bottom-left
                --jj;
                if((ii>=0)&&(jj>=0)&&(ii<nrVol)&&(jj<ncVol)){
                    z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                    (*res)+=calpha*beta*gamma*(*z);
                }
                ++kk;
                if(kk<nsVol){
                    --ii;
                    z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                    (*res)+=alpha*beta*cgamma*(*z);
                    ++jj;
                    if(jj<ncVol){
                        z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                        (*res)+=alpha*cbeta*cgamma*(*z);
                    }
                    //---bottom-right
                    ++ii;
                    if((ii>=0)&&(jj>=0)&&(ii<nrVol)&&(jj<ncVol)){
                        z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                        (*res)+=calpha*cbeta*cgamma*(*z);
                    }
                    //---bottom-left
                    --jj;
                    if((ii>=0)&&(jj>=0)&&(ii<nrVol)&&(jj<ncVol)){
                        z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                        (*res)+=calpha*beta*cgamma*(*z);
                    }
                }
            }
        }
    }
    return 0;
}

int warpVolume(double *volume, int nsVol, int nrVol, int ncVol, double *d1, int nslices, int nrows, int ncols, double *affinePre, double *affinePost, double *warped){
    double zero[3]={0.0,0.0,0.0};
    int sliceSizeVol=nrVol*ncVol;
    double *dx=d1;
    int offset=3;
    if(d1==NULL){
        dx=zero;
        offset=0;
    }
    double *res=warped;
    memset(warped, 0, sizeof(double)*nslices*nrows*ncols); 
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=offset, ++res){
                double dkk,dii,djj;
                if(affinePre!=NULL){
                    dkk=APPLY_AFFINE_X0(k,i,j,affinePre)+dx[0];
                    dii=APPLY_AFFINE_X1(k,i,j,affinePre)+dx[1];
                    djj=APPLY_AFFINE_X2(k,i,j,affinePre)+dx[2];
                }else{
                    dkk=k+dx[0];
                    dii=i+dx[1];
                    djj=j+dx[2];
                }
                if(affinePost!=NULL){
                    double tmp0=APPLY_AFFINE_X0(dkk,dii,djj,affinePost);
                    double tmp1=APPLY_AFFINE_X1(dkk,dii,djj,affinePost);
                    djj=APPLY_AFFINE_X2(dkk,dii,djj,affinePost);
                    dii=tmp1;
                    dkk=tmp0;
                }
                if((dii<0) || (djj<0) || (dkk<0) || (dii>nrVol-1)||(djj>ncVol-1)||(dkk>nsVol-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int kk=floor(dkk);
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (kk<0) || (ii>=nrVol)||(jj>=ncVol)||(kk>=nsVol)){//no one is affected
                    continue;
                }
                double cgamma=dkk-kk;
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                double gamma=1-cgamma;
                //---top-left
                (*res)=0;
                double *z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                (*res)+=alpha*beta*gamma*(*z);
                //---top-right
                ++jj;
                if(jj<ncVol){
                    z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                    (*res)+=alpha*cbeta*gamma*(*z);
                }
                //---bottom-right
                ++ii;
                if((ii>=0)&&(jj>=0)&&(ii<nrVol)&&(jj<ncVol)){
                    z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                    (*res)+=calpha*cbeta*gamma*(*z);
                }
                //---bottom-left
                --jj;
                if((ii>=0)&&(jj>=0)&&(ii<nrVol)&&(jj<ncVol)){
                    z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                    (*res)+=calpha*beta*gamma*(*z);
                }
                ++kk;
                if(kk<nsVol){
                    --ii;
                    z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                    (*res)+=alpha*beta*cgamma*(*z);
                    ++jj;
                    if(jj<ncVol){
                        z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                        (*res)+=alpha*cbeta*cgamma*(*z);
                    }
                    //---bottom-right
                    ++ii;
                    if((ii>=0)&&(jj>=0)&&(ii<nrVol)&&(jj<ncVol)){
                        z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                        (*res)+=calpha*cbeta*cgamma*(*z);
                    }
                    //---bottom-left
                    --jj;
                    if((ii>=0)&&(jj>=0)&&(ii<nrVol)&&(jj<ncVol)){
                        z=&volume[kk*sliceSizeVol+ii*ncVol+jj];
                        (*res)+=calpha*beta*cgamma*(*z);
                    }
                }
            }
        }
    }
    return 0;
}

/*
    Warp volume using Nearest Neighbor interpolation
*/
int warpVolumeNN(double *volume, int nsVol, int nrVol, int ncVol, double *d1, int nslices, int nrows, int ncols, double *affinePre, double *affinePost, double *warped){
    double zero[3]={0.0,0.0,0.0};
    int sliceSizeVol=nrVol*ncVol;
    double *dx=d1;
    int offset=3;
    if(d1==NULL){
        dx=zero;
        offset=0;
    }
    double *res=warped;
    memset(warped, 0, sizeof(double)*nslices*nrows*ncols); 
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=offset, ++res){
                double dkk,dii,djj;
                if(affinePre!=NULL){
                    dkk=APPLY_AFFINE_X0(k,i,j,affinePre)+dx[0];
                    dii=APPLY_AFFINE_X1(k,i,j,affinePre)+dx[1];
                    djj=APPLY_AFFINE_X2(k,i,j,affinePre)+dx[2];
                }else{
                    dkk=k+dx[0];
                    dii=i+dx[1];
                    djj=j+dx[2];
                }
                if(affinePost!=NULL){
                    double tmp0=APPLY_AFFINE_X0(dkk,dii,djj,affinePost);
                    double tmp1=APPLY_AFFINE_X1(dkk,dii,djj,affinePost);
                    djj=APPLY_AFFINE_X2(dkk,dii,djj,affinePost);
                    dii=tmp1;
                    dkk=tmp0;
                }
                if((dii<0) || (djj<0) || (dkk<0) || (dii>nrVol-1)||(djj>ncVol-1)||(dkk>nsVol-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int kk=floor(dkk);
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (kk<0) || (ii>=nrVol)||(jj>=ncVol)||(kk>=nsVol)){//no one is affected
                    continue;
                }
                double cgamma=dkk-kk;
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                double gamma=1-cgamma;
                if(gamma<cgamma){
                    ++kk;
                }
                if(alpha<calpha){
                    ++ii;
                }
                if(beta<cbeta){
                    ++jj;
                }
                if((ii<0) || (jj<0) || (kk<0) || (ii>=nrVol)||(jj>=ncVol)||(kk>=nsVol)){//no one is affected
                    continue;
                }else{
                    (*res)=volume[kk*sliceSizeVol + ii*ncVol + jj];
                }
            }
        }
    }
    return 0;
}

int warpDiscreteVolumeNNAffine(int *volume, int nsVol, int nrVol, int ncVol, double *affine, int *warped, int nsRef, int nrRef, int ncRef){
    int sliceSizeVol=nrVol*ncVol;
    int *res=warped;
    memset(warped, 0, sizeof(int)*nsRef*nrRef*ncRef);
    for(int k=0;k<nsRef;++k){
        for(int i=0;i<nrRef;++i){
            for(int j=0;j<ncRef;++j, ++res){
                double dkk,dii,djj;
                if(affine!=NULL){
                    dkk=APPLY_AFFINE_X0(k,i,j,affine);
                    dii=APPLY_AFFINE_X1(k,i,j,affine);
                    djj=APPLY_AFFINE_X2(k,i,j,affine);
                }else{
                    dkk=k;
                    dii=i;
                    djj=j;
                }
                if((dii<0) || (djj<0) || (dkk<0) || (dii>nrVol-1)||(djj>ncVol-1)||(dkk>nsVol-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int kk=floor(dkk);
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (kk<0) || (ii>=nrVol)||(jj>=ncVol)||(kk>=nsVol)){//no one is affected
                    continue;
                }
                double cgamma=dkk-kk;
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                double gamma=1-cgamma;
                if(gamma<cgamma){
                    ++kk;
                }
                if(alpha<calpha){
                    ++ii;
                }
                if(beta<cbeta){
                    ++jj;
                }
                if((ii<0) || (jj<0) || (kk<0) || (ii>=nrVol)||(jj>=ncVol)||(kk>=nsVol)){//no one is affected
                    continue;
                }else{
                    (*res)=volume[kk*sliceSizeVol + ii*ncVol + jj];
                }
            }
        }
    }
    return 0;
}


/*
    Warp volume using Nearest Neighbor interpolation
*/
int warpDiscreteVolumeNN(int *volume, int nsVol, int nrVol, int ncVol, double *d1, int nslices, int nrows, int ncols, double *affinePre, double *affinePost, int *warped){
    double zero[3]={0.0,0.0,0.0};
    int sliceSizeVol=nrVol*ncVol;
    double *dx=d1;
    int offset=3;
    if(d1==NULL){
        dx=zero;
        offset=0;
    }
    int *res=warped;
    memset(warped, 0, sizeof(int)*nslices*nrows*ncols); 
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=offset, ++res){
                double dkk,dii,djj;
                if(affinePre!=NULL){
                    dkk=APPLY_AFFINE_X0(k,i,j,affinePre)+dx[0];
                    dii=APPLY_AFFINE_X1(k,i,j,affinePre)+dx[1];
                    djj=APPLY_AFFINE_X2(k,i,j,affinePre)+dx[2];
                }else{
                    dkk=k+dx[0];
                    dii=i+dx[1];
                    djj=j+dx[2];
                }
                if(affinePost!=NULL){
                    double tmp0=APPLY_AFFINE_X0(dkk,dii,djj,affinePost);
                    double tmp1=APPLY_AFFINE_X1(dkk,dii,djj,affinePost);
                    djj=APPLY_AFFINE_X2(dkk,dii,djj,affinePost);
                    dii=tmp1;
                    dkk=tmp0;
                }
                if((dii<0) || (djj<0) || (dkk<0) || (dii>nrVol-1)||(djj>ncVol-1)||(dkk>nsVol-1)){//no one is affected
                    continue;
                }
                //find the top left index and the interpolation coefficients
                int kk=floor(dkk);
                int ii=floor(dii);
                int jj=floor(djj);
                if((ii<0) || (jj<0) || (kk<0) || (ii>=nrVol)||(jj>=ncVol)||(kk>=nsVol)){//no one is affected
                    continue;
                }
                double cgamma=dkk-kk;
                double calpha=dii-ii;//by definition these factors are nonnegative
                double cbeta=djj-jj;
                double alpha=1-calpha;
                double beta=1-cbeta;
                double gamma=1-cgamma;
                if(gamma<cgamma){
                    ++kk;
                }
                if(alpha<calpha){
                    ++ii;
                }
                if(beta<cbeta){
                    ++jj;
                }
                if((ii<0) || (jj<0) || (kk<0) || (ii>=nrVol)||(jj>=ncVol)||(kk>=nsVol)){//no one is affected
                    continue;
                }else{
                    (*res)=volume[kk*sliceSizeVol + ii*ncVol + jj];
                }
            }
        }
    }
    return 0;
}

int prependAffineToDisplacementField2D(double *d1, int nrows, int ncols, double *affine){
    if(affine==NULL){
        return 0;
    }
    double *dx=d1;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, dx+=2){
            dx[0]+=APPLY_AFFINE_2D_X0(i,j,affine)-i;
            dx[1]+=APPLY_AFFINE_2D_X1(i,j,affine)-j;
        }
    }
    return 0;
}

int prependAffineToDisplacementField3D(double *d1, int nslices, int nrows, int ncols, double *affine){
    if(affine==NULL){
        return 0;
    }
    double *dx=d1;
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=3){
                dx[0]+=APPLY_AFFINE_X0(k,i,j,affine)-k;
                dx[1]+=APPLY_AFFINE_X1(k,i,j,affine)-i;
                dx[2]+=APPLY_AFFINE_X2(k,i,j,affine)-j;
            }
        }
    }
    return 0;
}

int appendAffineToDisplacementField3D(double *d1, int nslices, int nrows, int ncols, double *affine){
    if(affine==NULL){
        return 0;
    }
    double *dx=d1;
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=3){
                double dkk=dx[0]+k,dii=dx[1]+i,djj=dx[2]+j;
                dx[0]=APPLY_AFFINE_X0(dkk,dii,djj,affine)-k;
                dx[1]=APPLY_AFFINE_X1(dkk,dii,djj,affine)-i;
                dx[2]=APPLY_AFFINE_X2(dkk,dii,djj,affine)-j;
            }
        }
    }
    return 0;
}

int appendAffineToDisplacementField2D(double *d1, int nrows, int ncols, double *affine){
    if(affine==NULL){
        return 0;
    }
    double *dx=d1;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, dx+=2){
            double dii=dx[0]+i,djj=dx[1]+j;
            dx[0]=APPLY_AFFINE_2D_X0(dii,djj,affine)-i;
            dx[1]=APPLY_AFFINE_2D_X1(dii,djj,affine)-j;
        }
    }
    return 0;
}

/*
    Interpolates the vector field d2 at d1: d2(d1(x)) (i.e. applies first d1, then d2 to the result)
    Seen as a linear operator, it is defined by d1 and the input vector is d2
*/
int vectorFieldInterpolation(double *d1, double *d2, int nrows, int ncols, double *comp){
    double *dx=d1;
    double *res=comp;
    memset(comp, 0, sizeof(double)*nrows*ncols*2); 
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, dx+=2, res+=2){
            double dii=i+dx[0];
            double djj=j+dx[1];
            int ii=floor(dii);
            int jj=floor(djj);
            if((ii<0)||(jj<0)||(ii>=nrows)||(jj>=ncols)){
                continue;
            }
            double calpha=dii-ii;//by definition these factors are nonnegative
            double cbeta=djj-jj;
            double alpha=1-calpha;
            double beta=1-cbeta;
            //---top-left
            double *z=&d2[2*(ii*ncols+jj)];
            res[0]+=alpha*beta*z[0];
            res[1]+=alpha*beta*z[1];
            //---top-right
            ++jj;
            if(jj<ncols){
                z=&d2[2*(ii*ncols+jj)];
                res[0]+=alpha*cbeta*z[0];
                res[1]+=alpha*cbeta*z[1];
            }
            //---bottom-right
            ++ii;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                z=&d2[2*(ii*ncols+jj)];
                res[0]+=calpha*cbeta*z[0];
                res[1]+=calpha*cbeta*z[1];
            }
            //---bottom-left
            --jj;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                z=&d2[2*(ii*ncols+jj)];
                res[0]+=calpha*beta*z[0];
                res[1]+=calpha*beta*z[1];
            }
        }
    }
    return 0;
}

/*
    Vector field interpolation taking as input the displacements from two separate arrays
*/
int vectorFieldInterpolation(double *d1r, double *d1c, double *d2r, double *d2c, int nrows, int ncols, double *compr, double *compc){
    double *dr=d1r;
    double *dc=d1c;
    double *resr=compr;
    double *resc=compc;
    memset(compr, 0, sizeof(double)*nrows*ncols); 
    memset(compc, 0, sizeof(double)*nrows*ncols); 
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, ++dr, ++dc, ++resr, ++resc){
            double dii=i+*dr;
            double djj=j+*dc;
            int ii=floor(dii);
            int jj=floor(djj);
            if((ii<0)||(jj<0)||(ii>=nrows)||(jj>=ncols)){
                continue;
            }
            double calpha=dii-ii;//by definition these factors are nonnegative
            double cbeta=djj-jj;
            double alpha=1-calpha;
            double beta=1-cbeta;
            //---top-left
            double *zr=&d2r[ii*ncols+jj];
            double *zc=&d2c[ii*ncols+jj];
            (*resr)+=alpha*beta*(*zr);
            (*resc)+=alpha*beta*(*zc);
            //---top-right
            ++jj;
            if(jj<ncols){
                zr=&d2r[ii*ncols+jj];
                zc=&d2c[ii*ncols+jj];
                (*resr)+=alpha*cbeta*(*zr);
                (*resc)+=alpha*cbeta*(*zc);
            }
            //---bottom-right
            ++ii;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                zr=&d2r[ii*ncols+jj];
                zc=&d2c[ii*ncols+jj];
                (*resr)+=calpha*cbeta*(*zr);
                (*resc)+=calpha*cbeta*(*zc);
            }
            //---bottom-left
            --jj;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                zr=&d2r[ii*ncols+jj];
                zc=&d2c[ii*ncols+jj];
                (*resr)+=calpha*beta*(*zr);
                (*resc)+=calpha*beta*(*zc);
            }
        }
    }
    return 0;
}


//Computes the Adjoint transformation associated to the interpolation transformation defined by d1, applied to d2.
//i.e. d1 defines the transformation and d2 is the input vector
int vectorFieldAdjointInterpolation(double *d1, double *d2, int nrows, int ncols, double *sol){
    double *dx=d1;
    double *z=d2;
    memset(sol, 0, sizeof(double)*nrows*ncols*2); 
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, dx+=2, z+=2){
            double dii=i+dx[0];
            double djj=j+dx[1];
            int ii=floor(dii);
            int jj=floor(djj);
            if((ii<0)||(jj<0)||(ii>=nrows)||(jj>=ncols)){
                continue;
            }
            double calpha=dii-ii;//by definition these factors are nonnegative
            double cbeta=djj-jj;
            double alpha=1-calpha;
            double beta=1-cbeta;
            //top left
            double *res=&sol[2*(ii*ncols+jj)];
            res[0]+=alpha*beta*z[0];
            res[1]+=alpha*beta*z[1];
            //---top-right
            ++jj;
            if(jj<ncols){
                double *res=&sol[2*(ii*ncols+jj)];
                res[0]+=alpha*cbeta*z[0];
                res[1]+=alpha*cbeta*z[1];
            }
            //---bottom-right
            ++ii;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                double *res=&sol[2*(ii*ncols+jj)];
                res[0]+=calpha*cbeta*z[0];
                res[1]+=calpha*cbeta*z[1];
            }
            //---bottom-left
            --jj;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                double *res=&sol[2*(ii*ncols+jj)];
                res[0]+=calpha*beta*z[0];
                res[1]+=calpha*beta*z[1];
            }
        }
    }
    return 0;
}

/*
    Vector field adjoint interpolation taking as input the displacements from two separate arrays
*/
int vectorFieldAdjointInterpolation(double *d1r, double *d1c, double *d2r, double *d2c, int nrows, int ncols, double *solr, double *solc){
    double *dr=d1r;
    double *dc=d1c;
    double *zr=d2r;
    double *zc=d2c;
    memset(solr, 0, sizeof(double)*nrows*ncols); 
    memset(solc, 0, sizeof(double)*nrows*ncols); 
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, ++dr, ++dc, ++zr, ++zc){
            double dii=i+(*dr);
            double djj=j+(*dc);
            int ii=floor(dii);
            int jj=floor(djj);
            if((ii<0)||(jj<0)||(ii>=nrows)||(jj>=ncols)){
                continue;
            }
            double calpha=dii-ii;//by definition these factors are nonnegative
            double cbeta=djj-jj;
            double alpha=1-calpha;
            double beta=1-cbeta;
            //top left
            double *resr=&solr[ii*ncols+jj];
            double *resc=&solc[ii*ncols+jj];
            (*resr)+=alpha*beta*(*zr);
            (*resc)+=alpha*beta*(*zc);
            //---top-right
            ++jj;
            if(jj<ncols){
                resr=&solr[ii*ncols+jj];
                resc=&solc[ii*ncols+jj];
                (*resr)+=alpha*cbeta*(*zr);
                (*resc)+=alpha*cbeta*(*zc);
            }
            //---bottom-right
            ++ii;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                resr=&solr[ii*ncols+jj];
                resc=&solc[ii*ncols+jj];
                (*resr)+=calpha*cbeta*(*zr);
                (*resc)+=calpha*cbeta*(*zc);
            }
            //---bottom-left
            --jj;
            if((ii>=0)&&(jj>=0)&&(ii<nrows)&&(jj<ncols)){
                resr=&solr[ii*ncols+jj];
                resc=&solc[ii*ncols+jj];
                (*resr)+=calpha*beta*(*zr);
                (*resc)+=calpha*beta*(*zc);
            }
        }
    }
    return 0;
}

int invertVectorField_TV_L2(double *forward, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *inv){
    int nsites=nrows*ncols;
    double *fr=new double[nsites];
    double *fc=new double[nsites];
    for(int i=0;i<nsites;++i){
        fr[i]=forward[2*i];
        fc[i]=forward[2*i+1];
    }
    double *tmpr=new double[nsites];
    double *tmpc=new double[nsites];
    double *tmpr2=new double[nsites];
    double *tmpc2=new double[nsites];
    double *prr=new double[nsites];
    double *prc=new double[nsites];
    double *pcr=new double[nsites];
    double *pcc=new double[nsites];
    double *qr=new double[nsites];
    double *qc=new double[nsites];
    double *sbarr=new double[nsites];
    double *sbarc=new double[nsites];
    double *sr=new double[nsites];
    double *sc=new double[nsites];
    double L=8;
    double sigma=1.0/L;
    double tau=sigma;
    double theta=0.8;
    double error=1+tolerance;
    int iter=0;
    /*memset(sbarr, 0, sizeof(double)*nrows*ncols);//initialize the inverse field
    memset(sbarc, 0, sizeof(double)*nrows*ncols);*/
    initializeNearestNeighborInverseField(forward, nrows, ncols, inv, NULL);
    for(int i=0;i<nsites;++i){
        sbarr[i]=inv[2*i];
        sbarc[i]=inv[2*i+1];
    }
    memset(sr, 0, sizeof(double)*nrows*ncols);//initialize the inverse field
    memset(sc, 0, sizeof(double)*nrows*ncols);
    memset(prr, 0, sizeof(double)*nrows*ncols);//initialize the dual Jacobian
    memset(prc, 0, sizeof(double)*nrows*ncols);
    memset(pcr, 0, sizeof(double)*nrows*ncols);
    memset(pcc, 0, sizeof(double)*nrows*ncols);
    memset(qr, 0, sizeof(double)*nrows*ncols);
    memset(qc, 0, sizeof(double)*nrows*ncols);
    double factorq=lambdaParam/(lambdaParam+sigma);
    //FILE *F=fopen("TVL2iterations.txt", "w");
    while((tolerance<error) && (iter<=maxIter)){
        ++iter;
        //update the dual p-variables
        computeGradient(sbarr, nrows, ncols, tmpr, tmpc);
        for(int i=nsites-1;i>=0;--i){
            prr[i]+=sigma*tmpr[i];
            prc[i]+=sigma*tmpc[i];
        }
        computeGradient(sbarc, nrows, ncols, tmpr, tmpc);
        for(int i=nsites-1;i>=0;--i){
            pcr[i]+=sigma*tmpr[i];
            pcc[i]+=sigma*tmpc[i];
            double nrm=(prr[i]*prr[i])+(prc[i]*prc[i])+(pcr[i]*pcr[i])+(pcc[i]*pcc[i]);
            if(nrm>1){
                nrm=sqrt(nrm);
                prr[i]/=nrm;
                prc[i]/=nrm;
                pcr[i]/=nrm;
                pcc[i]/=nrm;
            }
        }
        //update the dual q-variables
        vectorFieldInterpolation(fr, fc, sbarr, sbarc, nrows, ncols, tmpr, tmpc);
        for(int i=nsites-1;i>=0;--i){
            qr[i]=(qr[i]+sigma*(tmpr[i]+fr[i]))*factorq;
            qc[i]=(qc[i]+sigma*(tmpc[i]+fc[i]))*factorq;
        }
        //update primal variables and step
        vectorFieldAdjointInterpolation(fr, fc, qr, qc, nrows, ncols, tmpr, tmpc);
        for(int i=nsites-1;i>=0;--i){
            sbarr[i]=-theta*sr[i];//save -theta times the previous value of s
            sbarc[i]=-theta*sc[i];
            sr[i]-=tau*tmpr[i];
            sc[i]-=tau*tmpc[i];
            
        }
        computeDivergence(prr, prc, nrows, ncols, tmpr);
        computeDivergence(pcr, pcc, nrows, ncols, tmpc);
        for(int i=nsites-1;i>=0;--i){
            sr[i]+=tau*tmpr[i];
            sc[i]+=tau*tmpc[i];
        }
        //finish comuting the s-bar step
        for(int i=nsites-1;i>=0;--i){
            sbarr[i]+=(1.0+theta)*sr[i];
            sbarc[i]+=(1.0+theta)*sc[i];
        }
        //----compute error----
        double newError=0;
        vectorFieldInterpolation(fr, fc, sr, sc, nrows, ncols, tmpr, tmpc);
        for(int i=0;i<nsites;++i){
            double dr=tmpr[i]+fr[i];
            double dc=tmpc[i]+fc[i];
            newError+=dr*dr+dc*dc;
        }
        newError*=lambdaParam;
        computeGradient(sr, nrows, ncols, tmpr, tmpc);
        computeGradient(sc, nrows, ncols, tmpr2, tmpc2);
        for(int i=0;i<nsites;++i){
            newError+=sqrt(tmpr[i]*tmpr[i]+tmpc[i]*tmpc[i]+tmpr2[i]*tmpr2[i]+tmpc2[i]*tmpc2[i]);
        }
        //fprintf(F, "%d: %e\n", iter, newError);
        error=fabs(error-newError);
    }
    //fclose(F);
    double *g=inv;
    for(int i=0;i<nsites;++i, g+=2){
        g[0]=sr[i];
        g[1]=sc[i];
    }
    delete[] tmpr;
    delete[] tmpc;
    delete[] tmpr2;
    delete[] tmpc2;
    delete[] fr;
    delete[] fc;
    delete[] prr;
    delete[] prc;
    delete[] pcr;
    delete[] pcc;
    delete[] qr;
    delete[] qc;
    delete[] sbarr;
    delete[] sbarc;
    delete[] sr;
    delete[] sc;
    return 0;
}


int invertVectorFieldFixedPoint_deprecated(double *d, int nrows, int ncols, int maxIter, double tolerance, double *invd, double *start, double *stats){
    double error=1+tolerance;
    double substats[3];
    double *temp[3];
    temp[0]=new double[nrows*ncols*2];
    temp[1]=invd;
    temp[2]=new double[nrows*ncols*2];
    //memset(temp[0], 0, sizeof(double)*2*nrows*ncols);
    if(start!=NULL){
        memcpy(temp[0], start, sizeof(double)*2*nrows*ncols);
    }else{
        initializeNearestNeighborInverseField(d, nrows, ncols, temp[0], NULL);
    }
    
    int nsites=2*nrows*ncols;
    int iter;
    double epsilon=0.5;
    for(iter=0;(iter<maxIter) && (tolerance<error);++iter){
        composeVectorFields(temp[iter&1], nrows, ncols, d, nrows, ncols, temp[1-(iter&1)], substats);
        double difmag=0;
        error=0;
        double *p=temp[1-(iter&1)];
        for(int i=0;i<nsites;i+=2, p+=2){
            double mag=sqrt(p[0]*p[0]+p[1]*p[1]);
            error+=mag;
            if(difmag<mag){
                difmag=mag;
            }
        }
        error/=(nrows*ncols);
        p=temp[1-(iter&1)];
        double *q=temp[iter&1];
        for(int i=0;i<nsites;i+=2,p+=2,q+=2){
            p[0]=q[0]-epsilon*p[0];
            p[1]=q[1]-epsilon*p[1];
        }
    }
    if(iter&1){//then the last computation was stored at temp[0]
        memcpy(invd, temp[0], sizeof(double)*2*nrows*ncols);
    }
    delete[] temp[0];
    delete[] temp[2];
    stats[0]=substats[1];
    stats[1]=iter;
    return 0;
}

int invertVectorFieldFixedPoint(double *d, int nr1, int nc1, int nr2, int nc2, int maxIter, double tolerance, double *invd, double *start, double *stats){
    double error=1+tolerance;
    double substats[3];
    double *temp[2];
    temp[0]=new double[nr2*nc2*2];
    temp[1]=invd;
    if(start!=NULL){
        memcpy(temp[0], start, sizeof(double)*2*nr2*nc2);
    }else{
        memset(temp[0], 0, sizeof(double)*2*nr2*nc2);
    }
    
    int nsitesInverse=2*nr2*nc2;
    int iter;
    double epsilon=0.25;
    for(iter=0;(iter<maxIter) && (tolerance<error);++iter){
        composeVectorFields(temp[iter&1], nr1, nc1, d, nr2, nc2, temp[1-(iter&1)], substats);
        double difmag=0;
        error=0;
        double *p=temp[1-(iter&1)];
        for(int i=0;i<nsitesInverse;i+=2, p+=2){
            double mag=sqrt(p[0]*p[0]+p[1]*p[1]);
            error+=mag;
            if(difmag<mag){
                difmag=mag;
            }
        }
        error/=(nr2*nc2);
        p=temp[1-(iter&1)];
        double *q=temp[iter&1];
        for(int i=0;i<nsitesInverse;i+=2,p+=2,q+=2){
            p[0]=q[0]-epsilon*p[0];
            p[1]=q[1]-epsilon*p[1];
        }
    }
    if(iter&1){//then the last computation was stored at temp[0]
        memcpy(invd, temp[0], sizeof(double)*2*nr2*nc2);
    }
    delete[] temp[0];
    stats[0]=substats[1];
    stats[1]=iter;
    return 0;
}

int invertVectorFieldFixedPoint3D_deprecated(double *d, int nslices, int nrows, int ncols, int maxIter, double tolerance, double *invd, double *start, double *stats){
    double error=1+tolerance;
    double *temp[2];
    double substats[3];
    temp[0]=new double[nslices*nrows*ncols*3];
    temp[1]=invd;
    int nsites=3*nslices*nrows*ncols;
    if(start!=NULL){
        memcpy(temp[0], start, sizeof(double)*nsites);
    }else{
        memset(temp[0], 0, sizeof(double)*nsites);
    }
    int iter;
    double epsilon=0.5;
    for(iter=0;(iter<maxIter) && (tolerance<error);++iter){
        composeVectorFields3D(temp[iter&1], nslices, nrows, ncols, d, nslices, nrows, ncols, temp[1-(iter&1)], substats);
        double difmag=0;
        error=0;
        double *p=temp[1-(iter&1)];
        for(int i=0;i<nsites;i+=3, p+=3){
            double mag=sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
            error+=mag;
            if(difmag<mag){
                difmag=mag;
            }
        }
        error/=(nslices*nrows*ncols);
        p=temp[1-(iter&1)];
        double *q=temp[iter&1];
        for(int i=0;i<nsites;i+=3,p+=3,q+=3){
            p[0]=q[0]-epsilon*p[0];
            p[1]=q[1]-epsilon*p[1];
            p[2]=q[2]-epsilon*p[2];
        }
    }
    if(iter&1){//then the last computation was stored at temp[0]
        memcpy(invd, temp[0], sizeof(double)*nsites);
    }
    delete[] temp[0];
    stats[0]=error;
    stats[1]=iter;
    return 0;
}

int invertVectorFieldFixedPoint3D(double *d, int ns1, int nr1, int nc1, int ns2, int nr2, int nc2, int maxIter, double tolerance, double *invd, double *start, double *stats){
    double error=1+tolerance;
    double *temp[2];
    double substats[3];
    temp[0]=new double[ns2*nr2*nc2*3];
    temp[1]=invd;
    int nsitesInv=3*ns2*nr2*nc2;
    if(start!=NULL){
        memcpy(temp[0], start, sizeof(double)*nsitesInv);
    }else{
        memset(temp[0], 0, sizeof(double)*nsitesInv);
    }
    int iter;
    double epsilon=0.5;
    for(iter=0;(iter<maxIter) && (tolerance<error);++iter){
        composeVectorFields3D(temp[iter&1], ns2, nr2, nc2, d, ns1, nr1, nc1, temp[1-(iter&1)], substats);
        double difmag=0;
        error=0;
        double *p=temp[1-(iter&1)];
        for(int i=0;i<nsitesInv;i+=3, p+=3){
            double mag=sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
            error+=mag;
            if(difmag<mag){
                difmag=mag;
            }
        }
        error/=(ns2*nr2*nc2);
        p=temp[1-(iter&1)];
        double *q=temp[iter&1];
        for(int i=0;i<nsitesInv;i+=3,p+=3,q+=3){
            p[0]=q[0]-epsilon*p[0];
            p[1]=q[1]-epsilon*p[1];
            p[2]=q[2]-epsilon*p[2];
        }
    }
    if(iter&1){//then the last computation was stored at temp[0]
        memcpy(invd, temp[0], sizeof(double)*nsitesInv);
    }
    delete[] temp[0];
    stats[0]=error;
    stats[1]=iter;
    return 0;
}


int vectorFieldExponential(double *v, int nrows, int ncols, double *expv, double *invexpv){
    double EXP_EPSILON=0.1;//such that the vector field exponential is approx the identity
    //---compute the maximum norm---
    double stats[3];
    int nsites=nrows*ncols;
    double maxNorm=0;
    double *d=v;
    for(int i=0;i<nsites;++i, d+=2){
        double nn=d[0]*d[0]+d[1]*d[1];
        if(maxNorm<nn){
            maxNorm=nn;
        }
    }
    maxNorm=sqrt(maxNorm);
    int n=0;
    double factor=1.0;
    while(EXP_EPSILON<maxNorm){
        maxNorm*=0.5;
        factor*=0.5;
        ++n;
    }
    //---base case---
    if(n<1){
        memcpy(expv, v, sizeof(double)*2*nsites);
        if(invexpv!=NULL){
            invertVectorField(v, nrows, ncols, 0.5, 100, 1e-4, invexpv, stats);
        }
        return 0;
    }
    //---prepare memory buffers---
    double *tmp[2]={NULL, NULL};
    tmp[n%2]=new double[nsites*2];
    tmp[1-n%2]=expv;
    //---perform binary exponentiation: exponential---
    for(int i=2*nsites-1;i>=0;--i){
        tmp[1][i]=v[i]*factor;
    }
    for(int i=1;i<=n;++i){
        composeVectorFields(tmp[i&1], nrows, ncols, tmp[i&1], nrows, ncols, tmp[1-(i&1)], stats);
    }
    //---perform binary exponentiation: inverse---
    if(invexpv!=NULL){
        tmp[1-n%2]=invexpv;
        for(int i=2*nsites-1;i>=0;--i){
            tmp[0][i]=v[i]*factor;
        }
        invertVectorField(tmp[0], nrows, ncols, 0.1, 20, 1e-4, tmp[1], stats);
        for(int i=1;i<=n;++i){
            composeVectorFields(tmp[i&1], nrows, ncols, tmp[i&1], nrows, ncols, tmp[1-(i&1)], stats);
        }
    }
    delete[] tmp[n%2];
    return 0;
}

int vectorFieldExponential3D(double *v, int nslices, int nrows, int ncols, double *expv, double *invexpv){
    double EXP_EPSILON=0.001;//such that the vector field exponential is approx the identity
    //---compute the maximum norm---
    double stats[3];
    int nsites=nslices*nrows*ncols;
    double maxNorm=0;
    double *d=v;
    for(int i=0;i<nsites;++i, d+=3){
        double nn=d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
        if(maxNorm<nn){
            maxNorm=nn;
        }
    }
    maxNorm=sqrt(maxNorm);
    int n=0;
    double factor=1.0;
    while(EXP_EPSILON<maxNorm){
        maxNorm*=0.5;
        factor*=0.5;
        ++n;
    }
    //---base case---
    if(n<1){
        memcpy(expv, v, sizeof(double)*3*nsites);
        if(invexpv!=NULL){
            invertVectorField3D(v, nslices, nrows, ncols, 0.5, 100, 1e-4, invexpv, stats);
        }
        return 0;
    }
    //---prepare memory buffers---
    double *tmp[2]={NULL, NULL};
    tmp[n%2]=new double[nsites*3];
    tmp[1-n%2]=expv;
    //---perform binary exponentiation: exponential---
    for(int i=3*nsites-1;i>=0;--i){
        tmp[1][i]=v[i]*factor;
    }
    double substats[3];
    for(int i=1;i<=n;++i){
        composeVectorFields3D(tmp[i&1], nslices, nrows, ncols, tmp[i&1], nslices, nrows, ncols, tmp[1-(i&1)], substats);
    }
    //---perform binary exponentiation: inverse---
    if(invexpv!=NULL){
        tmp[1-n%2]=invexpv;
        for(int i=3*nsites-1;i>=0;--i){
            tmp[0][i]=v[i]*factor;
        }
        invertVectorField3D(tmp[0], nslices, nrows, ncols, 0.1, 20, 1e-4, tmp[1], stats);
        for(int i=1;i<=n;++i){
            composeVectorFields3D(tmp[i&1], nslices, nrows, ncols, tmp[i&1], nslices, nrows, ncols, tmp[1-(i&1)],substats);
        }
    }
    delete[] tmp[n%2];
    return 0;
}


int writeDoubleBuffer(double *buffer, int nDoubles, char *fname){
    FILE *F=fopen(fname, "wb");
    fwrite(buffer, sizeof(double), nDoubles, F);
    fclose(F);
    return 0;
}

int readDoubleBuffer(char *fname, int nDoubles, double *buffer){
    FILE *F=fopen(fname, "rb");
    size_t retVal=fread((void*)buffer, sizeof(double), nDoubles, F);
    fclose(F);
    return retVal!=size_t(nDoubles);
}


void createInvertibleDisplacementField(int nrows, int ncols, double b, double m, double *dField){
    int midRow=nrows/2;
    int midCol=ncols/2;
    double *d=dField;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, d+=2){
            int ii=i-midRow;
            int jj=j-midCol;
            double theta=atan2(ii,jj);
            d[0]=ii*(1.0/(1+b*cos(m*theta))-1.0);
            d[1]=jj*(1.0/(1+b*cos(m*theta))-1.0);
        }
    }
}


void countSupportingDataPerPixel(double *forward, int nrows, int ncols, int *counts){
    memset(counts, 0, sizeof(int)*nrows*ncols);
    double *f=forward;
    for(int i=0;i<nrows;++i){
        for(int j=0;j<ncols;++j, f+=2){
            double di=i+f[0];
            double dj=j+f[1];
            int ii=floor(di);
            int jj=floor(dj);
            if((ii<0) || (jj<0) || (ii>=nrows) || (jj>=ncols)){
                continue;
            }
            counts[ii*ncols+jj]++;
            ++jj;
            if(jj<ncols){
                counts[ii*ncols+jj]++;
            }
            ++ii;
            if(jj<ncols && ii<nrows){
                counts[ii*ncols+jj]++;
            }
            --jj;
            if(ii<nrows){
                counts[ii*ncols+jj]++;
            }
            
        }
    }
}


void consecutiveLabelMap(int *v, int n, int *out){
    set<int> labs;
    for(int i=0;i<n;++i){
        labs.insert(v[i]);
    }
    map<int, int> M;
    int nlabs=0;
    for(set<int>::iterator it=labs.begin(); it!=labs.end();++it){
        M[*it]=nlabs;
        ++nlabs;
    }
    for(int i=0;i<n;++i){
        out[i]=M[v[i]];
    }
}

void getVotingSegmentation(int *votes, int nslices, int nrows, int ncols, int nvotes, int *seg){
    int nsites=nslices*nrows*ncols;
    int *v=votes;
    for(int i=0;i<nsites;++i, v+=nvotes){
        map<int, int> cnt;
        for(int j=0;j<nvotes;++j){
           cnt[v[j]]++;
        }
        int best=-1;
        int bestCount=-1;
        for(map<int, int>::iterator it=cnt.begin(); it!=cnt.end();++it){
            if((best<0) || (it->second>bestCount)){
                best=it->first;
                bestCount=it->second;
            }
        }
        seg[i]=best;
    }
}

int getDisplacementRange(double *d, int nslices, int nrows, int ncols, double *affine, double *minVal, double *maxVal){
    double *dx=d;
    minVal[0]=maxVal[0]=d[0];
    minVal[1]=maxVal[1]=d[1];
    minVal[2]=maxVal[2]=d[2];
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j, dx+=3){
                double dkk,dii,djj;
                if(affine!=NULL){
                    dkk=APPLY_AFFINE_X0(k,i,j,affine)+dx[0];
                    dii=APPLY_AFFINE_X1(k,i,j,affine)+dx[1];
                    djj=APPLY_AFFINE_X2(k,i,j,affine)+dx[2];
                }else{
                    dkk=k+dx[0];
                    dii=i+dx[1];
                    djj=j+dx[2];
                }
                if(dkk>maxVal[0]){
                    maxVal[0]=dkk;
                }
                if(dii>maxVal[1]){
                    maxVal[1]=dii;
                }
                if(djj>maxVal[2]){
                    maxVal[2]=djj;
                }
            }
        }
    }
    return 0;
}


int computeJacard(int *A, int *B, int nslices, int nrows, int ncols, double *jacard, int nlabels){
    int *a=A;
    int *b=B;
    int *cnt=new int[nlabels*nlabels];
    int *sz=new int[nlabels];
    memset(cnt, 0, sizeof(int)*nlabels);
    memset(sz, 0, sizeof(int)*nlabels);
    memset(jacard, 0, sizeof(double)*nlabels);
    for(int k=0;k<nslices;++k){
        for(int i=0;i<nrows;++i){
            for(int j=0;j<ncols;++j,++a, ++b){
                int ii,jj;
                if((*a)<(*b)){
                    ii=*a;
                    jj=*b;
                }else{
                    ii=*b;
                    jj=*a;
                }
                sz[ii]++;
                if(ii==jj){
                    cnt[ii]++;
                }else{
                    sz[jj]++;
                }
            }
        }
    }
    for(int i=0;i<nlabels;++i){
        if(sz[i]==0){
            continue;
        }
        jacard[i]=double(cnt[i])/double(sz[i]);
    }
    delete[] cnt;
    delete[] sz;
    return 0;
}


int precomputeCCFactors3D(double *I, double *J, int ns, int nr, int nc, int radius, double *factors){
    const int SI=0;
    const int SI2=1;
    const int SJ=2;
    const int SJ2=3;
    const int SIJ=4;
    const int CNT=5;
    int sliceSize=nc*nr;
    int side=2*radius+1;
    double *lines[6];
    double sums[6];
    for(int k=0;k<6;++k){
        lines[k]=new double[side];
    }
    for(int r=0;r<nr;++r){
        int firstr=MAX(0, r-radius);
        int lastr=MIN(nr-1, r+radius);
        for(int c=0;c<nc;++c){
            int firstc=MAX(0, c-radius);
            int lastc=MIN(nc-1, c+radius);
            //compute factors for line [:,r,c]
            memset(sums, 0, sizeof(sums));
            //Compute all slices and set the sums on the fly
            for(int k=0;k<ns;++k){//compute each slice [k, i={r-radius..r+radius}, j={c-radius, c+radius}]
                int q=k%side;
                for(int t=0;t<6;++t){
                    sums[t]-=lines[t][q];
                    lines[t][q]=0;
                }
                for(int i=firstr;i<=lastr;++i){
                    for(int j=firstc;j<=lastc;++j){
                        int p=k*sliceSize + i*nc + j;
                        lines[SI][q]  += I[p];
                        lines[SI2][q] += I[p]*I[p];
                        lines[SJ][q]  += J[p];
                        lines[SJ2][q] += J[p]*J[p];
                        lines[SIJ][q] += I[p]*J[p];
                        lines[CNT][q] += 1;
                    }
                }
                memset(sums, 0, sizeof(sums));
                for(int t=0;t<6;++t){
                    for(int qq=0;qq<side;++qq){
                        sums[t]+=lines[t][qq];
                    }
                }
                if(k>=radius){
                    int s=k-radius;//s is the voxel that is affected by the cube with slices [s-radius..s+radius, :, :]
                    int pos=s*sliceSize + r*nc + c;
                    double *F=&factors[5*pos];
                    double Imean=sums[SI]/sums[CNT];
                    double Jmean=sums[SJ]/sums[CNT];
                    F[0] = I[pos] - Imean;
                    F[1] = J[pos] - Jmean;
                    F[2] = sums[SIJ] - Jmean * sums[SI] - Imean * sums[SJ] + sums[CNT] * Jmean * Imean;
                    F[3] = sums[SI2] - Imean * sums[SI] - Imean * sums[SI] + sums[CNT] * Imean * Imean;
                    F[4] = sums[SJ2] - Jmean * sums[SJ] - Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean;
                }
            }
            //Finally set the values at the end of the line
            for(int s=ns-radius;s<ns;++s){
                int k=s+radius;//this would be the last slice to be processed for voxel [s,r,c], if it existed
                int q=k%side;
                for(int t=0;t<6;++t){
                    sums[t]-=lines[t][q];
                }
                int pos=s*sliceSize + r*nc + c;
                double *F=&factors[5*pos];
                double Imean=sums[SI]/sums[CNT];
                double Jmean=sums[SJ]/sums[CNT];
                F[0] = I[pos] - Imean;
                F[1] = J[pos] - Jmean;
                F[2] = sums[SIJ] - Jmean * sums[SI] - Imean * sums[SJ] + sums[CNT] * Jmean * Imean;
                F[3] = sums[SI2] - Imean * sums[SI] - Imean * sums[SI] + sums[CNT] * Imean * Imean;
                F[4] = sums[SJ2] - Jmean * sums[SJ] - Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean;
            }
        }
    }
    for(int k=0;k<6;++k){
        delete[] lines[k];
    }
    return 0;
}

double computeCCForwardStep3D(double *gradFixed, double *gradMoving, int ns, int nr, int nc, double *factors, double *out){
    double *deriv=out;
    int nvox=ns*nr*nc;
    double *F=factors;
    double *gradI=gradFixed;
    double *gradJ=gradMoving;
    memset(out, 0, sizeof(double)*3*nvox);
    double energy=0;
    for(int s=0;s<ns;++s){
        for(int r=0;r<nr;++r){
            for(int c=0;c<nc;++c, deriv+=3, gradI+=3, gradJ+=3, F+=5){
                double Ii  = F[0];
                double Ji  = F[1];
                double sfm = F[2];
                double sff = F[3];
                double smm = F[4];
                if(sff==0.0 || smm==0.0){
                    continue;
                }
                double localCorrelation = 0;
                if(sff*smm>1e-5){
                    localCorrelation=sfm*sfm/(sff*smm);
                }
                if(localCorrelation<1){//avoid bad values...
                    energy-=localCorrelation;
                }
                double temp = 2.0 * sfm / (sff * smm) * ( Ji - sfm / sff * Ii );
                for(int qq=0;qq<3;++qq){
                    deriv[qq] -= temp*gradI[qq];
                }
            }
        }
    }
    return energy;
}

double computeCCBackwardStep3D(double *gradFixed, double *gradMoving, int ns, int nr, int nc, double *factors, double *out){
    double *deriv=out;
    int nvox=ns*nr*nc;
    double *F=factors;
    double *gradI=gradFixed;
    double *gradJ=gradMoving;
    memset(out, 0, sizeof(double)*3*nvox);
    double energy=0;
    for(int s=0;s<ns;++s){
        for(int r=0;r<nr;++r){
            for(int c=0;c<nc;++c, deriv+=3, gradI+=3, gradJ+=3, F+=5){
                double Ii  = F[0];
                double Ji  = F[1];
                double sfm = F[2];
                double sff = F[3];
                double smm = F[4];
                if(sff==0.0 || smm==0.0){
                    continue;
                }
                double localCorrelation = 0;
                if(sff*smm>1e-5){
                    localCorrelation=sfm*sfm/(sff*smm);
                }
                if(localCorrelation<1){//avoid bad values...
                    energy-=localCorrelation;
                }
                double temp = 2.0 * sfm / (sff * smm) * ( Ii - sfm / smm * Ji );
                for(int qq=0;qq<3;++qq){
                    deriv[qq] -= temp*gradJ[qq];
                }
            }
        }
    }
    return energy;
}
