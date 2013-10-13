/*# -*- coding: utf-8 -*-
Created on Fri Sep 20 19:03:32 2013

@author: khayyam
*/
#include <string.h>
#include <math.h>
#include <stdlib.h>
const int 		EXPONENT_MASK32=(255<<23);
const long long EXPONENT_MASK64=(2047LL<<52);
const int 		MANTISSA_MASK32=((1<<23)-1);
const long long MANTISSA_MASK64=((1LL<<52)-1);
const int 		SIGNALING_BIT32=(1<<22);
const long long SIGNALING_BIT64=(1LL<<51);
const int 		iQNAN32=EXPONENT_MASK32|((MANTISSA_MASK32+1)>>1);
const float 	QNAN32=*(float*)(&iQNAN32);
const long long iQNAN64=EXPONENT_MASK64|((MANTISSA_MASK64+1LL)>>1);
const double 	QNAN64=*(double*)(&iQNAN64);

const int 		iSNAN32=(EXPONENT_MASK32|MANTISSA_MASK32)&(~SIGNALING_BIT32);
const float 	SNAN32=*(float*)(&iSNAN32);
const long long iSNAN64=(EXPONENT_MASK64|MANTISSA_MASK64)&(~SIGNALING_BIT64);
const double 	SNAN64=*(double*)(&iSNAN64);

const float  	INF32=*((float*)(&EXPONENT_MASK32));
const double 	INF64=*((double*)(&EXPONENT_MASK64));

bool isNumber(float x){
	return ((*((int*)(&x)))&EXPONENT_MASK32)!=EXPONENT_MASK32;
}

bool isInfinite(float x){
	if((*((int*)(&x)))&MANTISSA_MASK32){
		return false;
	}
	return ((*((int*)(&x)))&EXPONENT_MASK32)==EXPONENT_MASK32;
}

bool isSNAN(float x){
	if(((*((int*)(&x)))&EXPONENT_MASK32)!=EXPONENT_MASK32){
		return false;
	}
	if((*((int*)(&x)))&MANTISSA_MASK32){
		return !((*((int*)(&x)))&SIGNALING_BIT32);
	}
	return false;
}

bool isQNAN(float x){
	if(((*((int*)(&x)))&EXPONENT_MASK32)!=EXPONENT_MASK32){
		return false;
	}
	return ((*((int*)(&x)))&SIGNALING_BIT32);
}

bool isNumber(double x){
	return ((*((long long*)(&x)))&EXPONENT_MASK64)!=EXPONENT_MASK64;
}

bool isInfinite(double x){
	if((*((long long*)(&x)))&MANTISSA_MASK64){
		return false;
	}
	return ((*((long long*)(&x)))&EXPONENT_MASK64)==EXPONENT_MASK64;
}

bool isSNAN(double x){
	if(((*((long long*)(&x)))&EXPONENT_MASK64)!=EXPONENT_MASK64){
		return false;
	}
	if((*((long long*)(&x)))&MANTISSA_MASK64){
		return !((*((long long*)(&x)))&SIGNALING_BIT64);
	}
	return false;
}

bool isQNAN(double x){
	if(((*((long long*)(&x)))&EXPONENT_MASK64)!=EXPONENT_MASK64){
		return false;
	}
	return ((*((long long*)(&x)))&SIGNALING_BIT64);
}

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


//#define GLOBAL_FIELD_CORRECTION
double iterateDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *previousDisplacement, double *displacementField, double *residual){
    const static int numNeighbors=4;
    const static int dRow[]={-1, 0, 1,  0, -1, 1,  1, -1};
    const static int dCol[]={ 0, 1, 0, -1,  1, 1, -1, -1};
    int nrows=dims[0];
    int ncols=dims[1];
    double *d=displacementField;
    double *prevd=previousDisplacement;
    double *g=gradientField;
    double y[2];
    double A[3];
    int pos=0;
    double maxDisplacement=0;
    for(int r=0;r<nrows;++r){
        for(int c=0;c<ncols;++c, ++pos, d+=2, prevd+=2, g+=2){
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
                ++nn;
                double *dneigh=&displacementField[2*(dr*ncols + dc)];
#ifdef GLOBAL_FIELD_CORRECTION
                double *prevNeigh=&previousDisplacement[2*(dr*ncols + dc)];
                y[0]+=dneigh[0]+prevNeigh[0];
                y[1]+=dneigh[1]+prevNeigh[1];
#else
                y[0]+=dneigh[0];
                y[1]+=dneigh[1];
#endif
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
#ifdef GLOBAL_FIELD_CORRECTION
                y[0]=(delta*g[0]) + sigma*lambdaParam*(y[0]-nn*prevd[0]);
                y[1]=(delta*g[1]) + sigma*lambdaParam*(y[1]-nn*prevd[1]);
#else
                y[0]=(delta*g[0]) + sigma*lambdaParam*y[0];
                y[1]=(delta*g[1]) + sigma*lambdaParam*y[1];
#endif
                A[0]=g[0]*g[0] + sigma*lambdaParam*nn;
                A[1]=g[0]*g[1];
                A[2]=g[1]*g[1] + sigma*lambdaParam*nn;
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

double iterateDisplacementField3DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *previousDisplacement, double *displacementField, double *residual){
    const static int numNeighbors=6;
    const static int dSlice[numNeighbors]={-1,  0, 0, 0,  0, 1};
    const static int dRow[numNeighbors]  ={ 0, -1, 0, 1,  0, 0};
    const static int dCol[numNeighbors]  ={ 0,  0, 1, 0, -1, 0};
    int nslices=dims[0];
    int nrows=dims[1];
    int ncols=dims[2];
    double *d=displacementField;
    double *prevd=previousDisplacement;
    double *g=gradientField;
    int sliceSize=ncols*nrows;
    double y[3];
    double A[6];
    int pos=0;
    double maxDisplacement=0;
    for(int s=0;s<nslices;++s){
        for(int r=0;r<nrows;++r){
            for(int c=0;c<ncols;++c, ++pos, d+=3, prevd+=3, g+=3){
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
#ifdef GLOBAL_FIELD_CORRECTION
                    double *prevNeigh=&previousDisplacement[3*(ds*sliceSize + dr*ncols + dc)];
                    y[0]+=dneigh[0]+prevNeigh[0];
                    y[1]+=dneigh[1]+prevNeigh[1];
                    y[2]+=dneigh[2]+prevNeigh[2];
#else
                    y[0]+=dneigh[0];
                    y[1]+=dneigh[1];
                    y[2]+=dneigh[2];
#endif
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
                    residual[pos]=0;
                }else{
#ifdef GLOBAL_FIELD_CORRECTION
                    y[0]=(delta*g[0]) + sigma*lambdaParam*(y[0]-nn*prevd[0]);
                    y[1]=(delta*g[1]) + sigma*lambdaParam*(y[1]-nn*prevd[1]);
                    y[1]=(delta*g[2]) + sigma*lambdaParam*(y[2]-nn*prevd[2]);
#else
                    y[0]=(delta*g[0]) + sigma*lambdaParam*y[0];
                    y[1]=(delta*g[1]) + sigma*lambdaParam*y[1];
                    y[2]=(delta*g[2]) + sigma*lambdaParam*y[2];
#endif
                    A[0]=g[0]*g[0] + sigma*lambdaParam*nn;
                    A[1]=g[0]*g[1];
                    A[2]=g[0]*g[2];
                    A[3]=g[1]*g[1] + sigma*lambdaParam*nn;
                    A[4]=g[1]*g[2];
                    A[5]=g[2]*g[2] + sigma*lambdaParam*nn;
                    double xx=d[0];
                    double yy=d[1];
                    double zz=d[2];
                    solve3DSymmetricPositiveDefiniteSystem(A,y,d, &residual[pos]);
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



