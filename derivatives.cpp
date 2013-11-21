/*! \file derivatives.cpp
	\author Omar Ocegueda
	\brief Definition of of numerical differentiation functions(derivatives, divergence, rotational, Jacobian, etc.)
*/
#include "derivatives.h"
#include "string.h"
int computeRowDerivative_Forward(double *f, int nrows, int ncols, double *dfdr, EBoundaryCondition ebc){
	int pos=0;
	for(int i=0;i<nrows-1;++i){
		for(int j=0;j<ncols;++j, ++pos){
			dfdr[pos]=f[pos+ncols]-f[pos];
		}
	}
	switch(ebc){
		case EBC_Circular:
			for(int j=0;j<ncols;++j){
				dfdr[(nrows-1)*ncols+j]=f[0*ncols+j]-f[(nrows-1)*ncols+j];
			}
		break;
		case EBC_DirichletZero:
			for(int j=0;j<ncols;++j){
				dfdr[(nrows-1)*ncols+j]=0-f[(nrows-1)*ncols+j];
			}
		break;
		case EBC_VonNeumanZero:
			for(int j=0;j<ncols;++j){
				dfdr[(nrows-1)*ncols+j]=0;
			}
		break;
	}
	return 0;
}

int computeColumnDerivative_Forward(double *f, int nrows, int ncols, double *dfdc, EBoundaryCondition ebc){
	int pos=0;
	for(int i=0;i<nrows;++i){
		for(int j=0;j<ncols-1;++j, ++pos){
			dfdc[pos]=f[pos+1]-f[pos];
		}
		++pos;
	}
	switch(ebc){
		case EBC_Circular:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols+ncols-1]=f[i*ncols+0]-f[i*ncols+ncols-1];
			}
		break;
		case EBC_DirichletZero:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols+ncols-1]=0-f[i*ncols+ncols-1];
			}
		break;
		case EBC_VonNeumanZero:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols+ncols-1]=0;
			}
		break;
	}
	return 0;
}

int computeSliceDerivative_Forward(double *f, int nslices, int nrows, int ncols, double *dfds, EBoundaryCondition ebc){
	int sliceSize=nrows*ncols;
	double *currentSlice=f;
	double *forwardSlice=f+sliceSize;
	double *currentDerivativeSlice=dfds;
	for(int s=0;s<nslices-1;++s){
		for(int i=0;i<sliceSize;++i){
			currentDerivativeSlice[i]=forwardSlice[i]-currentSlice[i];
		}
		currentSlice			+=sliceSize;
		forwardSlice			+=sliceSize;
		currentDerivativeSlice	+=sliceSize;
	}
	switch(ebc){
		case EBC_Circular:
			forwardSlice=f;
			for(int i=0;i<sliceSize;++i){
				currentDerivativeSlice[i]=forwardSlice[i]-currentSlice[i];
			}
		break;
		case EBC_DirichletZero:
			for(int i=0;i<sliceSize;++i){
				currentDerivativeSlice[i]=-currentSlice[i];
			}
		break;
		case EBC_VonNeumanZero:
			memset(currentDerivativeSlice, 0, sizeof(double)*sliceSize);
		break;
	}
	return 0;
}
//------------------------
int computeRowDerivative_Backward(double *f, int nrows, int ncols, double *dfdr, EBoundaryCondition ebc){
	int pos=ncols;
	for(int i=1;i<nrows;++i){
		for(int j=0;j<ncols;++j, ++pos){
			dfdr[pos]=f[pos]-f[pos-ncols];
		}
	}
	switch(ebc){
		case EBC_Circular:
			for(int j=0;j<ncols;++j){
				dfdr[j]=f[j]-f[(nrows-1)*ncols+j];
			}
		break;
		case EBC_DirichletZero:
			for(int j=0;j<ncols;++j){
				dfdr[j]=f[j];
			}
		break;
		case EBC_VonNeumanZero:
			for(int j=0;j<ncols;++j){
				dfdr[j]=0;
			}
		break;
	}
	return 0;
}

int computeColumnDerivative_Backward(double *f, int nrows, int ncols, double *dfdc, EBoundaryCondition ebc){
	int pos=0;
	for(int i=0;i<nrows;++i){
		++pos;
		for(int j=1;j<ncols;++j,++pos){
			dfdc[pos]=f[pos]-f[pos-1];
		}
	}
	switch(ebc){
		case EBC_Circular:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols]=f[i*ncols]-f[i*ncols+ncols-1];
			}
		break;
		case EBC_DirichletZero:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols]=f[i*ncols];
			}
		break;
		case EBC_VonNeumanZero:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols]=0;
			}
		break;
	}
	return 0;
}

int computeSliceDerivative_Backward(double *f, int nslices, int nrows, int ncols, double *dfds, EBoundaryCondition ebc){
	int sliceSize=nrows*ncols;
	double *currentSlice=f+sliceSize;
	double *currentDerivativeSlice=dfds+sliceSize;
	double *backwardslice=f;
	for(int s=1;s<nslices;++s){
		for(int i=0;i<sliceSize;++i){
			currentDerivativeSlice[i]=currentSlice[i]-backwardslice[i];
		}
		currentSlice+=sliceSize;
		currentDerivativeSlice+=sliceSize;
		backwardslice+=sliceSize;
	}
	currentSlice=f;
	currentDerivativeSlice=dfds;
	switch(ebc){
		case EBC_Circular:
			for(int i=0;i<sliceSize;++i){
				currentDerivativeSlice[i]=currentSlice[i]-backwardslice[i];
			}
		break;
		case EBC_DirichletZero:
			for(int i=0;i<sliceSize;++i){
				currentDerivativeSlice[i]=currentSlice[i];
			}
		break;
		case EBC_VonNeumanZero:
			memset(currentDerivativeSlice, 0, sizeof(double)*sliceSize);
		break;
	}
	return 0;
}

//---------------------------------

int computeGradient(double *f, int nrows, int ncols, double *dfdr, double *dfdc, EDerivativeType edt, EBoundaryCondition ebc){
	if((nrows<2) || (ncols<2)){
		return -1;
	}
	switch(edt){
		case EDT_Forward:
			computeRowDerivative_Forward(f, nrows, ncols, dfdr, ebc);
			computeColumnDerivative_Forward(f, nrows, ncols, dfdc, ebc);
		break;
		case EDT_Backward:
			computeRowDerivative_Backward(f, nrows, ncols, dfdr, ebc);
			computeColumnDerivative_Backward(f, nrows, ncols, dfdc, ebc);
		break;
	}
	return 0;
}

int computeDivergence_Forward(double *fr, double *fc, int nrows, int ncols, double *div, EBoundaryCondition ebc){
	int pos=0;
	for(int i=0;i<nrows-1;++i){
		for(int j=0;j<ncols-1;++j,++pos){
			div[pos]=(fr[pos+ncols]-fr[pos])+(fc[pos+1]-fc[pos]);
		}
		++pos;
	}
	switch(ebc){
		case EBC_Circular:
			div[nrows*ncols-1]=(fr[ncols-1]-fr[nrows*ncols-1])+(fc[(nrows-1)*ncols]-fc[nrows*ncols-1]);
			for(int i=0, pos=ncols-1;i<nrows-1;++i, pos+=ncols){//last column
				div[pos]=(fr[pos+ncols]-fr[pos]) + (fc[pos-(ncols-1)]-fc[pos]);
			}
			for(int j=0, pos=(nrows-1)*ncols;j<ncols-1;++j, ++pos){//last row
				div[pos]=(fr[j]-fr[pos]) + (fc[pos+1]-fc[pos]);
			}
		break;
		case EBC_DirichletZero:
			div[nrows*ncols-1]=(-fr[nrows*ncols-1])+(-fc[nrows*ncols-1]);
			for(int i=0, pos=ncols-1;i<nrows-1;++i, pos+=ncols){//last column
				div[pos]=(fr[pos+ncols]-fr[pos]) + (-fc[pos]);
			}
			for(int j=0, pos=(nrows-1)*ncols;j<ncols-1;++j, ++pos){//last row
				div[pos]=(-fr[pos]) + (fc[pos+1]-fc[pos]);
			}
		break;
		case EBC_VonNeumanZero:
			div[nrows*ncols-1]=0;
			for(int i=0, pos=ncols-1;i<nrows-1;++i, pos+=ncols){//last column
				div[pos]=(fr[pos+ncols]-fr[pos]);
			}
			for(int j=0, pos=(nrows-1)*ncols;j<ncols-1;++j, ++pos){//last row
				div[pos]=(fc[pos+1]-fc[pos]);
			}
		break;
	}
	return 0;
}

int computeDivergence_Forward(double *fs, double *fr, double *fc, int nslices, int nrows, int ncols, double *div, EBoundaryCondition ebc){
	int sliceSize=nrows*ncols;
	for(int s=0;s<nslices;++s){
		computeDivergence_Forward(&fr[s*sliceSize], &fc[s*sliceSize], nrows, ncols, &div[s*sliceSize], ebc);
	}
	double *currentSlice=fs;
	double *forwardSlice=fs+sliceSize;
	double *divSlice=div;
	for(int s=0;s<nslices-1;++s){
		for(int i=0;i<sliceSize;++i){
			divSlice[i]+=(forwardSlice[i]-currentSlice[i]);
		}
		currentSlice+=sliceSize;
		forwardSlice+=sliceSize;
		divSlice+=sliceSize;
	}

	switch(ebc){
		case EBC_Circular:
			forwardSlice=fs;
			for(int i=0;i<sliceSize;++i){
				divSlice[i]+=(forwardSlice[i]-currentSlice[i]);
			}
		break;
		case EBC_DirichletZero:
			for(int i=0;i<sliceSize;++i){
				divSlice[i]+=-currentSlice[i];
			}
		break;
		case EBC_VonNeumanZero:
			memset(divSlice, 0, sizeof(double)*sliceSize);
		break;
	}
	return 0;
}



int computeDivergence_Backward(double *fr, double *fc, int nrows, int ncols, double *div, EBoundaryCondition ebc){
	int pos=ncols;
	for(int i=1;i<nrows;++i){
		++pos;
		for(int j=1;j<ncols;++j,++pos){
			div[pos]=(fr[pos]-fr[pos-ncols])+(fc[pos]-fc[pos-1]);
		}
	}
	switch(ebc){
		case EBC_Circular:
			div[0]=(fr[0]-fr[(nrows-1)*ncols+0])+(fc[0]-fc[0+ncols-1]);
			for(int i=1;i<nrows;++i){//first column
				div[i*ncols+0]=(fr[i*ncols]-fr[(i-1)*ncols]) + (fc[i*ncols]-fc[i*ncols + ncols-1]);
			}
			for(int j=1;j<ncols;++j){//first row
				div[j]=(fr[j]-fr[(nrows-1)*ncols+j]) + (fc[j]-fc[j-1]);
			}
		break;
		case EBC_DirichletZero:
			div[0]=fr[0]+fc[0];
			for(int i=1;i<nrows;++i){//first column
				div[i*ncols+0]=(fr[i*ncols]-fr[(i-1)*ncols]) + fc[i*ncols];
			}
			for(int j=1;j<ncols;++j){//first row
				div[j]=fr[j] + (fc[j]-fc[j-1]);
			}
		break;
		case EBC_VonNeumanZero:
			div[0]=0;
			for(int i=1;i<nrows;++i){//first column
				div[i*ncols]=(fr[i*ncols]-fr[(i-1)*ncols]) ;
			}
			for(int j=1;j<ncols;++j){//first row
				div[j]=fc[j]-fc[j-1];
			}
		break;
	}
	return 0;
}

int computeDivergence_Backward(double *fs, double *fr, double *fc, int nslices, int nrows, int ncols, double *div, EBoundaryCondition ebc){
	int sliceSize=nrows*ncols;
	for(int s=0;s<nslices;++s){
		computeDivergence_Backward(&fr[s*sliceSize], &fc[s*sliceSize], nrows, ncols, &div[s*sliceSize], ebc);
	}
	double *currentSlice=fs+sliceSize;
	double *backwardSlice=fs;
	double *divSlice=div+sliceSize;
	for(int s=1;s<nslices;++s){
		for(int i=0;i<sliceSize;++i){
			divSlice[i]+=(currentSlice[i]-backwardSlice[i]);
		}
		currentSlice+=sliceSize;
		backwardSlice+=sliceSize;
		divSlice+=sliceSize;
	}
	currentSlice=fs;
	divSlice=div;
	switch(ebc){
		case EBC_Circular:
			for(int i=0;i<sliceSize;++i){
				divSlice[i]+=(currentSlice[i]-backwardSlice[i]);
			}
		break;
		case EBC_DirichletZero:
			for(int i=0;i<sliceSize;++i){
				divSlice[i]+=currentSlice[i];
			}
		break;
		case EBC_VonNeumanZero:
			memset(divSlice, 0, sizeof(double)*sliceSize);
		break;
	}
	return 0;
}


int computeDivergence(double *fr, double *fc, int nrows, int ncols, double *div, EDerivativeType edt, EBoundaryCondition ebc){
	if((nrows<2) || (ncols<2)){
		return -1;
	}
	switch(edt){
		case EDT_Forward:
			computeDivergence_Forward(fr, fc, nrows, ncols, div, ebc);
		break;
		case EDT_Backward:
			computeDivergence_Backward(fr, fc, nrows, ncols, div, ebc);
		break;
	}
	return 0;
}

int computeGradient(double *f, int nslices, int nrows, int ncols, double *dfds, double *dfdr, double *dfdc, EDerivativeType edt, EBoundaryCondition ebc){
	//--compute intra-slice derivatives--
	double *currentSlice_f=f;
	double *currentSlice_dfdr=dfdr;
	double *currentSlice_dfdc=dfdc;
	int sliceSize=nrows*ncols;
	for(int s=0;s<nslices;++s){
		computeGradient(currentSlice_f, nrows, ncols, currentSlice_dfdr, currentSlice_dfdc, edt, ebc);
		currentSlice_f+=sliceSize;
		currentSlice_dfdr+=sliceSize;
		currentSlice_dfdc+=sliceSize;
	}
	//-------------------------------------
	switch(edt){
		case EDT_Forward:
			computeSliceDerivative_Forward(f, nslices, nrows, ncols, dfds, ebc);
		break;
		case EDT_Backward:
			computeSliceDerivative_Backward(f, nslices, nrows, ncols, dfds, ebc);
		break;
	}
	return 0;
}

int computeDivergence(double *fs, double *fr, double *fc, int nslices, int nrows, int ncols, double *div, EDerivativeType edt, EBoundaryCondition ebc){
	if((nslices<2) || (nrows<2) || (ncols<2)){
		return -1;
	}
	switch(edt){
		case EDT_Forward:
			computeDivergence_Forward(fs, fr, fc, nslices, nrows, ncols, div, ebc);
		break;
		case EDT_Backward:
			computeDivergence_Backward(fs, fr, fc, nslices, nrows, ncols, div, ebc);
		break;
	}
	return 0;
}

int computeRotational_Forward(double *fr, double *fc, int nrows, int ncols, double *rot, EBoundaryCondition ebc){
	int pos=0;
	for(int i=0;i<nrows-1;++i){
		for(int j=0;j<ncols-1;++j,++pos){
			rot[pos]=(fc[pos+ncols]-fc[pos])-(fr[pos+1]-fr[pos]);
		}
		++pos;
	}
	switch(ebc){
		case EBC_Circular:
			rot[nrows*ncols-1]=(fc[ncols-1]-fc[nrows*ncols-1])-(fr[(nrows-1)*ncols]-fr[nrows*ncols-1]);
			for(int i=0, pos=ncols-1;i<nrows-1;++i, pos+=ncols){//last column
				rot[pos]=(fr[pos+ncols]-fr[pos]) + (fc[pos-(ncols-1)]-fc[pos]);
			}
			for(int j=0, pos=(nrows-1)*ncols;j<ncols-1;++j, ++pos){//last row
				rot[pos]=(fc[j]-fc[pos]) - (fr[pos+1]-fr[pos]);
			}
		break;
		case EBC_DirichletZero:
			rot[nrows*ncols-1]=(-fc[nrows*ncols-1])-(-fr[nrows*ncols-1]);
			for(int i=0, pos=ncols-1;i<nrows-1;++i, pos+=ncols){//last column
				rot[pos]=(fc[pos+ncols]-fc[pos]) - (-fr[pos]);
			}
			for(int j=0, pos=(nrows-1)*ncols;j<ncols-1;++j, ++pos){//last row
				rot[pos]=(-fc[pos]) - (fr[pos+1]-fr[pos]);
			}
		break;
		case EBC_VonNeumanZero:
			rot[nrows*ncols-1]=0;
			for(int i=0, pos=ncols-1;i<nrows-1;++i, pos+=ncols){//last column
				rot[pos]=(fc[pos+ncols]-fc[pos]);
			}
			for(int j=0, pos=(nrows-1)*ncols;j<ncols-1;++j, ++pos){//last row
				rot[pos]=-(fr[pos+1]-fr[pos]);
			}
		break;
	}
	return 0;
}

int computeRotational_Backward(double *fr, double *fc, int nrows, int ncols, double *rot, EBoundaryCondition ebc){
	int pos=ncols;
	for(int i=1;i<nrows;++i){
		++pos;
		for(int j=1;j<ncols;++j,++pos){
			rot[pos]=(fc[pos]-fc[pos-ncols])-(fr[pos]-fr[pos-1]);
		}
	}
	switch(ebc){
		case EBC_Circular:
			rot[0]=(fc[0]-fc[(nrows-1)*ncols+0])-(fr[0]-fr[0+ncols-1]);
			for(int i=1;i<nrows;++i){//first column
				rot[i*ncols+0]=(fc[i*ncols]-fc[(i-1)*ncols]) - (fr[i*ncols]-fr[i*ncols + ncols-1]);
			}
			for(int j=1;j<ncols;++j){//first row
				rot[j]=(fc[j]-fc[(nrows-1)*ncols+j]) - (fr[j]-fr[j-1]);
			}
		break;
		case EBC_DirichletZero:
			rot[0]=fc[0]-fr[0];
			for(int i=1;i<nrows;++i){//first column
				rot[i*ncols+0]=(fc[i*ncols]-fc[(i-1)*ncols]) - fr[i*ncols];
			}
			for(int j=1;j<ncols;++j){//first row
				rot[j]=fc[j] - (fr[j]-fr[j-1]);
			}
		break;
		case EBC_VonNeumanZero:
			rot[0]=0;
			for(int i=1;i<nrows;++i){//first column
				rot[i*ncols]=(fc[i*ncols]-fc[(i-1)*ncols]) ;
			}
			for(int j=1;j<ncols;++j){//first row
				rot[j]=-(fr[j]-fr[j-1]);
			}
		break;
	}
	return 0;
}


int computeRotational(double *fr, double *fc, int nrows, int ncols, double *rot, EDerivativeType edt, EBoundaryCondition ebc){
	if((nrows<2) || (ncols<2)){
		return -1;
	}
	switch(edt){
		case EDT_Forward:
			computeRotational_Forward(fr, fc, nrows, ncols, rot, ebc);
		break;
		case EDT_Backward:
			computeRotational_Backward(fr, fc, nrows, ncols, rot, ebc);
		break;
	}
	return 0;
}


//---symmetric tensors of order 2 and dimmension 2---
int computeDivergence(double *frr, double *fcc, double *frc, int nrows, int ncols, double *divr, double *divc, EDerivativeType edt, EBoundaryCondition ebc){
	int retVal=computeDivergence(frr, frc, nrows, ncols, divr, edt, ebc);
	computeDivergence(frc, fcc, nrows, ncols, divc, edt, ebc);
	return retVal;
}

int computeJacobian(double *fr, double *fc, int nrows, int ncols, double *dfdrr, double *dfdcc, double *dfdrc, EDerivativeType edt, EBoundaryCondition ebc){
	double *tmp=new double[nrows*ncols];
	int retVal=computeGradient(fr, nrows, ncols, dfdrr, dfdrc, edt, ebc);
	retVal=computeGradient(fc, nrows, ncols, tmp  , dfdcc, edt, ebc);
	int npix=nrows*ncols;
	for(int i=0;i<npix;++i){
		dfdrc[i]=0.5*(dfdrc[i]+tmp[i]);
	}
	delete[] tmp;
	return retVal;
}

//---symmetric tensors of order 2 and dimmension 3---
int computeDivergence(double *fss, double *frr, double *fcc, double *fsr, double *frc, double *fsc, int nslices, int nrows, int ncols, double *divs, double *divr, double *divc, EDerivativeType edt, EBoundaryCondition ebc){
	int retVal=computeDivergence(fss, fsr, fsc, nslices, nrows, ncols, divs, edt, ebc);
	retVal=computeDivergence(fsr, frr, frc, nslices, nrows, ncols, divr, edt, ebc);
	retVal=computeDivergence(fsc, frc, fcc, nslices, nrows, ncols, divc, edt, ebc);
	return retVal;
}

int computeJacobian(double *fs, double *fr, double *fc, int nslices, int nrows, int ncols, double *fss, double *frr, double *fcc, double *fsr, double *frc, double *fsc, EDerivativeType edt, EBoundaryCondition ebc){
	int volSize=nslices*nrows*ncols;
	double *tmp=new double[volSize];
	double *tmp2=new double[volSize];
	int retVal=computeGradient(fs, nslices, nrows, ncols, fss, fsr, fsc,edt, ebc);
	retVal=computeGradient(fr, nslices, nrows, ncols, tmp, frr, frc,edt, ebc);
	//--force fsr to be symmetric--
	for(int i=0;i<volSize;++i){
		fsr[i]=0.5*(fsr[i]+tmp[i]);
	}
	retVal=computeGradient(fc, nslices, nrows, ncols, tmp, tmp2, fcc,edt, ebc);
	//--force frc and fsc to be symmetric--
	for(int i=0;i<volSize;++i){
		fsc[i]=0.5*(fsc[i]+tmp[i]);
		frc[i]=0.5*(frc[i]+tmp2[i]);
	}
	delete[] tmp;
	delete[] tmp2;
	return retVal;
}
//==================derivatives of multi-band images with arbitrary topology=====================

//--auxiliary derivatives--
int computeRowDerivative_Forward(double *f, int *assignmentV, int nrows, int ncols, int maxCompartments, double *dfdr, EBoundaryCondition ebc){
	int pos=0;
	for(int i=0;i<nrows-1;++i){
		for(int j=0;j<ncols;++j){
			for(int k=0;k<maxCompartments;++k,++pos){
				int to=assignmentV[pos];
				dfdr[pos]=f[pos+ncols*maxCompartments-k+to]-f[pos];
			}
		}
	}
	switch(ebc){
		case EBC_Circular:
			pos=(nrows-1)*ncols*maxCompartments;
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k,++pos){
					int to=assignmentV[pos];
					dfdr[pos]=f[j*maxCompartments-k+to]-f[pos];
				}
			}
		break;
		case EBC_DirichletZero:
			pos=(nrows-1)*ncols*maxCompartments;
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k,++pos){
					dfdr[pos]=0-f[pos];
				}
			}
		break;
		case EBC_VonNeumanZero:
			pos=(nrows-1)*ncols*maxCompartments;
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k){
					dfdr[pos]=0;
				}
			}
		break;
	}
	return 0;
}

int computeColumnDerivative_Forward(double *f, int *assignmentH, int nrows, int ncols, int maxCompartments, double *dfdc, EBoundaryCondition ebc){
	int pos=0;
	for(int i=0;i<nrows;++i){
		for(int j=0;j<ncols-1;++j){
			for(int k=0;k<maxCompartments;++k, ++pos){
				int to=assignmentH[pos];
				dfdc[pos]=f[pos+maxCompartments-k+to]-f[pos];
			}
		}
		pos+=maxCompartments;
	}
	switch(ebc){
		case EBC_Circular:
			pos=(ncols-1)*maxCompartments;
			for(int i=0;i<nrows;++i){
				for(int k=0;k<maxCompartments;++k,++pos){
					int to=assignmentH[pos];
					dfdc[pos]=f[(i*ncols*maxCompartments)+to]-f[pos];
				}
				pos+=(ncols-1)*maxCompartments;
			}
		break;
		case EBC_DirichletZero:
			pos=(ncols-1)*maxCompartments;
			for(int i=0;i<nrows;++i){
				for(int k=0;k<maxCompartments;++k,++pos){
					dfdc[pos]=0-f[pos];
				}
				pos+=(ncols-1)*maxCompartments;
			}
		break;
		case EBC_VonNeumanZero:
			pos=(ncols-1)*maxCompartments;
			for(int i=0;i<nrows;++i){
				for(int k=0;k<maxCompartments;++k,++pos){
					dfdc[pos]=0;
				}
				pos+=(ncols-1)*maxCompartments;
			}
		break;
	}
	return 0;
}

int computeRowDerivative_Backward(double *f, int *assignmentV, int nrows, int ncols, int maxCompartments, double *dfdr, EBoundaryCondition ebc){
	int pos=ncols*maxCompartments;
	for(int i=1;i<nrows;++i){
		for(int j=0;j<ncols;++j){
			for(int k=0;k<maxCompartments;++k, ++pos){
				//k-th previous compartpent points to 'to' current compartment
				int to=assignmentV[pos-ncols*maxCompartments];
				dfdr[pos-k+to]=f[pos-k+to]-f[pos-ncols*maxCompartments];
			}
		}
	}
	int pos2;
	switch(ebc){
		case EBC_Circular:
			pos=0;
			pos2=(nrows-1)*ncols*maxCompartments;
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k, ++pos,++pos2){
					//k-th previous compartpent points to 'to' current compartment
					int to=assignmentV[pos2];
					dfdr[pos-k+to]=f[pos-k+to]-f[pos2];
				}
			}
		break;
		case EBC_DirichletZero:
			pos=0;
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k, ++pos){
					dfdr[pos]=f[pos];
				}
			}
		break;
		case EBC_VonNeumanZero:
			pos=0;
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k, ++pos){
					dfdr[pos]=0;
				}
			}
		break;
	}
	return 0;
}

int computeColumnDerivative_Backward(double *f, int *assignmentH, int nrows, int ncols, int maxCompartments, double *dfdc, EBoundaryCondition ebc){
	int pos=0;
	for(int i=0;i<nrows;++i){
		pos+=maxCompartments;
		for(int j=1;j<ncols;++j){
			for(int k=0;k<maxCompartments;++k,++pos){
				int to=assignmentH[pos-maxCompartments];
				dfdc[pos-k+to]=f[pos-k+to]-f[pos-maxCompartments];
			}
		}
	}
	int pos2;
	switch(ebc){
		case EBC_Circular:
			for(int i=0;i<nrows;++i){
				pos=i*ncols*maxCompartments;
				pos2=pos+(ncols-1)*maxCompartments;
				for(int k=0;k<maxCompartments;++k,++pos,++pos2){
					int to=assignmentH[pos2];
					dfdc[pos-k+to]=f[pos-k+to]-f[pos2];
				}
			}
		break;
		case EBC_DirichletZero:
			for(int i=0;i<nrows;++i){
				pos=i*ncols*maxCompartments;
				for(int k=0;k<maxCompartments;++k,++pos){
					dfdc[pos]=f[pos];
				}
			}
		break;
		case EBC_VonNeumanZero:
			for(int i=0;i<nrows;++i){
				pos=i*ncols*maxCompartments;
				for(int k=0;k<maxCompartments;++k,++pos){
					dfdc[pos]=0;
				}
			}
		break;
	}
	return 0;
}

int computeDivergence_Forward(double *fr, double *fc, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *div, EBoundaryCondition ebc){
	int pos=0;
	int posRight=maxCompartments;
	int posDown=ncols*maxCompartments;
	for(int i=0;i<nrows-1;++i){
		for(int j=0;j<ncols-1;++j){
			for(int k=0;k<maxCompartments;++k, ++pos, ++posDown, ++posRight){
				int toRight=assignmentH[pos];
				int toDown=assignmentV[pos];
				div[pos]=(fr[posDown-k+toDown]-fr[pos])+(fc[posRight-k+toRight]-fc[pos]);
			}
		}
		pos+=maxCompartments;
		posRight+=maxCompartments;
		posDown+=maxCompartments;
	}
	switch(ebc){
		case EBC_Circular:
			//---bottom-right corner---
			pos=(nrows*ncols-1)*maxCompartments;
			posDown=(ncols-1)*maxCompartments;
			posRight=pos-posDown;
			for(int k=0;k<maxCompartments;++k, ++pos, ++posRight, ++posDown){
				int toRight=assignmentH[pos];
				int toDown=assignmentV[pos];
				div[pos]=(fr[posDown-k+toDown]-fr[pos])+(fc[posRight-k+toRight]-fc[pos]);
			}
			//---last column (except bottom-right corner)---
			pos=(ncols-1)*maxCompartments;
			posRight=0;
			posDown=pos+ncols*maxCompartments;
			for(int i=0;i<nrows-1;++i){
				for(int k=0;k<maxCompartments;++k, ++pos, ++posRight, ++posDown){
					int toRight=assignmentH[pos];
					int toDown=assignmentV[pos];
					div[pos]=(fr[posDown-k+toDown]-fr[pos]) + (fc[posRight-k+toRight]-fc[pos]);
				}
				pos+=(ncols-1)*maxCompartments;
				posRight+=(ncols-1)*maxCompartments;
				posDown+=(ncols-1)*maxCompartments;
			}
			//---last row (except bottom-right corner)---
			pos=(nrows-1)*ncols*maxCompartments;
			posRight=pos+maxCompartments;
			posDown=0;
			for(int j=0;j<ncols-1;++j){
				for(int k=0;k<maxCompartments;++k, ++pos, ++posRight, ++posDown){
					int toRight=assignmentH[pos];
					int toDown=assignmentV[pos];
					div[pos]=(fr[posDown-k+toDown]-fr[pos]) + (fc[posRight-k+toRight]-fc[pos]);
				}
			}
		break;
		case EBC_DirichletZero:
			//---bottom-right corner---
			pos=(nrows*ncols-1)*maxCompartments;
			for(int k=0;k<maxCompartments;++k, ++pos){
				div[pos]=(-fr[pos])+(-fc[pos]);
			}
			//---last column (except bottom-right corner)---
			pos=(ncols-1)*maxCompartments;
			posDown=pos+ncols*maxCompartments;
			for(int i=0;i<nrows-1;++i){
				for(int k=0;k<maxCompartments;++k,++pos, ++posDown){
					int toDown=assignmentV[pos];
					div[pos]=(fr[posDown-k+toDown]-fr[pos]) + (-fc[pos]);
				}
				pos+=(ncols-1)*maxCompartments;
				posDown+=(ncols-1)*maxCompartments;
			}
			//---last row (except bottom-right corner)---
			pos=(nrows-1)*ncols*maxCompartments;
			posRight=pos+maxCompartments;
			for(int j=0;j<ncols-1;++j){
				for(int k=0;k<maxCompartments;++k, ++pos, ++posRight){
					int toRigh=assignmentH[pos];
					div[pos]=(-fr[pos]) + (fc[posRight-k+toRigh]-fc[pos]);
				}
			}
		break;
		case EBC_VonNeumanZero:
			//---bottom-right corner---
			pos=(nrows*ncols-1)*maxCompartments;
			for(int k=0;k<maxCompartments;++k, ++pos){
				div[pos]=0;
			}
			//---last column (except bottom-right corner)---
			pos=(ncols-1)*maxCompartments;
			posDown=pos+ncols*maxCompartments;
			for(int i=0;i<nrows-1;++i){
				for(int k=0;k<maxCompartments;++k){
					int toDown=assignmentV[pos];
					div[pos]=(fr[posDown-k+toDown]-fr[pos]);
				}
				pos+=(ncols-1)*maxCompartments;
				posDown+=(ncols-1)*maxCompartments;
			}
			//---last row (except bottom-right corner)---
			pos=(nrows-1)*ncols*maxCompartments;
			posRight=pos+maxCompartments;
			for(int j=0;j<ncols-1;++j){
				for(int k=0;k<maxCompartments;++k, ++pos, ++posRight){
					int toRight=assignmentH[pos];
					div[pos]=(fc[posRight-k+toRight]-fc[pos]);
				}
			}
		break;
	}
	return 0;
}

int computeDivergence_Backward(double *fr, double *fc, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *div, EBoundaryCondition ebc){
	int pos=ncols*maxCompartments;
	int posLeft=pos-maxCompartments;
	int posUp=0;
	for(int i=1;i<nrows;++i){
		pos+=maxCompartments;
		posLeft+=maxCompartments;
		posUp+=maxCompartments;
		for(int j=1;j<ncols;++j){
			for(int k=0;k<maxCompartments;++k){
				int fromLeft=0;
				int fromUp=0;
				while(assignmentH[posLeft+fromLeft]!=k){//find the compartment from left voxel assigned to the k-th compartment of the current voxel
					++fromLeft;
				}
				while(assignmentV[posUp+fromUp]!=k){//find the compartment from upside voxel assigned to the k-th compartment of the current voxel
					++fromUp;
				}
				div[pos+k]=(fr[pos+k]-fr[posUp+fromUp])+(fc[pos+k]-fc[posLeft+fromLeft]);
			}
			pos+=maxCompartments;
			posLeft+=maxCompartments;
			posUp+=maxCompartments;
		}
	}
	switch(ebc){
		case EBC_Circular:
			//---top-left corner---
			pos=0;
			posLeft=(ncols-1)*maxCompartments;
			posUp=(nrows-1)*ncols*maxCompartments;
			for(int k=0;k<maxCompartments;++k){
				int fromLeft=0;
				while(assignmentH[posLeft+fromLeft]!=k){
					++fromLeft;
				}
				int fromUp=0;
				while(assignmentV[posUp+fromUp]!=k){
					++fromUp;
				}
				div[pos+k]=(fr[pos+k]-fr[posUp+fromUp])+(fc[pos+k]-fc[posLeft+fromLeft]);
			}
			//---first column (except top-left corner)---
			posUp=0;
			pos=ncols*maxCompartments;
			posLeft=pos+(ncols-1)*maxCompartments;
			for(int i=1;i<nrows;++i){//first column
				for(int k=0;k<maxCompartments;++k){
					int fromLeft=0;
					while(assignmentH[posLeft+fromLeft]!=k){
						++fromLeft;
					}
					int fromUp=0;
					while(assignmentV[posUp+fromUp]!=k){
						++fromUp;
					}
					div[pos+k]=(fr[pos+k]-fr[posUp+fromUp]) + (fc[pos+k]-fc[posLeft+fromLeft]);
				}
				posUp+=ncols*maxCompartments;
				pos+=ncols*maxCompartments;
				posLeft+=ncols*maxCompartments;
			}
			//---first row (except top-left corner)---
			pos=maxCompartments;
			posLeft=0;
			posUp=pos+(nrows-1)*ncols*maxCompartments;
			for(int j=1;j<ncols;++j){//first row
				for(int k=0;k<maxCompartments;++k){
					int fromLeft=0;
					while(assignmentH[posLeft+fromLeft]!=k){
						++fromLeft;
					}
					int fromUp=0;
					while(assignmentV[posUp+fromUp]!=k){
						++fromUp;
					}
					div[pos+k]=(fr[pos+k]-fr[posUp+fromUp]) + (fc[pos+k]-fc[posLeft+fromLeft]);
				}
				pos+=maxCompartments;
				posLeft+=maxCompartments;
				posUp+=maxCompartments;
			}
		break;
		case EBC_DirichletZero:
			//---top-left corner---
			for(int k=0;k<maxCompartments;++k){
				div[k]=fr[k]+fc[k];
			}
			//---first column (except top-left corner)---
			posUp=0;
			pos=ncols*maxCompartments;
			for(int i=1;i<nrows;++i){//first column
				for(int k=0;k<maxCompartments;++k){
					int fromUp=0;
					while(assignmentV[posUp+fromUp]!=k){
						++fromUp;
					}
					div[pos+k]=(fr[pos+k]-fr[posUp+fromUp]) + fc[pos+k];
				}
				posUp+=ncols*maxCompartments;
				pos+=ncols*maxCompartments;
			}
			//---first row (except top-left corner)---
			pos=maxCompartments;
			posLeft=0;
			for(int j=1;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k){
					int fromLeft=0;
					while(assignmentH[posLeft+fromLeft]!=k){
						++fromLeft;
					}
					div[pos+k]=fr[pos+k] + (fc[pos+k]-fc[posLeft+fromLeft]);
				}
				pos+=maxCompartments;
				posLeft+=maxCompartments;
			}
		break;
		case EBC_VonNeumanZero:
			//---top-left corner---
			for(int k=0;k<maxCompartments;++k){
				div[k]=0;
			}
			//---first column (except top-left corner)---
			pos=ncols*maxCompartments;
			posUp=0;
			for(int i=1;i<nrows;++i){//first column
				for(int k=0;k<maxCompartments;++k){
					int fromUp=0;
					while(assignmentV[posUp+fromUp]!=k){
						++fromUp;
					}
					div[pos+k]=(fr[pos+k]-fr[posUp+fromUp]);
				}
				pos+=ncols*maxCompartments;
				posUp+=ncols*maxCompartments;
			}
			//---first row (except top-left corner)---
			pos=maxCompartments;
			posLeft=0;
			for(int j=1;j<ncols;++j){//first row
				for(int k=0;k<maxCompartments;++k){
					int fromLeft=0;
					while(assignmentH[posLeft+fromLeft]!=k){
						++fromLeft;
					}
					div[pos+k]=fc[pos+k]-fc[posLeft+fromLeft];
				}
				pos+=maxCompartments;
				posLeft+=maxCompartments;
			}
		break;
	}
	return 0;
}
//-------------------------

int computeMultiGradient(double *f, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *dfdr, double *dfdc, EDerivativeType edt, EBoundaryCondition ebc){
	if((nrows<2) || (ncols<2)){
		return -1;
	}
	switch(edt){
		case EDT_Forward:
			computeRowDerivative_Forward(f, assignmentV, nrows, ncols, maxCompartments, dfdr,ebc);
			computeColumnDerivative_Forward(f, assignmentH, nrows, ncols, maxCompartments, dfdc,ebc);
		break;
		case EDT_Backward:
			computeRowDerivative_Backward(f, assignmentV, nrows, ncols, maxCompartments, dfdr,ebc);
			computeColumnDerivative_Backward(f, assignmentH, nrows, ncols, maxCompartments, dfdc,ebc);
		break;
	}
	return 0;
}

int computeMultiDivergence(double *fr, double *fc, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *div, EDerivativeType edt, EBoundaryCondition ebc){
	if((nrows<2) || (ncols<2)){
		return -1;
	}
	switch(edt){
		case EDT_Forward:
			computeDivergence_Forward(fr, fc, assignmentV, assignmentH, nrows, ncols, maxCompartments, div, ebc);
		break;
		case EDT_Backward:
			computeDivergence_Backward(fr, fc, assignmentV, assignmentH, nrows, ncols, maxCompartments, div, ebc);
		break;
	}
	return 0;
}

int computeMultiDivergence(double *frr, double *fcc, double *frc, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *divr, double *divc, EDerivativeType edt, EBoundaryCondition ebc){
	int retVal=computeMultiDivergence(frr, frc, assignmentV, assignmentH, nrows, ncols, maxCompartments, divr, edt, ebc);
	computeMultiDivergence(frc, fcc, assignmentV, assignmentH, nrows, ncols, maxCompartments, divc, edt, ebc);
	return retVal;
}

int computeMultiJacobian(double *fr, double *fc, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *dfdrr, double *dfdcc, double *dfdrc, EDerivativeType edt, EBoundaryCondition ebc){
	double *tmp=new double[nrows*ncols*maxCompartments];
	int retVal=computeMultiGradient(fr, assignmentV, assignmentH, nrows, ncols, maxCompartments, dfdrr, dfdrc, edt, ebc);
	retVal=computeMultiGradient(fc, assignmentV, assignmentH, nrows, ncols, maxCompartments, tmp, dfdcc, edt, ebc);
	int nsites=nrows*ncols*maxCompartments;
	for(int i=0;i<nsites;++i){
		dfdrc[i]=0.5*(dfdrc[i]+tmp[i]);
	}
	delete[] tmp;
	return 0;
}


int computeMultiGradient(double *f, int *assignmentS, int *assignmentV, int *assignmentH, int nslices, int nrows, int ncols, int maxCompartments, double *dfds, double *dfdr, double *dfdc, EDerivativeType edt, EBoundaryCondition ebc){
	//TO-DO
	return 0;
}

int computeMultiDivergence(double *fs, double *fr, double *fc, int *assignmentS, int *assignmentV, int *assignmentH, int nslices, int nrows, int ncols, int maxCompartments, double *div, EDerivativeType edt, EBoundaryCondition ebc){
	//TO-DO
	return 0;
}

int computeMultiDivergence(double *fss, double *frr, double *fcc, double *fsr, double *frc, double *fsc, int *assignmentS, int *assignmentV, int *assignmentH, int nslices, int nrows, int ncols, int maxCompartments, double *divs, double *divr, double *divc, EDerivativeType edt, EBoundaryCondition ebc){
	//TO-DO
	return 0;
}

int computeMultiJacobian(double *fs, double *fr, double *fc, int *assignmentS, int *assignmentV, int *assignmentH, int nslices, int nrows, int ncols, int maxCompartments, double *fss, double *frr, double *fcc, double *fsr, double *frc, double *fsc, EDerivativeType edt, EBoundaryCondition ebc){
	//TO-DO
	return 0;
}