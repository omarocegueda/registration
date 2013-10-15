#include "bitsCPP.h"
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
