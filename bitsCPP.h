#ifndef BITS_H
#define BITS_H
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

bool isNumber(float x);
bool isInfinite(float x);
bool isSNAN(float x);
bool isQNAN(float x);
bool isNumber(double x);
bool isInfinite(double x);
bool isSNAN(double x);
bool isQNAN(double x);

#endif
