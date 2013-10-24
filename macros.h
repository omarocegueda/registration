#ifndef MACROS_H
#define MACROS_H
#include <math.h>
#include <string.h>
#define SQR(x) ((x)*(x))
#define ABS(x) (((x)<0)?(-(x)):(x))
const double EPSILON=1e-9;
const double EPSILON_SQR=EPSILON*EPSILON;
#ifndef MIN
	#define MIN(a,b) ((a)<(b)?(a):(b))
#endif
#ifndef MAX
	#define MAX(a,b) ((a)<(b)?(b):(a))
#endif
#define IS_ZERO_3(v) ((v[0]==0)&&(v[1]==0)&&(v[2]==0))
#define EQUALS_3(v,a,b,c) ((v[0]==(a))&&(v[1]==(b))&&(v[2]==(c)))
#define IN_RANGE(x, a, b) (((a)<=(x)) && ((x)<(b)))
#define SIGN(x) ((x)?(((x)>0)?1:-1):0)
#define DELETE_ARRAY(v) {if(v!=NULL){delete[] v; v=NULL;}}
#define DELETE_INSTANCE(v) {if(v!=NULL){delete v; v=NULL;}}

#define REPORT_PROGRESS(i,n,strm) {if(((10*(i))/(n))<((10*((i)+1))/n)){strm<<"Progress: "<<((100*((i)+1))/n)<<"%"<<endl;}}
#define REPORT_PROGRESS_100(i,n,strm) {if(((100*(i))/(n))<((100*((i)+1))/n)){strm<<"Progress: "<<((100*((i)+1))/n)<<"%"<<endl;}}
#define IS_ZERO(v) ((fabs(double(v[0]))<EPSILON) && (fabs(double(v[1]))<EPSILON) && (fabs(double(v[2]))<EPSILON))
#define MAX_STR_LEN 2048
#endif
