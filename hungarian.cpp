#include <vector>
#include <queue>
#include <string>
#include <set>
#include <map>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include "hungarian.h"
#include "macros.h"
using namespace std;
#define REP(i,n) for(int i=0, _n=(n);i<_n;++i)

Hungarian::Hungarian(int _n){
	allocate(_n);
}
void Hungarian::setSize(int _n){
	if(_n<=maxSize){
		n=_n;
	}else{//re-allocate
		free();
		allocate(_n);
	}
}

void Hungarian::allocate(int _n){
	n=_n;
	maxSize=n;
	columnChecked=new bool[n];
	rowChecked=new bool[n];
	starRowPosition=new int[n];
	starColumnPosition=new int[n];
	primeRowPosition=new int[n];
	primeColumnPosition=new int[n];
	A=new HUNGARIAN_DATA_TYPE[n*n];
	M=new HUNGARIAN_DATA_TYPE[n*n];
}

void Hungarian::free(void){
	DELETE_ARRAY(columnChecked);
	DELETE_ARRAY(rowChecked);
	DELETE_ARRAY(starRowPosition);
	DELETE_ARRAY(starColumnPosition);
	DELETE_ARRAY(primeRowPosition);
	DELETE_ARRAY(primeColumnPosition);
	DELETE_ARRAY(A);
	DELETE_ARRAY(M);
	maxSize=0;
	n=0;
}

Hungarian::~Hungarian(){
	free();
}

void Hungarian::hungarianStep1(void){
	HUNGARIAN_DATA_TYPE *currentRow=M;
	for(int i=0;i<n;++i, currentRow+=n){
		int p=0;
		for(int j=1;j<n;++j){
			if(currentRow[j]<currentRow[p]){
				p=j;
			}
		}
		HUNGARIAN_DATA_TYPE minVal=currentRow[p];
		REP(j,n) currentRow[j]-=minVal;
	}
}
	        
int Hungarian::hungarianStep2(void){
	int count=0;
	for(int i=0;i<n;++i){
		int pos=i*n;
		for(int j=0;j<n;++j,++pos){
			if(HUNGARIAN_IS_ZERO(M[pos]) && (starColumnPosition[i]==-1) && (starRowPosition[j]==-1)){
				columnChecked[j]=true;
				starRowPosition[j]=i;
				starColumnPosition[i]=j;
				count++;
				break;
			}
		}
	}
	return count;
}
	         
void Hungarian::hungarianStep6(void){
	HUNGARIAN_DATA_TYPE minActual=-1;
	int pos=0;
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j, ++pos)if((!rowChecked[i]) && (!columnChecked[j])){
			if((minActual==-1) || (M[pos]<minActual)){
				minActual=M[pos];
			}
		}
	}
	pos=0;
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j, ++pos){
			if(!rowChecked[i] && !columnChecked[j]){
				M[pos]-=minActual;
			}else if(rowChecked[i] && columnChecked[j]){
				M[pos]+=minActual;
			}
		}
	}
}
	         
	         
void Hungarian::hungarianStep4(void){//returns true iff an unchecked zero was found
	bool found=true;
	while(found){
		found=false;
		int pos=0;
		for(int i=0;i<n;++i)for(int j=0;j<n;++j,++pos){
			if(HUNGARIAN_IS_ZERO(M[pos]) && !rowChecked[i] && !columnChecked[j]){
				primeColumnPosition[i]=j;
				primeRowPosition[j]=i;
				if(starColumnPosition[i]!=-1){
					columnChecked[starColumnPosition[i]]=false;
					rowChecked[i]=true;
					i=j=n;
					found=true;
					break;
				}else{
					return;
				}
			}
		}
		if(!found){
			hungarianStep6();//construct some zeros =)
			found=true;
		}
	}
}
	         
int Hungarian::hungarianStep5(void){
	int ini=-1;
	REP(i,n){
		if((primeColumnPosition[i]!=-1) && (starColumnPosition[i]==-1) && (!rowChecked[i]) && (!columnChecked[primeColumnPosition[i]])){
			ini=i;
			break;
		}
	}
	int count=0;
	while(ini!=-1){
		//printf("step 5:%d\n",next);
		int next=starRowPosition[primeColumnPosition[ini]];
		//printf("step 5:%d\n\n",next);
		starRowPosition         [primeColumnPosition[ini]]=ini;
		starColumnPosition      [ini]                     =primeColumnPosition[ini];
		primeRowPosition        [primeColumnPosition[ini]]=-1;
		primeColumnPosition     [ini]                     =-1;
		count++;
		ini=next;
	}
	REP(i,n){
		rowChecked[i]=false;
		columnChecked[i]=(starRowPosition[i]!=-1);
	}
	return count>0;
}

HUNGARIAN_DATA_TYPE Hungarian::solve(void){
	//====initialize====
	memset(columnChecked, 0, sizeof(bool)*n);
	memset(rowChecked, 0, sizeof(bool)*n);
	memset(starRowPosition, -1, sizeof(int)*n);
	memset(starColumnPosition, -1, sizeof(int)*n);
	memset(primeRowPosition, -1, sizeof(int)*n);
	memset(primeColumnPosition, -1, sizeof(int)*n);
	memcpy(A, M, sizeof(HUNGARIAN_DATA_TYPE)*n*n);
	//==================
	hungarianStep1();
	int count=hungarianStep2();
	if(count==n){
		HUNGARIAN_DATA_TYPE total=0;
		REP(i,n)total+=A[i*n+starColumnPosition[i]];
		return total;
	}
	while(count<n){
		hungarianStep4();
		count+=hungarianStep5();
	}
	HUNGARIAN_DATA_TYPE total=0;
	REP(i,n)total+=A[i*n+starColumnPosition[i]];
	return total;
}

void Hungarian::setCost(int i, int j, HUNGARIAN_DATA_TYPE cost){
	M[i*n+j]=cost;
}

void Hungarian::setAllCosts(HUNGARIAN_DATA_TYPE val){
	int len=n*n;
	for(int i=0;i<len;++i){
		M[i]=val;
	}
}

void Hungarian::getAssignment(int *assignment){
	memcpy(assignment, starColumnPosition, sizeof(int)*n);
}
