#ifndef HUNGARIAN_H
#define HUNGARIAN_H
//-----ad-hoc implementation-----
#define HUNGARIAN_DATA_TYPE double
#define HUNGARIAN_EPSILON 1e-10
#define HUNGARIAN_IS_ZERO(x) (fabs(x)<HUNGARIAN_EPSILON)
#define HUNGARIAN_EQUALS(a,b) (fabs((a)-(b))<HUNGARIAN_EPSILON)
//------------------------
class Hungarian{
	private:
		bool *columnChecked;
		bool *rowChecked;
		int *starRowPosition;
		int *starColumnPosition;
		int *primeRowPosition;
		int *primeColumnPosition;
		HUNGARIAN_DATA_TYPE *A;
		HUNGARIAN_DATA_TYPE *M;
	protected:
		int n;
		int maxSize;
		void hungarianStep1(void);
		int hungarianStep2(void);
		void hungarianStep4(void);
		int hungarianStep5(void);
		void hungarianStep6(void);
		void free(void);
		void allocate(int _n);
	public:
		Hungarian(int _n);
		void setSize(int _n);
		void setCost(int i, int j, HUNGARIAN_DATA_TYPE cost);
		void setAllCosts(HUNGARIAN_DATA_TYPE val);
		HUNGARIAN_DATA_TYPE solve(void);
		void getAssignment(int *assignment);
		~Hungarian();
};

#endif
