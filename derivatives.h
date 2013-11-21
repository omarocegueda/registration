/*! \file derivatives.h
	\author Omar Ocegueda
	\brief Declaration of numerical differentiation functions (derivatives, divergence, rotational, Jacobian, etc.) and derivative types 
*/
#ifndef DERIVATIVES_H
#define DERIVATIVES_H
/*! \brief Available derivative types
	\author Omar Ocegueda
*/
enum EDerivativeType{	EDT_Forward, /*!< Derivative type value: Forward*/
						EDT_Backward /*!< Derivative type value: Backward*/
					};

/*! \brief Boundary conditions
	\author Omar Ocegueda
	These are the boundary conditions allowed when computing derivatives. 
*/
enum EBoundaryCondition{EBC_Circular,		/*!< Boundary condition value: Circular (assumes the function is circular)*/
						EBC_DirichletZero,	/*!< Boundary condition value: Dirichlet (assumes the function is zero beyond the boundary)*/
						EBC_VonNeumanZero	/*!< Boundary condition value: Von Neuman (assumes the derivative is zero at the boundary)*/
						};

/*! \brief Computes the 2D gradient field of a scalar image f.
	\author Omar Ocegueda
	\param f the image to be differentiated
	\param nrows the number of rows of image \a f
	\param ncols the number of columns of image \a f
	\param dfdr on exit contains the derivative of \a f w.r.t rows (vertical dimension)
	\param dfdc on exit contains the derivative of \a f w.r.t columns (horizontal dimension)
	\param edt the derivative type to be used 
	\param ebc the boundary condition to be used 
*/
int computeGradient(double *f, int nrows, int ncols, double *dfdr, double *dfdc, EDerivativeType edt=EDT_Forward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the divergence of a 2D vector field given by images fr, fc. The vector at position i is (fr[i], fc[i]).
	\author Omar Ocegueda
	\param fr the row (vertical) component of the vector field
	\param fc the column (horizontal) component of the vector field
	\param nrows the number of rows of the vector field (\a fr, \a fc)
	\param ncols the number of columns of the vector field (\a fr, \a fc)
	\param div on exit contains the divergence of the vector field (\a fr, \a fc)
	\param edt the derivative type to be used (both, for \a fr and \a fc)
	\param ebc the boundary condition to be used (both, for \a fr and \a fc)
*/
int computeDivergence(double *fr, double *fc, int nrows, int ncols, double *div, EDerivativeType edt=EDT_Backward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the 3D gradient field of a scalar image f.
	\author Omar Ocegueda
	\param f the image to be differentiated
	\param nslices the number of slices (z-dimension) of image \a f
	\param nrows the number of rows of image \a f
	\param ncols the number of columns of image \a f
	\param dfds on exit contains the derivative of \a f w.r.t slices (inter-slice (z) dimension)
	\param dfdr on exit contains the derivative of \a f w.r.t rows (vertical dimension)
	\param dfdc on exit contains the derivative of \a f w.r.t columns (horizontal dimension)
	\param edt the derivative type to be used 
	\param ebc the boundary condition to be used 
*/
int computeGradient(double *f, int nslices, int nrows, int ncols, double *dfds, double *dfdr, double *dfdc, EDerivativeType edt=EDT_Forward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the divergence of a 3D vector field given by images fs, fr, fc. The vector at position i is (fs[i], fr[i], fc[i]).
	\author Omar Ocegueda
	\param fs the slice (z) component of the vector field
	\param fr the row (vertical) component of the vector field
	\param fc the column (horizontal) component of the vector field
	\param nslices the number of slices (z-dimension) of the vector field
	\param nrows the number of rows of the vector field (\a fr, \a fc)
	\param ncols the number of columns of the vector field (\a fr, \a fc)
	\param div on exit contains the divergence of the vector field (\a fs, \a fr, \a fc)
	\param edt the derivative type to be used (for all \a fs, \a fr and \a fc)
	\param ebc the boundary condition to be used (for all \a fs, \a fr and \a fc)
*/
int computeDivergence(double *fs, double *fr, double *fc, int nslices, int nrows, int ncols, double *div, EDerivativeType edt=EDT_Backward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the rotational of a 2D vector field given by images fr, fc. The vector at position i is (fr[i], fc[i]).
	\author Omar Ocegueda
	\param fr the row (vertical) component of the vector field
	\param fc the column (horizontal) component of the vector field
	\param nrows the number of rows of the vector field (\a fr, \a fc)
	\param ncols the number of columns of the vector field (\a fr, \a fc)
	\param rot on exit contains the rotational of the vector field (\a fr, \a fc) (note that the rotational always points towards the z axis, so the output contains the signed-magnitude of the rotational only)
	\param edt the derivative type to be used (both, for \a fr and \a fc)
	\param ebc the boundary condition to be used (both, for \a fr and \a fc)
*/
int computeRotational(double *fr, double *fc, int nrows, int ncols, double *rot, EDerivativeType edt, EBoundaryCondition ebc);

/*! \brief Computes the divergence of a symmetric 2D tensor field of order 2 (a 2D 'symmetric matrix field') given by images frr, fcc, fcc. The tensor at position i is (frr[i], frc[i] ; frc[i], fcc[i]).
	\author Omar Ocegueda
	\param frr the (0,0) component of the tensor field
	\param fcc the (1,1) component of the tensor field
	\param frc the (0, 1) and (1, 0) components of the tensor field (it is assumed to be symmetric)
	\param nrows the number of rows of the tensor field 
	\param ncols the number of columns of the tensor field
	\param divr on exit contains the row-component of the divergence of the tensor field
	\param divc on exit contains the column-component of the divergence of the tensor field
	\param edt the derivative type to be used (for all components of the tensor fied)
	\param ebc the boundary condition to be used (for all components of the tensor fied)
*/
int computeDivergence(double *frr, double *fcc, double *frc, int nrows, int ncols, double *divr, double *divc, EDerivativeType edt=EDT_Backward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the Jacobian of a 2D vector field given by images fr, fc. The vector field at position i is (fr[i], fc[i]).
	\author Omar Ocegueda
	\param fr the row (vertical) component of the vector field
	\param fc the column (horizontal) component of the vector field
	\param nrows the number of rows of the vector field (\a fr, \a fc)
	\param ncols the number of columns of the vector field (\a fr, \a fc)
	\param dfdrr on exit contains the derivative of \a fr w.r.t. to rows (vertical dimension)
	\param dfdcc on exit contains the derivative of \a fc w.r.t. to columns (horizontal dimension)
	\param dfdrc on exit contains the average of the derivatives of \a fr w.r.t. to columns (horizontal dimension) and \a fc w.r.t. rows
	\param edt the derivative type to be used 
	\param ebc the boundary condition to be used 
*/
int computeJacobian(double *fr, double *fc, int nrows, int ncols, double *dfdrr, double *dfdcc, double *dfdrc, EDerivativeType edt=EDT_Forward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the divergence of a symmetric 3D tensor field of order 2 (a 3D 'symmetric matrix field') given by images fss, frr, fcc, fsr, frc and fsc. The tensor at position i is (fss[i], fsr[i], fsc[i] ; fsr[i], frr[i], frc[i] ; fsc[i], frc[i], fcc[i]).
	\author Omar Ocegueda
	\param fss the (0,0) component of the tensor field
	\param frr the (1,1) component of the tensor field
	\param fcc the (2,2) component of the tensor field
	\param fsr the (0, 1) and (1, 0) components of the tensor field (it is assumed to be symmetric)
	\param frc the (1, 2) and (2, 1) components of the tensor field (it is assumed to be symmetric)
	\param fsc the (0, 2) and (2, 0) components of the tensor field (it is assumed to be symmetric)
	\param nslices the number of slices of the tensor field 
	\param nrows the number of rows of the tensor field 
	\param ncols the number of columns of the tensor field
	\param divs on exit contains the slice-component of the divergence of the tensor field
	\param divr on exit contains the row-component of the divergence of the tensor field
	\param divc on exit contains the column-component of the divergence of the tensor field
	\param edt the derivative type to be used (for all components of the tensor fied)
	\param ebc the boundary condition to be used (for all components of the tensor fied)
*/
int computeDivergence(double *fss, double *frr, double *fcc, double *fsr, double *frc, double *fsc, int nslices, int nrows, int ncols, double *divs, double *divr, double *divc, EDerivativeType edt=EDT_Backward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the symmetric Jacobian of a 3D vector field given by images fs, fr, fc. The vector field at position i is (fs[i], fr[i], fc[i]).
	\author Omar Ocegueda
	\param fs the slice (z) component of the vector field
	\param fr the row (vertical) component of the vector field
	\param fc the column (horizontal) component of the vector field
	\param nslices the number of slices of the vector field (\a fs, \a fr, \a fc)
	\param nrows the number of rows of the vector field (\a fs, \a fr, \a fc)
	\param ncols the number of columns of the vector field (\a fs, \a fr, \a fc)
	\param fss on exit contains the derivative of \a fs w.r.t. to slices (z dimension)
	\param frr on exit contains the derivative of \a fr w.r.t. to rows (vertical dimension)
	\param fcc on exit contains the derivative of \a fc w.r.t. to columns (horizontal dimension)
	\param fsr on exit contains the average of the derivatives of \a fs w.r.t. to rows (vertical dimension) and \a fr w.r.t. slices (z-dimension)
	\param frc on exit contains the average of the derivative of \a fr w.r.t. to columns (horizontal dimension) and \a fc w.r.t. rows (vertical dimension)
	\param fsc on exit contains the average of the derivative of \a fs w.r.t. to columns (horizontal dimension) and \a fc w.r.t. slices (z-dimension)
	\param edt the derivative type to be used 
	\param ebc the boundary condition to be used 
*/
int computeJacobian(double *fs, double *fr, double *fc, int nslices, int nrows, int ncols, double *fss, double *frr, double *fcc, double *fsr, double *frc, double *fsc, EDerivativeType edt=EDT_Forward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the 2D multi-gradient field of a multi-scalar image f.
	After defining an assignment between compartments of neighboring vertices (f's pixels, or voxels), the numerical derivatives are computed by 
	taking the finite differences between assigned compartments only. The assignments are given by \a assignmentV and \a assignmentH.
	\author Omar Ocegueda
	\param f the multi-scalar image to be differentiated
	\param assignmentV the assignment between vertical neighbors
	\param assignmentH the assignment between horizontal neighbors
	\param nrows the number of rows of the multi-scalar image \a f
	\param ncols the number of columns of the multi-scalar image \a f
	\param maxCompartments the maximum number of compartments along the multi-scalar field \a f 
	\param dfdr on exit contains the multi-derivative of \a f w.r.t rows (vertical dimension)
	\param dfdc on exit contains the multi-derivative of \a f w.r.t columns (horizontal dimension)
	\param edt the derivative type to be used 
	\param ebc the boundary condition to be used 
*/
int computeMultiGradient(double *f, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *dfdr, double *dfdc, EDerivativeType edt=EDT_Forward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the multi-divergence of the 2D multi-vector field fiven by \a fr and \a fc (two multi-scalar images).
	After defining an assignment between compartments of neighboring vertices (f's pixels, or voxels), the numerical derivatives are computed by 
	taking the finite differences between assigned compartments only. The assignments are given by \a assignmentV and \a assignmentH.
	\author Omar Ocegueda
	\param fr the row-component of the multi-vector field
	\param fc the column-component of the multi-vector field
	\param assignmentV the assignment between vertical neighbors
	\param assignmentH the assignment between horizontal neighbors
	\param nrows the number of rows of the multi-scalar image \a f
	\param ncols the number of columns of the multi-scalar image \a f
	\param maxCompartments the maximum number of compartments along the multi-tensor field 
	\param div on exit contains the multi-divergence of the multi-vector field given
	\param edt the derivative type to be used 
	\param ebc the boundary condition to be used 
*/
int computeMultiDivergence(double *fr, double *fc, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *div, EDerivativeType edt=EDT_Backward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the multi-divergence of the 2D multi-tensor field of order 2 (a 'multi-matrix field') fiven by \a frr, \a fcc and \a frc (three multi-scalar images).
	After defining an assignment between compartments of neighboring vertices (f's pixels, or voxels), the numerical derivatives are computed by 
	taking the finite differences between assigned compartments only. The assignments are given by \a assignmentV and \a assignmentH.
	\author Omar Ocegueda
	\param frr the (0, 0) component of the multi-tensor field
	\param fcc the (1, 1) component of the multi-tensor field
	\param frc the (0, 1) and (1, 0) components of the multi-tensor field (it is assumed to be symmetric)
	\param assignmentV the assignment between vertical neighbors
	\param assignmentH the assignment between horizontal neighbors
	\param nrows the number of rows of the multi-tensor field
	\param ncols the number of columns of the multi-tensor field
	\param maxCompartments the maximum number of compartments along the multi-tensor field
	\param divr on exit contains the rows-component of the multi-divergence of the multi-tensor field given
	\param divc on exit contains the columns-component of the multi-divergence of the multi-tensor field given
	\param edt the derivative type to be used 
	\param ebc the boundary condition to be used 
*/
int computeMultiDivergence(double *frr, double *fcc, double *frc, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *divr, double *divc, EDerivativeType edt=EDT_Backward, EBoundaryCondition ebc=EBC_VonNeumanZero);

/*! \brief Computes the symmetric multi-Jacobian of a 2D multi-vector field given by images fr, fc. 
	After defining an assignment between compartments of neighboring vertices (f's pixels, or voxels), the numerical derivatives are computed by 
	taking the finite differences between assigned compartments only. The assignments are given by \a assignmentV and \a assignmentH.
	\author Omar Ocegueda
	\param fr the row (vertical) component of the vector field
	\param fc the column (horizontal) component of the vector field
	\param assignmentV the assignment between vertical neighbors
	\param assignmentH the assignment between horizontal neighbors
	\param nrows the number of rows of the vector field (\a fs, \a fr, \a fc)
	\param ncols the number of columns of the vector field (\a fs, \a fr, \a fc)
	\param maxCompartments the maximum number of compartments along the multi-vector field
	\param dfdrr on exit contains the (0, 0) component of the multi-Jacobian 
	\param dfdcc on exit contains the (1, 1) component of the multi-Jacobian 
	\param dfdrc on exit contains the (0, 1) and (1, 0) component of the multi-Jacobian (it is assumed to be symmetric)
	\param edt the derivative type to be used 
	\param ebc the boundary condition to be used 
*/
int computeMultiJacobian(double *fr, double *fc, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double *dfdrr, double *dfdcc, double *dfdrc, EDerivativeType edt=EDT_Forward, EBoundaryCondition ebc=EBC_VonNeumanZero);


int computeMultiGradient(double *f, int *assignmentS, int *assignmentV, int *assignmentH, int nslices, int nrows, int ncols, int maxCompartments, double *dfds, double *dfdr, double *dfdc, EDerivativeType edt=EDT_Forward, EBoundaryCondition ebc=EBC_VonNeumanZero);
int computeMultiDivergence(double *fs, double *fr, double *fc, int *assignmentS, int *assignmentV, int *assignmentH, int nslices, int nrows, int ncols, int maxCompartments, double *div, EDerivativeType edt=EDT_Backward, EBoundaryCondition ebc=EBC_VonNeumanZero);
int computeMultiDivergence(double *fss, double *frr, double *fcc, double *fsr, double *frc, double *fsc, int *assignmentS, int *assignmentV, int *assignmentH, int nslices, int nrows, int ncols, int maxCompartments, double *divs, double *divr, double *divc, EDerivativeType edt=EDT_Backward, EBoundaryCondition ebc=EBC_VonNeumanZero);
int computeMultiJacobian(double *fs, double *fr, double *fc, int *assignmentS, int *assignmentV, int *assignmentH, int nslices, int nrows, int ncols, int maxCompartments, double *fss, double *frr, double *fcc, double *fsr, double *frc, double *fsc, EDerivativeType edt=EDT_Forward, EBoundaryCondition ebc=EBC_VonNeumanZero);
#endif

