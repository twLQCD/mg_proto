/*
 * coarse_l1_blas.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: bjoo
 */

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_l1_blas.h"


namespace MG
{

namespace GlobalComm {
	void GlobalSum( double& my_summand )
	{
		return; // Return Summand Unchanged -- MPI version should use an MPI_ALLREDUCE

	}
	void GlobalSum( double* array, int array_length ) {
		return;  // Single Node for now. Return the untouched array. -- MPI Version should use allreduce
	}
}


/** Performs:
 *  x <- x - y;
 *  returns: norm(x) after subtraction
 *  Useful for computing residua, where r = b and y = Ax
 *  then n2 = xmyNorm(r,y); will leave r as the residuum and return its square norm
 *
 * @param x  - CoarseSpinor ref
 * @param y  - CoarseSpinor ref
 * @return   double containing the square norm of the difference
 *
 */
double XmyNorm2Vec(CoarseSpinor& x, const CoarseSpinor& y)
{
	double norm_diff = (double)0;

	const LatticeInfo& x_info = x.GetInfo();


	const LatticeInfo& y_info = y.GetInfo();
	AssertCompatible(x_info, y_info);


	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();

	// Loop over the sites and sum up the norm
#pragma omp parallel for collapse(2) reduction(+:norm_diff)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {


			// Identify the site
			float* x_site_data = x.GetSiteDataPtr(cb,cbsite);
			const float* y_site_data = y.GetSiteDataPtr(cb,cbsite);

			// Sum difference over the colorspins
			for(int cspin=0; cspin < num_colorspin; ++cspin) {
				double diff_re = x_site_data[ RE + n_complex*cspin ] - y_site_data[ RE + n_complex*cspin ];
				double diff_im = x_site_data[ IM + n_complex*cspin ] - y_site_data[ IM + n_complex*cspin ];
				x_site_data[RE + n_complex * cspin] = diff_re;
				x_site_data[IM + n_complex * cspin] = diff_im;

				norm_diff += diff_re*diff_re + diff_im*diff_im;

			}

		}
	} // End of Parallel for reduction

	// I would probably need some kind of global reduction here  over the nodes which for now I will ignore.
	MG::GlobalComm::GlobalSum(norm_diff);

	return norm_diff;
}

/** returns || x ||^2
 * @param x  - CoarseSpinor ref
 * @return   double containing the square norm of the difference
 *
 */
double Norm2Vec(const CoarseSpinor& x)
{
	double norm_sq = (double)0;

	const LatticeInfo& x_info = x.GetInfo();


	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();

	// Loop over the sites and sum up the norm
#pragma omp parallel for collapse(2) reduction(+:norm_sq)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {


			// Identify the site
			const float* x_site_data = x.GetSiteDataPtr(cb,cbsite);

			// Sum difference over the colorspins
			for(int cspin=0; cspin < num_colorspin; ++cspin) {
				double x_re = x_site_data[ RE + n_complex*cspin ];
				double x_im = x_site_data[ IM + n_complex*cspin ];

				norm_sq += x_re*x_re + x_im*x_im;

			}

		}
	} // End of Parallel for reduction

	// I would probably need some kind of global reduction here  over the nodes which for now I will ignore.
	MG::GlobalComm::GlobalSum(norm_sq);

	return norm_sq;
}

/** returns < x | y > = x^H . y
 * @param x  - CoarseSpinor ref
 * @param y  - CoarseSpinor ref
 * @return   double containing the square norm of the difference
 *
 */
std::complex<double> InnerProductVec(const CoarseSpinor& x, const CoarseSpinor& y)
{

	const LatticeInfo& x_info = x.GetInfo();
	const LatticeInfo& y_info = y.GetInfo();
		AssertCompatible(x_info, y_info);

	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();

	double iprod_re=(double)0;
	double iprod_im=(double)0;

	// Loop over the sites and sum up the norm
#pragma omp parallel for collapse(2) reduction(+:iprod_re) reduction(+:iprod_im)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {


			// Identify the site
			const float* x_site_data = x.GetSiteDataPtr(cb,cbsite);
			const float* y_site_data = y.GetSiteDataPtr(cb,cbsite);

			// Sum difference over the colorspins
			for(int cspin=0; cspin < num_colorspin; ++cspin) {

				iprod_re += x_site_data[ RE + n_complex*cspin ]*y_site_data[ RE + n_complex*cspin ]
							+ x_site_data[ IM + n_complex*cspin ]*y_site_data[ IM + n_complex*cspin ];

				iprod_im += x_site_data[ RE + n_complex*cspin ]*y_site_data[ IM + n_complex*cspin ]
							- x_site_data[ IM + n_complex*cspin ]*y_site_data[ RE + n_complex*cspin ];


			}

		}
	} // End of Parallel for reduction

	// Global Reduce
	double iprod_array[2] = { iprod_re, iprod_im };
	MG::GlobalComm::GlobalSum(iprod_array,2);

	std::complex<double> ret_val(iprod_array[0],iprod_array[1]);

	return ret_val;
}

void ZeroVec(CoarseSpinor& x)
{
	const LatticeInfo& x_info = x.GetInfo();


	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();
#pragma omp parallel for collapse(2)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {


			// Identify the site
			float* x_site_data = x.GetSiteDataPtr(cb,cbsite);

			// Sum difference over the colorspins
			for(int cspin=0; cspin < num_colorspin; ++cspin) {

				x_site_data[RE + n_complex*cspin] = 0;
				x_site_data[IM + n_complex*cspin] = 0;

			}

		}
	} // End of Parallel for region

}

void CopyVec(CoarseSpinor& x, const CoarseSpinor& y)
{


	const LatticeInfo& x_info = x.GetInfo();
	const LatticeInfo& y_info = y.GetInfo();
	AssertCompatible(x_info, y_info);


	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();
#pragma omp parallel for collapse(2)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {


			// Identify the site
			float* x_site_data = x.GetSiteDataPtr(cb,cbsite);
			const float* y_site_data = y.GetSiteDataPtr(cb,cbsite);

			// Sum difference over the colorspins
			for(int cspin=0; cspin < num_colorspin; ++cspin) {

				x_site_data[RE + n_complex*cspin] = y_site_data[RE + n_complex*cspin];
				x_site_data[IM + n_complex*cspin] = y_site_data[IM + n_complex*cspin];

			}

		}
	} // End of Parallel for region

}

void ScaleVec(const float alpha, CoarseSpinor& x)
{


	const LatticeInfo& x_info = x.GetInfo();

	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();

#pragma omp parallel for collapse(2)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {


			// Identify the site
			float* x_site_data = x.GetSiteDataPtr(cb,cbsite);

			for(int cspin=0; cspin < num_colorspin; ++cspin) {

				x_site_data[RE + n_complex*cspin] *= alpha;
				x_site_data[IM + n_complex*cspin] *= alpha;
			}

		}
	} // End of Parallel for region

}

void ScaleVec(const std::complex<float>& alpha, CoarseSpinor& x)
{


	const LatticeInfo& x_info = x.GetInfo();

	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();
#pragma omp parallel for collapse(2)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {


			// Identify the site
			float* x_site_data = x.GetSiteDataPtr(cb,cbsite);

			for(int cspin=0; cspin < num_colorspin; ++cspin) {
				std::complex<float> t( x_site_data[RE + n_complex*cspin],
									   x_site_data[IM + n_complex*cspin]);

				t *= alpha;
				x_site_data[RE + n_complex*cspin] = real(t);
				x_site_data[IM + n_complex*cspin] = imag(t);
			}

		}
	} // End of Parallel for reduction

}

void AxpyVec(const std::complex<float>& alpha, const CoarseSpinor& x, CoarseSpinor& y)
{
	const LatticeInfo& x_info = x.GetInfo();
	const LatticeInfo& y_info = y.GetInfo();
	AssertCompatible(x_info, y_info);


	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();

#pragma omp parallel for collapse(2)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {



			// Identify the site
			const float* x_site_data = x.GetSiteDataPtr(cb,cbsite);
			float* y_site_data = y.GetSiteDataPtr(cb,cbsite);

			// Sum difference over the colorspins
			for(int cspin=0; cspin < num_colorspin; ++cspin) {


				std::complex<float> c_x( x_site_data[RE + n_complex*cspin],
													   x_site_data[IM + n_complex*cspin]);

				std::complex<float> c_y( y_site_data[RE + n_complex*cspin],
													   y_site_data[IM + n_complex*cspin]);


				c_y += alpha*c_x;

				y_site_data[ RE + n_complex*cspin] = real(c_y);
				y_site_data[ IM + n_complex*cspin] = imag(c_y);

			}

		}
	} // End of Parallel for region
}

void AxpyVec(const float& alpha, const CoarseSpinor&x, CoarseSpinor& y) {
	const LatticeInfo& x_info = x.GetInfo();
	const LatticeInfo& y_info = y.GetInfo();
	AssertCompatible(x_info, y_info);


	IndexType num_cbsites = x_info.GetNumCBSites();
	IndexType num_colorspin = x.GetNumColorSpin();

#pragma omp parallel for collapse(2)
	for(int cb=0; cb < 2; ++cb ) {
		for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {



			// Identify the site
			const float* x_site_data = x.GetSiteDataPtr(cb,cbsite);
			float* y_site_data = y.GetSiteDataPtr(cb,cbsite);

			// Sum difference over the colorspins
			for(int cspin=0; cspin < num_colorspin; ++cspin) {


				y_site_data[ RE + n_complex*cspin] += alpha*x_site_data[ RE + n_complex*cspin];
				y_site_data[ IM + n_complex*cspin] += alpha*x_site_data[ IM + n_complex*cspin];

			}

		}
	} // End of Parallel for region
}



};

