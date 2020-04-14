/*
 * qphix_blas_wrappers.h
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"
#include "lattice/qphix/qphix_blas_wrappers.h"
#include <qphix/blas_full_spinor.h>
#include "lattice/coarse/subset.h"

using namespace QPhiX;

namespace MG
{


// x = x - y; followed by || x ||
template<typename ST>
inline
std::vector<double> XmyNorm2VecT(ST& x, const ST& y, const CBSubset& subset)
{
  IndexType ncol = x.GetNCol();
  std::vector<double> ret_norm(ncol);
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  for (int col=0; col < ncol; ++col)
    xmy2Norm2Spinor<>( y.get(col), x.get(col), ret_norm[col],  geom, n_blas_simt, subset.start, subset.end);
  return ret_norm;
}


std::vector<double> XmyNorm2Vec(QPhiXSpinor& x, const QPhiXSpinor& y, const CBSubset& subset )
{
  return XmyNorm2VecT(x,y,subset);
}

std::vector<double> XmyNorm2Vec(QPhiXSpinorF& x, const QPhiXSpinorF& y,const CBSubset& subset)
{
  return XmyNorm2VecT(x,y,subset);
}


template<typename ST>
inline
std::vector<double> Norm2VecT(const ST& x, const CBSubset& subset)
{
  IndexType ncol = x.GetNCol();
  std::vector<double> ret_norm(ncol);
  const typename ST::GeomT& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  for (int col=0; col < ncol; ++col)
    norm2Spinor(ret_norm[col], x.get(col), geom,n_blas_simt, subset.start, subset.end);
  return ret_norm;
}

std::vector<double>
Norm2Vec(const QPhiXSpinor& x, const CBSubset& subset)
{
  return Norm2VecT(x, subset);
}

std::vector<double>
Norm2Vec(const QPhiXSpinorF& x, const CBSubset& subset)
{
  return Norm2VecT(x, subset);
}



template<typename ST>
inline
std::vector<std::complex<double>> InnerProductVecT(const ST& x, const ST& y, const CBSubset& subset)
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  IndexType ncol = x.GetNCol();

  std::vector<std::complex<double>> ret(ncol);
  for (int col=0; col < ncol; ++col) {
    double result[2];
    innerProductSpinor(result,x.get(ncol),y.get(ncol),geom, n_blas_simt, subset.start, subset.end);
    ret[col] = std::complex<double>(result[0],result[1]);
  }

  return ret;
}

std::vector<std::complex<double>> InnerProductVec(const QPhiXSpinor& x, const QPhiXSpinor& y, const CBSubset& subset )
{
  return InnerProductVecT(x,y, subset);
}

std::vector<std::complex<double>> InnerProductVec(const QPhiXSpinorF& x, const QPhiXSpinorF& y, const CBSubset& subset )
{
  return InnerProductVecT(x,y, subset);
}

template<typename ST>
void ZeroVecT(ST& x, const CBSubset& subset)
{
  const typename ST::GeomT& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  IndexType ncol = x.GetNCol();
  for (int col=0; col < ncol; ++col)
    zeroSpinor(x.get(col),geom,n_blas_simt, subset.start, subset.end);
}

void ZeroVec(QPhiXSpinor& x, const CBSubset& subset ) { ZeroVecT(x,subset); }

void ZeroVec(QPhiXSpinorF& x, const CBSubset& subset) { ZeroVecT(x,subset); }

template<typename ST>
inline
void CopyVecT(ST& x, const ST& y,const CBSubset& subset)
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  IndexType ncol = x.GetNCol();
  for (int col=0; col < ncol; ++col)
    copySpinor(x.get(col),y.get(col),geom,n_blas_simt, subset.start, subset.end);
}

void CopyVec(QPhiXSpinor& x, const QPhiXSpinor& y, const CBSubset& subset ) { CopyVecT(x,y, subset); }
void CopyVec(QPhiXSpinorF& x, const QPhiXSpinorF& y, const CBSubset& subset) { CopyVecT(x,y, subset); }


template<typename ST>
inline
void AxVecT(const std::vector<double> alpha, ST& x, const CBSubset& subset )
{
  const typename ST::GeomT& geom = x.getGeom();
  int n_blas_simt = geom.getNSIMT();
  IndexType ncol = x.GetNCol();
  for (int col=0; col < ncol; ++col)
    axSpinor<>(alpha[col], x.get(col),geom,n_blas_simt, subset.start, subset.end);
}

void AxVec(const std::vector<double> alpha, QPhiXSpinor& x,const CBSubset& subset) { AxVecT(alpha,x,subset); }
void AxVec(const std::vector<double> alpha, QPhiXSpinorF& x, const CBSubset& subset) { AxVecT(alpha,x,subset); }


template<typename ST, typename T>
inline
void AxpyVecT(const std::vector<T>& alpha, const ST& x, ST& y, const CBSubset& subset )
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  IndexType ncol = x.GetNCol();

  for (int col=0; col < ncol; ++col) {
    double a[2] = { std::real(alpha[col]), std::imag(alpha[col]) };
    caxpySpinor(a, x.get(col), y.get(col), geom,n_blas_simt, subset.start, subset.end);
  }
}

void AxpyVec(const std::vector<double>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y,const CBSubset& subset)
{
   AxpyVecT(alpha,x,y,subset);
}

void AxpyVec(const std::vector<double>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y, const CBSubset& subset )
{
  AxpyVecT(alpha,x,y,subset);
}


void AxpyVec(const std::vector<std::complex<float>>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y, const CBSubset& subset )
{
  AxpyVecT(alpha,x,y,subset);
}

void AxpyVec(const std::vector<std::complex<float>>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y, const CBSubset& subset)
{
  AxpyVecT(alpha,x,y,subset);
}

void AxpyVec(const std::vector<std::complex<double>>& alpha, const QPhiXSpinor& x, QPhiXSpinor& y, const CBSubset& subset )
{
  AxpyVecT(alpha,x,y, subset);
}

void AxpyVec(const std::vector<std::complex<double>>& alpha, const QPhiXSpinorF& x, QPhiXSpinorF& y, const CBSubset& subset)
{
  AxpyVecT(alpha,x,y, subset);
}



template<typename ST>
void GaussianT(ST& v,const CBSubset& subset)
{
  IndexType ncol = v.GetNCol();
  LatticeFermion x;
  for (int col=0; col < ncol; ++col) {
    gaussian(x);
    QDPSpinorToQPhiXSpinor(x,v,col,subset);
  }

}

void Gaussian(QPhiXSpinor& v,const CBSubset& subset) {
  GaussianT(v, subset);
}

void Gaussian(QPhiXSpinorF& v,const CBSubset& subset) {
  GaussianT(v, subset);
}

template<typename S1, typename S2>
inline
void ConvertSpinorT(const S1& in, S2& out, const CBSubset& subset)
{
  const typename S1::GeomT& geom_in = in.getGeom();
  const typename S2::GeomT& geom_out = out.getGeom();
  const double scale_factor = 1;
  const int n_blas_threads = geom_out.getNSIMT();
  IndexType ncol = in.GetNCol();

  for (int col=0; col < ncol; ++col) {
     for(int cb=subset.start; cb < subset.end; ++cb) {
       // QPhiX conversions
       convert( out.getCB(col,cb).get(), scale_factor, in.getCB(col,cb).get(),
           geom_out, geom_in,n_blas_threads);
     }
  }
}

void ConvertSpinor(const QPhiXSpinor& in, QPhiXSpinorF& out, const CBSubset& subset)
{
  ConvertSpinorT(in,out,subset);
}

void ConvertSpinor(const QPhiXSpinorF& in, QPhiXSpinor& out, const CBSubset& subset)
{
  ConvertSpinorT(in,out,subset);
}


template<typename ST>
inline
void  YpeqXVecT(const ST& x, ST& y, const CBSubset& subset )
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  IndexType ncol = x.GetNCol();
  for (int col=0; col < ncol; ++col)
    ypeqxSpinor(x.get(col), y.get(col), geom,n_blas_simt, subset.start,subset.end);
}

void YpeqXVec(const QPhiXSpinor& x, QPhiXSpinor& y, const CBSubset& subset )
{
  YpeqXVecT(x,y, subset);
}

void YpeqXVec(const QPhiXSpinorF& x, QPhiXSpinorF& y, const CBSubset& subset )
{
  YpeqXVecT(x,y,subset);
}

template<typename ST>
inline
void  YmeqXVecT(const ST& x, ST& y,const CBSubset& subset )
{
  const typename ST::GeomT& geom = y.getGeom();
  int n_blas_simt = geom.getNSIMT();
  IndexType ncol = x.GetNCol();
  for (int col=0; col < ncol; ++col)
    ymeqxSpinor(x.get(col), y.get(col), geom,n_blas_simt, subset.start, subset.end);
}

void YmeqXVec(const QPhiXSpinor& x, QPhiXSpinor& y, const CBSubset& subset )
{
  YmeqXVecT(x,y, subset);
}

void YmeqXVec(const QPhiXSpinorF& x, QPhiXSpinorF& y,const CBSubset& subset )
{
  YmeqXVecT(x,y,subset);
}



} // namespace
