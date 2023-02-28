/*
 * qphix_transfer.h
 *
 *  Created on: Mar 15, 2018
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_TRANSFER_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_TRANSFER_H_

#include <MG_config.h>

#include <immintrin.h>

#include "MG_config.h"
#include "lattice/qphix/qphix_types.h"
#include <lattice/coarse/block.h>
#include <lattice/coarse/coarse_types.h>
#include <lattice/qphix/qphix_veclen.h>
#include <omp.h>
#include <utils/print_utils.h>
namespace MG {

#define MAX_VECS 96

    template <class QSpinor> class QPhiXTransfer {
    public:
        QPhiXTransfer(const std::vector<Block> &blocklist,
                      const std::vector<std::shared_ptr<QSpinor>> &vectors,
                      int r_threads_per_block = 1)
            : _n_blocks(blocklist.size()),
              _n_vecs(vectors.size()),
              _blocklist(blocklist),
              _vecs(vectors),
              _r_threads_per_block(r_threads_per_block) {
            if (blocklist.size() > 0) {
                _sites_per_block = blocklist[0].getNumSites();
            } else {
                MasterLog(ERROR, "Cannot create Restrictor Array for blocklist of size 0");
            }

            int num_colorspin = 2 * _n_vecs;
            int num_complex_colorspin = 2 * num_colorspin;
            int num_complex = _n_blocks * _sites_per_block * 2 * 3 * num_complex_colorspin;
            _data = (float *)MemoryAllocate(num_complex * sizeof(float));

            int num_coarse_cbsites = _n_blocks / 2;

            const LatticeInfo &fine_info = vectors[0]->GetInfo();

            MasterLog(INFO, "Importing Vectors");
#pragma omp parallel for collapse(2)
            for (int block_cb = 0; block_cb < n_checkerboard; ++block_cb) {
                for (int block_cbsite = 0; block_cbsite < num_coarse_cbsites; ++block_cbsite) {

                    // Identify the current block.
                    int block_idx = block_cbsite + block_cb * num_coarse_cbsites;
                    const Block &block = blocklist[block_idx];

                    // Get the list of fine sites in the blocks
                    auto block_sitelist = block.getCBSiteList();
                    auto num_sites_in_block = block_sitelist.size();

                    // Loop through the fine sites in the block
                    for (IndexType fine_site_idx = 0;
                         fine_site_idx < static_cast<IndexType>(num_sites_in_block);
                         fine_site_idx++) {

                        // Find the fine site
                        const CBSite &fine_cbsite = block_sitelist[fine_site_idx];
                        const int fine_site = (rb[fine_cbsite.cb].siteTable())[fine_cbsite.site];

                        // Upper components
                        for (int spin = 0; spin < 2; ++spin) {
                            for (int color = 0; color < 3; ++color) {
                                for (int vec = 0; vec < _n_vecs; ++vec) {
                                    (*this).index(block_idx, fine_site_idx, spin, color, 0, vec,
                                                  RE) =
                                        (*(vectors[vec]))(0, fine_cbsite.cb, fine_cbsite.site, spin,
                                                          color, RE);
                                    (*this).index(block_idx, fine_site_idx, spin, color, 0, vec,
                                                  IM) =
                                        -(*(vectors[vec]))(0, fine_cbsite.cb, fine_cbsite.site,
                                                           spin, color, IM);
                                }

                                for (int vec = 0; vec < _n_vecs; ++vec) {
                                    (*this).index(block_idx, fine_site_idx, spin, color, 1, vec,
                                                  RE) =
                                        (*(vectors[vec]))(0, fine_cbsite.cb, fine_cbsite.site,
                                                          2 + spin, color, RE);
                                    (*this).index(block_idx, fine_site_idx, spin, color, 1, vec,
                                                  IM) =
                                        -(*(vectors[vec]))(0, fine_cbsite.cb, fine_cbsite.site,
                                                           2 + spin, color, IM);
                                }
                            }
                        }
                    }
                }
            }

            // Set up the reverse map
            MasterLog(INFO, "Creating Reverse Map");

            for (int cb = 0; cb < n_checkerboard; ++cb) {
                reverse_map[cb].resize(fine_info.GetNumCBSites());
                reverse_transfer_row[cb].resize(fine_info.GetNumCBSites());
            }

#pragma omp parallel for collapse(2)
            for (int block_cb = 0; block_cb < n_checkerboard; ++block_cb) {
                for (int block_cbsite = 0; block_cbsite < num_coarse_cbsites; ++block_cbsite) {

                    // Identify the current block.
                    int block_idx = block_cbsite + block_cb * num_coarse_cbsites;
                    const Block &block = blocklist[block_idx];

                    // Get the list of fine sites in the blocks
                    auto block_sitelist = block.getCBSiteList();
                    auto num_sites_in_block = block_sitelist.size();

                    // Loop through the fine sites in the block
                    for (IndexType fine_site_idx = 0;
                         fine_site_idx < static_cast<IndexType>(num_sites_in_block);
                         fine_site_idx++) {
                        const CBSite &cbsite = block_sitelist[fine_site_idx];
                        reverse_map[cbsite.cb][cbsite.site].cb = block_cb;
                        reverse_map[cbsite.cb][cbsite.site].site = block_cbsite;
                        reverse_transfer_row[cbsite.cb][cbsite.site] = fine_site_idx;
                    }
                }
            }
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int n_threads = omp_get_num_threads();
                if (tid == 0) { _n_threads = n_threads; }
#pragma omp barrier
            }
        }

        inline float &index(int block, int blocksite, int spin, int color, int chiral, int vec,
                            int REIM) {
            return _data[REIM +
                         n_complex *
                             (vec + _n_vecs * (chiral +
                                               2 * (color +
                                                    3 * (spin + 2 * (blocksite +
                                                                     _sites_per_block * block)))))];
        }

        inline const float &index(int block, int blocksite, int spin, int color, int chiral,
                                  int vec, int REIM) const {
            return _data[REIM +
                         n_complex *
                             (vec + _n_vecs * (chiral +
                                               2 * (color +
                                                    3 * (spin + 2 * (blocksite +
                                                                     _sites_per_block * block)))))];
        }

        inline const float *indexPtr(int block, int blocksite, int spin, int color) const {
            return &(_data[2 * n_complex * _n_vecs *
                           (color + 3 * (spin + 2 * (blocksite + _sites_per_block * block)))]);
        }

        template <int num_coarse_color> void R_op(const QSpinor &fine_in, CoarseSpinor &out) const {
            assert(num_coarse_color == out.GetNumColor());
            assert(fine_in.GetNCol() == out.GetNCol());
            IndexType ncol = fine_in.GetNCol();

            const int num_coarse_cbsites = out.GetInfo().GetNumCBSites();

            const int num_coarse_colorspin = 2 * num_coarse_color;

            // Sanity check. The number of sites in the coarse spinor
            // Has to equal the number of blocks
            //  assert( n_checkerboard*num_coarse_cbsites == static_cast<const int>(blocklist.size()) );

            // The number of vectors has to eaqual the number of coarse colors
            assert(_n_vecs == num_coarse_color);
            assert(num_coarse_cbsites == _n_blocks / 2);

            // This will be a loop over blocks
#pragma omp parallel for collapse(3)
            for (int block_cb = 0; block_cb < n_checkerboard; ++block_cb) {
                for (int block_cbsite = 0; block_cbsite < num_coarse_cbsites; ++block_cbsite) {
                    for (int col = 0; col < ncol; ++col) {

                        // Identify the current block.
                        int block_idx = block_cbsite + block_cb * num_coarse_cbsites;
                        const Block &block = _blocklist[block_idx];

                        // Get the list of fine sites in the blocks
                        auto block_sitelist = block.getCBSiteList();
                        auto num_sites_in_block = block_sitelist.size();

                        // The coarse site spinor is where we will write the result
                        std::complex<float> *coarse_site_spinor =
                            reinterpret_cast<std::complex<float> *>(
                                out.GetSiteDataPtr(col, block_cb, block_cbsite));

                        // The accumulation goes here
                        std::complex<float> site_accum[2 * MAX_VECS];

                        // Zero the accumulated coarse site
                        for (int i = 0; i < 2 * num_coarse_color; ++i) { site_accum[i] = 0; }

                        // Loop through the fine sites in the block
                        for (IndexType fine_site_idx = 0;
                             fine_site_idx < static_cast<IndexType>(num_sites_in_block);
                             fine_site_idx++) {

                            // Find the fine site
                            const CBSite &fine_cbsite = block_sitelist[fine_site_idx];

                            // for each site we will loop over Ns/2 * Ncolor
                            for (int spin = 0; spin < 2; ++spin) {
                                for (int color = 0; color < 3; ++color) {

                                    std::complex<float> psi_upper(
                                        fine_in(col, fine_cbsite.cb, fine_cbsite.site, spin, color,
                                                RE),
                                        fine_in(col, fine_cbsite.cb, fine_cbsite.site, spin, color,
                                                IM));

                                    std::complex<float> psi_lower(
                                        fine_in(col, fine_cbsite.cb, fine_cbsite.site, spin + 2,
                                                color, RE),
                                        fine_in(col, fine_cbsite.cb, fine_cbsite.site, spin + 2,
                                                color, IM));

                                    const std::complex<float> *v =
                                        reinterpret_cast<const std::complex<float> *>(
                                            (*this).indexPtr(block_idx, fine_site_idx, spin,
                                                             color));

                                    for (int colorspin = 0; colorspin < num_coarse_color;
                                         colorspin++) {
                                        site_accum[colorspin] += v[colorspin] * psi_upper;
                                    }

                                    const int offset = num_coarse_color;
                                    for (int colorspin = 0; colorspin < num_coarse_color;
                                         colorspin++) {
                                        site_accum[colorspin + offset] +=
                                            v[colorspin + offset] * psi_lower;
                                    }

                                } // color
                            }     // spin

                        } // fine sites in block

                        for (int colorspin = 0; colorspin < 2 * num_coarse_color; ++colorspin) {
                            coarse_site_spinor[colorspin] = site_accum[colorspin];
                        }
                    } //col
                }     // block CBSITE
            }         // block CB
        }

        template <int num_coarse_color>
        void R_op(const QSpinor &fine_in, int source_cb, CoarseSpinor &out) const {
            assert(num_coarse_color == out.GetNumColor());
            assert(fine_in.GetNCol() == out.GetNCol());
            IndexType ncol = fine_in.GetNCol();

            const int num_coarse_cbsites = out.GetInfo().GetNumCBSites();

            const int num_coarse_colorspin = 2 * num_coarse_color;

            // Sanity check. The number of sites in the coarse spinor
            // Has to equal the number of blocks
            //  assert( n_checkerboard*num_coarse_cbsites == static_cast<const int>(blocklist.size()) );

            // The number of vectors has to eaqual the number of coarse colors
            assert(_n_vecs == num_coarse_color);
            assert(num_coarse_cbsites == _n_blocks / 2);

            // This will be a loop over blocks
#pragma omp parallel for collapse(3)
            for (int block_cb = 0; block_cb < n_checkerboard; ++block_cb) {
                for (int block_cbsite = 0; block_cbsite < num_coarse_cbsites; ++block_cbsite) {
                    for (int col = 0; col < ncol; ++col) {

                        // Identify the current block.
                        int block_idx = block_cbsite + block_cb * num_coarse_cbsites;
                        const Block &block = _blocklist[block_idx];

                        // Get the list of fine sites in the blocks
                        auto block_sitelist = block.getCBSiteList();
                        auto num_sites_in_block = block_sitelist.size();

                        // The coarse site spinor is where we will write the result
                        std::complex<float> *coarse_site_spinor =
                            reinterpret_cast<std::complex<float> *>(
                                out.GetSiteDataPtr(col, block_cb, block_cbsite));

                        // The accumulation goes here
                        std::complex<float> site_accum[2 * MAX_VECS] __attribute__((aligned(64)));

                        // Zero the accumulated coarse site
                        for (int i = 0; i < 2 * num_coarse_color; ++i) { site_accum[i] = 0; }

                        // Loop through the fine sites in the block
                        for (IndexType fine_site_idx = 0;
                             fine_site_idx < static_cast<IndexType>(num_sites_in_block);
                             fine_site_idx++) {

                            // Find the fine site
                            const CBSite &fine_cbsite = block_sitelist[fine_site_idx];

                            if (fine_cbsite.cb == source_cb) {

                                // for each site we will loop over Ns/2 * Ncolor
                                for (int spin = 0; spin < 2; ++spin) {
                                    for (int color = 0; color < 3; ++color) {

                                        std::complex<float> psi_upper(
                                            fine_in(col, source_cb, fine_cbsite.site, spin, color,
                                                    RE),
                                            fine_in(col, source_cb, fine_cbsite.site, spin, color,
                                                    IM));

                                        std::complex<float> psi_lower(
                                            fine_in(col, source_cb, fine_cbsite.site, spin + 2,
                                                    color, RE),
                                            fine_in(col, source_cb, fine_cbsite.site, spin + 2,
                                                    color, IM));

                                        const std::complex<float> *v =
                                            reinterpret_cast<const std::complex<float> *>(
                                                (*this).indexPtr(block_idx, fine_site_idx, spin,
                                                                 color));

                                        for (int colorspin = 0; colorspin < num_coarse_color;
                                             colorspin++) {
                                            site_accum[colorspin] += v[colorspin] * psi_upper;
                                        }

                                        const int offset = num_coarse_color;
                                        for (int colorspin = 0; colorspin < num_coarse_color;
                                             colorspin++) {
                                            site_accum[colorspin + offset] +=
                                                v[colorspin + offset] * psi_lower;
                                        }

                                    } // color
                                }     // spin
                            }         // source cb
                        }             // fine sites in block

                        for (int colorspin = 0; colorspin < 2 * num_coarse_color; ++colorspin) {
                            coarse_site_spinor[colorspin] = site_accum[colorspin];
                        }
                    } //col
                }     // block CBSITE
            }         // block CB
        }

        void R(const QSpinor &fine_in, CoarseSpinor &out) const {
            int num_color = out.GetNumColor();
            if (num_color == 6) {
                R_op<6>(fine_in, out);
            } else if (num_color == 8) {
                R_op<8>(fine_in, out);
            } else if (num_color == 12) {
		R_op<12>(fine_in, out);
            } else if (num_color == 16) {
                R_op<16>(fine_in, out);
            } else if (num_color == 24) {
                R_op<24>(fine_in, out);
            } else if (num_color == 32) {
                R_op<32>(fine_in, out);
            } else if (num_color == 40) {
                R_op<40>(fine_in, out);
            } else if (num_color == 48) {
                R_op<48>(fine_in, out);
            } else if (num_color == 56) {
                R_op<56>(fine_in, out);
            } else if (num_color == 64) {
                R_op<64>(fine_in, out);
            } else if (num_color == 96) {
                R_op<96>(fine_in, source_cb, out);
            } else {
                MasterLog(
                    ERROR,
                    "Unhandled dispatch size %d. Num coarse color must be divisible by 8 and <=64",
                    num_color);
            }
            return;
        }

        void R(const QSpinor &fine_in, int source_cb, CoarseSpinor &out) const {
            int num_color = out.GetNumColor();
            if (num_color == 6) {
                R_op<6>(fine_in, source_cb, out);
            } else if (num_color == 8) {
                R_op<8>(fine_in, source_cb, out);
            } else if (num_color == 12) {
                R_op<12>(fine_in, source_cb, out);
            } else if (num_color == 16) {
                R_op<16>(fine_in, source_cb, out);
            } else if (num_color == 24) {
                R_op<24>(fine_in, source_cb, out);
            } else if (num_color == 32) {
                R_op<32>(fine_in, source_cb, out);
            } else if (num_color == 40) {
                R_op<40>(fine_in, source_cb, out);
            } else if (num_color == 48) {
                R_op<48>(fine_in, source_cb, out);
            } else if (num_color == 56) {
                R_op<56>(fine_in, source_cb, out);
            } else if (num_color == 64) {
                R_op<64>(fine_in, source_cb, out);
            } else if (num_color == 96) {
                R_op<96>(fine_in, source_cb, out);
            } else {
                MasterLog(
                    ERROR,
                    "Unhandled dispatch size %d. Num coarse color must be divisible by 8 and <=64",
                    num_color);
            }
            return;
        }

#if 1
        template <int num_coarse_color>
        void P_op(const CoarseSpinor &coarse_in, QSpinor &fine_out) const {

            const LatticeInfo &fine_info = fine_out.GetInfo();
            const LatticeInfo &coarse_info = coarse_in.GetInfo();
            assert(coarse_in.GetNCol() == fine_out.GetNCol());
            IndexType ncol = coarse_in.GetNCol();

            int num_soa = fine_info.GetNumCBSites() / QPHIX_SOALEN;

#    pragma omp parallel for collapse(3)
            for (int cb = 0; cb < n_checkerboard; ++cb) {
                for (int oblock = 0; oblock < num_soa; ++oblock) {
                    for (int col = 0; col < ncol; ++col) {

                        auto *out_blocks = (fine_out.getCB(col, cb)).get();
                        // Current FourSpinorBlock
                        float *cur_block = static_cast<float *>(&out_blocks[oblock][0][0][0][0]);

                        // Somewhere to put our results.
                        float oblock_result[4 * 3 * 2 * QPHIX_SOALEN];

                        for (int i = 0; i < 4 * 3 * 2 * QPHIX_SOALEN; ++i) { oblock_result[i] = 0; }

                        for (int isite = 0; isite < QPHIX_SOALEN; ++isite) {

                            // First thing to do is work out my unblocked site
                            int qdp_cbsite = isite + QPHIX_SOALEN * oblock;

                            // Next thing to do is identify the block the site is in.
                            // These two to index the the input coarse vector
                            int block_cb = reverse_map[cb][qdp_cbsite].cb;
                            int block_cbsite = reverse_map[cb][qdp_cbsite].site;

                            const float *coarse_site_spinor = reinterpret_cast<const float *>(
                                coarse_in.GetSiteDataPtr(col, block_cb, block_cbsite));

                            // These two to index the V-s
                            int block_idx = block_cbsite + block_cb * coarse_info.GetNumCBSites();
                            int fine_site_idx = reverse_transfer_row[cb][qdp_cbsite];

                            for (int spin = 0; spin < 2; ++spin) {
                                for (int color = 0; color < 3; ++color) {

                                    // v is 2 x num_coarse_color complexes.
                                    const float *v = reinterpret_cast<const float *>(
                                        (*this).indexPtr(block_idx, fine_site_idx, spin, color));

                                    float reduce_upper_re = 0;
                                    float reduce_upper_im = 0;

                                    float reduce_lower_re = 0;
                                    float reduce_lower_im = 0;
                                    const int offset = 2 * num_coarse_color;

                                    float vec_re[2 * num_coarse_color] __attribute__((aligned(64)));
                                    float vec_im[2 * num_coarse_color] __attribute__((aligned(64)));

                                    for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                        vec_re[i] = v[i] * coarse_site_spinor[i];
                                    }

                                    // This is meant to be a shuffle
                                    // But we fold the minus sign in so at the end
                                    // we can do a straight sum
                                    for (int i = 0; i < num_coarse_color; ++i) {
                                        vec_im[2 * i] = v[2 * i] * coarse_site_spinor[2 * i + 1];
                                        vec_im[2 * i + 1] =
                                            -v[2 * i + 1] * coarse_site_spinor[2 * i];
                                    }

                                    // Horizontal sum for real part
                                    for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                        reduce_upper_re += vec_re[i];
                                    }

                                    // Horizontal sum for imag part
                                    for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                        reduce_upper_im += vec_im[i];
                                    }

                                    //   Second Chirality
                                    for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                        vec_re[i] = v[i + offset] * coarse_site_spinor[i + offset];
                                    }

                                    // This is the shuffle for the imaginary part of the sum
                                    // we fold i the -ve sign
                                    for (int i = 0; i < num_coarse_color; ++i) {
                                        vec_im[2 * i] = v[2 * i + offset] *
                                                        coarse_site_spinor[2 * i + 1 + offset];
                                        vec_im[2 * i + 1] = -v[2 * i + 1 + offset] *
                                                            coarse_site_spinor[2 * i + offset];
                                    }

                                    // Horizontal sum for the real part
                                    for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                        reduce_lower_re += vec_re[i];
                                    }

                                    // Horizontal sum for the imag part
                                    for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                        reduce_lower_im += vec_im[i];
                                    }

                                    oblock_result[isite +
                                                  QPHIX_SOALEN *
                                                      (RE + n_complex * (spin + 4 * color))] =
                                        reduce_upper_re;
                                    oblock_result[isite +
                                                  QPHIX_SOALEN *
                                                      (IM + n_complex * (spin + 4 * color))] =
                                        reduce_upper_im;

                                    oblock_result[isite +
                                                  QPHIX_SOALEN *
                                                      (RE + n_complex * ((2 + spin) + 4 * color))] =
                                        reduce_lower_re;
                                    oblock_result[isite +
                                                  QPHIX_SOALEN *
                                                      (IM + n_complex * ((2 + spin) + 4 * color))] =
                                        reduce_lower_im;
                                }
                            }
                        } //isite

                        for (int i = 0; i < 4 * 3 * 2 * QPHIX_SOALEN; ++i) {
                            cur_block[i] = oblock_result[i];
                        }
                    } // col
                }     // oblock
            }         // cb
        }             // function

        template <int num_coarse_color>
        void P_op(const CoarseSpinor &coarse_in, int target_cb, QSpinor &fine_out) const {

            const LatticeInfo &fine_info = fine_out.GetInfo();
            const LatticeInfo &coarse_info = coarse_in.GetInfo();
            assert(coarse_in.GetNCol() == fine_out.GetNCol());
            IndexType ncol = coarse_in.GetNCol();

            int num_soa = fine_info.GetNumCBSites() / QPHIX_SOALEN;

            int cb = target_cb;
#    pragma omp parallel for

            for (int oblock = 0; oblock < num_soa; ++oblock) {
                for (int col = 0; col < ncol; ++col) {

                    auto *out_blocks = (fine_out.getCB(col, cb)).get();
                    // Current FourSpinorBlock
                    float *cur_block = static_cast<float *>(&out_blocks[oblock][0][0][0][0]);

                    // Somewhere to put our results.
                    float oblock_result[4 * 3 * 2 * QPHIX_SOALEN];

                    for (int i = 0; i < 4 * 3 * 2 * QPHIX_SOALEN; ++i) { oblock_result[i] = 0; }

                    for (int isite = 0; isite < QPHIX_SOALEN; ++isite) {

                        // First thing to do is work out my unblocked site
                        int qdp_cbsite = isite + QPHIX_SOALEN * oblock;

                        // Next thing to do is identify the block the site is in.
                        // These two to index the the input coarse vector
                        int block_cb = reverse_map[cb][qdp_cbsite].cb;
                        int block_cbsite = reverse_map[cb][qdp_cbsite].site;

                        const float *coarse_site_spinor = reinterpret_cast<const float *>(
                            coarse_in.GetSiteDataPtr(col, block_cb, block_cbsite));

                        // These two to index the V-s
                        int block_idx = block_cbsite + block_cb * coarse_info.GetNumCBSites();
                        int fine_site_idx = reverse_transfer_row[cb][qdp_cbsite];

                        for (int spin = 0; spin < 2; ++spin) {
                            for (int color = 0; color < 3; ++color) {

                                // v is 2 x num_coarse_color complexes.
                                const float *v = reinterpret_cast<const float *>(
                                    (*this).indexPtr(block_idx, fine_site_idx, spin, color));

                                float reduce_upper_re = 0;
                                float reduce_upper_im = 0;

                                float reduce_lower_re = 0;
                                float reduce_lower_im = 0;
                                const int offset = 2 * num_coarse_color;

                                float vec_re[2 * num_coarse_color] __attribute__((aligned(64)));
                                float vec_im[2 * num_coarse_color] __attribute__((aligned(64)));

                                for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                    vec_re[i] = v[i] * coarse_site_spinor[i];
                                }

                                // This is meant to be a shuffle
                                // But we fold the minus sign in so at the end
                                // we can do a straight sum
                                for (int i = 0; i < num_coarse_color; ++i) {
                                    vec_im[2 * i] = v[2 * i] * coarse_site_spinor[2 * i + 1];
                                    vec_im[2 * i + 1] = -v[2 * i + 1] * coarse_site_spinor[2 * i];
                                }

                                // Horizontal sum for real part
                                for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                    reduce_upper_re += vec_re[i];
                                }

                                // Horizontal sum for imag part
                                for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                    reduce_upper_im += vec_im[i];
                                }

                                //   Second Chirality
                                for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                    vec_re[i] = v[i + offset] * coarse_site_spinor[i + offset];
                                }

                                // This is the shuffle for the imaginary part of the sum
                                // we fold i the -ve sign
                                for (int i = 0; i < num_coarse_color; ++i) {
                                    vec_im[2 * i] =
                                        v[2 * i + offset] * coarse_site_spinor[2 * i + 1 + offset];
                                    vec_im[2 * i + 1] =
                                        -v[2 * i + 1 + offset] * coarse_site_spinor[2 * i + offset];
                                }

                                // Horizontal sum for the real part
                                for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                    reduce_lower_re += vec_re[i];
                                }

                                // Horizontal sum for the imag part
                                for (int i = 0; i < 2 * num_coarse_color; ++i) {
                                    reduce_lower_im += vec_im[i];
                                }

                                oblock_result[isite + QPHIX_SOALEN *
                                                          (RE + n_complex * (spin + 4 * color))] =
                                    reduce_upper_re;
                                oblock_result[isite + QPHIX_SOALEN *
                                                          (IM + n_complex * (spin + 4 * color))] =
                                    reduce_upper_im;

                                oblock_result[isite +
                                              QPHIX_SOALEN *
                                                  (RE + n_complex * ((2 + spin) + 4 * color))] =
                                    reduce_lower_re;
                                oblock_result[isite +
                                              QPHIX_SOALEN *
                                                  (IM + n_complex * ((2 + spin) + 4 * color))] =
                                    reduce_lower_im;
                            }
                        }
                    } //isite

                    for (int i = 0; i < 4 * 3 * 2 * QPHIX_SOALEN; ++i) {
                        cur_block[i] = oblock_result[i];
                    }
                } //col
            }     // oblock

        } // function
#endif

        void P(const CoarseSpinor &coarse_in, QSpinor &fine_out) const {
            int num_color = coarse_in.GetNumColor();

            if (num_color == 6) {
                P_op<6>(coarse_in, fine_out);
            } else if (num_color == 8) {
                P_op<8>(coarse_in, fine_out);
            } else if (num_color == 12) {
                P_op<12>(coarse_in, fine_out);
            } else if (num_color == 16) {
                P_op<16>(coarse_in, fine_out);
            } else if (num_color == 24) {
                P_op<24>(coarse_in, fine_out);
            } else if (num_color == 32) {
                P_op<32>(coarse_in, fine_out);
            } else if (num_color == 40) {
                P_op<40>(coarse_in, fine_out);
            } else if (num_color == 48) {
                P_op<48>(coarse_in, fine_out);
            } else if (num_color == 56) {
                P_op<56>(coarse_in, fine_out);
            } else if (num_color == 64) {
                P_op<64>(coarse_in, fine_out);
            } else if (num_color == 96) {
                P_op<96>(coarse_in, target_cb, fine_out);
            } else {
                MasterLog(ERROR,
                          "Unhandled dispatch size %d. Num vecs must be divisible by 8 and <= 64",
                          num_color);
            }
            return;
        }

        void P(const CoarseSpinor &coarse_in, int target_cb, QSpinor &fine_out) const {
            int num_color = coarse_in.GetNumColor();

            if (num_color == 6) {
                P_op<6>(coarse_in, target_cb, fine_out);
            } else if (num_color == 8) {
                P_op<8>(coarse_in, target_cb, fine_out);
            } else if (num_color == 12) {
                P_op<12>(coarse_in, fine_out);
            } else if (num_color == 16) {
                P_op<16>(coarse_in, target_cb, fine_out);
            } else if (num_color == 24) {
                P_op<24>(coarse_in, target_cb, fine_out);
            } else if (num_color == 32) {
                P_op<32>(coarse_in, target_cb, fine_out);
            } else if (num_color == 40) {
                P_op<40>(coarse_in, target_cb, fine_out);
            } else if (num_color == 48) {
                P_op<48>(coarse_in, target_cb, fine_out);
            } else if (num_color == 56) {
                P_op<56>(coarse_in, target_cb, fine_out);
            } else if (num_color == 64) {
                P_op<64>(coarse_in, target_cb, fine_out);
            } else if (num_color == 96) {
                P_op<96>(coarse_in, target_cb, fine_out);
            } else {
                MasterLog(ERROR,
                          "Unhandled dispatch size %d. Num vecs must be divisible by 8 and <= 64",
                          num_color);
            }
            return;
        }

        ~QPhiXTransfer() { MemoryFree(_data); }

    private:
        int _n_blocks;
        int _sites_per_block;
        int _n_vecs;
        const std::vector<Block> &_blocklist;
        std::vector<CBSite> reverse_map[2];
        std::vector<int> reverse_transfer_row[2];
        float *_data;
        const std::vector<std::shared_ptr<QSpinor>> &_vecs;
        int _n_threads;
        int _r_threads_per_block;
    };

} // namespace

#endif /* INCLUDE_LATTICE_QPHIX_QPHIX_TRANSFER_H_ */
