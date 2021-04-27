/*
 * qphix_ali.h
 *
 *  Created on: July 11, 2020
 *      Author: Eloy Romero <eloy@cs.wm.edu>
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_

#include "lattice/coarse/coarse_deflation.h" // computeDeflation
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_transfer.h"                // CoarseTransfer
#include "lattice/coarse/invfgmres_coarse.h"               // UnprecFGMRESSolverCoarseWrapper
#include "lattice/coloring.h"                              // Coloring
#include "lattice/eigs_common.h"                           // EigsParams
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"            // SetupParams
#include "lattice/qphix/mg_level_qphix.h"                  // QPhiXMultigridLevels
#include "lattice/qphix/qphix_eo_clover_linear_operator.h" // QPhiXWilsonCloverEOLinearOperatorF
#include "lattice/qphix/qphix_mgdeflation.h"               // MGDeflation
#include "lattice/qphix/qphix_transfer.h"                  // QPhiXTransfer
#include "lattice/qphix/qphix_types.h"                     // QPhiXSpinorF
#include "lattice/qphix/vcycle_recursive_qphix.h"          // VCycleRecursiveQPhiXEO2
#include <MG_config.h>
#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>

#ifdef MG_QMP_COMMS
#    include <qmp.h>
#endif

namespace MG {

    namespace GlobalComm {

#ifdef MG_QMP_COMMS
        inline void GlobalSum(double &array) { QMP_sum_double_array(&array, 1); }
#else
        inline void GlobalSum(double &array) {}
#endif
    } // namespace GlobalComm

    template <typename T> inline double sum(const std::vector<T> &v) {
        return std::accumulate(v.begin(), v.end(), 0.0);
    }

    template <typename T> class vector3d : public std::vector<T> {
    public:
        vector3d(unsigned int m, unsigned int n, unsigned int o)
            : std::vector<T>(m * n * o), _m(m), _n(n), _o(o) {}
        const T &operator()(unsigned int i, unsigned int j, unsigned int k) const {
            return std::vector<T>::operator[](i *_n *_o + j * _o + k);
        }
        T &operator()(unsigned int i, unsigned int j, unsigned int k) {
            return std::vector<T>::operator[](i *_n *_o + j * _o + k);
        }

    private:
        unsigned int _m, _n, _o;
    };

    /*
     * Solve a linear system using the Approximate Lattice Inverse as a preconditioner
     *
     * If K is the ALI preconditioner and P is an approximate projector on the lower part of A's
     * spectrum, then the linear system A*x = b is solved as x = y + A^{-1}*P*b where K*A*y =
     * K*(I-P)*b. The preconditioner K approximates the links of A^{-1} for near neighbor sites. The
     * approach is effective if |[(I-P)*A^{-1}]_ij| decays quickly as i and j are further apart
     * sites.
     *
     * The projector is built using multigrid deflation (see MGDeflation) and K is reconstructed
     * with probing based on coloring the graph lattice.
     */

    class ALIPrec : public ImplicitLinearSolver<QPhiXSpinor>,
                    public LinearSolver<QPhiXSpinorF>,
                    public AuxiliarySpinors<CoarseSpinor> {
        using AuxQ = AuxiliarySpinors<QPhiXSpinor>;
        using AuxQF = AuxiliarySpinors<QPhiXSpinorF>;
        using AuxC = AuxiliarySpinors<CoarseSpinor>;

    public:
        /*
         * Constructor
         *
         * \param info: lattice info
         * \param M_fine: linear system operator (A)
         * \param defl_p: Multigrid parameters used to build the multgrid deflation
         * \param defl_solver_params: linear system parameters to build the multigrid deflation
         * \param defl_eigs_params: eigensolver parameters to build the multigrid deflation
         * \param prec_p: Multigrid parameters used to build the preconditioner
         * \param K_distance: maximum distance of the approximated links
         * \param probing_distance: maximum distance for probing
         *
         * The parameters defl_p, defl_solver_params and defl_eigs_params are passed to MGDeflation
         * to build the projector P. The interesting values of (I-P)*A^{-1} are reconstructed with a
         * probing scheme that remove contributions from up to 'probing_distance' sites.
         */

        ALIPrec(const std::shared_ptr<LatticeInfo> info,
                const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> M_fine,
                SetupParams defl_p, LinearSolverParamsBase defl_solver_params,
                EigsParams defl_eigs_params, SetupParams prec_p,
                std::vector<MG::VCycleParams> prec_vcycle_params,
                LinearSolverParamsBase prec_solver_params, unsigned int K_distance,
                unsigned int probing_distance, const CBSubset subset, unsigned int mode = 2)
            : ImplicitLinearSolver<QPhiXSpinor>(*info, subset, prec_solver_params),
              LinearSolver<QPhiXSpinorF>(*M_fine, prec_solver_params),
              _info(info),
              _M_fine(M_fine),
              _K_distance(K_distance),
              _op(*info),
              _subset(subset),
              _mode(mode) {

            MasterLog(INFO, "ALI Solver constructor: mode= %d BEGIN", _mode);

            // Create projector
            if (_mode < 2) {
                _mg_deflation = std::make_shared<MGDeflation>(_info, _M_fine, defl_p,
                                                              defl_solver_params, defl_eigs_params);
            }

            // Create Multigrid preconditioner
            _mg_levels = std::make_shared<QPhiXMultigridLevelsEO>();
            SetupQPhiXMGLevels(prec_p, *_mg_levels, _M_fine);
            _vcycle = std::make_shared<VCycleRecursiveQPhiXEO2>(prec_vcycle_params, *_mg_levels);

            // Create a Multigrid without smoothers
            if (_mode >= 2) {
                std::vector<MG::VCycleParams> defl_vcycle_params(prec_vcycle_params);
                defl_vcycle_params[0].cycle_params.MaxIter = 1;
                defl_vcycle_params[0].pre_smoother_params.MaxIter = 0;
                defl_vcycle_params[0].post_smoother_params.MaxIter = 0;
                _vcycle_defl =
                    std::make_shared<VCycleRecursiveQPhiXEO2>(defl_vcycle_params, *_mg_levels);
            }

            // Build K
            build_K(prec_p, prec_vcycle_params, prec_solver_params, K_distance, probing_distance);
            MasterLog(INFO, "ALI Solver constructor: END", _mode);

            // Hack vcycle
            set_smoother();

            AuxQ::clear();
            AuxQF::clear();
            AuxC::clear();
        }

        /*
         * Apply the preconditioner onto 'in'.
         *
         * \param out: returned vectors
         * \param in: input vectors
         *
         * It applies the deflation on the input vectors and return the results on 'out'.
         *
         *    out = [M^{-1}*Q + K*(I-Q)] * in,
         *
         * where Q = M_oo^{-1}*P*M_oo, P is a projector on M, and K approximates M^{-1}_oo*M_oo.
         */

        std::vector<LinearSolverResults>
        operator()(QPhiXSpinor &out, const QPhiXSpinor &in, ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const override {
            // test_defl(in);
            // applyK(out, in);
            (*_vcycle)(out, in, resid_type, guess);
            IndexType ncol = out.GetNCol();
            return std::vector<LinearSolverResults>(ncol, LinearSolverResults());
        }

        /*
         * Apply the preconditioner onto 'in'.
         *
         * \param out: returned vectors
         * \param in: input vectors
         *
         * It applies the deflation on the input vectors and return the results on 'out'.
         *
         *    out = [M^{-1}*Q + K*(I-Q)] * in,
         *
         * where Q = M_oo^{-1}*P*M_oo, P is a projector on M, and K approximates M^{-1}_oo*M_oo.
         */

        std::vector<LinearSolverResults>
        operator()(QPhiXSpinorF &out, const QPhiXSpinorF &in, ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const override {
            // test_defl(in);
            // applyK(out, in);
            (*_vcycle)(out, in, resid_type, guess);
            IndexType ncol = out.GetNCol();
            return std::vector<LinearSolverResults>(ncol, LinearSolverResults());
        }

        template <typename Spinor>
        void apply_precon(Spinor &out, const Spinor &in, int recursive = 0) const {
            // // TEMP!!!
            // double norm2_cb0 = sqrt(Norm2Vec(in, SUBSET_EVEN)[0]);
            // double norm2_cb1 = sqrt(Norm2Vec(in, SUBSET_ODD)[0]);
            // MasterLog(INFO,"MG Level 0: ALI Solver operator(): || v_e ||=%16.8e || v_o
            // ||=%16.8e", norm2_cb0, norm2_cb1);

            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            if (_K_distance == 0 && (!_mg_deflation || _mg_deflation->GetRank() <= 0)) {
                CopyVec(out, in, _subset);
                return;
            }
            applyK(out, in);

            // std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(*_info, ncol);
            // ZeroVec(*in_f);
            // ConvertSpinor(in, *in_f, _subset);

            // // I_Q_in = (I-Q)*in
            // std::shared_ptr<QPhiXSpinorF> I_Q_in_f = AuxQF::tmp(*_info, ncol);
            // std::shared_ptr<QPhiXSpinorF> VV_in_f = AuxQF::tmp(*_info, ncol);
            // apply_complQ(*I_Q_in_f, *in_f, VV_in_f.get());
            // in_f.reset();

            // // M_oe*(M_ee^{-1}*M_eo*VV_in_o + VV_in_e)
            // if (_mode == 0) {
            //     std::shared_ptr<QPhiXSpinorF> strange_VV_in_f = AuxQF::tmp(*_info, ncol);
            //     _M_fine->strangeOp(*strange_VV_in_f, *VV_in_f);
            //     if (recursive > 0) {
            //         std::shared_ptr<QPhiXSpinorF> Minv_oo_strange_VV_in_f =
            //             AuxQF::tmp(*_info, ncol);
            //         apply_precon(*Minv_oo_strange_VV_in_f, *strange_VV_in_f, recursive - 1);
            //         YpeqXVec(*Minv_oo_strange_VV_in_f, *VV_in_f, _subset);
            //     } else {
            //         YpeqXVec(*strange_VV_in_f, *I_Q_in_f, _subset);
            //     }
            // }

            // // out_f = K * (I-Q)*in
            // std::shared_ptr<QPhiXSpinorF> out_f = AuxQF::tmp(*_info, ncol);
            // applyK(*out_f, *I_Q_in_f);
            // I_Q_in_f.reset();

            // // out += VV_in
            // YpeqXVec(*VV_in_f, *out_f, _subset);
            // ZeroVec(out);
            // ConvertSpinor(*out_f, out, _subset);
        }

        /**
         * Return M^{-1}_oo * Q * in
         *
         * \param eo_solver: invertor on _M_fine
         * \param out: (out) output vector
         * \param in: input vector
         */

        template <typename Spinor> void apply_invM_Q(Spinor &out, const Spinor &in) const {
            assert(in.GetNCol() == out.GetNCol());
            int ncol = in.GetNCol();

            if (_mode < 2) {
                Spinor in0(*_info, ncol);
                ZeroVec(in0);
                CopyVec(in0, in, _subset);
                _mg_deflation->VV(out, in0);
            } else {
                (*_vcycle_defl)(out, in);
            }

            // out_o + K * M_oe*(M_ee^{-1}*M_eo*out_o + out_e)
            if (_mode == 0 && _K_vals) {
                std::shared_ptr<QPhiXSpinorF> VV_in_f = AuxQF::tmp(*_info, ncol);
                ConvertSpinor(out, *VV_in_f);
                std::shared_ptr<QPhiXSpinorF> strange_VV_in_f = AuxQF::tmp(*_info, ncol);
                _M_fine->strangeOp(*strange_VV_in_f, *VV_in_f);
                std::shared_ptr<QPhiXSpinorF> K_f = AuxQF::tmp(*_info, ncol);
                applyK(*K_f, *strange_VV_in_f);
                YpeqXVec(*K_f, *VV_in_f, _subset);
                ConvertSpinor(*VV_in_f, out, _subset);
            }

            ZeroVec(out, _subset.complementary());
        }

        void test_defl(const QPhiXSpinor &in) const {
            IndexType ncol = in.GetNCol();

            // I_Q_in = (I-Q)*in = in - L^{-1} * P * in
            std::shared_ptr<QPhiXSpinor> I_Q_in = AuxQ::tmp(*_info, ncol);
            std::shared_ptr<QPhiXSpinor> invM_Q_in = AuxQ::tmp(*_info, ncol);
            apply_complQ(*I_Q_in, in);
            apply_invM_Q(*invM_Q_in, in);

            std::shared_ptr<QPhiXSpinorF> invM_Q_in_f = AuxQF::tmp(*_info, ncol);
            ZeroVec(*invM_Q_in_f);
            ConvertSpinor(*invM_Q_in, *invM_Q_in_f, _subset);
            std::shared_ptr<QPhiXSpinorF> MinvM_Q_in_f = AuxQF::tmp(*_info, ncol);
            (*_M_fine)(*MinvM_Q_in_f, *invM_Q_in_f);
            std::shared_ptr<QPhiXSpinor> MinvM_Q_in = AuxQ::tmp(*_info, ncol);
            ConvertSpinor(*MinvM_Q_in_f, *MinvM_Q_in, SUBSET_ODD);

            YpeqXVec(*MinvM_Q_in, *I_Q_in, SUBSET_ODD);
            YmeqXVec(in, *I_Q_in, SUBSET_ODD);
            std::vector<double> n_in = Norm2Vec(in);
            std::vector<double> n_diff = Norm2Vec(*I_Q_in);
            for (int col = 0; col < ncol; col++)
                MasterLog(INFO, "MG Level 0: ALI Solver test_defl: error= %16.8e",
                          sqrt(n_diff[col] / n_in[col]));
        }

        const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> GetM() const {
            return _M_fine;
        }
        const std::shared_ptr<MGDeflation> GetMGDeflation() const { return _mg_deflation; }

        /**
         * Return the lattice information
         */

        const LatticeInfo &GetInfo() const { return *_info; }

        /**
         * Return the support of the operator (SUBSET_EVEN, SUBSET_ODD, SUBSET_ALL)
         */

        const CBSubset &GetSubset() const { return _subset; }

    private:
        struct S : public ImplicitLinearSolver<QPhiXSpinorF> {
            S(const ALIPrec &aliprec)
                : ImplicitLinearSolver<QPhiXSpinorF>(aliprec.GetInfo(), aliprec.GetSubset()),
                  _aliprec(aliprec) {}

            std::vector<LinearSolverResults>
            operator()(QPhiXSpinorF &out, const QPhiXSpinorF &in,
                       ResiduumType resid_type = RELATIVE,
                       InitialGuess guess = InitialGuessNotGiven) const override {
                (void)resid_type;
                (void)guess;
                _aliprec.apply_precon(out, in);
                return std::vector<LinearSolverResults>(in.GetNCol(), LinearSolverResults());
            }

            const ALIPrec &_aliprec;
        };

        void set_smoother() {
            if (_K_distance <= 0) return;

            _antipostsmoother = std::make_shared<const S>(*this);
            _vcycle->SetAntePostSmoother(_antipostsmoother.get());
            //_vcycle->GetPostSmoother()->SetPrec(_antipostsmoother.get());
        }

        /**
         * Return (I-Q) * in
         *
         * \param out: (out) output vector
         * \param in: input vector
         *
         * NOTE: Assuming 'in' is properly zeroed
         */

        void apply_complQ(QPhiXSpinor &out, const QPhiXSpinor &in) const {
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(*_info, ncol);
            ZeroVec(*in_f);
            ConvertSpinor(in, *in_f, _subset);
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQF::tmp(*_info, ncol);
            apply_complQ(*out_f, *in_f);
            ZeroVec(out);
            ConvertSpinor(*out_f, out, _subset);
        }

        void apply_complQ(QPhiXSpinorF &out, const QPhiXSpinorF &in,
                          QPhiXSpinorF *VVin = nullptr) const {
            assert(out.GetNCol() == in.GetNCol());
            assert(!VVin || VVin->GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            if (_mode == 0) {
                // Q = (I-P)
                std::shared_ptr<QPhiXSpinorF> VVin0;
                if (!VVin) {
                    VVin0 = AuxQF::tmp(*_info, ncol);
                    VVin = VVin0.get();
                }
                _mg_deflation->VV(*VVin, in);
                std::shared_ptr<QPhiXSpinorF> AVVin = AuxQF::tmp(*_info, ncol);
                _M_fine->unprecOp(*AVVin, *VVin);
                VVin0.reset();
                VVin = nullptr;

                ZeroVec(out);
                CopyVec(out, in, _subset);
                YmeqXVec(*AVVin, out, _subset);
            } else if (_mode == 1) {
                // Q = (I-L^{-1}*P*L)
                std::shared_ptr<QPhiXSpinorF> VVin0;
                if (!VVin) {
                    VVin0 = AuxQF::tmp(*_info, ncol);
                    VVin = VVin0.get();
                }
                _mg_deflation->VV(*VVin, in);
                std::shared_ptr<QPhiXSpinorF> AVVin = AuxQF::tmp(*_info, ncol);
                (*_M_fine)(*AVVin, *VVin);
                VVin0.reset();
                VVin = nullptr;

                ZeroVec(out);
                CopyVec(out, in, _subset);
                YmeqXVec(*AVVin, out, _subset);
            } else if (_mode == 2) {
                std::shared_ptr<QPhiXSpinorF> VVin0;
                if (!VVin) {
                    VVin0 = AuxQF::tmp(*_info, ncol);
                    VVin = VVin0.get();
                }
                (*_vcycle_defl)(*VVin, in);
                std::shared_ptr<QPhiXSpinorF> AVVin = AuxQF::tmp(*_info, ncol);
                (*_M_fine)(*AVVin, *VVin);
                VVin0.reset();
                VVin = nullptr;

                ZeroVec(out);
                CopyVec(out, in, _subset);
                YmeqXVec(*AVVin, out, _subset);
            } else {
                assert(false);
            }
        }

        /**
         * Return M^{-1}_oo * (I-Q) * in
         *
         * \param eo_solver: invertor on _M_fine
         * \param out: (out) output vector
         * \param in: input vector
         */

        void apply_invM_after_defl(const FGMRESSolverQPhiXF &eo_solver, QPhiXSpinorF &out,
                                   const QPhiXSpinorF &in) const {
            assert(in.GetNCol() == out.GetNCol());
            int ncol = in.GetNCol();

            // I_Q_in = (I-Q)*in = in - M_oo^{-1} * P * in
            std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(*_info, ncol);
            ZeroVec(*in_f);
            ConvertSpinor(in, *in_f, _subset);
            std::shared_ptr<QPhiXSpinorF> I_Q_in = AuxQF::tmp(*_info, ncol);
            apply_complQ(*I_Q_in, *in_f);

            // out = M^{-1} * (I-Q) * in
            ZeroVec(out);
            std::vector<MG::LinearSolverResults> res = eo_solver(out, *I_Q_in, RELATIVE);
        }

        void build_K(SetupParams p, std::vector<MG::VCycleParams> vcycle_params,
                     LinearSolverParamsBase solver_params, unsigned int K_distance,
                     unsigned int probing_distance, unsigned int blocking = 32) {
            if (K_distance == 0) return;
            if (K_distance > 1) throw std::runtime_error("Not implemented 'K_distance' > 1");

            FGMRESSolverQPhiXF eo_solver(*_M_fine, solver_params, _vcycle.get());

            std::shared_ptr<Coloring> coloring =
                get_good_coloring(eo_solver, probing_distance, solver_params.RsdTarget * 2);
            unsigned int num_probing_vecs = coloring->GetNumSpinColorColors();
            for (unsigned int col = 0, nc = std::min(num_probing_vecs, blocking);
                 col < num_probing_vecs;
                 col += nc, nc = std::min(num_probing_vecs - col, blocking)) {

                // p(i) is the probing vector for the color col+i
                std::shared_ptr<QPhiXSpinorF> p = AuxQF::tmp(*_info, nc);
                coloring->GetProbingVectors(*p, col);

                // sol = inv(M_fine) * (I-P) * p
                std::shared_ptr<QPhiXSpinorF> sol = AuxQF::tmp(*_info, nc);
                apply_invM_after_defl(eo_solver, *sol, *p);
                p.reset();

                // Update K from sol
                update_K_from_probing_vecs(*coloring, col, sol);
            }

            test_K(eo_solver);
        }

        void update_K_from_probing_vecs(const Coloring &coloring, unsigned int c0,
                                        const std::shared_ptr<QPhiXSpinorF> sol) {
            if (!_K_vals) {
                _K_vals = std::make_shared<CoarseGauge>(*_info);
                ZeroGauge(*_K_vals);
            }

            IndexType num_cbsites = _info->GetNumCBSites();
            IndexType num_color = _info->GetNumColors();
            IndexType num_spin = _info->GetNumSpins();
            IndexType ncol = sol->GetNCol();
            CBSubset subset = SUBSET_ALL;

            // Loop over the sites and sum up the norm
#pragma omp parallel for collapse(3) schedule(static)
            for (int cb = subset.start; cb < subset.end; ++cb) {
                for (int cbsite = 0; cbsite < num_cbsites; ++cbsite) {
                    for (int col = 0; col < ncol; ++col) {
                        // Decompose the color into the node's color and the spin-color components
                        IndexType col_spin, col_color, node_color;
                        coloring.SpinColorColorComponents(c0 + col, node_color, col_spin,
                                                          col_color);
                        unsigned int colorj = coloring.GetColorCBIndex(cb, cbsite);

                        // Process this site if its color is the same as the color of the probing
                        // vector
                        if (colorj != node_color) continue;

                        // Get diag
                        for (int color = 0; color < num_color; ++color) {
                            for (int spin = 0; spin < num_spin; ++spin) {
                                int g5 = (spin < num_spin / 2 ? 1 : -1) *
                                         (col_spin < num_spin / 2 ? 1 : -1);
                                _K_vals->GetSiteDiagData(cb, cbsite, col_spin, col_color, spin,
                                                         color, RE) +=
                                    (*sol)(col, cb, cbsite, spin, color, 0) / 2;
                                _K_vals->GetSiteDiagData(cb, cbsite, col_spin, col_color, spin,
                                                         color, IM) +=
                                    (*sol)(col, cb, cbsite, spin, color, 1) / 2;
                                _K_vals->GetSiteDiagData(cb, cbsite, spin, color, col_spin,
                                                         col_color, RE) +=
                                    (*sol)(col, cb, cbsite, spin, color, 0) / 2 * g5;
                                _K_vals->GetSiteDiagData(cb, cbsite, spin, color, col_spin,
                                                         col_color, IM) -=
                                    (*sol)(col, cb, cbsite, spin, color, 1) / 2 * g5;
                            }
                        }
                    }
                }
            }
        }

        std::shared_ptr<Coloring> get_good_coloring(const FGMRESSolverQPhiXF &eo_solver,
                                                    unsigned int max_probing_distance, double tol) {
            // Returned coloring
            std::shared_ptr<Coloring> coloring;

            // Build probing vectors to get the exact first columns for sites [1 0 0 0]
            IndexArray site = {{1, 0, 0, 0}}; // This is an ODD site
            std::shared_ptr<QPhiXSpinorF> e = AuxQF::tmp(*_info, _info->GetNumColorSpins());
            ZeroVec(*e);
            std::vector<IndexType> the_cbsite = Coloring::GetKDistNeighbors(site, 0, *_info);
            for (unsigned int i = 0; i < the_cbsite.size(); ++i) {
                for (int color = 0; color < _info->GetNumColors(); ++color) {
                    for (int spin = 0; spin < _info->GetNumSpins(); ++spin) {
                        int sc = color * _info->GetNumSpins() + spin;
                        (*e)(sc, ODD, the_cbsite[i], spin, color, 0) = 1.0;
                    }
                }
            }
            std::vector<IndexType> cbsites_dist_k = Coloring::GetKDistNeighbors(site, _K_distance-1, *_info);

            // sol_e = inv(M_fine) * (I-P) * e
            std::shared_ptr<QPhiXSpinorF> sol_e = AuxQF::tmp(*_info, _info->GetNumColorSpins());
            apply_invM_after_defl(eo_solver, *sol_e, *e);
            ZeroVec(*sol_e, _subset.complementary());

            unsigned int probing_distance = 1;
            while (probing_distance <= max_probing_distance) {
                // Create coloring
                coloring = std::make_shared<Coloring>(_info, probing_distance, SUBSET_ODD);

                // Get the probing vectors for "site" 
                double color_node = 0;
                if (the_cbsite.size() > 0)
                    color_node = coloring->GetColorCBIndex(ODD, the_cbsite[0]);
                GlobalComm::GlobalSum(color_node);
                std::shared_ptr<QPhiXSpinorF> p = AuxQF::tmp(*_info, _info->GetNumColorSpins());
                coloring->GetProbingVectors(
                    *p, coloring->GetSpinColorColor((unsigned int)color_node, 0, 0));

                // sol_p = inv(M_fine) * (I-P) * p
                std::shared_ptr<QPhiXSpinorF> sol_p = AuxQF::tmp(*_info, _info->GetNumColorSpins());
                apply_invM_after_defl(eo_solver, *sol_p, *p);

                // Compute sol_F = \sum |sol_e[i,j]|^2 over the K nonzero pattern.
                // Compute diff_F = \sum |sol_e[i,j]-sol_p[i,j]|^2 over the K nonzero pattern.
                double sol_F = 0.0, diff_F = 0.0;
                vector3d<std::complex<float>> sol_p_00(
                    cbsites_dist_k.size(), _info->GetNumColorSpins(), _info->GetNumColorSpins());
                for (unsigned int i = 0; i < cbsites_dist_k.size(); ++i) {
                    for (int colorj = 0; colorj < _info->GetNumColors(); ++colorj) {
                        for (int spinj = 0; spinj < _info->GetNumSpins(); ++spinj) {
                            int sc_e_j = colorj * _info->GetNumSpins() + spinj;
                            int sc_p_j = coloring->GetSpinColorColor(0, spinj, colorj);
                            for (int color = 0; color < _info->GetNumColors(); ++color) {
                                for (int spin = 0; spin < _info->GetNumSpins(); ++spin) {
                                    std::complex<float> sol_e_ij(
                                        (*sol_e)(sc_e_j, ODD, i, spin, color, 0),
                                        (*sol_e)(sc_e_j, ODD, i, spin, color, 1));
                                    std::complex<float> sol_p_ij(
                                        (*sol_p)(sc_p_j, ODD, i, spin, color, 0),
                                        (*sol_p)(sc_p_j, ODD, i, spin, color, 1));
                                    sol_F += (sol_e_ij * std::conj(sol_e_ij)).real();
                                    std::complex<float> diff_ij = sol_e_ij - sol_p_ij;
                                    diff_F += (diff_ij * std::conj(diff_ij)).real();

                                    int sc_p_i = coloring->GetSpinColorColor(0, spin, color);
                                    sol_p_00(i, sc_p_j, sc_p_i) = sol_p_ij;
                                }
                            }
                        }
                    }
                }

                GlobalComm::GlobalSum(diff_F);
                GlobalComm::GlobalSum(sol_F);

                // Zero sol_p[i,j] that are not on the K nonzero pattern
                ZeroVec(*sol_p, SUBSET_ALL);
                for (unsigned int i = 0; i < cbsites_dist_k.size(); ++i) {
                    for (int colorj = 0; colorj < _info->GetNumColors(); ++colorj) {
                        for (int spinj = 0; spinj < _info->GetNumSpins(); ++spinj) {
                            int sc_p_j = coloring->GetSpinColorColor(0, spinj, colorj);
                            for (int color = 0; color < _info->GetNumColors(); ++color) {
                                for (int spin = 0; spin < _info->GetNumSpins(); ++spin) {
                                    int sc_p_i = coloring->GetSpinColorColor(0, spin, color);
                                    (*sol_p)(sc_p_j, ODD, i, spin, color, 0) =
                                        sol_p_00(i, sc_p_j, sc_p_i).real();
                                    (*sol_p)(sc_p_j, ODD, i, spin, color, 1) =
                                        sol_p_00(i, sc_p_j, sc_p_i).imag();
                                }
                            }
                        }
                    }
                }

                // Compute norm_sol_e = |sol_e|_F
                std::shared_ptr<QPhiXSpinorF> aux = AuxQF::tmp(*_info, _info->GetNumColorSpins());
                double norm_sol_e = sqrt(sum(Norm2Vec(*sol_e, SUBSET_ODD)));

                // Compute norm_diff = |sol_p-sol_e|_F
                CopyVec(*aux, *sol_e);
                double norm_diff = sqrt(sum(XmyNorm2Vec(*aux, *sol_p, SUBSET_ODD)));

                // Compute norm_F = |A*(sol_p - sol_e)|_F
                (*_M_fine)(*sol_p, *aux);
                double norm_F = sqrt(sum(Norm2Vec(*sol_p, SUBSET_ODD)));

                MasterLog(INFO,
                          "K probing error with %d distance coloring: %d colors "
                          "||M^{-1}_00-K_00||_F/||M^{-1}_00||_F= "
                          "%g ||M^{-1}_0-K_0||_F/||M^{-1}_0||_F= %g   ||M*K-I||= %g",
                          probing_distance, (int)coloring->GetNumSpinColorColors(),
                          sqrt(diff_F / sol_F), norm_diff / norm_sol_e, norm_F);

                if (diff_F <= sol_F * tol * tol) break;

                probing_distance++;
                // Coloring produces distinct coloring schemes for even distances only (excepting
                // 1-distance)
                if (probing_distance % 2 == 1) probing_distance++;
            }

            return coloring;
        }

        void test_K(const FGMRESSolverQPhiXF &eo_solver) {
            // Build probing vectors to get the exact first columns for ODD site 0
            const int nc = _info->GetNumColorSpins();
            std::shared_ptr<QPhiXSpinorF> e = AuxQF::tmp(*_info, nc);
            ZeroVec(*e);
            if (_info->GetNodeInfo().NodeID() == 0) {
                for (int color = 0; color < _info->GetNumColors(); ++color) {
                    for (int spin = 0; spin < _info->GetNumSpins(); ++spin) {
                        int sc = color * _info->GetNumSpins() + spin;
                        (*e)(sc, ODD, 0, spin, color, 0) = 1.0;
                    }
                }
            }

            // sol_e = inv(M_fine) * (I-Q) * e
            std::shared_ptr<QPhiXSpinorF> sol_e = AuxQF::tmp(*_info, nc);
            // apply_invM_after_defl(eo_solver, *sol_e, *e);
            eo_solver(*sol_e, *e);

            // sol_p \approx inv(M_fine) * (I-Q) * e
            std::shared_ptr<QPhiXSpinorF> sol_p = AuxQF::tmp(*_info, nc);
            std::shared_ptr<QPhiXSpinor> e_d = AuxQ::tmp(*_info, nc);
            std::shared_ptr<QPhiXSpinor> sol_p_d = AuxQ::tmp(*_info, nc);
            ConvertSpinor(*e, *e_d);
            (*this)(*sol_p_d, *e_d);
            ConvertSpinor(*sol_p_d, *sol_p);

            double norm_e = sqrt(sum(Norm2Vec(*sol_e, SUBSET_ODD)));
            double norm_diff = sqrt(sum(XmyNorm2Vec(*sol_e, *sol_p, SUBSET_ODD)));
            MasterLog(INFO, "K probing error : ||M^{-1}-K||_F= %g", norm_diff / norm_e);
        }

        /*
         * Apply K. out = K * in.
         *
         * \param out: returned vectors
         * \param in: input vectors
         */

        template <typename Spinor> void applyK(Spinor &out, const Spinor &in) const {
            assert(out.GetNCol() == in.GetNCol());
            int ncol = in.GetNCol();

            if (_K_distance == 0) {
                // If no K, copy 'in' into 'out'
                ZeroVec(out, _subset);

            } else if (_K_distance == 1) {
                // Apply the diagonal of K
                std::shared_ptr<CoarseSpinor> in_c = AuxC::tmp(*_info, ncol);
                std::shared_ptr<CoarseSpinor> out_c = AuxC::tmp(*_info, ncol);
                ConvertSpinor(in, *in_c, _subset);
#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    for (int cb = _subset.start; cb < _subset.end; ++cb) {
                        _op.M_diag(*out_c, *_K_vals, *in_c, cb, LINOP_OP, tid);
                    }
                }
                ConvertSpinor(*out_c, out, _subset);

            } else if (_K_distance == 2) {
                // Apply the whole operator
                std::shared_ptr<CoarseSpinor> in_c = AuxC::tmp(*_info, ncol);
                std::shared_ptr<CoarseSpinor> out_c = AuxC::tmp(*_info, ncol);
                ConvertSpinor(in, *in_c);
#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    for (int cb = _subset.start; cb < _subset.end; ++cb) {
                        _op.unprecOp(*out_c, *_K_vals, *in_c, cb, LINOP_OP, tid);
                    }
                }
                ConvertSpinor(*out_c, out);

            } else {
                assert(false);
            }
        }

        const std::shared_ptr<LatticeInfo> _info;
        const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> _M_fine;
        std::shared_ptr<MGDeflation> _mg_deflation;
        std::shared_ptr<CoarseGauge> _K_vals;
        unsigned int _K_distance;
        const CoarseDiracOp _op;
        const CBSubset _subset;
        const unsigned int _mode;
        std::shared_ptr<QPhiXMultigridLevelsEO> _mg_levels;
        std::shared_ptr<VCycleRecursiveQPhiXEO2> _vcycle;
        std::shared_ptr<VCycleRecursiveQPhiXEO2> _vcycle_defl;
        std::shared_ptr<const S> _antipostsmoother;
    };
} // namespace MG

#endif // INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_