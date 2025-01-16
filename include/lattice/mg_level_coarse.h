/*
 * mg_level_coarse.h
 *
 *  Created on: Mar 15, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_MG_LEVEL_COARSE_H_
#define INCLUDE_LATTICE_MG_LEVEL_COARSE_H_

#include "lattice/coarse/block.h"
#include "lattice/coarse/coarse_deflation.h"
#include "lattice/coarse/coarse_eo_wilson_clover_linear_operator.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_wilson_clover_linear_operator.h"
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/solver.h"
#include "utils/timer.h"
#include <cstring>
#include <lattice/coarse/invbicgstab_coarse.h>
#include <memory>
#include <iostream>
namespace MG {
    template <typename SolverT, typename LinOpT> struct MGLevelCoarseT {
        using Solver = SolverT;
        using LinOp = LinOpT;
        std::shared_ptr<const LatticeInfo> info;
        std::shared_ptr<CoarseGauge> gauge;
        std::vector<std::shared_ptr<CoarseSpinor>> null_vecs; // NULL Vectors
        std::vector<Block> blocklist;
        std::shared_ptr<const SolverT> null_solver; // Solver for NULL on this level;
        std::shared_ptr<const LinOpT> M;

        ~MGLevelCoarseT() {}
    };

    // Unpreconditioned levels
    using MGLevelCoarse = MGLevelCoarseT<BiCGStabSolverCoarse, CoarseWilsonCloverLinearOperator>;

    // Preconditioned levels
    using MGLevelCoarseEO =
        MGLevelCoarseT<UnprecBiCGStabSolverCoarseWrapper, CoarseEOWilsonCloverLinearOperator>;

    template <typename CoarseLevelT>
    void SetupCoarseToCoarseStreamingSVDT(const SetupParams &p, const typename CoarseLevelT::LinOp &M_fine,
                              int fine_level_id, CoarseLevelT &fine_level,
                              CoarseLevelT &coarse_level) {
        // Info should already be created

        // Null solver is BiCGStab. Let us make a parameter struct for it.
        LinearSolverParamsBase params = p.null_solver_params[fine_level_id];

        // Zero RHS and randomize the initial guess
        const LatticeInfo &fine_info = *(fine_level.info);
        //int num_vecs = p.n_vecs[fine_level_id] + p.n_vecs_keep[fine_level_id];
	int num_vecs = 2*p.n_vecs[fine_level_id];
	fine_level.null_vecs.resize(num_vecs);
	for (int k = 0; k < num_vecs; k++) {fine_level.null_vecs[k] = std::make_shared<CoarseSpinor>(fine_info);}

	IndexArray blocked_lattice_dims;
        IndexArray blocked_lattice_orig;
        CreateBlockList(fine_level.blocklist, blocked_lattice_dims, blocked_lattice_orig,
                        fine_level.info->GetLatticeDimensions(), p.block_sizes[fine_level_id],
                        fine_level.info->GetLatticeOrigin());
	

        num_vecs = p.n_vecs[fine_level_id];
	for (int i = 0; i < p.n_streams[fine_level_id]; i++) {
		
		std::shared_ptr<CoarseSpinor> x;
		//num_vecs = ( i == 0 ? p.n_vecs[fine_level_id] : p.n_vecs[fine_level_id] + p.n_vecs_keep[fine_level_id]);
		//num_vecs = ( i == 0 ? p.n_vecs[fine_level_id] : 2*p.n_vecs[fine_level_id]);

		x = std::make_shared<CoarseSpinor>(fine_info, num_vecs);
		Gaussian(*x);
		CoarseSpinor b(fine_info, num_vecs);

        	ZeroVec(b);

		fine_level.null_solver =
                std::make_shared<typename CoarseLevelT::Solver>(M_fine, params);
            	assert(fine_level.null_solver->GetSubset() == SUBSET_ALL);

            	// Solve the linear systems
            	std::vector<LinearSolverResults> res =
                (*(fine_level.null_solver))(*x, b, ABSOLUTE, InitialGuessGiven);
            	assert(res.size() == (unsigned int)num_vecs);

            	if (num_vecs > 0)
                	MasterLog(INFO, "Level %d: Solver Took: %d iterations", fine_level_id,
                        res[0].n_count);

		int n_test_vecs = 0;
		for (int k = ( i == 0 ? 0 : p.n_vecs[fine_level_id]); k < (i == 0 ? p.n_vecs[fine_level_id] : 2*num_vecs); ++k) {
            		CopyVec(*fine_level.null_vecs[k], 0, 1, *x, n_test_vecs, SUBSET_ALL);
			double norm2_cb0 = sqrt(Norm2Vec(*(fine_level.null_vecs[k]), SUBSET_EVEN)[0]);
                        double norm2_cb1 = sqrt(Norm2Vec(*(fine_level.null_vecs[k]), SUBSET_ODD)[0]);
			MasterLog(
                        INFO,
                        "Coarse Level: BiCGStab Solver Took: %d iterations: || v_e ||=%16.8e || v_o "
                        "||=%16.8e",
                        res[0].n_count, norm2_cb0, norm2_cb1);
			n_test_vecs++;
        	}
		
		if ( i == 0 ) {
		std::vector<std::shared_ptr<CoarseSpinor>> tmp(fine_level.null_vecs.begin(), fine_level.null_vecs.begin() + p.n_vecs[fine_level_id]);
		streamingChiralSVD(tmp, fine_level.blocklist, false);
		} else {
		streamingChiralSVD(fine_level.null_vecs, fine_level.blocklist, (i == p.n_streams[fine_level_id]-1 ? true : false));
		}

	} //n_streams

	M_fine.clear();

	fine_level.null_vecs.resize(p.n_vecs_keep[fine_level_id]);

        num_vecs = fine_level.null_vecs.size();

        // Create the blocked Clover and Gauge Fields
        // This service needs the blocks, the vectors and is a convenience
        // Function of the M
        coarse_level.info =
            std::make_shared<LatticeInfo>(blocked_lattice_orig, blocked_lattice_dims, 2, num_vecs,
                                          fine_info.GetNodeInfo(), fine_level_id + 1);

        coarse_level.gauge = std::make_shared<CoarseGauge>(*(coarse_level.info));

        M_fine.generateCoarse(fine_level.blocklist, fine_level.null_vecs, *(coarse_level.gauge));

        coarse_level.M = std::make_shared<const typename CoarseLevelT::LinOp>(coarse_level.gauge);

        const char *coarse_prefix_name = std::getenv("MG_COARSE_FILENAME");
        if (coarse_prefix_name != nullptr && std::strlen(coarse_prefix_name) > 0) {
            std::string filename = std::string(coarse_prefix_name) + "_level" +
                                   std::to_string(fine_level_id + 1) + ".bin";
            MasterLog(INFO, "CoarseEOCloverLinearOperator: Writing coarse operator in %s",
                      filename.c_str());
            CoarseDiracOp::write(*(coarse_level.gauge), filename);
        }


    } //func

    template <typename CoarseLevelT>
    void SetupCoarseToCoarseT(const SetupParams &p, const typename CoarseLevelT::LinOp &M_fine,
                              int fine_level_id, CoarseLevelT &fine_level,
                              CoarseLevelT &coarse_level) {
        // Info should already be created

        // Null solver is BiCGStab. Let us make a parameter struct for it.
        LinearSolverParamsBase params = p.null_solver_params[fine_level_id];

        // Zero RHS and randomize the initial guess
        const LatticeInfo &fine_info = *(fine_level.info);
        int num_vecs = p.n_vecs[fine_level_id];

        std::shared_ptr<CoarseSpinor> x;
        if (params.RsdTarget > 0) {
            
	    x = std::make_shared<CoarseSpinor>(fine_info, num_vecs);
            CoarseSpinor b(fine_info, num_vecs);
	

            ZeroVec(b);
            Gaussian(*x);

            fine_level.null_solver =
                std::make_shared<typename CoarseLevelT::Solver>(M_fine, params);
            assert(fine_level.null_solver->GetSubset() == SUBSET_ALL);

            // Solve the linear systems
            std::vector<LinearSolverResults> res =
                (*(fine_level.null_solver))(*x, b, ABSOLUTE, InitialGuessGiven);
            assert(res.size() == (unsigned int)num_vecs);


            if (num_vecs > 0)
                MasterLog(INFO, "Level %d: Solver Took: %d iterations", fine_level_id,
                          res[0].n_count);
        } else {
            params.RsdTarget = fabs(params.RsdTarget);
            fine_level.null_solver =
                std::make_shared<typename CoarseLevelT::Solver>(M_fine, params);
            std::vector<float> vals;
            EigsParams eigs_params;
            eigs_params.MaxIter = 0;
            eigs_params.MaxNumEvals = num_vecs;
            eigs_params.RsdTarget = params.RsdTarget;
            eigs_params.VerboseP = true;
            computeDeflation(fine_info, *fine_level.null_solver, eigs_params, x, vals);
            if (p.purpose == SetupParams::INVERT) { ScaleVec(vals, *x); }
        }
        M_fine.clear();

        // Generate individual vectors
        fine_level.null_vecs.resize(num_vecs);
        for (int k = 0; k < num_vecs; ++k) {
            fine_level.null_vecs[k] = std::make_shared<CoarseSpinor>(fine_info);
            CopyVec(*fine_level.null_vecs[k], 0, 1, *x, k, SUBSET_ALL);
	    double norm2_cb0 = sqrt(Norm2Vec(*(fine_level.null_vecs[k]), SUBSET_EVEN)[0]);
            double norm2_cb1 = sqrt(Norm2Vec(*(fine_level.null_vecs[k]), SUBSET_ODD)[0]);
            MasterLog(
            INFO,
            "Coarse Level: || v_e ||=%16.8e || v_o "
            "||=%16.8e",
            norm2_cb0, norm2_cb1);
        }

        IndexArray blocked_lattice_dims;
        IndexArray blocked_lattice_orig;
        CreateBlockList(fine_level.blocklist, blocked_lattice_dims, blocked_lattice_orig,
                        fine_level.info->GetLatticeDimensions(), p.block_sizes[fine_level_id],
                        fine_level.info->GetLatticeOrigin());

	const CBSubset &subset = fine_level.null_solver->GetSubset();

	//need to orthonormalize vectors before hand
	//orthonormalizeVecs(fine_level.null_vecs, subset);

	//do the svd on each partition separately
	if (p.do_psvd[fine_level_id]){
	MasterLog(INFO, "Performing SVD of Local Blocks on Level %d for each partition of the near null vectors",fine_level_id);
	partitionedChiralSVD(fine_level.null_vecs, fine_level.blocklist, p.n_partitions[fine_level_id]);
        //orthonormalizeBlockAggregates(fine_level.null_vecs, fine_level.blocklist);
        //orthonormalizeBlockAggregates(fine_level.null_vecs, fine_level.blocklist);
	}
	//do the svd on all partitions simultaneously
	if (p.do_lsvd[fine_level_id] && !p.do_lsq[fine_level_id]){
	MasterLog(INFO, "Performing SVD of Local Blocks on Level %d for all partitions of the near null vectors",fine_level_id);
	chiralSVD(fine_level.null_vecs, fine_level.blocklist, p.n_vecs_keep[fine_level_id]);
	fine_level.null_vecs.resize(p.n_vecs_keep[fine_level_id]);
	}
	if (p.do_lsq[fine_level_id] && p.do_lsvd[fine_level_id]) {
	MasterLog(INFO, "MG Level %d: Performing SVD followed by Least Squares Interpolation on Local Blocks of all near null vectors", fine_level_id);	       
	leastSquaresInterp(fine_level.null_vecs, p.n_vecs_keep[fine_level_id], fine_level.blocklist);
        orthonormalizeBlockAggregates(fine_level.null_vecs, fine_level.blocklist);
        orthonormalizeBlockAggregates(fine_level.null_vecs, fine_level.blocklist);
	} 
	if (!p.do_psvd[fine_level_id] && !p.do_lsvd[fine_level_id] && !p.do_lsq[fine_level_id]) { //should be the default

        // Orthonormalize the vectors -- I heard once that for GS stability is improved
        // if you do it twice.
        orthonormalizeBlockAggregates(fine_level.null_vecs, fine_level.blocklist);

        orthonormalizeBlockAggregates(fine_level.null_vecs, fine_level.blocklist);
	}
	//now have a different number of near null vectors (potentially) so change
	//num_vecs to be equal to the number of near null vectors
	num_vecs = fine_level.null_vecs.size();

        // Create the blocked Clover and Gauge Fields
        // This service needs the blocks, the vectors and is a convenience
        // Function of the M
        coarse_level.info =
            std::make_shared<LatticeInfo>(blocked_lattice_orig, blocked_lattice_dims, 2, num_vecs,
                                          fine_info.GetNodeInfo(), fine_level_id + 1);

        coarse_level.gauge = std::make_shared<CoarseGauge>(*(coarse_level.info));

        M_fine.generateCoarse(fine_level.blocklist, fine_level.null_vecs, *(coarse_level.gauge));

        coarse_level.M = std::make_shared<const typename CoarseLevelT::LinOp>(coarse_level.gauge);

        const char *coarse_prefix_name = std::getenv("MG_COARSE_FILENAME");
        if (coarse_prefix_name != nullptr && std::strlen(coarse_prefix_name) > 0) {
            std::string filename = std::string(coarse_prefix_name) + "_level" +
                                   std::to_string(fine_level_id + 1) + ".bin";
            MasterLog(INFO, "CoarseEOCloverLinearOperator: Writing coarse operator in %s",
                      filename.c_str());
            CoarseDiracOp::write(*(coarse_level.gauge), filename);
        }
    }

    template <typename CoarseLevelT>
    void ModifyCoarseOpT(CoarseLevelT &coarse_level)
    {
    coarse_level.M = std::make_shared<const typename CoarseLevelT::LinOp>(coarse_level.gauge);
    }
    // These need to be moved into a .cc file. Right now they are with QDPXX (shriek!!!)
    void SetupCoarseToCoarse(const SetupParams &p,
                             std::shared_ptr<const CoarseWilsonCloverLinearOperator> M_fine,
                             int fine_level_id, MGLevelCoarse &fine_level,
                             MGLevelCoarse &coarse_level);

    void SetupCoarseToCoarse(const SetupParams &p,
                             std::shared_ptr<const CoarseEOWilsonCloverLinearOperator> M_fine,
                             int fine_level_id, MGLevelCoarseEO &fine_level,
                             MGLevelCoarseEO &coarse_level);

    void SetupCoarseToCoarseStreamingSVD(const SetupParams &p,
                             std::shared_ptr<const CoarseWilsonCloverLinearOperator> M_fine,
                             int fine_level_id, MGLevelCoarse &fine_level,
                             MGLevelCoarse &coarse_level);

    void SetupCoarseToCoarseStreamingSVD(const SetupParams &p,
                             std::shared_ptr<const CoarseEOWilsonCloverLinearOperator> M_fine,
                             int fine_level_id, MGLevelCoarseEO &fine_level,
                             MGLevelCoarseEO &coarse_level);

    void ModifyCoarseOp(MGLevelCoarse &coarse_level);

    void ModifyCoarseOp(MGLevelCoarseEO &coarse_level);

} // namespace MG

#endif /* INCLUDE_LATTICE_MG_LEVEL_COARSE_H_ */
