/*
 * mg_params_qdpxx.h
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_FINE_QDPXX_MG_PARAMS_QDPXX_H_
#define INCLUDE_LATTICE_FINE_QDPXX_MG_PARAMS_QDPXX_H_

#include "lattice/constants.h"
#include "lattice/fgmres_common.h"
#include "lattice/mr_params.h"
#include "lattice/solver.h"

namespace MG {

    struct VCycleParams {
        // Pre Smoother Params
        LinearSolverParamsBase pre_smoother_params;
        LinearSolverParamsBase bottom_solver_params;
        LinearSolverParamsBase post_smoother_params;
        LinearSolverParamsBase cycle_params;
    };

    struct SetupParams {
        int n_levels;
        std::vector<int> n_vecs;
	std::vector<bool> do_psvd;
	std::vector<bool> do_lsvd;
	std::vector<int> n_vecs_keep;
	std::vector<int> n_partitions;
        std::vector<IndexArray> block_sizes;
        std::vector<LinearSolverParamsBase> null_solver_params;
        enum { INVERT, INVARIANT_SPACE } purpose;
        SetupParams() : n_levels(0), purpose(INVERT) {}
    };

} // Namespace

#endif /* INCLUDE_LATTICE_FINE_QDPXX_MG_PARAMS_QDPXX_H_ */
