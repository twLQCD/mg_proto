/*
 * mg_level_qphix.cpp
 *
 *  Created on: Oct 20, 2017
 *      Author: bjoo
 */

#include <lattice/qphix/mg_level_qphix.h>
#include <lattice/qphix/qphix_blas_wrappers.h>
#include <lattice/qphix/qphix_aggregate.h>
#include <lattice/qphix/qphix_qdp_utils.h>

namespace MG
{
template<typename SpinorT, typename SolverT, typename LinOpT>
void SetupQPhiXToCoarseGenerateVecsT(const SetupParams& p,  std::shared_ptr<LinOpT> M_fine,
              MGLevelQPhiXT<SpinorT,SolverT,LinOpT>& fine_level,
              MGLevelCoarse& coarse_level)
{
  // Check M
  if ( !M_fine ) {
    MasterLog(ERROR, "%s: M_fine is null...", __FUNCTION__);
  }

  fine_level.M = M_fine;

  // OK we will need the level for this info
  fine_level.info = std::make_shared<LatticeInfo>((M_fine->GetInfo()).GetLatticeDimensions());


  // Null solver is BiCGStabF. Let us make a parameter struct for it.
  LinearSolverParamsBase params;
  params.MaxIter = p.null_solver_max_iter[0];
  params.RsdTarget = p.null_solver_rsd_target[0];
  params.VerboseP = p.null_solver_verboseP[0];

  fine_level.null_solver = std::make_shared<const SolverT>(*M_fine, params);

  // Zero RHS
  SpinorT b(*(fine_level.info));
  ZeroVec(b);


  // Generate the vectors
  int num_vecs = p.n_vecs[0];

  fine_level.null_vecs.resize(num_vecs);
  for(int k=0; k < num_vecs; ++k) {

    fine_level.null_vecs[k]= std::make_shared<SpinorT>(*(fine_level.info));
    Gaussian(*(fine_level.null_vecs[k]));
  }


  for(int k=0; k < num_vecs; ++k ) {
    LinearSolverResults res = (*(fine_level.null_solver))(*(fine_level.null_vecs[k]),b, ABSOLUTE);

    MasterLog(INFO,"MG Level 0: BiCGStab Solver Took: %d iterations",res.n_count);
  }


}

template<typename SpinorT, typename SolverT, typename LinOpT>
void SetupQPhiXToCoarseVecsInT(const SetupParams& p,  std::shared_ptr<LinOpT> M_fine,
              MGLevelQPhiXT<SpinorT,SolverT,LinOpT>& fine_level,
              MGLevelCoarse& coarse_level)
{
  // Check M
  if ( ! fine_level.info ) {
    fine_level.info = std::make_shared<LatticeInfo>(M_fine->GetInfo().GetLatticeDimensions());
  }

  if ( ! fine_level.M ) {
    fine_level.M = M_fine;
  }

  if( ! fine_level.null_solver) {
    // Null solver is BiCGStabF. Let us make a parameter struct for it.
     LinearSolverParamsBase params;
     params.MaxIter = p.null_solver_max_iter[0];
     params.RsdTarget = p.null_solver_rsd_target[0];
     params.VerboseP = p.null_solver_verboseP[0];

     fine_level.null_solver = std::make_shared<const SolverT>(*M_fine, params);
  }

  int num_vecs = p.n_vecs[0];
  if ( num_vecs != fine_level.null_vecs.size()) {
    MasterLog(ERROR, "Expected %d vecs but got %d", num_vecs, fine_level.null_vecs.size());
  }

  const IndexArray& latdims = fine_level.info->GetLatticeDimensions();

  MasterLog(INFO, "MG Level 0: Creating BlockList");
  IndexArray blocked_lattice_dims;
  IndexArray blocked_lattice_orig;
  CreateBlockList(fine_level.blocklist,
      blocked_lattice_dims,
      blocked_lattice_orig,
      latdims,
      p.block_sizes[0],
      fine_level.info->GetLatticeOrigin());

  // Orthonormalize the vectors -- I heard once that for GS stability is improved
  // if you do it twice.
  MasterLog(INFO, "MG Level 0: Block Orthogonalizing Aggregates");



  orthonormalizeBlockAggregates(fine_level.null_vecs,
                    fine_level.blocklist);

  orthonormalizeBlockAggregates(fine_level.null_vecs,
                    fine_level.blocklist);




  // Create the blocked Clover and Gauge Fields
  // This service needs the blocks, the vectors and is a convenience
    // Function of the M
  coarse_level.info = std::make_shared<const LatticeInfo>(blocked_lattice_orig,
                            blocked_lattice_dims,
                            2, num_vecs, NodeInfo());

  coarse_level.gauge = std::make_shared<CoarseGauge>(*(coarse_level.info));



  M_fine->generateCoarse(fine_level.blocklist, fine_level.null_vecs, *(coarse_level.gauge));

  coarse_level.M = std::make_shared< const CoarseWilsonCloverLinearOperator>(coarse_level.gauge,1);

}

// NON EO VERSIONS
void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p,
									const std::shared_ptr<QPhiXWilsonCloverLinearOperator>& M_fine,
             MGLevelQPhiXLinOp<QPhiXWilsonCloverLinearOperator>& fine_level,
			 MGLevelCoarse& coarse_level)
{
  SetupQPhiXToCoarseGenerateVecsT<>(p,M_fine,fine_level,coarse_level);
}

 void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p,
		 	 const std::shared_ptr<QPhiXWilsonCloverLinearOperatorF>& M_fine,
               MGLevelQPhiXLinOpF<QPhiXWilsonCloverLinearOperatorF>& fine_level,
			   MGLevelCoarse& coarse_level)
 {
   SetupQPhiXToCoarseGenerateVecsT<>(p,M_fine,fine_level,coarse_level);
 }

 void SetupQPhiXToCoarseVecsIn(const SetupParams& p,
		 	 const std::shared_ptr<QPhiXWilsonCloverLinearOperator>& M_fine,
              MGLevelQPhiXLinOp<QPhiXWilsonCloverLinearOperator>& fine_level,
			  MGLevelCoarse& coarse_level)
 {
   SetupQPhiXToCoarseVecsInT<>(p,M_fine,fine_level,coarse_level);
 }

  void SetupQPhiXToCoarseVecsIn(const SetupParams& p,
		  	  	  const std::shared_ptr<QPhiXWilsonCloverLinearOperatorF>& M_fine,
				  MGLevelQPhiXLinOpF<QPhiXWilsonCloverLinearOperatorF>& fine_level,
				  MGLevelCoarse& coarse_level)
  {
    SetupQPhiXToCoarseVecsInT<>(p,M_fine,fine_level,coarse_level);
  }

  void SetupQPhiXToCoarse(const SetupParams& p,
		  	  const std::shared_ptr<QPhiXWilsonCloverLinearOperator>& M_fine,
               MGLevelQPhiXLinOp<QPhiXWilsonCloverLinearOperator>& fine_level,
			   MGLevelCoarse& coarse_level)
  {
    SetupQPhiXToCoarseGenerateVecs(p, M_fine, fine_level, coarse_level);
    SetupQPhiXToCoarseVecsIn(p, M_fine, fine_level, coarse_level);
  }

  void SetupQPhiXToCoarse(const SetupParams& p,
		  	  const std::shared_ptr<QPhiXWilsonCloverLinearOperatorF>& M_fine,
			  MGLevelQPhiXLinOpF<QPhiXWilsonCloverLinearOperatorF>& fine_level,
			  MGLevelCoarse& coarse_level)
   {
     SetupQPhiXToCoarseGenerateVecs(p, M_fine, fine_level, coarse_level);
     SetupQPhiXToCoarseVecsIn(p, M_fine, fine_level, coarse_level);
   }



  // EO VERSIONS



	void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p,
				const std::shared_ptr<QPhiXWilsonCloverEOLinearOperator>& M_fine,
 	             MGLevelQPhiXLinOp<QPhiXWilsonCloverEOLinearOperator>& fine_level,
				 MGLevelCoarse& coarse_level)
 	{
 	  SetupQPhiXToCoarseGenerateVecsT<>(p,M_fine,fine_level,coarse_level);
 	}

 	 void SetupQPhiXToCoarseGenerateVecs(const SetupParams& p,
 			 	 const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF>& M_fine,
 	               MGLevelQPhiXLinOpF<QPhiXWilsonCloverEOLinearOperatorF>& fine_level, MGLevelCoarse& coarse_level)
 	 {
 	   SetupQPhiXToCoarseGenerateVecsT<>(p,M_fine,fine_level,coarse_level);
 	 }

 	 void SetupQPhiXToCoarseVecsIn(const SetupParams& p,
 			 	 const std::shared_ptr<QPhiXWilsonCloverEOLinearOperator>& M_fine,
 	              MGLevelQPhiXLinOp<QPhiXWilsonCloverEOLinearOperator>& fine_level,
				  MGLevelCoarse& coarse_level)
 	 {
 	   SetupQPhiXToCoarseVecsInT<>(p,M_fine,fine_level,coarse_level);
 	 }

 	  void SetupQPhiXToCoarseVecsIn(const SetupParams& p,
 			  	  const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF>& M_fine,
 	                MGLevelQPhiXLinOpF<QPhiXWilsonCloverEOLinearOperatorF>& fine_level, MGLevelCoarse& coarse_level)
 	  {
 	    SetupQPhiXToCoarseVecsInT<>(p,M_fine,fine_level,coarse_level);
 	  }

 	  void SetupQPhiXToCoarse(const SetupParams& p,
 			  	  const std::shared_ptr<QPhiXWilsonCloverEOLinearOperator>& M_fine,
 	               MGLevelQPhiXLinOp<QPhiXWilsonCloverEOLinearOperator>& fine_level, MGLevelCoarse& coarse_level)
 	  {
 	    SetupQPhiXToCoarseGenerateVecs(p, M_fine, fine_level, coarse_level);
 	    SetupQPhiXToCoarseVecsIn(p, M_fine, fine_level, coarse_level);
 	  }

 	  void SetupQPhiXToCoarse(const SetupParams& p,
 			  	  	  	  	  const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF>& M_fine,
 	                MGLevelQPhiXLinOpF<QPhiXWilsonCloverEOLinearOperatorF>& fine_level, MGLevelCoarse& coarse_level)
 	   {
 	     SetupQPhiXToCoarseGenerateVecs(p, M_fine, fine_level, coarse_level);
 	     SetupQPhiXToCoarseVecsIn(p, M_fine, fine_level, coarse_level);
 	   }



 	  void SetupQPhiXMGLevels(const SetupParams& p, QPhiXMultigridLevels& mg_levels,
 			  const std::shared_ptr< QPhiXWilsonCloverLinearOperatorF >& M_fine)
 	  {

 		  mg_levels.n_levels = p.n_levels;
 		  if( mg_levels.n_levels < 2 ){
 			  MasterLog(ERROR, "Number of Multigrid Levels < 2");
 		  }

 		  int n_coarse_levels = mg_levels.n_levels-1;
 		  mg_levels.coarse_levels.resize(n_coarse_levels);

 		  MasterLog(INFO, "QPhiXMG Setup Level 0 and 1");
 		  SetupQPhiXToCoarse(p,M_fine, mg_levels.fine_level, mg_levels.coarse_levels[0]);

 		  for(int coarse_level=1; coarse_level < n_coarse_levels; ++coarse_level ) {

 			  MasterLog(INFO, "Setup Level %d and %d", coarse_level, coarse_level+1);
 			  SetupCoarseToCoarse(p,mg_levels.coarse_levels[coarse_level-1].M,
 					  coarse_level, mg_levels.coarse_levels[coarse_level-1],
					  mg_levels.coarse_levels[coarse_level]);

 		  }


 	  }

 	  void SetupQPhiXMGLevels(const SetupParams& p,
 			  	  QPhiXMultigridLevelsEO& mg_levels,
				  const std::shared_ptr< QPhiXWilsonCloverEOLinearOperatorF >& M_fine)
 	  {

 		  mg_levels.n_levels = p.n_levels;
 		  if( mg_levels.n_levels < 2 ){
 			  MasterLog(ERROR, "Number of Multigrid Levels < 2");
 		  }

 		  int n_coarse_levels = mg_levels.n_levels-1;
 		  mg_levels.coarse_levels.resize(n_coarse_levels);

 		  MasterLog(INFO, "QPhiXMG Setup Level 0 and 1");
 		  SetupQPhiXToCoarse(p,M_fine, mg_levels.fine_level, mg_levels.coarse_levels[0]);

 		  for(int coarse_level=1; coarse_level < n_coarse_levels; ++coarse_level ) {

 			  MasterLog(INFO, "Setup Level %d and %d", coarse_level, coarse_level+1);
 			  SetupCoarseToCoarse(p,mg_levels.coarse_levels[coarse_level-1].M,
 					  coarse_level, mg_levels.coarse_levels[coarse_level-1],
					  mg_levels.coarse_levels[coarse_level]);

 		  }


 	  }
}

