//===--- polly/DependenceInfo.h - Polyhedral dependency analysis *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Calculate the data dependency relations for a Scop using ISL.
//
// The integer set library (ISL) from Sven has an integrated dependency analysis
// to calculate data dependences. This pass takes advantage of this and
// calculates those dependences of a Scop.
//
// The dependences in this pass are exact in terms that for a specific read
// statement instance only the last write statement instance is returned. In
// case of may-writes, a set of possible write instances is returned. This
// analysis will never produce redundant dependences.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_DEPENDENCE_INFO_H
#define POLLY_DEPENDENCE_INFO_H

#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Support/ScopStmt.h"
#include "llvm/ADT/DenseMap.h"
#include "isl/ctx.h"
#include "isl/isl-noexceptions.h"

namespace polymer {

using namespace llvm;

typedef IslScop Scop;

/// The accumulated dependence information for a SCoP.
///
/// The Dependences struct holds all dependence information we collect and
/// compute for one SCoP. It also offers an interface that allows users to
/// query only specific parts.
class Dependences final {
public:
  // Granularities of the current dependence analysis
  enum AnalysisLevel {
    AL_Statement = 0,
    // Distinguish accessed memory references in the same statement
    AL_Reference,
    // Distinguish memory access instances in the same statement
    AL_Access,

    NumAnalysisLevels
  };

  /// Map type for reduction dependences.
  using ReductionDependencesMapTy = DenseMap<MemoryAccess *, isl_map *>;

  /// Map type to associate statements with schedules.
  using StatementToIslMapTy = DenseMap<ScopStmt *, isl::map>;

  /// The type of the dependences.
  ///
  /// Reduction dependences are separated from RAW/WAW/WAR dependences because
  /// we can ignore them during the scheduling. That's because the order
  /// in which the reduction statements are executed does not matter. However,
  /// if they are executed in parallel we need to take additional measures
  /// (e.g, privatization) to ensure a correct result. The (reverse) transitive
  /// closure of the reduction dependences are used to check for parallel
  /// executed reduction statements during code generation. These dependences
  /// connect all instances of a reduction with each other, they are therefore
  /// cyclic and possibly "reversed".
  enum Type {
    // Write after read
    TYPE_WAR = 1 << 0,

    // Read after write
    TYPE_RAW = 1 << 1,

    // Write after write
    TYPE_WAW = 1 << 2,

    // Reduction dependences
    TYPE_RED = 1 << 3,

    // Transitive closure of the reduction dependences (& the reverse)
    TYPE_TC_RED = 1 << 4,
  };

  const std::shared_ptr<isl_ctx> &getSharedIslCtx() const { return IslCtx; }

  /// Get the dependences of type @p Kinds.
  ///
  /// @param Kinds This integer defines the different kinds of dependences
  ///              that will be returned. To return more than one kind, the
  ///              different kinds are 'ored' together.
  isl::union_map getDependences(int Kinds) const;

  /// Report if valid dependences are available.
  bool hasValidDependences() const;

  /// Return the reduction dependences caused by @p MA.
  ///
  /// @return The reduction dependences caused by @p MA or nullptr if none.
  __isl_give isl_map *getReductionDependences(MemoryAccess *MA) const;

  /// Return all reduction dependences.
  const ReductionDependencesMapTy &getReductionDependences() const {
    return ReductionDependences;
  }

  /// Check if a partial schedule is parallel wrt to @p Deps.
  ///
  /// @param Schedule       The subset of the schedule space that we want to
  ///                       check.
  /// @param Deps           The dependences @p Schedule needs to respect.
  /// @param MinDistancePtr If not nullptr, the minimal dependence distance will
  ///                       be returned at the address of that pointer
  ///
  /// @return Returns true, if executing parallel the outermost dimension of
  ///         @p Schedule is valid according to the dependences @p Deps.
  bool isParallel(__isl_keep isl_union_map *Schedule,
                  __isl_take isl_union_map *Deps,
                  __isl_give isl_pw_aff **MinDistancePtr = nullptr) const;

  /// Check if a new schedule is valid.
  ///
  /// @param S             The current SCoP.
  /// @param NewSchedules  The new schedules
  ///
  /// @return True if the new schedule is valid, false if it reverses
  ///         dependences.
  bool isValidSchedule(Scop &S, const StatementToIslMapTy &NewSchedules) const;

  /// Return true of the schedule @p NewSched is a schedule for @S that does not
  /// violate any dependences.
  bool isValidSchedule(Scop &S, isl::schedule NewSched) const;

  /// Print the stored dependence information.
  void print(llvm::raw_ostream &OS) const;

  /// Dump the dependence information stored to the dbgs stream.
  void dump() const;

  /// Return the granularity of this dependence analysis.
  AnalysisLevel getDependenceLevel() { return Level; }

  /// Allow the DependenceInfo access to private members and methods.
  ///
  /// To restrict access to the internal state, only the DependenceInfo class
  /// is able to call or modify a Dependences struct.
  friend struct DependenceAnalysis;
  friend struct DependenceInfoPrinterPass;
  friend class DependenceInfo;
  friend class DependenceInfoWrapperPass;

  /// Destructor that will free internal objects.
  ~Dependences() { releaseMemory(); }

public:
  /// Create an empty dependences struct.
  explicit Dependences(const std::shared_ptr<isl_ctx> &IslCtx,
                       AnalysisLevel Level)
      : RAW(nullptr), WAR(nullptr), WAW(nullptr), RED(nullptr), TC_RED(nullptr),
        IslCtx(IslCtx), Level(Level) {}

  /// Calculate and add at the privatization dependences.
  void addPrivatizationDependences();

  /// Calculate the dependences for a certain SCoP @p S.
  void calculateDependences(Scop &S);

private:
  /// Set the reduction dependences for @p MA to @p Deps.
  void setReductionDependences(MemoryAccess *MA, __isl_take isl_map *Deps);

  /// Free the objects associated with this Dependences struct.
  ///
  /// The Dependences struct will again be "empty" afterwards.
  void releaseMemory();

  /// The different basic kinds of dependences we calculate.
  isl_union_map *RAW;
  isl_union_map *WAR;
  isl_union_map *WAW;

  /// The special reduction dependences.
  isl_union_map *RED;

  /// The (reverse) transitive closure of reduction dependences.
  isl_union_map *TC_RED;

  /// Mapping from memory accesses to their reduction dependences.
  ReductionDependencesMapTy ReductionDependences;

  /// Isl context from the SCoP.
  std::shared_ptr<isl_ctx> IslCtx;

  /// Granularity of this dependence analysis.
  const AnalysisLevel Level;
};

} // namespace polymer

// ############### PPCG DECLS BEGIN ###############
// clang-format off
#define		PPCG_TARGET_C		0
#define		PPCG_TARGET_CUDA	1
#define		PPCG_TARGET_OPENCL      2

struct ppcg_debug_options {
	int verbose;
};

struct pet_scop {};
struct ppcg_options {
	int live_range_reordering;
	int target;
	struct ppcg_debug_options *debug;
};

struct ppcg_scop {
	struct ppcg_options *options;

	unsigned start;
	unsigned end;

	isl_set *context;
	isl_union_set *domain;
	isl_union_set *call;
	isl_union_map *tagged_reads;
	isl_union_map *atagged_reads;
	isl_union_map *reads;
	isl_union_map *live_in;
	isl_union_map *tagged_may_writes;
	isl_union_map *atagged_may_writes;
	isl_union_map *may_writes;
	isl_union_map *tagged_must_writes;
	isl_union_map *atagged_must_writes;
	isl_union_map *must_writes;
	isl_union_map *live_out;
	isl_union_map *tagged_must_kills;
	isl_union_map *atagged_must_kills;
	isl_union_map *must_kills;
	isl_union_map *async_reads;
	isl_union_map *async_must_writes;

	isl_union_pw_multi_aff *tagger;
	isl_union_pw_multi_aff *atagger;

	isl_union_map *independence;

	isl_union_map *dep_flow;
	isl_union_map *tagged_dep_flow;
	isl_union_map *atagged_dep_flow;
	isl_union_map *dep_false;
	isl_union_map *dep_forced;
	isl_union_map *dep_order;
	isl_union_map *tagged_dep_order;
  isl_union_map *dep_async;
	isl_schedule *schedule;

	isl_id_to_ast_expr *names;

	struct pet_scop *pet;

  isl_union_map *array_order;

  isl_union_map *access_to_array;
};
// clang-format on
// ############### PPCG DECLS END ###############

ppcg_scop *computeDeps(polymer::Scop &S);

#endif
