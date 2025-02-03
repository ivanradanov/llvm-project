//===- DependenceInfo.cpp - Calculate dependency information for a Scop. --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Calculate the data dependency relations for a Scop using ISL.
//
// The integer set library (ISL) from Sven, has a integrated dependency analysis
// to calculate data dependences. This pass takes advantage of this and
// calculate those dependences a Scop.
//
// The dependences in this pass are exact in terms that for a specific read
// statement instance only the last write statement instance is returned. In
// case of may writes a set of possible write instances is returned. This
// analysis will never produce redundant dependences.
//
//===----------------------------------------------------------------------===//
//

#include "DependenceInfo.h"
#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Support/ScopStmt.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Builders.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLTools.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "isl/aff.h"
#include "isl/aff_type.h"
#include "isl/ctx.h"
#include "isl/flow.h"
#include "isl/map.h"
#include "isl/schedule.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "isl/union_map_type.h"
#include "isl/union_set.h"

using namespace polymer;
using namespace llvm;

#include "polly/Support/PollyDebug.h"
#define DEBUG_TYPE "polymer-dependence"

static cl::OptionCategory
    PolymerCategory("Polymer Options", "Configure the polymer loop optimizer");

static cl::opt<int> OptComputeOut(
    "polymer-dependences-computeout",
    cl::desc("Bound the dependence analysis by a maximal amount of "
             "computational steps (0 means no bound)"),
    cl::Hidden, cl::init(0), cl::cat(PolymerCategory));

static cl::opt<bool>
    LegalityCheckDisabled("disable-polymer-legality",
                          cl::desc("Disable polly legality check"), cl::Hidden,
                          cl::cat(PolymerCategory));

static cl::opt<bool>
    UseReductions("polymer-dependences-use-reductions",
                  cl::desc("Exploit reductions in dependence analysis"),
                  cl::Hidden, cl::init(true), cl::cat(PolymerCategory));

enum AnalysisType { VALUE_BASED_ANALYSIS, MEMORY_BASED_ANALYSIS };

static cl::opt<enum AnalysisType> OptAnalysisType(
    "polymer-dependences-analysis-type",
    cl::desc("The kind of dependence analysis to use"),
    cl::values(clEnumValN(VALUE_BASED_ANALYSIS, "value-based",
                          "Exact dependences without transitive dependences"),
               clEnumValN(MEMORY_BASED_ANALYSIS, "memory-based",
                          "Overapproximation of dependences")),
    cl::Hidden, cl::init(VALUE_BASED_ANALYSIS), cl::cat(PolymerCategory));

static cl::opt<Dependences::AnalysisLevel> OptAnalysisLevel(
    "polymer-dependences-analysis-level",
    cl::desc("The level of dependence analysis"),
    cl::values(clEnumValN(Dependences::AL_Statement, "statement-wise",
                          "Statement-level analysis"),
               clEnumValN(Dependences::AL_Reference, "reference-wise",
                          "Memory reference level analysis that distinguish"
                          " accessed references in the same statement"),
               clEnumValN(Dependences::AL_Access, "access-wise",
                          "Memory reference level analysis that distinguish"
                          " access instructions in the same statement")),
    cl::Hidden, cl::init(Dependences::AL_Statement), cl::cat(PolymerCategory));

//===----------------------------------------------------------------------===//

/// Tag the @p Relation domain with @p TagId
static __isl_give isl_map *tag(__isl_take isl_map *Relation,
                               __isl_take isl_id *TagId) {
  isl_space *Space = isl_map_get_space(Relation);
  Space = isl_space_drop_dims(Space, isl_dim_out, 0,
                              isl_map_dim(Relation, isl_dim_out));
  Space = isl_space_set_tuple_id(Space, isl_dim_out, TagId);
  isl_multi_aff *Tag = isl_multi_aff_domain_map(Space);
  Relation = isl_map_preimage_domain_multi_aff(Relation, Tag);
  return Relation;
}

/// Tag the @p Relation domain with either MA->getArrayId() or
///        MA->getId() based on @p TagLevel
static __isl_give isl_map *tag(__isl_take isl_map *Relation, MemoryAccess *MA,
                               Dependences::AnalysisLevel TagLevel) {
  if (TagLevel == Dependences::AL_Reference)
    return tag(Relation, MA->getArrayId().release());

  if (TagLevel == Dependences::AL_Access)
    return tag(Relation, MA->getId().release());

  // No need to tag at the statement level.
  return Relation;
}

static void collectAsyncCopyInfo(Scop &S, isl_union_map *&CopyMustWrite,
                                 isl_union_map *&CopyRead) {
  isl_space *Space = S.getParamSpace().release();
  CopyRead = isl_union_map_empty(isl_space_copy(Space));
  CopyMustWrite = isl_union_map_empty(isl_space_copy(Space));

  for (ScopStmt &Stmt : S) {
    if (!Stmt.isValidAsyncCopy())
      continue;

    for (MemoryAccess *MA : Stmt) {
      isl_set *domcp = Stmt.getDomain().release();
      isl_map *accdom = MA->getAccessRelation().release();

      accdom = isl_map_intersect_domain(accdom, domcp);

      if (MA->isRead())
        CopyRead = isl_union_map_add_map(CopyRead, accdom);
      else if (MA->isMayWrite())
        llvm_unreachable("may writes not allowed for copies");
      else if (MA->isMustWrite())
        CopyMustWrite = isl_union_map_add_map(CopyMustWrite, accdom);
      else if (MA->isKill())
        llvm_unreachable("kills not allowed for copies");
      else
        llvm_unreachable("unknown access type");
    }
  }

  CopyRead = isl_union_map_coalesce(CopyRead);
  CopyMustWrite = isl_union_map_coalesce(CopyMustWrite);
}

/// Collect information about the SCoP @p S.
static void collectInfo(Scop &S, isl_union_map *&Read,
                        isl_union_map *&MustWrite, isl_union_map *&MayWrite,
                        isl_union_map *&Kill, isl_union_map *&ReductionTagMap,
                        isl_union_set *&TaggedStmtDomain,
                        Dependences::AnalysisLevel Level) {
  isl_space *Space = S.getParamSpace().release();
  Read = isl_union_map_empty(isl_space_copy(Space));
  MustWrite = isl_union_map_empty(isl_space_copy(Space));
  MayWrite = isl_union_map_empty(isl_space_copy(Space));
  Kill = isl_union_map_empty(isl_space_copy(Space));
  ReductionTagMap = isl_union_map_empty(isl_space_copy(Space));
  isl_union_map *StmtSchedule = isl_union_map_empty(Space);

  SmallPtrSet<const ScopArrayInfo *, 8> ReductionArrays;
  // FIXME reductions
  // if (UseReductions)
  //   for (ScopStmt &Stmt : S)
  //     for (MemoryAccess *MA : Stmt)
  //       if (MA->isReductionLike())
  //         ReductionArrays.insert(MA->getScopArrayInfo());

  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      isl_set *domcp = Stmt.getDomain().release();
      isl_map *accdom = MA->getAccessRelation().release();

      accdom = isl_map_intersect_domain(accdom, domcp);

      if (ReductionArrays.count(MA->getScopArrayInfo())) {
        // Wrap the access domain and adjust the schedule accordingly.
        //
        // An access domain like
        //   Stmt[i0, i1] -> MemAcc_A[i0 + i1]
        // will be transformed into
        //   [Stmt[i0, i1] -> MemAcc_A[i0 + i1]] -> MemAcc_A[i0 + i1]
        //
        // We collect all the access domains in the ReductionTagMap.
        // This is used in Dependences::calculateDependences to create
        // a tagged Schedule tree.

        ReductionTagMap =
            isl_union_map_add_map(ReductionTagMap, isl_map_copy(accdom));
        accdom = isl_map_range_map(accdom);
      } else {
        accdom = tag(accdom, MA, Level);
        if (Level > Dependences::AL_Statement) {
          isl_map *StmtScheduleMap = Stmt.getSchedule().release();
          assert(StmtScheduleMap &&
                 "Schedules that contain extension nodes require special "
                 "handling.");
          isl_map *Schedule = tag(StmtScheduleMap, MA, Level);
          StmtSchedule = isl_union_map_add_map(StmtSchedule, Schedule);
        }
      }

      if (MA->isRead())
        Read = isl_union_map_add_map(Read, accdom);
      else if (MA->isMayWrite())
        MayWrite = isl_union_map_add_map(MayWrite, accdom);
      else if (MA->isMustWrite())
        MustWrite = isl_union_map_add_map(MustWrite, accdom);
      else if (MA->isKill())
        Kill = isl_union_map_add_map(Kill, accdom);
      else
        llvm_unreachable("unknown access type");
    }

    if (!ReductionArrays.empty() && Level == Dependences::AL_Statement)
      StmtSchedule =
          isl_union_map_add_map(StmtSchedule, Stmt.getSchedule().release());
  }

  StmtSchedule = isl_union_map_intersect_params(
      StmtSchedule, S.getAssumedContext().release());
  TaggedStmtDomain = isl_union_map_domain(StmtSchedule);

  ReductionTagMap = isl_union_map_coalesce(ReductionTagMap);
  Read = isl_union_map_coalesce(Read);
  MustWrite = isl_union_map_coalesce(MustWrite);
  MayWrite = isl_union_map_coalesce(MayWrite);
}

/// Fix all dimension of @p Zero to 0 and add it to @p user
static void fixSetToZero(isl::set Zero, isl::union_set *User) {
  for (auto i : polly::rangeIslSize(0, Zero.tuple_dim()))
    Zero = Zero.fix_si(isl::dim::set, i, 0);
  *User = User->unite(Zero);
}

/// Compute the privatization dependences for a given dependency @p Map
///
/// Privatization dependences are widened original dependences which originate
/// or end in a reduction access. To compute them we apply the transitive close
/// of the reduction dependences (which maps each iteration of a reduction
/// statement to all following ones) on the RAW/WAR/WAW dependences. The
/// dependences which start or end at a reduction statement will be extended to
/// depend on all following reduction statement iterations as well.
/// Note: "Following" here means according to the reduction dependences.
///
/// For the input:
///
///  S0:   *sum = 0;
///        for (int i = 0; i < 1024; i++)
///  S1:     *sum += i;
///  S2:   *sum = *sum * 3;
///
/// we have the following dependences before we add privatization dependences:
///
///   RAW:
///     { S0[] -> S1[0]; S1[1023] -> S2[] }
///   WAR:
///     {  }
///   WAW:
///     { S0[] -> S1[0]; S1[1024] -> S2[] }
///   RED:
///     { S1[i0] -> S1[1 + i0] : i0 >= 0 and i0 <= 1022 }
///
/// and afterwards:
///
///   RAW:
///     { S0[] -> S1[i0] : i0 >= 0 and i0 <= 1023;
///       S1[i0] -> S2[] : i0 >= 0 and i0 <= 1023}
///   WAR:
///     {  }
///   WAW:
///     { S0[] -> S1[i0] : i0 >= 0 and i0 <= 1023;
///       S1[i0] -> S2[] : i0 >= 0 and i0 <= 1023}
///   RED:
///     { S1[i0] -> S1[1 + i0] : i0 >= 0 and i0 <= 1022 }
///
/// Note: This function also computes the (reverse) transitive closure of the
///       reduction dependences.
void Dependences::addPrivatizationDependences() {
  isl_union_map *PrivRAW, *PrivWAW, *PrivWAR;

  // The transitive closure might be over approximated, thus could lead to
  // dependency cycles in the privatization dependences. To make sure this
  // will not happen we remove all negative dependences after we computed
  // the transitive closure.
  TC_RED = isl_union_map_transitive_closure(isl_union_map_copy(RED), nullptr);

  // FIXME: Apply the current schedule instead of assuming the identity schedule
  //        here. The current approach is only valid as long as we compute the
  //        dependences only with the initial (identity schedule). Any other
  //        schedule could change "the direction of the backward dependences" we
  //        want to eliminate here.
  isl_union_set *UDeltas = isl_union_map_deltas(isl_union_map_copy(TC_RED));
  isl_union_set *Universe = isl_union_set_universe(isl_union_set_copy(UDeltas));
  isl::union_set Zero =
      isl::manage(isl_union_set_empty(isl_union_set_get_space(Universe)));

  for (isl::set Set : isl::manage_copy(Universe).get_set_list())
    fixSetToZero(Set, &Zero);

  isl_union_map *NonPositive =
      isl_union_set_lex_le_union_set(UDeltas, Zero.release());

  TC_RED = isl_union_map_subtract(TC_RED, NonPositive);

  TC_RED = isl_union_map_union(
      TC_RED, isl_union_map_reverse(isl_union_map_copy(TC_RED)));
  TC_RED = isl_union_map_coalesce(TC_RED);

  isl_union_map **Maps[] = {&RAW, &WAW, &WAR};
  isl_union_map **PrivMaps[] = {&PrivRAW, &PrivWAW, &PrivWAR};
  for (unsigned u = 0; u < 3; u++) {
    isl_union_map **Map = Maps[u], **PrivMap = PrivMaps[u];

    *PrivMap = isl_union_map_apply_range(isl_union_map_copy(*Map),
                                         isl_union_map_copy(TC_RED));
    *PrivMap = isl_union_map_union(
        *PrivMap, isl_union_map_apply_range(isl_union_map_copy(TC_RED),
                                            isl_union_map_copy(*Map)));

    *Map = isl_union_map_union(*Map, *PrivMap);
  }

  isl_union_set_free(Universe);
}

static __isl_give isl_union_flow *buildFlow(__isl_keep isl_union_map *Snk,
                                            __isl_keep isl_union_map *Src,
                                            __isl_keep isl_union_map *MaySrc,
                                            __isl_keep isl_union_map *Kill,
                                            __isl_keep isl_schedule *Schedule) {
  isl_union_access_info *AI;

  AI = isl_union_access_info_from_sink(isl_union_map_copy(Snk));
  if (MaySrc)
    AI = isl_union_access_info_set_may_source(AI, isl_union_map_copy(MaySrc));
  if (Src)
    AI = isl_union_access_info_set_must_source(AI, isl_union_map_copy(Src));
  if (Kill)
    AI = isl_union_access_info_set_kill(AI, isl_union_map_copy(Kill));
  AI = isl_union_access_info_set_schedule(AI, isl_schedule_copy(Schedule));
  auto Flow = isl_union_access_info_compute_flow(AI);
  POLLY_DEBUG(if (!Flow) dbgs()
                  << "last error: "
                  << isl_ctx_last_error(isl_schedule_get_ctx(Schedule)) << " "
                  << isl_ctx_last_error_msg(isl_schedule_get_ctx(Schedule))
                  << '\n';);
  return Flow;
}

void Dependences::calculateDependences(Scop &S) {
  isl_union_map *Read, *MustWrite, *MayWrite, *ReductionTagMap, *Kill;
  isl_schedule *Schedule;
  isl_union_set *TaggedStmtDomain;

  // POLLY_DEBUG(dbgs() << "Scop: \n" << S << "\n");

  collectInfo(S, Read, MustWrite, MayWrite, Kill, ReductionTagMap,
              TaggedStmtDomain, Level);

  bool HasReductions = !isl_union_map_is_empty(ReductionTagMap);

  POLLY_DEBUG(
      dbgs() << "Read: " << isl_union_map_to_str(Read) << '\n';
      dbgs() << "MustWrite: " << isl_union_map_to_str(MustWrite) << '\n';
      dbgs() << "MayWrite: " << isl_union_map_to_str(MayWrite) << '\n';
      dbgs() << "Kill: " << isl_union_map_to_str(Kill) << '\n';
      dbgs() << "ReductionTagMap: " << isl_union_map_to_str(ReductionTagMap)
             << '\n';
      dbgs() << "TaggedStmtDomain: " << isl_union_set_to_str(TaggedStmtDomain)
             << '\n';);

  Schedule = S.getScheduleTree().release();

  if (!HasReductions) {
    isl_union_map_free(ReductionTagMap);
    // Tag the schedule tree if we want fine-grain dependence info
    if (Level > AL_Statement) {
      auto TaggedMap =
          isl_union_set_unwrap(isl_union_set_copy(TaggedStmtDomain));
      auto Tags = isl_union_map_domain_map_union_pw_multi_aff(TaggedMap);
      Schedule = isl_schedule_pullback_union_pw_multi_aff(Schedule, Tags);
    }
  } else {
    isl_union_map *IdentityMap;
    isl_union_pw_multi_aff *ReductionTags, *IdentityTags, *Tags;

    // Extract Reduction tags from the combined access domains in the given
    // SCoP. The result is a map that maps each tagged element in the domain to
    // the memory location it accesses. ReductionTags = {[Stmt[i] ->
    // Array[f(i)]] -> Stmt[i] }
    ReductionTags =
        isl_union_map_domain_map_union_pw_multi_aff(ReductionTagMap);

    // Compute an identity map from each statement in domain to itself.
    // IdentityTags = { [Stmt[i] -> Stmt[i] }
    IdentityMap = isl_union_set_identity(isl_union_set_copy(TaggedStmtDomain));
    IdentityTags = isl_union_pw_multi_aff_from_union_map(IdentityMap);

    Tags = isl_union_pw_multi_aff_union_add(ReductionTags, IdentityTags);

    // By pulling back Tags from Schedule, we have a schedule tree that can
    // be used to compute normal dependences, as well as 'tagged' reduction
    // dependences.
    Schedule = isl_schedule_pullback_union_pw_multi_aff(Schedule, Tags);
  }

  POLLY_DEBUG(dbgs() << "Read: " << isl_union_map_to_str(Read) << '\n';
              dbgs() << "MustWrite: " << isl_union_map_to_str(MustWrite)
                     << '\n';
              dbgs() << "MayWrite: " << isl_union_map_to_str(MayWrite) << '\n';
              dbgs() << "Kill: " << isl_union_map_to_str(Kill) << '\n';
              dbgs() << "Schedule: " << isl_schedule_to_str(Schedule) << "\n");

  isl_union_map *StrictWAW = nullptr;
  {
    polly::IslMaxOperationsGuard MaxOpGuard(IslCtx.get(), OptComputeOut);

    RAW = WAW = WAR = RED = nullptr;
    isl_union_map *Write = isl_union_map_union(isl_union_map_copy(MustWrite),
                                               isl_union_map_copy(MayWrite));

    // We are interested in detecting reductions that do not have intermediate
    // computations that are captured by other statements.
    //
    // Example:
    // void f(int *A, int *B) {
    //     for(int i = 0; i <= 100; i++) {
    //
    //            *-WAR (S0[i] -> S0[i + 1] 0 <= i <= 100)------------*
    //            |                                                   |
    //            *-WAW (S0[i] -> S0[i + 1] 0 <= i <= 100)------------*
    //            |                                                   |
    //            v                                                   |
    //     S0:    *A += i; >------------------*-----------------------*
    //                                        |
    //         if (i >= 98) {          WAR (S0[i] -> S1[i]) 98 <= i <= 100
    //                                        |
    //     S1:        *B = *A; <--------------*
    //         }
    //     }
    // }
    //
    // S0[0 <= i <= 100] has a reduction. However, the values in
    // S0[98 <= i <= 100] is captured in S1[98 <= i <= 100].
    // Since we allow free reordering on our reduction dependences, we need to
    // remove all instances of a reduction statement that have data dependences
    // originating from them.
    // In the case of the example, we need to remove S0[98 <= i <= 100] from
    // our reduction dependences.
    //
    // When we build up the WAW dependences that are used to detect reductions,
    // we consider only **Writes that have no intermediate Reads**.
    //
    // `isl_union_flow_get_must_dependence` gives us dependences of the form:
    // (sink <- must_source).
    //
    // It *will not give* dependences of the form:
    // 1. (sink <- ... <- may_source <- ... <- must_source)
    // 2. (sink <- ... <- must_source <- ... <- must_source)
    //
    // For a detailed reference on ISL's flow analysis, see:
    // "Presburger Formulas and Polyhedral Compilation" - Approximate Dataflow
    //  Analysis.
    //
    // Since we set "Write" as a must-source, "Read" as a may-source, and ask
    // for must dependences, we get all Writes to Writes that **do not flow
    // through a Read**.
    //
    // ScopInfo::checkForReductions makes sure that if something captures
    // the reduction variable in the same basic block, then it is rejected
    // before it is even handed here. This makes sure that there is exactly
    // one read and one write to a reduction variable in a Statement.
    // Example:
    //     void f(int *sum, int A[N], int B[N]) {
    //       for (int i = 0; i < N; i++) {
    //         *sum += A[i]; < the store and the load is not tagged as a
    //         B[i] = *sum;  < reduction-like access due to the overlap.
    //       }
    //     }

    isl_union_flow *Flow = buildFlow(Write, Write, Read, nullptr, Schedule);
    StrictWAW = isl_union_flow_get_must_dependence(Flow);
    isl_union_flow_free(Flow);

    if (OptAnalysisType == VALUE_BASED_ANALYSIS) {
      Flow = buildFlow(Read, MustWrite, MayWrite, nullptr, Schedule);
      RAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, MustWrite, MayWrite, nullptr, Schedule);
      WAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      // ISL now supports "kills" in approximate dataflow analysis, we can
      // specify the MustWrite as kills, Read as source and Write as sink.
      Flow = buildFlow(Write, nullptr, Read, MustWrite, Schedule);
      WAR = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);
    } else {
      Flow = buildFlow(Read, nullptr, Write, nullptr, Schedule);
      RAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, nullptr, Read, nullptr, Schedule);
      WAR = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, nullptr, Write, nullptr, Schedule);
      WAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);
    }

    isl_union_map_free(Write);
    isl_union_map_free(MustWrite);
    isl_union_map_free(MayWrite);
    isl_union_map_free(Read);
    isl_schedule_free(Schedule);

    RAW = isl_union_map_coalesce(RAW);
    WAW = isl_union_map_coalesce(WAW);
    WAR = isl_union_map_coalesce(WAR);

    // End of max_operations scope.
  }

  if (isl_ctx_last_error(IslCtx.get()) == isl_error_quota) {
    isl_union_map_free(RAW);
    isl_union_map_free(WAW);
    isl_union_map_free(WAR);
    isl_union_map_free(StrictWAW);
    RAW = WAW = WAR = StrictWAW = nullptr;
    isl_ctx_reset_error(IslCtx.get());
  }

  // Drop out early, as the remaining computations are only needed for
  // reduction dependences or dependences that are finer than statement
  // level dependences.
  if (!HasReductions && Level == AL_Statement) {
    RED = isl_union_map_empty(isl_union_map_get_space(RAW));
    TC_RED = isl_union_map_empty(isl_union_set_get_space(TaggedStmtDomain));
    isl_union_set_free(TaggedStmtDomain);
    isl_union_map_free(StrictWAW);
    return;
  }

  isl_union_map *STMT_RAW, *STMT_WAW, *STMT_WAR;
  STMT_RAW = isl_union_map_intersect_domain(
      isl_union_map_copy(RAW), isl_union_set_copy(TaggedStmtDomain));
  STMT_WAW = isl_union_map_intersect_domain(
      isl_union_map_copy(WAW), isl_union_set_copy(TaggedStmtDomain));
  STMT_WAR =
      isl_union_map_intersect_domain(isl_union_map_copy(WAR), TaggedStmtDomain);
  POLLY_DEBUG({
    dbgs() << "Wrapped Dependences:\n";
    dump();
    dbgs() << "\n";
  });

  // To handle reduction dependences we proceed as follows:
  // 1) Aggregate all possible reduction dependences, namely all self
  //    dependences on reduction like statements.
  // 2) Intersect them with the actual RAW & WAW dependences to the get the
  //    actual reduction dependences. This will ensure the load/store memory
  //    addresses were __identical__ in the two iterations of the statement.
  // 3) Relax the original RAW, WAW and WAR dependences by subtracting the
  //    actual reduction dependences. Binary reductions (sum += A[i]) cause
  //    the same, RAW, WAW and WAR dependences.
  // 4) Add the privatization dependences which are widened versions of
  //    already present dependences. They model the effect of manual
  //    privatization at the outermost possible place (namely after the last
  //    write and before the first access to a reduction location).

  // Step 1)
  RED = isl_union_map_empty(isl_union_map_get_space(RAW));
  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isReductionLike())
        continue;
      isl_set *AccDomW = isl_map_wrap(MA->getAccessRelation().release());
      isl_map *Identity =
          isl_map_from_domain_and_range(isl_set_copy(AccDomW), AccDomW);
      RED = isl_union_map_add_map(RED, Identity);
    }
  }

  // Step 2)
  RED = isl_union_map_intersect(RED, isl_union_map_copy(RAW));
  RED = isl_union_map_intersect(RED, StrictWAW);

  if (!isl_union_map_is_empty(RED)) {

    // Step 3)
    RAW = isl_union_map_subtract(RAW, isl_union_map_copy(RED));
    WAW = isl_union_map_subtract(WAW, isl_union_map_copy(RED));
    WAR = isl_union_map_subtract(WAR, isl_union_map_copy(RED));

    // Step 4)
    addPrivatizationDependences();
  } else
    TC_RED = isl_union_map_empty(isl_union_map_get_space(RED));

  POLLY_DEBUG({
    dbgs() << "Final Wrapped Dependences:\n";
    dump();
    dbgs() << "\n";
  });

  // RED_SIN is used to collect all reduction dependences again after we
  // split them according to the causing memory accesses. The current assumption
  // is that our method of splitting will not have any leftovers. In the end
  // we validate this assumption until we have more confidence in this method.
  isl_union_map *RED_SIN = isl_union_map_empty(isl_union_map_get_space(RAW));

  // For each reduction like memory access, check if there are reduction
  // dependences with the access relation of the memory access as a domain
  // (wrapped space!). If so these dependences are caused by this memory access.
  // We then move this portion of reduction dependences back to the statement ->
  // statement space and add a mapping from the memory access to these
  // dependences.
  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isReductionLike())
        continue;

      isl_set *AccDomW = isl_map_wrap(MA->getAccessRelation().release());
      isl_union_map *AccRedDepU = isl_union_map_intersect_domain(
          isl_union_map_copy(TC_RED), isl_union_set_from_set(AccDomW));
      if (isl_union_map_is_empty(AccRedDepU)) {
        isl_union_map_free(AccRedDepU);
        continue;
      }

      isl_map *AccRedDep = isl_map_from_union_map(AccRedDepU);
      RED_SIN = isl_union_map_add_map(RED_SIN, isl_map_copy(AccRedDep));
      AccRedDep = isl_map_zip(AccRedDep);
      AccRedDep = isl_set_unwrap(isl_map_domain(AccRedDep));
      setReductionDependences(MA, AccRedDep);
    }
  }

  assert(isl_union_map_is_equal(RED_SIN, TC_RED) &&
         "Intersecting the reduction dependence domain with the wrapped access "
         "relation is not enough, we need to loosen the access relation also");
  isl_union_map_free(RED_SIN);

  RAW = isl_union_map_zip(RAW);
  WAW = isl_union_map_zip(WAW);
  WAR = isl_union_map_zip(WAR);
  RED = isl_union_map_zip(RED);
  TC_RED = isl_union_map_zip(TC_RED);

  POLLY_DEBUG({
    dbgs() << "Zipped Dependences:\n";
    dump();
    dbgs() << "\n";
  });

  RAW = isl_union_set_unwrap(isl_union_map_domain(RAW));
  WAW = isl_union_set_unwrap(isl_union_map_domain(WAW));
  WAR = isl_union_set_unwrap(isl_union_map_domain(WAR));
  RED = isl_union_set_unwrap(isl_union_map_domain(RED));
  TC_RED = isl_union_set_unwrap(isl_union_map_domain(TC_RED));

  POLLY_DEBUG({
    dbgs() << "Unwrapped Dependences:\n";
    dump();
    dbgs() << "\n";
  });

  RAW = isl_union_map_union(RAW, STMT_RAW);
  WAW = isl_union_map_union(WAW, STMT_WAW);
  WAR = isl_union_map_union(WAR, STMT_WAR);

  RAW = isl_union_map_coalesce(RAW);
  WAW = isl_union_map_coalesce(WAW);
  WAR = isl_union_map_coalesce(WAR);
  RED = isl_union_map_coalesce(RED);
  TC_RED = isl_union_map_coalesce(TC_RED);

  POLLY_DEBUG(dump());
}

bool Dependences::isValidSchedule(Scop &S, isl::schedule NewSched) const {
  // TODO: Also check permutable/coincident flags as well.

  StatementToIslMapTy NewSchedules;
  for (auto NewMap : NewSched.get_map().get_map_list()) {
    auto Stmt = reinterpret_cast<ScopStmt *>(
        NewMap.get_tuple_id(isl::dim::in).get_user());
    NewSchedules[Stmt] = NewMap;
  }

  return isValidSchedule(S, NewSchedules);
}

bool Dependences::isValidSchedule(
    Scop &S, const StatementToIslMapTy &NewSchedule) const {
  if (LegalityCheckDisabled)
    return true;

  isl::union_map Dependences = getDependences(TYPE_RAW | TYPE_WAW | TYPE_WAR);
  isl::union_map Schedule = isl::union_map::empty(S.getIslCtx());

  isl::space ScheduleSpace;

  for (ScopStmt &Stmt : S) {
    isl::map StmtScat;

    auto Lookup = NewSchedule.find(&Stmt);
    if (Lookup == NewSchedule.end())
      StmtScat = Stmt.getSchedule();
    else
      StmtScat = Lookup->second;
    assert(!StmtScat.is_null() &&
           "Schedules that contain extension nodes require special handling.");

    if (ScheduleSpace.is_null())
      ScheduleSpace = StmtScat.get_space().range();

    Schedule = Schedule.unite(StmtScat);
  }

  Dependences = Dependences.apply_domain(Schedule);
  Dependences = Dependences.apply_range(Schedule);

  isl::set Zero = isl::set::universe(ScheduleSpace);
  for (auto i : polly::rangeIslSize(0, Zero.tuple_dim()))
    Zero = Zero.fix_si(isl::dim::set, i, 0);

  isl::union_set UDeltas = Dependences.deltas();
  isl::set Deltas = polly::singleton(UDeltas, ScheduleSpace);

  isl::space Space = Deltas.get_space();
  isl::map NonPositive = isl::map::universe(Space.map_from_set());
  NonPositive =
      NonPositive.lex_le_at(isl::multi_pw_aff::identity_on_domain(Space));
  NonPositive = NonPositive.intersect_domain(Deltas);
  NonPositive = NonPositive.intersect_range(Zero);

  return NonPositive.is_empty();
}

// Check if the current scheduling dimension is parallel.
//
// We check for parallelism by verifying that the loop does not carry any
// dependences.
//
// Parallelism test: if the distance is zero in all outer dimensions, then it
// has to be zero in the current dimension as well.
//
// Implementation: first, translate dependences into time space, then force
// outer dimensions to be equal. If the distance is zero in the current
// dimension, then the loop is parallel. The distance is zero in the current
// dimension if it is a subset of a map with equal values for the current
// dimension.
bool Dependences::isParallel(__isl_keep isl_union_map *Schedule,
                             __isl_take isl_union_map *Deps,
                             __isl_give isl_pw_aff **MinDistancePtr) const {
  isl_set *Deltas, *Distance;
  isl_map *ScheduleDeps;
  unsigned Dimension;
  bool IsParallel;

  Deps = isl_union_map_apply_range(Deps, isl_union_map_copy(Schedule));
  Deps = isl_union_map_apply_domain(Deps, isl_union_map_copy(Schedule));

  if (isl_union_map_is_empty(Deps)) {
    isl_union_map_free(Deps);
    return true;
  }

  ScheduleDeps = isl_map_from_union_map(Deps);
  Dimension = isl_map_dim(ScheduleDeps, isl_dim_out) - 1;

  for (unsigned i = 0; i < Dimension; i++)
    ScheduleDeps = isl_map_equate(ScheduleDeps, isl_dim_out, i, isl_dim_in, i);

  Deltas = isl_map_deltas(ScheduleDeps);
  Distance = isl_set_universe(isl_set_get_space(Deltas));

  // [0, ..., 0, +] - All zeros and last dimension larger than zero
  for (unsigned i = 0; i < Dimension; i++)
    Distance = isl_set_fix_si(Distance, isl_dim_set, i, 0);

  Distance = isl_set_lower_bound_si(Distance, isl_dim_set, Dimension, 1);
  Distance = isl_set_intersect(Distance, Deltas);

  IsParallel = isl_set_is_empty(Distance);
  if (IsParallel || !MinDistancePtr) {
    isl_set_free(Distance);
    return IsParallel;
  }

  Distance = isl_set_project_out(Distance, isl_dim_set, 0, Dimension);
  Distance = isl_set_coalesce(Distance);

  // This last step will compute a expression for the minimal value in the
  // distance polyhedron Distance with regards to the first (outer most)
  // dimension.
  *MinDistancePtr = isl_pw_aff_coalesce(isl_set_dim_min(Distance, 0));

  return false;
}

static void printDependencyMap(raw_ostream &OS, __isl_keep isl_union_map *DM) {
  // FIXME free the string
  if (DM)
    OS << isl_union_map_to_str(DM) << "\n";
  else
    OS << "n/a\n";
}

void Dependences::print(raw_ostream &OS) const {
  OS << "\tRAW dependences:\n\t\t";
  printDependencyMap(OS, RAW);
  OS << "\tWAR dependences:\n\t\t";
  printDependencyMap(OS, WAR);
  OS << "\tWAW dependences:\n\t\t";
  printDependencyMap(OS, WAW);
  OS << "\tReduction dependences:\n\t\t";
  printDependencyMap(OS, RED);
  OS << "\tTransitive closure of reduction dependences:\n\t\t";
  printDependencyMap(OS, TC_RED);
}

void Dependences::dump() const { print(dbgs()); }

void Dependences::releaseMemory() {
  isl_union_map_free(RAW);
  isl_union_map_free(WAR);
  isl_union_map_free(WAW);
  isl_union_map_free(RED);
  isl_union_map_free(TC_RED);

  RED = RAW = WAR = WAW = TC_RED = nullptr;

  for (auto &ReductionDeps : ReductionDependences)
    isl_map_free(ReductionDeps.second);
  ReductionDependences.clear();
}

isl::union_map Dependences::getDependences(int Kinds) const {
  assert(hasValidDependences() && "No valid dependences available");
  isl::space Space = isl::manage_copy(RAW).get_space();
  isl::union_map Deps = Deps.empty(Space.ctx());

  if (Kinds & TYPE_RAW)
    Deps = Deps.unite(isl::manage_copy(RAW));

  if (Kinds & TYPE_WAR)
    Deps = Deps.unite(isl::manage_copy(WAR));

  if (Kinds & TYPE_WAW)
    Deps = Deps.unite(isl::manage_copy(WAW));

  if (Kinds & TYPE_RED)
    Deps = Deps.unite(isl::manage_copy(RED));

  if (Kinds & TYPE_TC_RED)
    Deps = Deps.unite(isl::manage_copy(TC_RED));

  Deps = Deps.coalesce();
  Deps = Deps.detect_equalities();
  return Deps;
}

bool Dependences::hasValidDependences() const {
  return (RAW != nullptr) && (WAR != nullptr) && (WAW != nullptr);
}

__isl_give isl_map *
Dependences::getReductionDependences(MemoryAccess *MA) const {
  return isl_map_copy(ReductionDependences.lookup(MA));
}

void Dependences::setReductionDependences(MemoryAccess *MA,
                                          __isl_take isl_map *D) {
  assert(ReductionDependences.count(MA) == 0 &&
         "Reduction dependences set twice!");
  ReductionDependences[MA] = D;
}

// ############### PPCG BEGIN ###############
// clang-format off

/* Given a union of "tagged" access relations of the form
 *
 *	[S_i[...] -> R_j[]] -> A_k[...]
 *
 * project out the "tags" (R_j[]).
 * That is, return a union of relations of the form
 *
 *	S_i[...] -> A_k[...]
 */
static __isl_give isl_union_map *project_out_tags(
	__isl_take isl_union_map *umap)
{
	return isl_union_map_domain_factor_domain(umap);
}

/* Construct a function from tagged iteration domains to the corresponding
 * untagged iteration domains with as range of the wrapped map in the domain
 * the reference tags that appear in any of the reads, writes or kills.
 * Store the result in ps->tagger.
 *
 * For example, if the statement with iteration space S[i,j]
 * contains two array references R_1[] and R_2[], then ps->tagger will contain
 *
 *	{ [S[i,j] -> R_1[]] -> S[i,j]; [S[i,j] -> R_2[]] -> S[i,j] }
 */
static void compute_tagger(struct ppcg_scop *ps)
{
	isl_union_map *tagged;
	isl_union_pw_multi_aff *tagger;

	tagged = isl_union_map_copy(ps->tagged_reads);
	tagged = isl_union_map_union(tagged,
				isl_union_map_copy(ps->tagged_may_writes));
	tagged = isl_union_map_union(tagged,
				isl_union_map_copy(ps->tagged_must_kills));
	tagged = isl_union_map_universe(tagged);
	tagged = isl_union_set_unwrap(isl_union_map_domain(tagged));

	tagger = isl_union_map_domain_map_union_pw_multi_aff(tagged);

	ps->tagger = tagger;
}

/* Compute the live out accesses, i.e., the writes that are
 * potentially not killed by any kills or any other writes, and
 * store them in ps->live_out.
 *
 * We compute the "dependence" of any "kill" (an explicit kill
 * or a must write) on any may write.
 * The elements accessed by the may writes with a "depending" kill
 * also accessing the element are definitely killed.
 * The remaining may writes can potentially be live out.
 *
 * The result of the dependence analysis is
 *
 *	{ IW -> [IK -> A] }
 *
 * with IW the instance of the write statement, IK the instance of kill
 * statement and A the element that was killed.
 * The range factor range is
 *
 *	{ IW -> A }
 *
 * containing all such pairs for which there is a kill statement instance,
 * i.e., all pairs that have been killed.
 */
static void compute_live_out(struct ppcg_scop *ps)
{
	isl_schedule *schedule;
	isl_union_map *kills;
	isl_union_map *exposed;
	isl_union_map *covering;
	isl_union_access_info *access;
	isl_union_flow *flow;

	schedule = isl_schedule_copy(ps->schedule);
	kills = isl_union_map_union(isl_union_map_copy(ps->must_writes),
				    isl_union_map_copy(ps->must_kills));
	access = isl_union_access_info_from_sink(kills);
	access = isl_union_access_info_set_may_source(access,
				    isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	covering = isl_union_flow_get_full_may_dependence(flow);
	isl_union_flow_free(flow);

	covering = isl_union_map_range_factor_range(covering);
	exposed = isl_union_map_copy(ps->may_writes);
	exposed = isl_union_map_subtract(exposed, covering);
	ps->live_out = exposed;
}

/* Compute the tagged flow dependences and the live_in accesses and store
 * the results in ps->tagged_dep_flow and ps->live_in.
 *
 * Both must-writes and must-kills are allowed to kill dependences
 * from earlier writes to subsequent reads.
 * The must-kills are not included in the potential sources, though.
 * The flow dependences with a must-kill as source would
 * reflect possibly uninitialized reads.
 * No dependences need to be introduced to protect such reads
 * (other than those imposed by potential flows from may writes
 * that follow the kill).  Those flow dependences are therefore not needed.
 * The dead code elimination also assumes
 * the flow sources are non-kill instances.
 */
static void compute_tagged_flow_dep_only(struct ppcg_scop *ps)
{
	isl_union_pw_multi_aff *tagger;
	isl_schedule *schedule;
	isl_union_map *live_in;
	isl_union_access_info *access;
	isl_union_flow *flow;
	isl_union_map *must_source;
	isl_union_map *kills;
	isl_union_map *tagged_flow;

	tagger = isl_union_pw_multi_aff_copy(ps->tagger);
	schedule = isl_schedule_copy(ps->schedule);
	schedule = isl_schedule_pullback_union_pw_multi_aff(schedule, tagger);
	kills = isl_union_map_copy(ps->tagged_must_kills);
	must_source = isl_union_map_copy(ps->tagged_must_writes);
	kills = isl_union_map_union(kills, must_source);
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(ps->tagged_reads));
	access = isl_union_access_info_set_kill(access, kills);
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->tagged_may_writes));
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	tagged_flow = isl_union_flow_get_may_dependence(flow);
	ps->tagged_dep_flow = tagged_flow;
	live_in = isl_union_flow_get_may_no_source(flow);
	ps->live_in = project_out_tags(live_in);
	isl_union_flow_free(flow);
}

/* Compute ps->dep_flow from ps->tagged_dep_flow
 * by projecting out the reference tags.
 */
static void derive_flow_dep_from_tagged_flow_dep(struct ppcg_scop *ps)
{
	ps->dep_flow = isl_union_map_copy(ps->tagged_dep_flow);
	ps->dep_flow = isl_union_map_factor_domain(ps->dep_flow);
}

/* Compute the flow dependences and the live_in accesses and store
 * the results in ps->dep_flow and ps->live_in.
 * A copy of the flow dependences, tagged with the reference tags
 * is stored in ps->tagged_dep_flow.
 *
 * We first compute ps->tagged_dep_flow, i.e., the tagged flow dependences
 * and then project out the tags.
 */
static void compute_tagged_flow_dep(struct ppcg_scop *ps)
{
	compute_tagged_flow_dep_only(ps);
	derive_flow_dep_from_tagged_flow_dep(ps);
}

/* Compute the order dependences that prevent the potential live ranges
 * from overlapping.
 *
 * In particular, construct a union of relations
 *
 *	[R[...] -> R_1[]] -> [W[...] -> R_2[]]
 *
 * where [R[...] -> R_1[]] is the range of one or more live ranges
 * (i.e., a read) and [W[...] -> R_2[]] is the domain of one or more
 * live ranges (i.e., a write).  Moreover, the read and the write
 * access the same memory element and the read occurs before the write
 * in the original schedule.
 * The scheduler allows some of these dependences to be violated, provided
 * the adjacent live ranges are all local (i.e., their domain and range
 * are mapped to the same point by the current schedule band).
 *
 * Note that if a live range is not local, then we need to make
 * sure it does not overlap with _any_ other live range, and not
 * just with the "previous" and/or the "next" live range.
 * We therefore add order dependences between reads and
 * _any_ later potential write.
 *
 * We also need to be careful about writes without a corresponding read.
 * They are already prevented from moving past non-local preceding
 * intervals, but we also need to prevent them from moving past non-local
 * following intervals.  We therefore also add order dependences from
 * potential writes that do not appear in any intervals
 * to all later potential writes.
 * Note that dead code elimination should have removed most of these
 * dead writes, but the dead code elimination may not remove all dead writes,
 * so we need to consider them to be safe.
 *
 * The order dependences are computed by computing the "dataflow"
 * from the above unmatched writes and the reads to the may writes.
 * The unmatched writes and the reads are treated as may sources
 * such that they would not kill order dependences from earlier
 * such writes and reads.
 */
static void compute_order_dependences(struct ppcg_scop *ps)
{
	isl_union_map *reads;
	isl_union_map *shared_access;
	isl_union_set *matched;
	isl_union_map *unmatched;
	isl_union_pw_multi_aff *tagger;
	isl_schedule *schedule;
	isl_union_access_info *access;
	isl_union_flow *flow;

	tagger = isl_union_pw_multi_aff_copy(ps->tagger);
	schedule = isl_schedule_copy(ps->schedule);
	schedule = isl_schedule_pullback_union_pw_multi_aff(schedule, tagger);
	reads = isl_union_map_copy(ps->tagged_reads);
	matched = isl_union_map_domain(isl_union_map_copy(ps->tagged_dep_flow));
	unmatched = isl_union_map_copy(ps->tagged_may_writes);
	unmatched = isl_union_map_subtract_domain(unmatched, matched);
	reads = isl_union_map_union(reads, unmatched);
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(ps->tagged_may_writes));
	access = isl_union_access_info_set_may_source(access, reads);
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	shared_access = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);

	ps->tagged_dep_order = isl_union_map_copy(shared_access);
	ps->dep_order = isl_union_map_factor_domain(shared_access);
}

/* Compute those validity dependences of the program represented by "scop"
 * that should be unconditionally enforced even when live-range reordering
 * is used.
 *
 * In particular, compute the external false dependences
 * as well as order dependences between sources with the same sink.
 * The anti-dependences are already taken care of by the order dependences.
 * The external false dependences are only used to ensure that live-in and
 * live-out data is not overwritten by any writes inside the scop.
 * The independences are removed from the external false dependences,
 * but not from the order dependences between sources with the same sink.
 *
 * In particular, the reads from live-in data need to precede any
 * later write to the same memory element.
 * As to live-out data, the last writes need to remain the last writes.
 * That is, any earlier write in the original schedule needs to precede
 * the last write to the same memory element in the computed schedule.
 * The possible last writes have been computed by compute_live_out.
 * They may include kills, but if the last access is a kill,
 * then the corresponding dependences will effectively be ignored
 * since we do not schedule any kill statements.
 *
 * Note that the set of live-in and live-out accesses may be
 * an overapproximation.  There may therefore be potential writes
 * before a live-in access and after a live-out access.
 *
 * In the presence of may-writes, there may be multiple live-ranges
 * with the same sink, accessing the same memory element.
 * The sources of these live-ranges need to be executed
 * in the same relative order as in the original program
 * since we do not know which of the may-writes will actually
 * perform a write.  Consider all sources that share a sink and
 * that may write to the same memory element and compute
 * the order dependences among them.
 */
static void compute_forced_dependences(struct ppcg_scop *ps)
{
	isl_union_map *shared_access;
	isl_union_map *exposed;
	isl_union_map *live_in;
	isl_union_map *sink_access;
	isl_union_map *shared_sink;
	isl_union_access_info *access;
	isl_union_flow *flow;
	isl_schedule *schedule;

	exposed = isl_union_map_copy(ps->live_out);
	schedule = isl_schedule_copy(ps->schedule);
	access = isl_union_access_info_from_sink(exposed);
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	shared_access = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);
	ps->dep_forced = shared_access;

	schedule = isl_schedule_copy(ps->schedule);
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->live_in));
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	live_in = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);

	ps->dep_forced = isl_union_map_union(ps->dep_forced, live_in);
	ps->dep_forced = isl_union_map_subtract(ps->dep_forced,
				isl_union_map_copy(ps->independence));

	schedule = isl_schedule_copy(ps->schedule);
	sink_access = isl_union_map_copy(ps->tagged_dep_flow);
	sink_access = isl_union_map_range_product(sink_access,
				isl_union_map_copy(ps->tagged_may_writes));
	sink_access = isl_union_map_domain_factor_domain(sink_access);
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(sink_access));
	access = isl_union_access_info_set_may_source(access, sink_access);
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	shared_sink = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);
	ps->dep_forced = isl_union_map_union(ps->dep_forced, shared_sink);
}

/* Remove independence from the tagged flow dependences.
 * Since the user has guaranteed that source and sink of an independence
 * can be executed in any order, there cannot be a flow dependence
 * between them, so they can be removed from the set of flow dependences.
 * However, if the source of such a flow dependence is a must write,
 * then it may have killed other potential sources, which would have
 * to be recovered if we were to remove those flow dependences.
 * We therefore keep the flow dependences that originate in a must write,
 * even if it corresponds to a known independence.
 */
static void remove_independences_from_tagged_flow(struct ppcg_scop *ps)
{
	isl_union_map *tf;
	isl_union_set *indep;
	isl_union_set *mw;

	tf = isl_union_map_copy(ps->tagged_dep_flow);
	tf = isl_union_map_zip(tf);
	indep = isl_union_map_wrap(isl_union_map_copy(ps->independence));
	tf = isl_union_map_intersect_domain(tf, indep);
	tf = isl_union_map_zip(tf);
	mw = isl_union_map_domain(isl_union_map_copy(ps->tagged_must_writes));
	tf = isl_union_map_subtract_domain(tf, mw);
	ps->tagged_dep_flow = isl_union_map_subtract(ps->tagged_dep_flow, tf);
}

/* Compute the dependences of the program represented by "scop"
 * in case live range reordering is allowed.
 *
 * We compute the actual live ranges and the corresponding order
 * false dependences.
 *
 * The independences are removed from the flow dependences
 * (provided the source is not a must-write) as well as
 * from the external false dependences (by compute_forced_dependences).
 */
static void compute_live_range_reordering_dependences(struct ppcg_scop *ps)
{
	compute_tagged_flow_dep_only(ps);
	remove_independences_from_tagged_flow(ps);
	derive_flow_dep_from_tagged_flow_dep(ps);
	compute_order_dependences(ps);
	compute_forced_dependences(ps);
}

/* Compute the potential flow dependences and the potential live in
 * accesses.
 *
 * Both must-writes and must-kills are allowed to kill dependences
 * from earlier writes to subsequent reads, as in compute_tagged_flow_dep_only.
 */
static void compute_flow_dep(struct ppcg_scop *ps)
{
	isl_union_access_info *access;
	isl_union_flow *flow;
	isl_union_map *kills, *must_writes;

	access = isl_union_access_info_from_sink(isl_union_map_copy(ps->reads));
	kills = isl_union_map_copy(ps->must_kills);
	must_writes = isl_union_map_copy(ps->must_writes);
	kills = isl_union_map_union(kills, must_writes);
	access = isl_union_access_info_set_kill(access, kills);
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_schedule(access,
				isl_schedule_copy(ps->schedule));
	flow = isl_union_access_info_compute_flow(access);

	ps->dep_flow = isl_union_flow_get_may_dependence(flow);
	ps->live_in = isl_union_flow_get_may_no_source(flow);
	isl_union_flow_free(flow);
}

static void compute_async_dependences(struct ppcg_scop *scop) {
	isl_union_access_info *access;
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(scop->reads));
	access = isl_union_access_info_set_kill(access,
				isl_union_map_union(isl_union_map_copy(scop->must_writes), isl_union_map_copy(scop->must_kills)));
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(scop->async_must_writes));
	access = isl_union_access_info_set_schedule(access,
				isl_schedule_copy(scop->schedule));
	isl_union_flow *flow;
	flow = isl_union_access_info_compute_flow(access);

	scop->dep_async = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);
}

/* Compute the dependences of the program represented by "scop".
 * Store the computed potential flow dependences
 * in scop->dep_flow and the reads with potentially no corresponding writes in
 * scop->live_in.
 * Store the potential live out accesses in scop->live_out.
 * Store the potential false (anti and output) dependences in scop->dep_false.
 *
 * If live range reordering is allowed, then we compute a separate
 * set of order dependences and a set of external false dependences
 * in compute_live_range_reordering_dependences.
 */
static void compute_dependences(struct ppcg_scop *scop)
{
	isl_union_map *may_source;
	isl_union_access_info *access;
	isl_union_flow *flow;

	if (!scop)
		return;

	compute_async_dependences(scop);
	compute_live_out(scop);

	if (scop->options->live_range_reordering)
		compute_live_range_reordering_dependences(scop);
	else if (scop->options->target != PPCG_TARGET_C)
		compute_tagged_flow_dep(scop);
	else
		compute_flow_dep(scop);

	may_source = isl_union_map_union(isl_union_map_copy(scop->may_writes),
					isl_union_map_copy(scop->reads));
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(scop->may_writes));
	access = isl_union_access_info_set_kill(access,
				isl_union_map_copy(scop->must_writes));
	access = isl_union_access_info_set_may_source(access, may_source);
	access = isl_union_access_info_set_schedule(access,
				isl_schedule_copy(scop->schedule));
	flow = isl_union_access_info_compute_flow(access);

	scop->dep_false = isl_union_flow_get_may_dependence(flow);
	scop->dep_false = isl_union_map_coalesce(scop->dep_false);
	isl_union_flow_free(flow);
}

/* Report the eliminated dead code,
 * if there is any and if the verbose option is set.
 */
static void report_dead_code(struct ppcg_scop *ps,
	__isl_keep isl_union_set *live)
{
	isl_ctx *ctx;
	isl_printer *p;
	isl_union_set *dead;

	if (!ps->options->debug->verbose)
		return;
	if (isl_union_set_is_equal(ps->domain, live))
		return;

	ctx = isl_union_set_get_ctx(live);
	dead = isl_union_set_subtract(isl_union_set_copy(ps->domain),
					isl_union_set_copy(live));

	p = isl_printer_to_file(ctx, stdout);
	p = isl_printer_print_str(p, "Eliminated dead instances: ");
	p = isl_printer_print_union_set(p, dead);
	p = isl_printer_end_line(p);
	isl_printer_free(p);

	isl_union_set_free(dead);
}

/* Eliminate dead code from ps->domain.
 *
 * In particular, intersect both ps->domain and the domain of
 * ps->schedule with the (parts of) iteration
 * domains that are needed to produce the output or for statement
 * iterations that call functions.
 * Also intersect the range of the dataflow dependences with
 * this domain such that the removed instances will no longer
 * be considered as targets of dataflow.
 *
 * We start with the iteration domains that call functions
 * and the set of iterations that last write to an array
 * (except those that are later killed).
 *
 * Then we add those statement iterations that produce
 * something needed by the "live" statements iterations.
 * We keep doing this until no more statement iterations can be added.
 * To ensure that the procedure terminates, we compute the affine
 * hull of the live iterations (bounded to the original iteration
 * domains) each time we have added extra iterations.
 */
static void eliminate_dead_code(struct ppcg_scop *ps)
{
	isl_union_set *live;
	isl_union_map *dep;
	isl_union_pw_multi_aff *tagger;

	live = isl_union_map_domain(isl_union_map_copy(ps->live_out));
	if (!isl_union_set_is_empty(ps->call)) {
		live = isl_union_set_union(live, isl_union_set_copy(ps->call));
		live = isl_union_set_coalesce(live);
	}

	dep = isl_union_map_copy(ps->dep_flow);
	dep = isl_union_map_reverse(dep);

	for (;;) {
		isl_union_set *extra;

		extra = isl_union_set_apply(isl_union_set_copy(live),
					    isl_union_map_copy(dep));
		if (isl_union_set_is_subset(extra, live)) {
			isl_union_set_free(extra);
			break;
		}

		live = isl_union_set_union(live, extra);
		live = isl_union_set_affine_hull(live);
		live = isl_union_set_intersect(live,
					    isl_union_set_copy(ps->domain));
	}

	isl_union_map_free(dep);

	report_dead_code(ps, live);

	ps->domain = isl_union_set_intersect(ps->domain,
						isl_union_set_copy(live));
	ps->schedule = isl_schedule_intersect_domain(ps->schedule,
						isl_union_set_copy(live));
	ps->dep_flow = isl_union_map_intersect_range(ps->dep_flow,
						isl_union_set_copy(live));
	tagger = isl_union_pw_multi_aff_copy(ps->tagger);
	live = isl_union_set_preimage_union_pw_multi_aff(live, tagger);
	ps->tagged_dep_flow = isl_union_map_intersect_range(ps->tagged_dep_flow,
						live);
}

// clang-format on
// ############### PPCG END ###############

bool gpu_array_can_be_private(ScopArrayInfo &sai) {
  auto ty = dyn_cast<mlir::MemRefType>(sai.val.getType());
  return ty && mlir::nvgpu::NVGPUDialect::hasGlobalMemoryAddressSpace(ty);
}

void collect_access_to_array(Scop &S, isl_union_map *&access_to_array) {
  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      isl::id accessId = MA->getId();
      isl::map rel = MA->getAccessRelation();
      isl::space arraySpace = rel.range().space();
      isl::set arraySet = arraySpace.universe_set();
      arraySpace = arraySpace.drop_dims(
          isl::dim::set, 0, unsignedFromIslSize(arraySpace.dim(isl::dim::set)));
      isl::set accessSet =
          arraySpace.set_tuple_id(isl::dim::set, accessId).universe_set();
      isl::map map = isl::map::from_domain_and_range(accessSet, arraySet);
      if (!access_to_array)
        access_to_array = map.to_union_map().release();
      else
        access_to_array = isl_union_map_add_map(access_to_array, map.release());
    }
  }
}

void collect_order_dependences(Scop &S, ppcg_scop *scop) {
  isl_space *space;
  isl_union_map *accesses;
  isl_union_map *array_order;

  space = isl_union_map_get_space(scop->reads);
  array_order = isl_union_map_empty(space);

  accesses = isl_union_map_copy(scop->tagged_reads);
  accesses = isl_union_map_union(accesses,
                                 isl_union_map_copy(scop->tagged_may_writes));
  accesses = isl_union_map_universe(accesses);

  for (auto &array : S.arrays) {
    isl_set *set;
    isl_union_set *uset;
    isl_union_map *order;

    set = isl_set_universe(array.space.copy());
    uset = isl_union_set_from_set(set);
    uset = isl_union_map_domain(
        isl_union_map_intersect_range(isl_union_map_copy(accesses), uset));
    order = isl_union_map_copy(scop->tagged_dep_order);
    order = isl_union_map_intersect_domain(order, uset);
    order = isl_union_map_zip(order);
    order = isl_union_set_unwrap(isl_union_map_domain(order));
    order = isl_union_map_subtract(order, S.getIndependence().release());
    array.dep_order = isl::manage(order);

    LLVM_DEBUG(dbgs() << "dep_order for " << array.name << " ";
               isl_union_map_dump(array.dep_order.get()));

    if (gpu_array_can_be_private(array))
      continue;

    array_order = isl_union_map_union(array_order, array.dep_order.copy());
  }

  isl_union_map_free(accesses);

  scop->array_order = array_order;
}

ppcg_scop *computeDeps(Scop &S, polymer::Dependences::AnalysisLevel level) {
  isl_union_map *ReductionTagMap;
  isl_schedule *Schedule;
  isl_union_set *TaggedStmtDomain;

  // TODO leaking
  bool verbose = false;
  LLVM_DEBUG(verbose = true);
  ppcg_debug_options *debug_options = new ppcg_debug_options{verbose};
  ppcg_options *options = new ppcg_options{1, PPCG_TARGET_CUDA, debug_options};
  ppcg_scop *ps = new ppcg_scop;
  (*ps) = ppcg_scop{0};
  (*ps).options = options;

  collectInfo(S, ps->reads, ps->must_writes, ps->may_writes, ps->must_kills,
              ReductionTagMap, TaggedStmtDomain,
              polymer::Dependences::AL_Statement);
  collectInfo(S, ps->tagged_reads, ps->tagged_must_writes,
              ps->tagged_may_writes, ps->tagged_must_kills, ReductionTagMap,
              TaggedStmtDomain, level);

  // In ppcg, the must writes are a subset of the may writes
  ps->may_writes =
      isl_union_map_union(isl_union_map_copy(ps->must_writes), ps->may_writes);
  ps->tagged_may_writes = isl_union_map_union(
      isl_union_map_copy(ps->tagged_must_writes), ps->tagged_may_writes);

  collectAsyncCopyInfo(S, ps->async_must_writes, ps->async_reads);

  POLLY_DEBUG(dbgs() << "ReductionTagMap: "
                     << isl_union_map_to_str(ReductionTagMap) << '\n';
              dbgs() << "TaggedStmtDomain: "
                     << isl_union_set_to_str(TaggedStmtDomain) << '\n';);

  Schedule = S.getScheduleTree().release();

  ps->schedule = Schedule;
  ps->domain = S.getScheduleTree().get_domain().release();
  ps->context =
      isl_set_universe(S.getScheduleTree().get_domain().get_space().release());
  // See PENCIL support in pet and PPCG (Verdoolaege 2015)
  ps->independence = S.getIndependence().release();

  compute_tagger(ps);
  compute_dependences(ps);
  eliminate_dead_code(ps);

  collect_order_dependences(S, ps);

  return ps;
}

ppcg_scop *computeDeps(Scop &S) {
  ppcg_scop *ps = computeDeps(S, polymer::Dependences::AL_Access);
  ppcg_scop *array_ps = computeDeps(S, polymer::Dependences::AL_Reference);

  ps->atagged_reads = isl_union_map_copy(array_ps->tagged_reads);
  ps->atagged_may_writes = isl_union_map_copy(array_ps->tagged_may_writes);
  ps->atagged_must_writes = isl_union_map_copy(array_ps->tagged_must_writes);
  ps->atagged_must_kills = isl_union_map_copy(array_ps->tagged_must_kills);
  ps->atagger = isl_union_pw_multi_aff_copy(array_ps->tagger);
  ps->atagged_dep_flow = isl_union_map_copy(array_ps->tagged_dep_flow);

  collect_access_to_array(S, ps->access_to_array);

#define PPCGSCOPDUMP(field)                                                    \
  dbgs() << #field << " " << isl_union_map_to_str(ps->field) << '\n'
  POLLY_DEBUG({
    PPCGSCOPDUMP(tagged_reads);
    PPCGSCOPDUMP(atagged_reads);
    PPCGSCOPDUMP(reads);
    PPCGSCOPDUMP(async_reads);
    PPCGSCOPDUMP(tagged_may_writes);
    PPCGSCOPDUMP(atagged_may_writes);
    PPCGSCOPDUMP(may_writes);
    PPCGSCOPDUMP(tagged_must_writes);
    PPCGSCOPDUMP(atagged_must_writes);
    PPCGSCOPDUMP(must_writes);
    PPCGSCOPDUMP(async_must_writes);
    PPCGSCOPDUMP(tagged_must_kills);
    PPCGSCOPDUMP(atagged_must_kills);
    PPCGSCOPDUMP(must_kills);
    PPCGSCOPDUMP(live_in);
    PPCGSCOPDUMP(live_out);
    PPCGSCOPDUMP(independence);
    PPCGSCOPDUMP(dep_flow);
    PPCGSCOPDUMP(tagged_dep_flow);
    PPCGSCOPDUMP(atagged_dep_flow);
    PPCGSCOPDUMP(dep_false);
    PPCGSCOPDUMP(dep_forced);
    PPCGSCOPDUMP(dep_order);
    PPCGSCOPDUMP(tagged_dep_order);
    PPCGSCOPDUMP(dep_async);
    PPCGSCOPDUMP(array_order);
    PPCGSCOPDUMP(access_to_array);
    dbgs() << "tagger" << " " << isl_union_pw_multi_aff_to_str(ps->tagger)
           << '\n';
    dbgs() << "atagger" << " " << isl_union_pw_multi_aff_to_str(ps->atagger)
           << '\n';
    dbgs() << "schedule" << '\n';
    isl_schedule_dump(ps->schedule);
  });
#undef PPCGSCOPDUMP

  return ps;
}
