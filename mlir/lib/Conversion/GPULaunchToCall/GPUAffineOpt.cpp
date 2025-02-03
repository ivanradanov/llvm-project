#include "LoopUndistribute.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Target/ISL.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "polly/Support/GICHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/ErrorHandling.h"

#include "isl/aff.h"
#include "isl/ast.h"
#include "isl/ast_build.h"
#include "isl/constraint.h"
#include "isl/id.h"
#include "isl/isl-noexceptions.h"
#include "isl/map.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/union_set.h"

#include "DependenceInfo.h"
#include "GPULowering.h"
#include "ISLUtils.h"
#include "LoopDistribute.h"
#include "Utils.h"

using namespace mlir;

#define DEBUG_TYPE "gpu-affine-opt"
#define DBGS (llvm::dbgs() << "gpu-affine-opt: ")

#define ISL_DUMP(X) LLVM_DEBUG(llvm::dbgs() << #X << ": "; dumpIslObj(X));

STATISTIC(numKernels, "number of kernels");
STATISTIC(newSchedulesComputed, "how many new schedules we computed");

namespace mlir {
#define GEN_PASS_DEF_GPUAFFINEOPTPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using polly::dumpIslObj;

namespace mlir {
namespace gpu {
namespace affine_opt {

struct Copy {
  affine::AffineReadOpInterface load;
  affine::AffineWriteOpInterface store;
  Operation *loadOp;
  Operation *storeOp;
};

struct AccessInfo {
  SmallVector<Value, 8> regionSymbols;
  SmallVector<Value, 4> lbs, ubs;
  SmallVector<AffineMap, 4> lbMaps, ubMaps;
  unsigned rank;
};

struct VectorStore {
  affine::AffineParallelOp par;
  affine::AffineStoreOp store;
  Value val;
};
struct VectorLoad {
  affine::AffineParallelOp par;
  affine::AffineLoadOp load;
  Value val;
};

bool isBlockPar(Operation *op) {
  if (!!op->getAttr("gpu.par.block")) {
    assert(op->getNumRegions() == 1);
    return true;
  }
  return false;
}
bool isGridPar(Operation *op) {
  if (!!op->getAttr("gpu.par.grid")) {
    assert(op->getNumRegions() == 1);
    return true;
  }
  return false;
}
affine::AffineParallelOp isAffineGridPar(Operation *op) {
  auto gridPar = dyn_cast_or_null<affine::AffineParallelOp>(op);
  if (!gridPar)
    return nullptr;
  if (gridPar->getAttr("gpu.par.grid"))
    return gridPar;
  return nullptr;
}
affine::AffineParallelOp isAffineBlockPar(Operation *op) {
  auto blockPar = dyn_cast_or_null<affine::AffineParallelOp>(op);
  if (!blockPar)
    return nullptr;
  if (blockPar->getAttr("gpu.par.block"))
    return blockPar;
  return nullptr;
}

static std::optional<VectorStore> isVectorStore(affine::AffineStoreOp store) {
  if (!store->getParentOp()->hasAttr("affine.vector.store"))
    return {};
  return VectorStore{
      cast<affine::AffineParallelOp>(store->getParentOp()), store,
      cast<vector::ExtractOp>(store.getValueToStore().getDefiningOp())
          .getVector()};
}

static std::optional<VectorLoad> isVectorLoad(affine::AffineParallelOp par) {
  if (!par->hasAttr("affine.vector.load"))
    return {};
  return VectorLoad{
      par,
      cast<affine::AffineLoadOp>(
          cast<vector::BroadcastOp>(
              par.getBody()->getTerminator()->getOperand(0).getDefiningOp())
              .getSource()
              .getDefiningOp()),
      par->getResult(0)};
}

template <typename T>
static Value computeMap(RewriterBase &rewriter, Location loc, AffineMap map,
                        ValueRange operands) {
  if (map.getNumResults() > 1)
    return rewriter.create<T>(loc, map, operands);
  else if (map.getNumResults() == 1)
    return rewriter.create<affine::AffineApplyOp>(loc, map, operands);
  else
    llvm_unreachable("map with 0 results");
}

static inline void islAssert(const isl_size &size) {
  assert(size != isl_size_error);
}
[[maybe_unused]] static inline unsigned
unsignedFromIslSize(const isl::size &size) {
  assert(!size.is_error());
  return static_cast<unsigned>(size);
}
[[maybe_unused]] static inline unsigned
unsignedFromIslSize(const isl_size &size) {
  islAssert(size);
  return static_cast<unsigned>(size);
}

static __isl_give isl_schedule_constraints *
construct_schedule_constraints(struct ppcg_scop *scop, polymer::Scop &S) {
  isl_union_set *domain;
  isl_union_map *dep_raw, *dep;
  isl_union_map *validity, *proximity, *coincidence, *anti_proximity;
  isl_schedule_constraints *sc;

  domain = isl_union_set_copy(scop->domain);
  sc = isl_schedule_constraints_on_domain(domain);
  sc = isl_schedule_constraints_set_context(sc, isl_set_copy(scop->context));
  if (scop->options->live_range_reordering) {
    sc = isl_schedule_constraints_set_conditional_validity(
        sc, isl_union_map_copy(scop->tagged_dep_flow),
        isl_union_map_copy(scop->tagged_dep_order));
    proximity = isl_union_map_copy(scop->dep_flow);
    validity = isl_union_map_copy(proximity);
    validity =
        isl_union_map_union(validity, isl_union_map_copy(scop->dep_forced));
// According to "Scheduling for PPCG": "Note that as explained in Section
// 4.3 below, the false dependence relation is only used for historical
// reasons." Try disabling for now.
#if 0
    proximity =
        isl_union_map_union(proximity, isl_union_map_copy(scop->dep_false));
#endif
    coincidence = isl_union_map_copy(validity);
    coincidence = isl_union_map_subtract(
        coincidence, isl_union_map_copy(scop->independence));
// TODO this is introducing unwanted dependencies, like these, dont know
// why, need to investigate
//   "[P0] -> { RS1[i0, 0, 0] -> RS0[o0, 0, 0] : i0 >= 0 and i0 < o0 < P0;
//   RS0_affine_parallel[i0, 0, 0] -> RS1_affine_parallel[i0, 0, 0] : 0 <=
//   i0 < P0 }"
// Disable for now.
#if 0
    coincidence =
        isl_union_map_union(coincidence, isl_union_map_copy(scop->array_order));
#endif
  } else {
    dep_raw = isl_union_map_copy(scop->dep_flow);
    dep = isl_union_map_copy(scop->dep_false);
    dep = isl_union_map_union(dep, dep_raw);
    dep = isl_union_map_coalesce(dep);
    proximity = isl_union_map_copy(dep);
    coincidence = isl_union_map_copy(dep);
    validity = dep;
  }
  anti_proximity = isl_union_map_copy(scop->dep_async);
  sc = isl_schedule_constraints_set_validity(sc, validity);
  sc = isl_schedule_constraints_set_coincidence(sc, coincidence);
  sc = isl_schedule_constraints_set_proximity(sc, proximity);
  sc = isl_schedule_constraints_set_anti_proximity(sc, anti_proximity);
  isl_union_map *lrs = isl_union_map_copy(scop->atagged_dep_flow);
  sc =
      isl_schedule_constraints_set_live_range_span(sc, isl_union_map_copy(lrs));
  // FIXME we would like to put this maximal path computation in isl and not
  // here, but the function is currently written in c++ so that doesn't work.
  lrs = get_maximal_paths(isl::manage(lrs)).release();
  sc = isl_schedule_constraints_set_live_range_maximal_span(sc, lrs);

  isl_union_set *arrays = isl_union_map_range(isl_union_set_unwrap(
      isl_union_map_domain(isl_union_map_copy(scop->atagged_dep_flow))));
  arrays = isl_union_set_universe(arrays);
  isl::union_map arraySizes =
      isl::manage(isl_union_map_empty(isl_union_set_get_space(arrays)));

  auto r = isl::manage(arrays).foreach_set([&](isl::set set) -> isl::stat {
    Value v = Value::getFromOpaquePointer(set.get_tuple_id().get_user());
    auto *sai = S.getArray(v);
    assert(sai);
    auto size = sai->getSize();
    if (!size)
      return isl::stat::error();

    auto space = set.get_space();
    space = space.add_dims(isl::dim::out, 1);
    auto sizeSet = isl::set::universe(isl::space(set.ctx(), 0, 1));
    auto cst =
        isl::constraint::alloc_equality(isl::local_space(sizeSet.get_space()));
    cst = cst.set_constant_si(-*size);
    cst = cst.set_coefficient_si(isl::dim::out, 0, 1);
    sizeSet = sizeSet.add_constraint(cst);
    isl::map map = isl::map::from_domain_and_range(set, sizeSet);
    arraySizes = arraySizes.unite(isl::union_map(map));

    return isl::stat::ok();
  });

  if (r.is_ok()) {
    // TODO get this from the gpu module target info
    // 48kB is it KiB or KB???
    const int maxShmemPerBlock = 48 * 1000;
    sc = isl_schedule_constraints_set_caches(sc, 1, &maxShmemPerBlock);
    sc = isl_schedule_constraints_set_array_sizes(sc, arraySizes.release());
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Computing array sizes failed\n");
    return nullptr;
  }

  sc = isl_schedule_constraints_set_access_to_array(
      sc, isl_union_map_copy(scop->access_to_array));

  return sc;
}

static isl::schedule insertGridPar(isl::schedule schedule) {
  auto root = schedule.get_root();

  auto nChildren = root.n_children();
  if (nChildren.is_error())
    return {};
  if ((unsigned)nChildren != 1)
    return {};

  auto child = root.child(0);

  if (!child.isa<isl::schedule_node_band>())
    return {};
  auto band = child.as<isl::schedule_node_band>();

  // TODO need to split
  auto nMember = band.n_member();
  assert(!nMember.is_error());
  if ((unsigned)nMember > 3)
    return {};

  bool allCoincident = true;
  for (unsigned i = 0; i < (unsigned)nMember; i++)
    allCoincident = allCoincident && band.member_get_coincident(i);

  if (!allCoincident)
    return {};

  isl::id gridMark = isl::id::alloc(schedule.ctx(), polymer::gridParallelMark,
                                    (void *)(uintptr_t)(unsigned)nMember);

  // The generated grid parallel loop must be atomic for all members
  for (unsigned i = 0; i < (unsigned)nMember; i++)
    band = band.member_set_ast_loop_atomic(i);

  isl::schedule_node node = band.insert_mark(gridMark);
  return node.get_schedule();
}

struct PrepScheduleInfo {
  isl::union_set allArrays;
  isl::union_set unallocatedArrays;
  isl::union_set allocatedArrays;
  isl::union_map lrs;
  isl::union_map asyncDeps;
};

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

static isl::union_map tag(isl::union_map umap, isl::id id) {
  isl::union_map taggedMap =
      isl::manage(isl_union_map_empty(umap.get_space().release()));
  umap.foreach_map([&](isl::map map) {
    isl::map tagged = isl::manage(tag(map.release(), id.copy()));
    taggedMap = taggedMap.unite(tagged.to_union_map());
    return isl::stat::ok();
  });
  return taggedMap;
}

static isl::stat
isl_id_to_id_foreach(isl_id_to_id *id_to_id,
                     std::function<isl::stat(isl::id, isl::id)> f) {
  struct Data {
    decltype(f) &fn;
  } data{f};
  auto lambda2 = [](isl_id *id, isl_id *target_id, void *user) -> isl_stat {
    return static_cast<struct Data *>(user)
        ->fn(isl::manage(id), isl::manage(target_id))
        .release();
  };
  return isl::manage(isl_id_to_id_foreach(id_to_id, lambda2, &data));
}

static bool isMark(isl::schedule_node node, StringRef mark) {
  if (!node.isa<isl::schedule_node_mark>())
    return false;
  isl::id id = node.as<isl::schedule_node_mark>().get_id();
  return mark.str() == id.get_name();
}

static bool are_coincident(isl::union_map umap, isl::union_map schedule) {
  isl::union_map applied = umap;
  applied = applied.apply_domain(schedule);
  applied = applied.apply_range(schedule);

  LLVM_DEBUG(llvm::dbgs() << "Applied schedule "; dumpIslObj(applied));

  isl::union_set deltas = applied.deltas();
  LLVM_DEBUG(llvm::dbgs() << "Deltas "; dumpIslObj(deltas));
  isl::union_map deltas_map =
      isl::manage(isl_union_map_deltas_map(applied.copy()));
  LLVM_DEBUG(llvm::dbgs() << "Deltas map "; dumpIslObj(deltas_map));

  bool coincident = true;
  deltas.foreach_set([&](isl::set set) {
    unsigned size = unsignedFromIslSize(set.dim(isl::dim::set));
    bool allZero = true;
    for (unsigned i = 0; i < size; i++) {
      isl::val v = set.plain_get_val_if_fixed(isl::dim::set, i);
      allZero &= v.is_zero();
    }
    coincident &= allZero;
    return isl::stat::ok();
  });
  return coincident;
}

static isl::schedule_node insertArrayExpansion(isl::schedule_node node,
                                               PrepScheduleInfo &psi) {

  auto nChildren = unsignedFromIslSize(node.n_children());

  isl::union_map schedule = isl::manage(
      isl_schedule_node_get_prefix_and_node_schedule_union_map(node.get()));

  LLVM_DEBUG(llvm::dbgs() << "Prefix schedule "; dumpIslObj(schedule);
             dumpIslObj(node.get_prefix_schedule_relation()); dumpIslObj(node));

  isl::union_set toAllocate =
      isl::manage(isl_union_set_empty(psi.allArrays.get_space().release()));
  psi.unallocatedArrays.foreach_set([&](isl::set set) {
    isl::id array = set.get_tuple_id();
    isl::union_map taggedSchedule = tag(schedule, array);
    LLVM_DEBUG(llvm::dbgs() << "Tagged prefix schedule ";
               dumpIslObj(taggedSchedule));

    bool coincident = are_coincident(psi.lrs, taggedSchedule);

    LLVM_DEBUG(llvm::dbgs() << "Array ";
               dumpIslObj(set.get_space().get_tuple_id(isl::dim::set));
               llvm::dbgs() << " coincidence: " << coincident << "\n");

    if (!coincident)
      toAllocate =
          toAllocate.unite(isl::set::universe(set.get_space()).to_union_set());

    return isl::stat::ok();
  });

  psi.allocatedArrays = psi.allocatedArrays.unite(toAllocate);
  psi.unallocatedArrays = psi.unallocatedArrays.subtract(toAllocate);

  LLVM_DEBUG(llvm::dbgs() << "Unallocated "; dumpIslObj(psi.unallocatedArrays));
  LLVM_DEBUG(llvm::dbgs() << "Allocated "; dumpIslObj(psi.allocatedArrays));

  isl::union_set toExpand =
      isl::manage(isl_union_set_empty(psi.allArrays.get_space().release()));
  std::map<isl_id *, std::vector<unsigned>> toExpandMap;
  if (node.isa<isl::schedule_node_band>()) {
    auto band = node.as<isl::schedule_node_band>();
    unsigned members = unsignedFromIslSize(band.n_member());
    for (unsigned i = 0; i < members; i++) {
      isl_id_to_id *expansion =
          isl_schedule_node_band_member_get_array_expansion(node.get(), i);
      auto getFactors = [&](isl::id arrayId, isl::id expansionId) -> isl::stat {
        // if (!toAllocate
        //     .intersect(isl::set::universe(toAllocate.get_space())
        //                .set_tuple_id(isl::manage_copy(arrayId))
        //                .to_union_set())
        //     .is_empty())
        unsigned factor = (unsigned)(uintptr_t)expansionId.get_user();
        // TODO put the expansion number in here
        auto set = isl::set::universe(toAllocate.get_space())
                       .set_tuple_id(arrayId)
                       .to_union_set();
        toExpand = toExpand.unite(set);
        if (i == 0)
          toExpandMap.insert({arrayId.get(), {factor}});
        else
          toExpandMap[arrayId.get()].push_back(factor);
        return isl::stat::ok();
      };
      if (isl_id_to_id_foreach(expansion, getFactors).is_error())
        return {};
    }
  }
  for (auto it = toExpandMap.begin(); it != toExpandMap.end();) {
    auto expand = it->second;
    bool expanded =
        !llvm::all_of(expand, [&](unsigned expand) { return expand == 1; });
    if (!expanded)
      it = toExpandMap.erase(it);
    else
      it++;
  }
  LLVM_DEBUG(llvm::dbgs() << "ToExpand "; dumpIslObj(toExpand));

  for (unsigned i = 0; i < (unsigned)nChildren; i++) {
    auto child = node.child(i);
    auto newChild = insertArrayExpansion(child, psi);
    node = newChild.parent();
  }

  if (!toAllocate.is_empty()) {

    if (!toExpand.is_empty()) {
      assert(node.isa<isl::schedule_node_band>() &&
             "Expansion in non-band node?");
      auto band = node.as<isl::schedule_node_band>();
      unsigned members = unsignedFromIslSize(band.n_member());
      // FIXME this is a workaround because our algorithm for determining what
      // specific index of the expansion dim is accessed depends on the band
      // being generated as a single loop.
      //
      // This needs to be fixed.
      //
      // Also we don't need all dimensions to be atomic.
      //
      // Also this will change when we separate the prologue, epilogue of
      // expansion bands.
      for (unsigned i = 0; i < members; i++)
        band = band.member_set_ast_loop_atomic(i);
      node = band;
    }

    // TODO add free function
    polymer::AllocateArrayMarkInfo *aam =
        new polymer::AllocateArrayMarkInfo{toAllocate, toExpandMap};
    isl::id allocateArrayMark =
        isl::id::alloc(schedule.ctx(), polymer::allocateArrayMark, (void *)aam);
    node = node.insert_mark(allocateArrayMark);
  }

  return node;
}
static isl::schedule insertArrayExpansion(isl::schedule schedule,
                                          PrepScheduleInfo &psi) {
  auto root = schedule.get_root();
  return insertArrayExpansion(root, psi).get_schedule();
}

static isl::schedule_node
insertAsyncCopySynchronisation(isl::schedule_node node, PrepScheduleInfo &psi) {
  auto nChildren = unsignedFromIslSize(node.n_children());

  for (unsigned i = 0; i < (unsigned)nChildren; i++) {
    auto child = node.child(i);
    auto newChild = insertAsyncCopySynchronisation(child, psi);
    node = newChild.parent();
  }

  // TODO need to figure out if we will _always_ have a single-set domain at
  // some point and there will be no filter with two statements as a leaf for
  // example. I think that is the case because a schedule should give a complete
  // ordering of the statements in its domain but still.

  auto domain = node.get_domain();
  if (!node.isa<isl::schedule_node_leaf>())
    return node;

  assert(domain.isa_set());
  isl::set asyncRange = domain.as_set();
  ISL_DUMP(asyncRange);

  if (asyncRange.is_subset(psi.asyncDeps.range())) {
    unsigned waitNum = [&]() -> unsigned {
      isl::union_map intersectedAsyncDeps =
          psi.asyncDeps.intersect_range(asyncRange);
      ISL_DUMP(intersectedAsyncDeps);
      isl::union_set asyncDomainUset =
          asyncRange.apply(psi.asyncDeps.reverse());
      ISL_DUMP(asyncDomainUset);
      if (!asyncDomainUset.isa_set())
        return 0;
      isl::set asyncDomain = asyncDomainUset.as_set();

      ISL_DUMP(asyncDomain);

      isl::schedule_node nd = node;
      while (true) {
        ISL_DUMP(nd);
        if (!nd.has_parent())
          break;
        ISL_DUMP(nd.domain());
        if (!asyncDomain.to_union_set().intersect(nd.domain()).is_empty() &&
            nd.isa<isl::schedule_node_band>()) {
          auto band = nd.as<isl::schedule_node_band>();
          auto _partial = band.get_partial_schedule();
          ISL_DUMP(_partial);
          // Bands should have been Cconverted to single dim when splitting the
          // pipeilne loops
          if (unsignedFromIslSize(_partial.dim(isl::dim::out)) != 1)
            return 0;
          auto partial = _partial.get_at(0).as_union_map();
          ISL_DUMP(partial);
          isl::union_map applied = intersectedAsyncDeps;
          applied = applied.apply_domain(partial);
          applied = applied.apply_range(partial);
          isl::union_set deltas = applied.deltas();
          ISL_DUMP(deltas);
          assert(deltas.isa_set());
          isl::val v = deltas.as_set().plain_get_val_if_fixed(isl::dim::set, 0);
          assert(v.is_int());
          assert(v.get_den_si() == 1);
          return v.get_num_si() - 1;
          // FIXME this does not always need the (-1), that depends on the
          // sequence schedule in the band
        }
        nd = nd.parent();
      }
      return 0;
    }();

    // TODO add free function
    polymer::AsyncWaitGroupInfo *awg = new polymer::AsyncWaitGroupInfo{waitNum};
    isl::id waitGroupMark =
        isl::id::alloc(node.ctx(), polymer::asyncWaitGroupMark, (void *)awg);
    node = node.insert_mark(waitGroupMark);
  }
  return node;
}
static isl::schedule insertAsyncCopySynchronisation(isl::schedule schedule,
                                                    PrepScheduleInfo &psi) {
  auto root = schedule.get_root();
  return insertAsyncCopySynchronisation(root, psi).get_schedule();
}

static isl::schedule_node splitPipelineLoops(isl::schedule_node node,
                                             PrepScheduleInfo &psi) {
  LLVM_DEBUG(llvm::dbgs() << "splitPipelineLoops for\n"; dumpIslObj(node));
  if (node.isa<isl::schedule_node_band>()) {
    auto band = node.as<isl::schedule_node_band>();

    isl::union_map upToSchedule = isl::manage(
        isl_schedule_node_get_prefix_and_node_schedule_union_map(node.get()));
    isl::union_map partialSchedule =
        isl::union_map::from(band.get_partial_schedule());

    LLVM_DEBUG(llvm::dbgs() << "asyncDeps: "; dumpIslObj(psi.asyncDeps));
    LLVM_DEBUG(llvm::dbgs() << "schedule: "; dumpIslObj(upToSchedule));
    LLVM_DEBUG(llvm::dbgs() << "partialSchedule: ";
               dumpIslObj(partialSchedule));

    bool coincident = are_coincident(psi.asyncDeps, upToSchedule);
    LLVM_DEBUG(llvm::dbgs() << "coincident: " << coincident << "\n");

    if (!coincident) {
      isl::union_map reverseSchedule = partialSchedule.reverse();
      LLVM_DEBUG(llvm::dbgs() << "reverseSchedule: ";
                 dumpIslObj(reverseSchedule));

      isl::union_set domain = node.get_domain();
      LLVM_DEBUG(llvm::dbgs() << "domain: "; dumpIslObj(domain));

      isl::set universe = domain.get_space().universe_set();

      isl::set iterationFull;
      domain.foreach_set([&](isl::set set) {
        LLVM_DEBUG(llvm::dbgs() << "domain set: "; dumpIslObj(set));
        LLVM_DEBUG(llvm::dbgs() << "domain set . schedule: ";
                   dumpIslObj(set.apply(partialSchedule).as_set()));
        isl::set applied = set.apply(partialSchedule).as_set();
        if (iterationFull.is_null()) {
          iterationFull = applied;
        } else {
          iterationFull = iterationFull.unite(applied);
        }
        LLVM_DEBUG(llvm::dbgs() << "iterationFull: ";
                   dumpIslObj(iterationFull));
        return isl::stat::ok();
      });

      isl::union_set asyncDomain =
          domain.intersect(psi.asyncDeps.domain())
              .unite(domain.intersect(psi.asyncDeps.range()));
      LLVM_DEBUG(llvm::dbgs() << "asyncDomain: "; dumpIslObj(asyncDomain));
      isl::set main;
      asyncDomain.foreach_set([&](isl::set set) {
        LLVM_DEBUG(llvm::dbgs() << "domain set: "; dumpIslObj(set));
        LLVM_DEBUG(llvm::dbgs() << "domain set . schedule: ";
                   dumpIslObj(set.apply(partialSchedule).as_set()));
        isl::set applied = set.apply(partialSchedule).as_set();
        if (main.is_null()) {
          main = applied;
        } else {
          main = main.intersect(applied);
        }
        LLVM_DEBUG(llvm::dbgs() << "iterationIntersection: "; dumpIslObj(main));
        return isl::stat::ok();
      });

      // FIXME we would want to split the band in 1-member bands as
      // preprocessing before this transformation
      assert(unsignedFromIslSize(band.n_member()) == 1);

      isl::union_set rest = iterationFull.subtract(main);
      ISL_DUMP(rest);

      auto lexmin = main.lexmin_pw_multi_aff();
      auto lexmax = main.lexmax_pw_multi_aff();
      ISL_DUMP(lexmin);
      ISL_DUMP(lexmax);

      isl::aff identity = isl::manage(isl_multi_aff_identity_on_domain_space(
                                          main.get_space().release()))
                              .get_at(0);

      isl::set prologue, epilogue;
      lexmin.foreach_piece([&](isl::set set, isl::multi_aff ma) {
        assert(unsignedFromIslSize(ma.n_piece()) == 1);
        isl::aff aff = ma.get_at(0).add_dims(isl::dim::in, 1).as_aff();
        ISL_DUMP(identity);
        ISL_DUMP(aff);
        assert(prologue.is_null());
        prologue = identity.lt_set(aff);
        return isl::stat::ok();
      });
      lexmax.foreach_piece([&](isl::set set, isl::multi_aff ma) {
        assert(unsignedFromIslSize(ma.n_piece()) == 1);
        isl::aff aff = ma.get_at(0).add_dims(isl::dim::in, 1).as_aff();
        ISL_DUMP(identity);
        ISL_DUMP(aff);
        assert(epilogue.is_null());
        epilogue = identity.gt_set(aff);
        return isl::stat::ok();
      });
      ISL_DUMP(prologue);
      ISL_DUMP(main);
      ISL_DUMP(epilogue);

      isl::union_set prologueDomain = prologue.apply(reverseSchedule);
      isl::union_set mainDomain = main.apply(reverseSchedule);
      isl::union_set epilogueDomain = epilogue.apply(reverseSchedule);
      ISL_DUMP(prologueDomain);
      ISL_DUMP(mainDomain);
      ISL_DUMP(epilogueDomain);

      bool inSeq = node.parent().isa<isl::schedule_node_filter>();
      node = node.order_after(epilogueDomain);
      ISL_DUMP(node);
      if (node.parent().isa<isl::schedule_node_filter>()) {
        node = node.parent().next_sibling().child(0);
        for (unsigned i = 0;
             i <
             unsignedFromIslSize(node.as<isl::schedule_node_band>().n_member());
             i++)
          node =
              node.as<isl::schedule_node_band>().member_set_ast_loop_unroll(i);
        node = node.parent().previous_sibling().child(0);
      } else {
        llvm_unreachable("???");
      }

      node = node.order_after(mainDomain);
      ISL_DUMP(node);
      for (unsigned i = 0;
           i <
           unsignedFromIslSize(node.as<isl::schedule_node_band>().n_member());
           i++)
        node = node.as<isl::schedule_node_band>().member_set_ast_loop_unroll(i);

      // FIXME need to check if the async deps are fully carried
      bool fullyCarried = true;
      if (fullyCarried) {
        if (inSeq)
          return node;
        else
          return node.parent().parent();
      }
      // FIXME need to call this func recursively on the main band
      llvm_unreachable("...");
    }
  }

  if (!node.has_children())
    return node;

  isl::schedule_node child = node.child(0);
  while (true) {
    child = splitPipelineLoops(child, psi);
    if (child.has_next_sibling())
      child = child.next_sibling();
    else
      break;
  }
  return child.parent();
}
static isl::schedule splitPipelineLoops(isl::schedule schedule,
                                        PrepScheduleInfo &psi) {
  LLVM_DEBUG(llvm::dbgs() << "Splitting pipeline loops\n");
  auto root = schedule.get_root();
  return splitPipelineLoops(root, psi).get_schedule();
}

static isl::schedule prepareScheduleForGPU(isl::schedule schedule,
                                           PrepScheduleInfo &psi) {
  schedule = insertGridPar(schedule);
  schedule = insertArrayExpansion(schedule, psi);
  schedule = splitPipelineLoops(schedule, psi);
  schedule = insertAsyncCopySynchronisation(schedule, psi);
  return schedule;
}

// TODO we can do flow analysis on the operation to check that:
// 1. all live-in memory come from global memory
// 2. all live-out memory locations are in shared memory
// 3. any live-out memory location that does not flow from global memory is
//    either undefined, or flows from a 0-constant (because cuda async copy
//    supports padding with 0s)
//
// then, it is a valid async copy
static bool isValidAsyncCopy(Operation *op) {
  return !op->walk([&](Operation *nested) {
              if (isa<affine::AffineParallelOp, affine::AffineForOp,
                      affine::AffineYieldOp, affine::AffineIfOp,
                      LLVM::BitcastOp, LLVM::IntToPtrOp, LLVM::PtrToIntOp>(
                      nested))
                return WalkResult::advance();
              if (isa<affine::AffineVectorStoreOp, affine::AffineStoreOp,
                      affine::AffineVectorLoadOp, affine::AffineLoadOp>(
                      nested)) {
                affine::MemRefAccess access(nested);
                auto memrefTy = cast<MemRefType>(access.memref.getType());
                if (access.isStore() &&
                    !nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(memrefTy))
                  return WalkResult::interrupt();
                // FIXME Currently assume that memrefs with the default memory
                // space (no memory space) are global, we need to actually check
                // - e.g. they come from kernel arguments
                bool isGlobalMemref =
                    nvgpu::NVGPUDialect::hasGlobalMemoryAddressSpace(
                        memrefTy) ||
                    (!memrefTy.getMemorySpace());
                if (access.isLoad() && !isGlobalMemref)
                  return WalkResult::interrupt();
                return WalkResult::advance();
              }
              return WalkResult::interrupt();
            }).wasInterrupted();
}

template <typename OpTy>
static bool isNormalLoop(OpTy op) {
  // TODO
  return true;
}

// template <typename ParTy>
static std::pair<affine::AffineParallelOp, affine::AffineIfOp>
convertToSingleThreadFor(affine::AffineParallelOp par, RewriterBase &rewriter) {
  assert(par->getNumResults() == 0);
  auto loc = par->getLoc();
  rewriter.setInsertionPoint(par);
  auto newPar =
      cast<affine::AffineParallelOp>(rewriter.cloneWithoutRegions(*par));
  Block *oldParBody = &par->getRegion(0).front();
  rewriter.createBlock(&newPar->getRegion(0), newPar->getRegion(0).begin(),
                       oldParBody->getArgumentTypes(),
                       oldParBody->getArgumentLocs());
  clearBlock(newPar.getBody(), rewriter);
  rewriter.setInsertionPoint(rewriter.create<affine::AffineYieldOp>(loc));

  unsigned dims = par.getNumDims();

  SmallVector<AffineExpr> ifExprs;
  SmallVector<bool> ifEqs;
  SmallVector<Value> ifOperands;
  MLIRContext *ctx = loc.getContext();
  assert(isNormalLoop(newPar));
  for (unsigned i = 0; i < dims; i++) {
    ifOperands.push_back(newPar.getIVs()[i]);
    ifExprs.push_back(getAffineDimExpr(i, ctx));
    ifEqs.push_back(true);
  }
  IntegerSet set = IntegerSet::get(dims, 0, ifExprs, ifEqs);
  auto affineIfOp =
      rewriter.create<affine::AffineIfOp>(loc, TypeRange(), set, ifOperands,
                                          /*hasElse=*/false);
  rewriter.setInsertionPointToStart(affineIfOp.getThenBlock());

  affine::AffineForOp forOp;
  Block *forBody;
  for (unsigned i = 0; i < dims; i++) {
    forOp = rewriter.create<affine::AffineForOp>(
        loc, par.getLowerBoundsOperands(), par.getLowerBoundMap(i),
        par.getUpperBoundsOperands(), par.getUpperBoundMap(i),
        par.getSteps()[i]);
    forBody = &forOp->getRegion(0).front();
    clearBlock(forBody, rewriter);
    rewriter.setInsertionPointToStart(forBody);
    rewriter.replaceAllUsesWith(par.getIVs()[i], forOp.getInductionVar());
    rewriter.setInsertionPoint(rewriter.create<affine::AffineYieldOp>(loc));
  }
  rewriter.eraseOp(oldParBody->getTerminator());
  rewriter.inlineBlockBefore(oldParBody, forBody,
                             forBody->getTerminator()->getIterator(),
                             SmallVector<Value, 3>(dims, nullptr));
  rewriter.eraseOp(par);
  return {newPar, affineIfOp};
}

template <bool UseSingleThreadCopy = false>
static void convertToAsync(polymer::IslScop &scop,
                           polymer::IslScop::ApplyScheduleRes &applied) {
  Operation *root = applied.newFunc;

  SmallVector<memref::AllocaOp> shmemAllocas;
  llvm::SmallSetVector<MemrefValue, 4> shmemMemrefs;
  root->walk([&](memref::AllocaOp alloca) {
    if (nvgpu::NVGPUDialect::isSharedMemoryAddressSpace(
            alloca.getType().getMemorySpace())) {
      shmemAllocas.push_back(alloca);
      shmemMemrefs.insert(cast<MemrefValue>(alloca.getMemref()));
    }
  });

  auto isGlobalMemref = [&](MemrefValue m) {
    return !nvgpu::NVGPUDialect::isSharedMemoryAddressSpace(
               m.getType().getMemorySpace()) &&
           m.getParentRegion()->getParentOp() == root;
  };

  llvm::SmallSetVector<Operation *, 4> syncsInserted;

  auto getBlockPar = [&](Operation *op) -> Operation * {
    while ((op = op->getParentOp()))
      if (isBlockPar(op))
        return op;
    return nullptr;
  };

  DenseSet<Operation *> asyncCopyStmts;

  SmallVector<Copy> copies;
  root->walk([&](Operation *op) {
    auto par = isAffineBlockPar(op);
    if (!par || !op->getAttr("polymer.stmt.async.copy"))
      return WalkResult::advance();

    asyncCopyStmts.insert(par);
    par->walk([&](Operation *opInst) {
      if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(opInst)) {
        if (!nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(
                cast<MemRefType>(writeOp.getMemRef().getType())))
          return;
        if (auto store = dyn_cast<affine::AffineStoreOp>(opInst)) {
          llvm_unreachable("?");
        } else if (auto store = dyn_cast<affine::AffineVectorStoreOp>(opInst)) {
          if (!store.getValueToStore().hasOneUse())
            return;
          auto load = dyn_cast_or_null<affine::AffineReadOpInterface>(
              store.getValueToStore().getDefiningOp());
          if (!load)
            return;
          if (isGlobalMemref(cast<MemrefValue>(load.getMemRef())))
            copies.push_back({load, store, load, store});
          return;
        }
      } else {
        return;
      }
    });
    return WalkResult::skip();
  });
  // TODO check that we have converted all load/stores in the block par. We
  // actually don't need to convert _everything_, we could keep some of the
  // accesses sync. However, in that case the convert to single thread for
  // function needs to be smarter and only put the async copies on a single
  // thread.

  for (auto copy : copies) {

    Location loc = copy.load.getLoc();

    Operation *blockPar = getBlockPar(copy.store);
    assert(blockPar->getAttr("gpu.par.block"));

    IRRewriter rewriter(copy.storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Found copy\n");

    auto vty = dyn_cast<VectorType>(copy.load.getValue().getType());
    if (!vty) {
      LLVM_DEBUG(llvm::dbgs() << "Need a vector load/store\n");
      continue;
    }
    if (vty.getShape().size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "Need 1d vector load/store\n");
      continue;
    }
    if (vty.getElementType() != rewriter.getI8Type()) {
      LLVM_DEBUG(llvm::dbgs() << "Need i8 vector\n");
      continue;
    }
    unsigned copySize = vty.getShape()[0];
    if (!(copySize == 4 || copySize == 8 || copySize == 16)) {
      LLVM_DEBUG(llvm::dbgs() << "Need 4/8/16 size\n");
      continue;
    }

    // TODO need to align stuff to 4, 8, or 16-byte chunks
    rewriter.setInsertionPoint(copy.store);
    SmallVector<Value> storeIdxs = {computeMap<affine::AffineApplyOp>(
        rewriter, copy.store.getLoc(), copy.store.getAffineMap(),
        copy.store.getMapOperands())};
    SmallVector<Value> loadIdxs = {computeMap<affine::AffineApplyOp>(
        rewriter, copy.load.getLoc(), copy.load.getAffineMap(),
        copy.load.getMapOperands())};
    rewriter.create<nvgpu::DeviceAsyncCopyOp>(
        loc, copy.store.getMemRef(), storeIdxs, copy.load.getMemRef(), loadIdxs,
        rewriter.getIndexAttr(copySize), nullptr, nullptr);
    rewriter.eraseOp(copy.store);
    rewriter.eraseOp(copy.load);
  }

  IRRewriter rewriter(root->getContext());
  for (Operation *stmtOp : asyncCopyStmts) {
    if constexpr (UseSingleThreadCopy) {
      if (auto affinePar = dyn_cast<affine::AffineParallelOp>(stmtOp)) {
        auto [newAffinePar, affineIf] =
            convertToSingleThreadFor(affinePar, rewriter);
        rewriter.setInsertionPoint(affineIf.getThenBlock()->getTerminator());
        rewriter.create<NVVM::CpAsyncCommitGroupOp>(affineIf->getLoc());
      } else {
        llvm_unreachable("scf.for not supported yet");
      }
    } else {
      rewriter.setInsertionPoint(getSingleBlock(stmtOp)->getTerminator());
      rewriter.create<NVVM::CpAsyncCommitGroupOp>(stmtOp->getLoc());
    }
  }

  // TODO temporary hack - maybe we should have a custom codegen callback at
  // the points because it would be nice to keep the polyhedral logic
  // (IslScop.cc) separate from the GPU intrinsic specifics. Currently IslScop
  // puts a NVVM::CpAsyncWaitGroupOp at the place we instructed it to which is
  // outside the block parallel statements. Those have memory effects so
  // unditribute loops fails as it doesnt know what to do with them.
  SmallVector<NVVM::CpAsyncWaitGroupOp> waitOps;
  root->walk(
      [&](NVVM::CpAsyncWaitGroupOp waitOp) { waitOps.push_back(waitOp); });
  for (auto waitOp : waitOps) {
    affine::AffineParallelOp par;
    Operation *op = waitOp->getNextNode();
    while (op && !(par = isAffineBlockPar(op)))
      op = op->getNextNode();
    assert(par);
    waitOp->moveBefore(&par.getBody()->front());
  }
}

void transform(LLVM::LLVMFuncOp f) {
  using namespace polymer;
  std::unique_ptr<polymer::IslScop> scop = polymer::createIslFromFuncOp(f);
  scop->buildSchedule();
  LLVM_DEBUG({
    llvm::dbgs() << "Schedule:\n";
    isl_schedule_dump(scop->getScheduleTree().get());
    llvm::dbgs() << "Accesses:\n";
    scop->dumpAccesses(llvm::dbgs());
  });

  scop->rescopeStatements(isAffineBlockPar, isValidAsyncCopy);

  scop->buildSchedule();
  LLVM_DEBUG({
    llvm::dbgs() << "Schedule:\n";
    isl_schedule_dump(scop->getScheduleTree().get());
    llvm::dbgs() << "Accesses:\n";
    scop->dumpAccesses(llvm::dbgs());
  });

  ppcg_scop *ps = computeDeps(*scop);
  isl_schedule_constraints *sc = construct_schedule_constraints(ps, *scop);
  if (!sc) {
    LLVM_DEBUG(llvm::dbgs() << "Computing schedule constraints failed\n");
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Schedule constraints:\n";
    isl_schedule_constraints_dump(sc);
  });

  // FIXME TODO this is a temporary work around to avoid using the scc
  // clustering part of the isl_scheduler. The incremental scheduling (described
  // in Scheduling for PPCG) creates clusters from the isl_sched_graph and we do
  // not properly convey the arrays and live ranges etc information when
  // creating new clusters and when we merge them together. Once we implemnet
  // that this can be turned off.
  isl_stat r;
  r = isl_options_set_schedule_whole_component(scop->getIslCtx(), 1);
  assert(r == isl_stat_ok);
  r = isl_options_set_schedule_serialize_sccs(scop->getIslCtx(), 1);
  assert(r == isl_stat_ok);

  PrepScheduleInfo psi;
  psi.unallocatedArrays = psi.allArrays =
      isl::manage(isl_schedule_constraints_get_array_size(sc)).domain();
  psi.allocatedArrays =
      isl::manage(isl_union_set_empty(psi.allArrays.get_space().release()));
  psi.lrs = isl::manage(isl_schedule_constraints_get_live_range_span(sc));
  psi.asyncDeps = isl::manage_copy(ps->dep_async);

  isl::schedule newSchedule;
  if (getenv("GPU_AFFINE_OPT_ROUNDTRIP")) {
    newSchedule = scop->getScheduleTree();
  } else {
    newSchedule = isl::manage(isl_schedule_constraints_compute_schedule(
        isl_schedule_constraints_copy(sc)));
    LLVM_DEBUG({
      llvm::dbgs() << "New Schedule:\n";
      isl_schedule_dump(newSchedule.get());
    });

    // TODO we want to collect statistics and emit remarks about this kind of
    // stuff
    if (newSchedule.is_null())
      return;

    newSchedulesComputed++;

    // TODO add a round-trip mode where we codegen the original schedule

    newSchedule = prepareScheduleForGPU(newSchedule, psi);

    LLVM_DEBUG({
      llvm::dbgs() << "New Schedule Prepared for GPU:\n";
      isl_schedule_dump(newSchedule.get());
    });
  }

  /// XXX do not hardcode 32
  auto applied = scop->applySchedule(newSchedule.copy(), psi.lrs.copy(), f, 32);
// FIXME we are getting some double frees/invalid read/writes due to these...
#if 0
  isl_schedule_constraints_free(sc);
#endif
  auto g = cast<LLVM::LLVMFuncOp>(applied.newFunc);
  LLVM_DEBUG({
    llvm::dbgs() << "New func:\n";
    llvm::dbgs() << *g << "\n";
  });
  if (!getenv("GPU_AFFINE_OPT_DISABLE_CONVERT_TO_ASYNC"))
    convertToAsync(*scop, applied);
  LLVM_DEBUG({
    llvm::dbgs() << "Converted to async:\n";
    llvm::dbgs() << *g << "\n";
  });

  scop->cleanup(g);

  if (g) {
    for (auto &b : llvm::make_early_inc_range(f.getRegion().getBlocks()))
      b.erase();

    f.getRegion().getBlocks().splice(f.getRegion().getBlocks().begin(),
                                     g.getRegion().getBlocks());
    g->erase();
  }
}

} // namespace affine_opt
} // namespace gpu
} // namespace mlir

namespace {

struct RegisterAllocaReduce : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp ao,
                                PatternRewriter &rewriter) const override {
    auto mt = ao.getType();
    if (mt.getMemorySpaceAsInt() == loop_distribute::registerMemorySpace)
      return rewriter.notifyMatchFailure(ao,
                                         "Not register memory address space");

    SmallVector<affine::AffineLoadOp> loads;
    SmallVector<affine::AffineStoreOp> stores;

    for (auto &use : ao.getResult().getUses()) {
      Operation *op = use.getOwner();
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        if (loadOp.getMemRef() == ao.getMemref()) {
          loads.push_back(loadOp);
        } else {
          return rewriter.notifyMatchFailure(loadOp,
                                             "alloca not used as the memref?");
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (storeOp.getMemRef() == ao.getMemref()) {
          stores.push_back(storeOp);
        } else {
          return rewriter.notifyMatchFailure(storeOp,
                                             "alloca not used as the memref?");
        }
      } else {
        return rewriter.notifyMatchFailure(op, "non access use");
      }
    }

    affine::AffineParallelOp blockPar = nullptr;
    ao->getParentOp()->walk([&](Operation *op) {
      if (auto par = gpu::affine_opt::isAffineBlockPar(op)) {
        assert(!blockPar);
        blockPar = par;
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    assert(blockPar);

    rewriter.setInsertionPoint(&blockPar.getBody()->front());
    auto newAo = rewriter.create<memref::AllocaOp>(
        ao.getLoc(), MemRefType::get({}, mt.getElementType()));
    auto newMemref = newAo.getMemref();

    for (auto loadOp : loads) {
      rewriter.setInsertionPoint(loadOp);
      rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, newMemref,
                                                  ValueRange{});
    }
    for (auto storeOp : stores) {
      rewriter.setInsertionPoint(storeOp);
      rewriter.replaceOpWithNewOp<memref::StoreOp>(
          storeOp, storeOp.getValueToStore(), newMemref, ValueRange{});
    }

    rewriter.replaceOp(ao, newAo->getResults());

    return success();
  }
};

struct SharedMemrefAllocaToGlobal : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp ao,
                                PatternRewriter &rewriter) const override {
    auto mt = ao.getType();
    if (!nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(mt))
      return rewriter.notifyMatchFailure(ao, "Not shared memory address space");

    auto type = MemRefType::get(mt.getShape(), mt.getElementType(), {},
                                /* memspace */ 3);
    auto loc = ao->getLoc();

    unsigned counter = 0;
    SmallString<20> name = SymbolTable::generateSymbolName<20>(

        "shared_mem",
        [&](llvm::StringRef candidate) {
          return SymbolTable::lookupNearestSymbolFrom(
                     ao, StringAttr::get(ao.getContext(), candidate)) !=
                 nullptr;
        },
        counter);

    auto mod = ao->getParentOfType<gpu::GPUModuleOp>();
    if (!mod)
      return rewriter.notifyMatchFailure(ao, "Could not find gpu module");

    rewriter.setInsertionPointToStart(mod.getBody());

    auto initialValue = rewriter.getUnitAttr();
    rewriter.create<memref::GlobalOp>(
        loc, rewriter.getStringAttr(name),
        /* sym_visibility */ mlir::StringAttr(), mlir::TypeAttr::get(type),
        initialValue, mlir::UnitAttr(), /* alignment */ nullptr);
    rewriter.setInsertionPoint(ao);
    auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(loc, type, name);

    rewriter.replaceOp(ao, getGlobalOp->getResults());

    return success();
  }
};

} // namespace

namespace mlir {
void populateRemoveIVPatterns(RewritePatternSet &patterns);
}

// TODO we should not loop distribute non-affine ops, as we cannot reason about
// them.
//
// e.g.
//
// affine.parallel {
//   scf.for {
//     A
//     barrier
//     B
//   }
// } {gpu.par.block}
//
// In this case we should _not_ interchange and distribute as that would leave
// us with non-affine code (when considering block parallels as statements.

struct SingleRegion {
  Block::iterator begin, end;
};

struct GPUAffineOptPass : impl::GPUAffineOptPassBase<GPUAffineOptPass> {
  using impl::GPUAffineOptPassBase<GPUAffineOptPass>::GPUAffineOptPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    auto removeIVs = [&](Operation *op) {
      RewritePatternSet patterns(context);
      populateRemoveIVPatterns(patterns);
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    auto expandSubView = [&](Operation *op) {
      RewritePatternSet patterns(context);
      memref::populateExpandStridedMetadataPatterns(patterns);
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    auto sharedMemrefAllocaToGlobal = [&](Operation *op) {
      // TODO need to forward register stores to loads.
      // Check if the llvm mem2reg does that for us
      RewritePatternSet patterns(context);
      patterns.insert<SharedMemrefAllocaToGlobal>(context);
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    auto registerAllocaReduce = [&](Operation *op) {
      // TODO need to forward register stores to loads.
      // Check if the llvm mem2reg does that for us
      RewritePatternSet patterns(context);
      patterns.insert<RegisterAllocaReduce>(context);
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    auto gpuify = [&](Operation *op) {
      if (mlir::undistributeLoops(op).failed())
        return failure();
      IRRewriter rewriter(context);
      op->walk([&](affine::AffineBarrierOp barrier) {
        rewriter.setInsertionPoint(barrier);
        rewriter.replaceOpWithNewOp<NVVM::Barrier0Op>(barrier);
      });
      op->walk([&](memref::AllocaOp alloca) {});
      return success();
    };
    auto lowerAffine = [&](Operation *op) {
      assert(!op->walk([&](affine::AffineScopeOp op) {
                  return WalkResult::interrupt();
                }).wasInterrupted());
      RewritePatternSet patterns(&getContext());
      populateAffineToStdConversionPatterns(patterns);
      populateAffineToVectorConversionPatterns(patterns);
      affine::populateAffineExpandIndexOpsPatterns(patterns);
      ConversionTarget target(getContext());
      target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                             scf::SCFDialect, vector::VectorDialect>();
      target.addDynamicallyLegalDialect<affine::AffineDialect>(
          [&](Operation *op) { return isa<affine::AffineScopeOp>(op); });
      if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    };
    auto lowerAccesses = [&](Operation *op) {
      auto dl = dataLayoutAnalysis.getAtOrAbove(op);
      LowerToLLVMOptions options(&getContext(),
                                 dataLayoutAnalysis.getAtOrAbove(op));
      // TODO need to tweak options.indexBitwidth in some cases? consult
      // LowerGpuOpsToNVVMOpsPass
      options.useBarePtrCallConv = true;
      unsigned indexBitwidth = 32;
      if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
        options.overrideIndexBitwidth(indexBitwidth);

      // TODO do we need this?
      // options.dataLayout = llvm::DataLayout(this->dataLayout);

      LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);

      RewritePatternSet patterns(context);
      converter.addConversion([&](MemRefType type) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(type.getContext(),
                                          type.getMemorySpaceAsInt());
      });
      populateGPULoweringPatterns(patterns, converter);
      arith::populateArithToLLVMConversionPatterns(converter, patterns);
      ConversionTarget target(getContext());
      target.addIllegalDialect<affine::AffineDialect>();
      target.addIllegalDialect<memref::MemRefDialect>();
      target.addIllegalDialect<vector::VectorDialect>();
      target.addIllegalDialect<nvgpu::NVGPUDialect>();
      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addLegalDialect<NVVM::NVVMDialect>();
      if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    };
    auto canonicalize = [&](Operation *op) {
      PassManager pm(&getContext());
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());
      if (failed(pm.run(op))) {
        signalPassFailure();
        return;
      }
    };
    auto unrollLoops = [&](Operation *op) {
      bool changed;
      do {
        changed = op->walk([&](affine::AffineForOp forOp) {
                      if (affine::loopUnrollFull(forOp).succeeded())
                        return WalkResult::interrupt();
                      return WalkResult::advance();
                    }).wasInterrupted();
      } while (changed);
    };
    op->walk([&](mlir::gpu::GPUModuleOp gpuModule) {
      const mlir::DataLayoutAnalysis dl(gpuModule);
      gpuModule->walk([&](mlir::LLVM::LLVMFuncOp func) {
        if (func->getAttr("gpu.par.kernel")) {
          numKernels++;
          LLVM_DEBUG(DBGS << "Before opt:\n" << func << "\n");
          removeIVs(func);
          LLVM_DEBUG(DBGS << "Removed IVs:\n" << func << "\n");
          (void)mlir::convertLLVMToAffineAccess(func, dl, false);
          LLVM_DEBUG(DBGS << "To Affine:\n" << func << "\n");
          (void)distributeParallelLoops(func, "distribute.mincut",
                                        &getContext());
          LLVM_DEBUG(DBGS << "Distributed:\n" << func << "\n");
          canonicalize(func);
          LLVM_DEBUG(DBGS << "Canonicalized:\n" << func << "\n");
          if (!getenv("GPU_AFFINE_OPT_ROUNDTRIP"))
            mlir::gpu::affine_opt::transform(func);
          LLVM_DEBUG(DBGS << "After opt:\n" << func << "\n");
          if (!getenv("GPU_AFFINE_OPT_DISABLE_UNROLL_LOOPS"))
            unrollLoops(func);
          LLVM_DEBUG(DBGS << "After unroll loops:\n" << func << "\n");
          sharedMemrefAllocaToGlobal(func);
          LLVM_DEBUG(DBGS << "After shmem to alloca:\n" << func << "\n");
          (void)gpuify(func);
          LLVM_DEBUG(DBGS << "After gpuify:\n" << func << "\n");
          expandSubView(func);
          LLVM_DEBUG(DBGS << "After expand subview:\n" << func << "\n");
          registerAllocaReduce(func);
          LLVM_DEBUG(DBGS << "After rar:\n" << func << "\n");
          lowerAffine(func);
          LLVM_DEBUG(DBGS << "After lower affine:\n" << func << "\n");
          lowerAccesses(func);
          LLVM_DEBUG(DBGS << "After lower accesses:\n" << func << "\n");
          canonicalize(func);
          LLVM_DEBUG(DBGS << "Canonicalized:\n" << func << "\n");
        }
      });
      // This is for lowering the memref.globals
      lowerAccesses(gpuModule);
      LLVM_DEBUG(DBGS << "After gpu module lower accesses:" << gpuModule
                      << "\n");
    });
  }
};

std::unique_ptr<Pass> mlir::createGPUAffineOptPass() {
  return std::make_unique<GPUAffineOptPass>();
}
