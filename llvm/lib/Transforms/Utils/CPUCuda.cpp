//===-- CPUCuda.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.

// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CPUCuda.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#include <queue>
#include <vector>
#include <string>
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "cpucudapass"

// TODO handle lifetimes which get split into different subkernels if needed?

// TODO split the pass in two parts - before and after replacing the dim3 getter
// calls with arguments, and optimise the code in between

// TODO I think we should be passing all dim3's around using pointers - it might
// be the most ABI stable solution

// TODO handle dynamic shared mem


namespace {

using std::vector;
using std::set;
using std::map;
using std::queue;

typedef vector<BasicBlock *> BBVector;
typedef queue<BasicBlock *> BBQueue;
typedef set<BasicBlock *> BBSet;

typedef vector<Value *> ValueVector;
typedef set<Value *> ValueSet;
typedef set<Instruction *> InstSet;
typedef set<GlobalVariable *> GlobalVarSet;
typedef vector<Type *> TypeVector;
typedef vector<Instruction *> InstVector;
typedef vector<Argument *> ArgVector;
typedef vector<GlobalVariable *> GlobalVarVector;

typedef int SubkernelIdType;
typedef int BBIdType;

const int MAX_CUDA_THREADS = 1024;

const TargetTransformInfo::TargetCostKind TARGET_COST_KIND = TargetTransformInfo::TCK_RecipThroughput;

const vector<std::string> Dim3Names = {
  "gridDim",
  "blockIdx",
  "blockDim",
  "threadIdx"
};

const map<std::string, vector<std::string>> Dim3GetterIntrinsicNames = {
  {Dim3Names[0], {
      "llvm.nvvm.read.ptx.sreg.nctaid.x",
      "llvm.nvvm.read.ptx.sreg.nctaid.y",
      "llvm.nvvm.read.ptx.sreg.nctaid.z",
    }
  },
  {Dim3Names[1], {
      "llvm.nvvm.read.ptx.sreg.ctaid.x",
      "llvm.nvvm.read.ptx.sreg.ctaid.y",
      "llvm.nvvm.read.ptx.sreg.ctaid.z",
    }
  },
  {Dim3Names[2], {
      "llvm.nvvm.read.ptx.sreg.ntid.x",
      "llvm.nvvm.read.ptx.sreg.ntid.y",
      "llvm.nvvm.read.ptx.sreg.ntid.z",
    }
  },
  {Dim3Names[3], {
      "llvm.nvvm.read.ptx.sreg.tid.x",
      "llvm.nvvm.read.ptx.sreg.tid.y",
      "llvm.nvvm.read.ptx.sreg.tid.z",
    }
  },
};

struct UsedValVars {
  ValueSet usedVals;
  ValueSet definedLater;
  GlobalVarSet usedSharedVars;
};

struct {
  // Whether to use self contained kernel (with included loops for blocks in
  // grid)
  bool UseSelfContainedKernel = false;
  // Do we use a single or triple thread loop NOTE turns out using a linear loop
  // reduces performance by about a factor of 2
  bool SingleDimThreadLoop = false;
  // Do we use malloc or alloca for the preserved data array - I think we
  // might actually overflow the stack with alloca so should be malloc
  // TODO Should we malloc the shared data as well?
  bool MallocPreservedDataArray = true;
  // Do we allocate for all 1024 threads or only as many as we have run the
  // kernel with
  bool DynamicPreservedDataArray = false;
  // Manually inline the subkernels in the driver function - the optimisations
  // following this pass should do it anyways if it is deemed profitable
  bool InlineSubkernels = true;
  // Actually they always have to be inlined because otherwise we would get
  // undefined references when linking, so not really an option currently
  bool InlineDim3Fs = true;
} Options;

} // anonymous namespace

namespace llvm {

class FunctionTransformer {
public:
  Module *M;
  TargetTransformInfo *TTI;

  Function *F;
  Function *OriginalF;

  struct {
    Function *ConstructorF;
    Function *Getterx;
    Function *Gettery;
    Function *Getterz;
    Function *Dim3ToArg;
  } Dim3Fs;

  std::set<BasicBlock *> BlocksAfterBarriers;

  set<SubkernelIdType> SubkernelIds;
  map<SubkernelIdType, BBVector> SubkernelBBs;
  map<SubkernelIdType, Function *> SubkernelFs;
  map<SubkernelIdType, map<SubkernelIdType, ValueVector>> SubkernelUsedVals;
  map<SubkernelIdType, map<BasicBlock *, BBIdType>> SubkernelBBIds;
  map<BBIdType, BasicBlock *> OriginalFunBBs;
  SubkernelIdType EntrySubkernel;

  map<SubkernelIdType, map<Value *, int>> IndexInCombinedDataType;
  map<SubkernelIdType, ValueVector> CombinedUsedVals;
  StructType *CombinedDataType;

  map<GlobalVariable *, int> IndexInCombinedSharedVarsDataType;
  GlobalVarVector CombinedSharedVars;
  StructType *SharedVarsDataType;

  GlobalVariable *DynamicSharedVar;

  Function *DriverF;
  Function *WrapperF;
  Function *SelfContainedF;

  // Label type for which BB id we should continue from after we return or we
  // have come from
  IntegerType *LLVMBBIdType;
  IntegerType *LLVMSubkernelIdType;
  StructType *SubkernelReturnType;
  IntegerType *GepIndexType;
  IntegerType *I32Type;
  IntegerType *Dim3FieldType;
  IntegerType *SizeTType;
  Type *Dim3Type;
  Type *Dim3PtrType;

  void splitBlocksAroundBarriers(Function &F);
  bool blockIsAfterBarrier(BasicBlock *BB);
  bool blockIsAfterBarrier(SubkernelIdType SK, BasicBlock *BB);
  void _findSubkernelBBs(BasicBlock *BB, BBSet &visited);
  void findSubkernelUsedValsDom();
  SubkernelIdType findSubkernelFromBB(BBIdType BB);
  void createSubkernelFunctionClones();
  set<SubkernelIdType> getSubkernelSuccessors(SubkernelIdType SK);
  Type *getSubkernelReturnDataFieldType(SubkernelIdType FromSK, SubkernelIdType SuccSK);
  StructType *getSubkernelsReturnType();
  void assignBBIds();
  TypeVector getSubkernelParams(SubkernelIdType SK);
  vector<StringRef> getSubkernelParamNames(SubkernelIdType SK);
  void transformSubkernels(SubkernelIdType SK);
  void findSubkernelBBs();
  void findSharedVars();
  void handleAllocas(Function *F);
  void createSubkernels();
  Type *getCombinedDataType();
  int getValIndexInCombinedDataType(SubkernelIdType SK, Value *Val);
  void sortValueVector(SubkernelIdType SK, ValueVector &VV, map<Value *, int> &Indices);
  void removeReferencesInPhi(const BBVector &BBsToRemove);
  void createDriverFunction();
  void createSelfContainedFunction();
  void optimizeUsedVals();
  void indexUsedVals();
  void createWrapperFunction();
  void replaceDim3Usages();
  void getDim3StructType();
  void getDim3Fs();
  ValueVector convertDim3ToArgs(Value *D, Instruction *After);
  void cleanup();

  FunctionTransformer(Module *M, Function *F, TargetTransformInfo *TTI);

};

}

// TODO fix this ugly hack.

// Since we do not know how the dim3 structure will be represented in LLVM IR
// (It might depend on architecture or OS? I am not sure) this is a function
// which takes 3 arguments x, y, z and returns a dim3 structure

// As of now it works for amd64 on linux

void maybeAssignFunctionWithNameTo(Module *M, Function *&Assign, std::string String) {
  Assign = nullptr;
  for (auto &F : *M)
    if (F.getName() == String) {
      Assign = &F;
      break;
    }
}

void assignFunctionWithNameTo(Module *M, Function *&Assign, std::string String) {
  maybeAssignFunctionWithNameTo(M, Assign, String);
  assert(Assign);
}

bool callIsBarrier(CallInst *callInst) {
  if (Function *calledFunction = callInst->getCalledFunction()) {
    return calledFunction->getName() == "llvm.nvvm.barrier0";
  } else {
    return false;
  }
}

bool instrIsBarrier(Instruction *I) {
  auto &instruction = *I;
  if (CallInst *callInst = dyn_cast<CallInst>(&instruction)) {
    if (callIsBarrier(callInst)) {
      return true;
    }
  }
  return false;
}

void FunctionTransformer::splitBlocksAroundBarriers(Function &F) {
  while ([&]() -> bool {
    for (auto &bb : F) {
      for (auto &instruction : bb) {
        if (instrIsBarrier(&instruction)) {
          auto newbb = SplitBlock(&bb, &instruction);
          BlocksAfterBarriers.insert(newbb);
          instruction.eraseFromParent();
          return true;
        }
      }
    }
    return false;
  }());
}

bool FunctionTransformer::blockIsAfterBarrier(BasicBlock *BB) {
  return BlocksAfterBarriers.find(BB) != BlocksAfterBarriers.end();
}

bool FunctionTransformer::blockIsAfterBarrier(SubkernelIdType SK, BasicBlock *BB) {
  BasicBlock *OriginalFunBB = OriginalFunBBs[SubkernelBBIds[SK][BB]];
  return BlocksAfterBarriers.find(OriginalFunBB) != BlocksAfterBarriers.end();
}

template <class T>
bool in_vector(std::vector<T> &v, T key) {
  return v.end() != std::find(v.begin(), v.end(), key);
}

template <class T>
bool in_set(std::set<T> &v, T key) {
  return v.end() != std::find(v.begin(), v.end(), key);
}

template <class T>
void print_container(T &c) {
  for (auto &el : c) {
    errs() << el << ", ";
  }
  errs() << "\n";
}

ValueVector findValuesUsedInAndDefinedOutsideBBs(Function *f, BBVector bbs) {
  ValueVector defined_outside;
  ValueVector used_inside;
  // Function arguments
  for (auto &arg: f->args()) {
    defined_outside.push_back(static_cast<Value *>(&arg));
  }
  // Values in basic blocks
  for (auto &bb : *f) {
    if (in_vector(bbs, &bb)) {
      for (auto &inst : bb) {
        for (Use &u : inst.operands()) {
          used_inside.push_back(u.get());
        }
      }
    } else {
      for (auto &inst : bb) {
        defined_outside.push_back(static_cast<Value *>(&inst));
      }
    }
  }
  std::sort(defined_outside.begin(), defined_outside.end());
  std::sort(used_inside.begin(), used_inside.end());
  ValueVector intersection;
  std::set_intersection(defined_outside.begin(), defined_outside.end(),
                        used_inside.begin(), used_inside.end(),
                        std::back_inserter(intersection));
  return intersection;
}

BasicBlock *convertBasicBlock(BasicBlock *BB, Function *FOld, Function *FNew) {
  unsigned id = 0;
  for (auto it = FOld->begin(); it != FOld->end(); ++it, ++id) {
    BasicBlock *bb = &(*it);
    if (bb == BB)
      break;
  }
  assert(id != FOld->size());
  unsigned id2 = 0;
  for (auto it = FNew->begin(); it != FNew->end(); ++it, ++id2) {
    BasicBlock *bb = &(*it);
    if (id == id2)
      return bb;
  }
  assert(false && "A BB with the same index must exist in the new function as well");
  return NULL;
}

// Converts a list of bbs to the corresponding list of bbs in the newly cloned
// function
//
// TODO this depends on the representation of blocks in a function - is
// there a better way to do it?
BBVector convert_bb_vector(BBVector &vold, Function *fold, Function *fnew) {
  BBVector vnew;
  for (auto BB : vold)
    vnew.push_back(convertBasicBlock(BB, fold, fnew));
  return vnew;
}

// TODO: additionally to the BB id to continue from, will probably need to
// return value to indicate the type of barrier that was hit

// NOTE actually the type of barrier we hit can be figured out from the returned
// BB id which we came from
struct TransformTerminator : public InstVisitor<TransformTerminator> {

  SubkernelIdType SK;
  LLVMContext &C;
  FunctionTransformer *Pass;

  Type *StructIndexType;

  TransformTerminator(SubkernelIdType SK, FunctionTransformer *Pass):
      SK(SK),
      C(Pass->M->getContext()),
      Pass(Pass) {

    StructIndexType = Type::getInt32Ty(C);
  }

  BasicBlock *createNewRetBB(BasicBlock *BBFrom, SubkernelIdType ContSK) {
    Function *F = Pass->SubkernelFs[SK];

    BasicBlock *NewBB = BasicBlock::Create(F->getContext(), "generated_ret_block", F);
    Constant *FromLabel = llvm::ConstantInt::get(Pass->LLVMBBIdType,
                                                 /* value */ Pass->SubkernelBBIds[SK][BBFrom],
                                                 /* isSigned */ true);
    Constant *ContLabel = llvm::ConstantInt::get(Pass->LLVMSubkernelIdType,
                                                 /* value */ ContSK,
                                                 /* isSigned */ true);
    Constant *ReturnVal = ConstantStruct::get(Pass->getSubkernelsReturnType(), {FromLabel, ContLabel});

    ReturnInst::Create(C, ReturnVal, NewBB);

    return NewBB;
  }

  void visitReturnInst(ReturnInst &I) {
    LLVM_DEBUG(dbgs() << "Transforming ReturnInst " << I << "\n");
    Value *ReturnStructVal = static_cast<Value *>(UndefValue::get(Pass->SubkernelReturnType));
    Constant *ContLabel = llvm::ConstantInt::get(Pass->LLVMSubkernelIdType,
                                                 /* value */ -1,
                                                 /* isSigned */ true);
    ReturnStructVal = InsertValueInst::Create(ReturnStructVal, ContLabel, ArrayRef<unsigned>({1}), "", I.getParent());
    ReturnInst::Create(I.getContext(), ReturnStructVal, I.getParent());
    I.eraseFromParent();
  }

  void visitBranchInst(BranchInst &I) {
    LLVM_DEBUG(dbgs() << "Transforming BranchInst " << I << "\n");
    Function *F = I.getFunction();
    assert(F == Pass->SubkernelFs[SK]);
    for (unsigned i = 0; i < I.getNumSuccessors(); i++) {
      BasicBlock *succ = I.getSuccessor(i);
      if (Pass->blockIsAfterBarrier(SK, succ)) {
        // If the succesor block is after a barrier, the branch instruction that
        // jumps to it should be an unconditional one generated when we split
        // the blocks around the barriers
        assert(I.getNumSuccessors() == 1);
        auto SuccSK = Pass->findSubkernelFromBB(Pass->SubkernelBBIds[SK][succ]);
        assert(SuccSK != -1 && "There should always be a subkernel from a BB after a barrier");
        BasicBlock *RetBB = createNewRetBB(I.getParent(), SuccSK);
        I.setSuccessor(i, RetBB);
      }
    }
  }

  // Below terminators do not need to be handled - only unconditinal branches
  // terminate ends of kernels before barriers, and only returns terminate the
  // final subkernel of a kernel

  //void visitSwitchInst(SwitchInst &I) {
  //  assert(false && "Not yet implemented");
  //}

  //void visitIndirectBrInst(IndirectBrInst &I) {
  //  assert(false && "Not yet implemented");
  //}

  //void visitResumeInst(ResumeInst &I) {
  //  assert(false && "Not yet implemented");
  //}

  //void visitUnreachableInst(UnreachableInst &I) {
  //  assert(false && "Not yet implemented");
  //}

  //void visitCleanupReturnInst(CleanupReturnInst &I) {
  //  assert(false && "Not yet implemented");
  //}

  //void visitCatchReturnInst(CatchReturnInst &I) {
  //  assert(false && "Not yet implemented");
  //}

  //void visitCatchSwitchInst(CatchSwitchInst &I) {
  //  assert(false && "Not yet implemented");
  //}

};

void FunctionTransformer::findSubkernelBBs() {
  std::set<BasicBlock *> visited;
  _findSubkernelBBs(&F->getEntryBlock(), visited);

  for (auto SK : SubkernelIds)
    if (&F->getEntryBlock() == SubkernelBBs[SK][0])
      EntrySubkernel = SK;
}

void FunctionTransformer::_findSubkernelBBs(BasicBlock *BB, BBSet &visited) {
  if (visited.find(BB) != visited.end())
    return;
  visited.insert(BB);

  // BBs which are reachable without crossing a barrier from the current BB
  BBVector func_bbs;

  std::queue<BasicBlock *> to_walk;
  to_walk.push(BB);

  while (!to_walk.empty()) {
    auto bb = to_walk.front();
    to_walk.pop();

    if (in_vector(func_bbs, bb))
      continue;
    func_bbs.push_back(bb);

    for (auto succBB : successors(bb)) {
      if (blockIsAfterBarrier(succBB)) {
        // We crossed a barrier: start a new search at that successor
        _findSubkernelBBs(succBB, visited);
      } else {
        to_walk.push(succBB);
      }
    }
  }

  SubkernelIdType SK = SubkernelIds.size();
  SubkernelIds.insert(SK);
  SubkernelBBs[SK] = func_bbs;
  assert(BB == func_bbs[0] && (blockIsAfterBarrier(BB) || &F->getEntryBlock() == BB));
}

SubkernelIdType FunctionTransformer::findSubkernelFromBB(BBIdType BB) {
  for (auto SK : SubkernelIds) {
    if (SubkernelBBIds[SK][SubkernelBBs[SK][0]] == BB)
      return SK;
  }
  return -1;
}

void FunctionTransformer::createSubkernelFunctionClones() {
  for (auto SK : SubkernelIds) {
    LLVM_DEBUG(dbgs() << "FunctionTransformer - generating new subkernel " << SK << "\n");
    ValueToValueMapTy VMap;
    // Clone the function to get a clone of the basic blocks
    Function *NF = CloneFunction(F, VMap);
    SubkernelFs[SK] = NF;
    SubkernelBBs[SK] = convert_bb_vector(SubkernelBBs[SK], F, NF);
  }
}

void FunctionTransformer::sortValueVector(SubkernelIdType SK, ValueVector &VV, map<Value *, int> &Indices) {
  InstVector IV;
  ArgVector AV;
  GlobalVarVector GV;
  for (auto Val : VV) {
    if (Instruction *I = dyn_cast<Instruction>(Val))
      IV.push_back(I);
    else if (Argument *A = dyn_cast<Argument>(Val))
      AV.push_back(A);
    else if (GlobalVariable *G = dyn_cast<GlobalVariable>(Val))
      GV.push_back(G);
    else
      assert(false && "Used vals must be only instructions, arguments, or globals");
  }
  std::sort(IV.begin(), IV.end(), [&](Instruction *A, Instruction *B) {
    BasicBlock *BBA = A->getParent();
    BasicBlock *BBB = B->getParent();
    if (BBA != BBB) {
      return this->SubkernelBBIds[SK][BBA] < this->SubkernelBBIds[SK][BBB];
    } else {
      return A->comesBefore(B);
    }
  });
  std::sort(AV.begin(), AV.end(), [](Argument *A, Argument *B) {
    return A->getArgNo() < B->getArgNo();
  });
  ValueVector SortedVV;
  for (auto G : GV) {
    Value *Val = static_cast<Value *>(G);
    Indices[Val] = SortedVV.size();
    SortedVV.push_back(G);
  }
  for (auto Arg : AV) {
    Value *Val = static_cast<Value *>(Arg);
    Indices[Val] = SortedVV.size();
    SortedVV.push_back(Val);
  }
  for (auto Inst : IV) {
    Value *Val = static_cast<Value *>(Inst);
    Indices[Val] = SortedVV.size();
    SortedVV.push_back(Val);
  }
  VV = SortedVV;
}

bool isSharedVar(GlobalVariable *G) {
  return G->hasAttribute(Attribute::CPUCUDAShared);
}

bool isStaticSharedVar(GlobalVariable *G) {
  return G->hasAttribute(Attribute::CPUCUDAShared) && !G->isDeclaration();
}

bool isDynamicSharedVar(GlobalVariable *G) {
  return G->hasAttribute(Attribute::CPUCUDAShared) && G->isDeclaration();
}

class DomAnalysis {
public:
  SubkernelIdType SK;
  FunctionTransformer *Pass;
  Function *F;
  ValueToValueMapTy VMap;
  std::unique_ptr<DominatorTree> DomTree;

  DomAnalysis(SubkernelIdType SK, SubkernelIdType _SK, FunctionTransformer *Pass):
      SK(SK), Pass(Pass) {

    Function *OriginalF = Pass->SubkernelFs[SK];

    // Clone the function to get a clone of the basic blocks
    F = CloneFunction(OriginalF, VMap);

    BasicBlock *_OriginalEntryBB = convertBasicBlock(Pass->SubkernelBBs[_SK][0], Pass->SubkernelFs[_SK], Pass->SubkernelFs[SK]);
    BasicBlock *OriginalEntryBB = convertBasicBlock(_OriginalEntryBB, OriginalF, F);

    // Remove all branch instructions jumping to blocks after barriers
    for (auto &BB : *OriginalF) {
      if (Pass->blockIsAfterBarrier(SK, &BB)) {
        BasicBlock *NewBB = dyn_cast<BasicBlock>(&*VMap[&BB]);
        // There is only one unconditional predecessor because a block after a
        // barrier should be the result of SplitBlock()
        BasicBlock *PredBB = NewBB->getSinglePredecessor();
        assert(PredBB && "Block after a barrier must have a single predecessor");
        Instruction *Term = PredBB->getTerminator();
        if (auto Branch = dyn_cast<BranchInst>(Term)) {
          Branch->eraseFromParent();
          ReturnInst::Create(PredBB->getContext(), nullptr, PredBB);
        } else {
          assert(false && "Block after a barrier cannot be jumped to by anything other than an unconditional branch");
        }
      }
    }

    // Make the original entry bb the entry
    BasicBlock *EntryBB = BasicBlock::Create(F->getContext(), "generated_entry_block", F, &F->getEntryBlock());
    BranchInst::Create(OriginalEntryBB, EntryBB);

    DomTree.reset(new DominatorTree(*F));
  }

  DomAnalysis(SubkernelIdType SK, FunctionTransformer *Pass):
      DomAnalysis(SK, SK, Pass) {}

  bool dominates(Instruction *ValD, Instruction *User) {
    return DomTree->dominates(VMap[ValD], dyn_cast<Instruction>(&*VMap[User]));
  }

  bool dominates(Instruction &_I, unsigned i) {
    Instruction *I = dyn_cast<Instruction>(&*VMap[&_I]);
    Instruction *Op = cast<Instruction>(I->getOperand(i));
    const Use &U = I->getOperandUse(i);
    return DomTree->dominates(Op, U);
  }

  ~DomAnalysis() {
    F->eraseFromParent();
  }

};

// Currently only tracks registers and not values written to memory

// TODO I think we would have to transform Alloca's and their lifetime tags into
// malloc and frees or our custom implementations of them.

// I have a sneaking suspicion this implementation calling dominates() for every
// instruction in the function SubkernelNum^2 times is terribly, terribly slow
void FunctionTransformer::findSubkernelUsedValsDom() {
  for (auto SK : SubkernelIds) {
    for (auto _SK : SubkernelIds) {
      DomAnalysis DA(SK, _SK, this);
      ValueSet UsedInsts;
      for (auto &BB : *SubkernelFs[SK]) {
        BasicBlock *ConvertedBB = convertBasicBlock(&BB, SubkernelFs[SK], SubkernelFs[_SK]);
        if (in_vector(SubkernelBBs[_SK], ConvertedBB)) {
          for (auto &I : BB) {
            for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
              auto *Op = I.getOperand(i);
              if (isa<Instruction>(Op) && !DA.dominates(I, i)) {
                UsedInsts.insert(Op);
              }
            }
          }
        }
      }

      SubkernelUsedVals[SK][_SK] = ValueVector(UsedInsts.begin(), UsedInsts.end());

    }
  }
}

set<SubkernelIdType> FunctionTransformer::getSubkernelSuccessors(SubkernelIdType SK) {
  set<SubkernelIdType> Successors;
  for (auto BB : SubkernelBBs[SK]) {
    for (auto SuccBB : successors(BB)) {
      BBIdType SuccBBId = SubkernelBBIds[SK][SuccBB];
      if (!blockIsAfterBarrier(SK, SuccBB))
        continue;
      for (auto _SK : SubkernelIds) {
        BBIdType _SKEntryBB = SubkernelBBIds[_SK][SubkernelBBs[_SK][0]];
        if (_SKEntryBB == SuccBBId) {
          Successors.insert(_SK);
          break;
        }
      }
    }
  }
  return Successors;
}

Type *FunctionTransformer::getCombinedDataType() {
  return CombinedDataType;
}

int FunctionTransformer::getValIndexInCombinedDataType(SubkernelIdType SK, Value *Val) {
  return IndexInCombinedDataType[SK][Val];
}

StructType *FunctionTransformer::getSubkernelsReturnType() {
  vector<Type *> types;
  // The BB id we are coming from (for phi instruction handling)
  types.push_back(LLVMBBIdType);
  // The next subkernel to call
  types.push_back(LLVMSubkernelIdType);
  // Resulting in { from: blockIdType, to: blockIdType }
  return StructType::get(M->getContext(), ArrayRef<Type *>(types));
}

void FunctionTransformer::assignBBIds() {
  for (auto SK : SubkernelIds) {
    Function *F = SubkernelFs[SK];
    int id = 0;
    for (auto it = F->begin(); it != F->end(); ++it, ++id) {
      BasicBlock *bb = &(*it);
      SubkernelBBIds[SK][bb] = id;
    }
  }
  int id = 0;
  for (auto it = F->begin(); it != F->end(); ++it, ++id) {
    BasicBlock *bb = &(*it);
    OriginalFunBBs[id] = bb;
  }
}

vector<StringRef> FunctionTransformer::getSubkernelParamNames(SubkernelIdType SK) {
  vector<StringRef> names = {
    "from_bb_id",
    "preserved_data",
    "static_shared_data",
    "dynamic_shared_data"
  };
  for (auto &Arg : F->args()) {
    names.push_back(Arg.getName());
  }
  return names;
}

TypeVector FunctionTransformer::getSubkernelParams(SubkernelIdType SK) {
  // Construct the data struct param
  std::vector<Type *> valparams;

  // TODO Are the address spaces for these correct?

  TypeVector params;
  // The id of the BB we returned from in the previous subkernel (for phi instr)
  params.push_back(LLVMBBIdType);
  // Values to be preserved between subkernel calls
  params.push_back(PointerType::get(CombinedDataType, SubkernelFs[SK]->getAddressSpace()));
  // Pointer to the struct with shared variables
  params.push_back(PointerType::get(SharedVarsDataType, SubkernelFs[SK]->getAddressSpace()));
  // Pointer to the dynamically allocated shared memory TODO actually implement
  // it
  params.push_back(PointerType::get(IntegerType::getInt8Ty(M->getContext()), SubkernelFs[SK]->getAddressSpace()));

  // The original arguments
  for (auto &Arg : F->args()) {
    params.push_back(Arg.getType());
  }

  return params;
}

void FunctionTransformer::removeReferencesInPhi(const BBVector &BBsToRemove) {
  for (auto &BB : BBsToRemove) {
    BBVector Successors;
    for (auto SuccBB : successors(BB)) {
      Successors.push_back(SuccBB);
    }
    for (auto SuccBB : Successors) {
      vector<PHINode *> Phis;
      for (auto &Phi : SuccBB->phis()) {
        Phis.push_back(&Phi);
      }
      for (auto &Phi : Phis) {
        while (true) {
          int BBIndex = Phi->getBasicBlockIndex(BB);
          if (BBIndex != -1)
            Phi->removeIncomingValue(BBIndex);
          else
            break;
        }
      }
    }
  }
}

// TODO optimise when usedVals gets populated by simple struct member accesses,
// for example, currently accesses of dim3 members get added to usedVals
void FunctionTransformer::transformSubkernels(SubkernelIdType SK) {

  Function *_nf = SubkernelFs[SK];

  BBVector nfunc_bbs = SubkernelBBs[SK];

  auto usedVals = SubkernelUsedVals[SK][SK];

  TypeVector params = getSubkernelParams(SK);
  vector<StringRef> paramNames = getSubkernelParamNames(SK);

  // Make a new function which will be the subkernel
  FunctionType *nfty = FunctionType::get(
      /* return type */ SubkernelReturnType,
      /* params */ params,
      /* isVarArg */ false);
  Function *nf = Function::Create(nfty, F->getLinkage(), F->getAddressSpace(),
                                  F->getName(), F->getParent());

  auto NamesIt = paramNames.begin();
  for (auto &Arg : nf->args()) {
    Arg.setName(*NamesIt++);
  }

  // Insert the cloned basic blocks
  nf->getBasicBlockList().splice(nf->begin(), _nf->getBasicBlockList());
  nf->takeName(_nf);
  SubkernelFs[SK] = nf;

  // Get the BBs we have to remove before adding new ones which would interfere
  // with this construction
  BBVector BBsToRemove;
  for (auto &bb : *nf) {
    if (!in_vector(nfunc_bbs, &bb))
      BBsToRemove.insert(BBsToRemove.begin(), &bb);
  }
  BBVector OriginalBBs;
  for (auto &BB : *nf) {
    OriginalBBs.push_back(&BB);
  }

  // Store the used vals for later subkernels
  {
    Value *DataStructPtr = nf->getArg(1);

    ConstantInt *Zero = ConstantInt::get(I32Type, 0);

    for (auto &Val : CombinedUsedVals[SK]) {
      Instruction *Inst = dyn_cast<Instruction>(Val);
      BasicBlock *ValBB = Inst->getParent();
      // We are only interested in values which are defined in the current
      // subkernel
      if (!in_vector(SubkernelBBs[SK], ValBB))
        continue;

      ConstantInt *Index = ConstantInt::get(
          I32Type, getValIndexInCombinedDataType(SK, Val));
      Instruction *NextInst = Inst->getNextNonDebugInstruction();
      // If next inst is Phi, we have to get the first following non-phi
      // instruction because all Phi's must be bunched at the start of a BB
      if (dyn_cast<PHINode>(NextInst)) {
        NextInst = NextInst->getParent()->getFirstNonPHI();
      }
      GetElementPtrInst *Gep = GetElementPtrInst::Create(
          getCombinedDataType(), DataStructPtr, {Zero, Index}, "", NextInst);
      assert(NextInst && "The Inst must not be a terminator instruction so a next instruction has to exist");
      new StoreInst(Val, Gep, NextInst);
    }

  }

  // Construct the entry block which sets up the usedVals params and handles phi
  // instructions
  {
    ConstantInt *Zero = ConstantInt::get(Type::getInt32Ty(nf->getContext()), 0);

    // The index at which the original arguments start
    unsigned i = 4;
    for (auto &Arg : _nf->args()) {
      Arg.replaceAllUsesWith(nf->getArg(i));
      ++i;
    }

    DomAnalysis DA(SK, this);

    BasicBlock *EntryBB = BasicBlock::Create(nf->getContext(), "generated_entry_block", nf, &nf->getEntryBlock());

    {
      // Transfer usages of the usedVals to the arguments to the function
      auto I = usedVals.begin(), E = usedVals.end();
      // Unpack args from data struct param and replace usages with them
      for (; I != E; ++I) {
        // The second argument of the function is the structure of usedVals
        Value *Val = (*I);
        ConstantInt *Index = ConstantInt::get(
            GepIndexType, getValIndexInCombinedDataType(SK, Val));
        GetElementPtrInst *Gep = GetElementPtrInst::Create(
            getCombinedDataType(), nf->getArg(1), {Zero, Index}, "", EntryBB);
        LoadInst *UnpackedVal = new LoadInst(Val->getType(), Gep, "", EntryBB);

        // Replace only if the val does not already dominate the use - sometimes
        // a value dominates only part of the uses in the subkernel - it happens
        // for example when a subkernel starts execution after a barrier and a
        // value is passed back to an earlier BB using a PHI node
        Val->replaceUsesWithIf(UnpackedVal, [&](Use &U) {
          Instruction *I = dyn_cast<Instruction>(U.getUser());
          return !DA.dominates(dyn_cast<Instruction>(Val), I);
        });
        UnpackedVal->takeName(Val);
      }
    }

    {
      auto usedSharedVars = CombinedSharedVars;
      // Transfer usages of the used shared vars to the arguments to the function
      auto It = usedSharedVars.begin(), E = usedSharedVars.end();
      // Unpack args from data struct param and replace usages with them
      for (unsigned i = 0; It != E; ++It, ++i) {
        GlobalVariable *G = (*It);
        ConstantInt *Index = ConstantInt::get(
            GepIndexType, IndexInCombinedSharedVarsDataType[G]);
        // The third argument of the function is the structure of shared variables
        GetElementPtrInst *Gep = GetElementPtrInst::Create(
            SharedVarsDataType, nf->getArg(2), {Zero, Index}, "", EntryBB);

        // Perhaps we could use RemapFunction() here to only remap usages within
        // the function
        G->replaceUsesWithIf(Gep, [&](Use &U) {
          Instruction *I = dyn_cast<Instruction>(U.getUser());
          assert(I && "There should only be Instruction users by now");
          return I->getParent()->getParent() == nf;
        });
        Gep->takeName(G);
        // TODO clean up the shared variables when we are done with all subkernels
        // G->eraseFromParent();
      }
    }

    // Replace usages of the dynamic shared variable if it exists
    if (DynamicSharedVar) {
      auto SharedVarArg = nf->getArg(3);
      auto SharedVarArgCast = new BitCastInst(
          SharedVarArg, DynamicSharedVar->getType(), "cast_dyn_shared_var", EntryBB);
      DynamicSharedVar->replaceUsesWithIf(SharedVarArgCast, [&](Use &U) {
        Instruction *I = dyn_cast<Instruction>(U.getUser());
        assert(I && "There should only be Instruction users by now");
        return I->getParent()->getParent() == nf;
      });
    }

    // Add return from exiting blocks
    for (auto &bb : nfunc_bbs) {
      Instruction *term = bb->getTerminator();
      TransformTerminator transformer(SK, this);
      transformer.visit(term);
    }


    // List of BBs which are actually used in phi instructions
    BBVector ToHandle;
    BasicBlock *OriginalEntryBB = SubkernelBBs[SK][0];
    for (auto &BB : OriginalBBs) {
      BasicBlock *PhiHandlerBB = BasicBlock::Create(nf->getContext(), "generated_phi_handler_block", nf);
      BranchInst::Create(OriginalEntryBB, PhiHandlerBB);
      // Find usages of BB in phi instructions to be transformed
      for (User *U : BB->users()) {
        if (PHINode *Phi = dyn_cast<PHINode>(U)) {

          // We are only interested in Phi Instructions in the original entry
          // block TODO maybe we have to remove references to deleted BBs from
          // phi instructions not in the original entry block?
          if (Phi->getParent() != OriginalEntryBB)
            continue;

          int BBIndex = Phi->getBasicBlockIndex(BB);
          assert(BBIndex != -1);
          if (in_vector(nfunc_bbs, BB)) {
            // If the BB already exists in the Subkernel add an additional case
            // for the new handler block
            Phi->addIncoming(Phi->getIncomingValue(BBIndex), PhiHandlerBB);
          } else {
            // If the BB does not exist in the SK, just replace its usage with
            // the new handler block
            Phi->replaceIncomingBlockWith(BB, PhiHandlerBB);
          }

          if (!in_vector(ToHandle, BB))
            ToHandle.push_back(BB);
        }
      }
    }

    removeReferencesInPhi(BBsToRemove);

    // The first argument of the function is the BBId label indicating which BB
    // we came from
    SwitchInst *Switch = SwitchInst::Create(nf->getArg(0), OriginalEntryBB, 0, EntryBB);
    for (auto &BB : ToHandle) {
      ConstantInt *CaseConst = llvm::ConstantInt::get(LLVMBBIdType,
                                                      SubkernelBBIds[SK][BB],
                                                      /* isSigned */ true);
      Switch->addCase(CaseConst, OriginalEntryBB);
    }
  }

  // Clone metadata from the old function
  {
    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    _nf->getAllMetadata(MDs);
    for (auto MD : MDs)
      nf->addMetadata(MD.first, *MD.second);
  }

  // Erase unneeded basic blocks
  {
    // Empty placeholder BB to replace BB usages
    BasicBlock *EmptyBB = BasicBlock::Create(M->getContext(), "empty_block", nf);
    for (auto &bb : BBsToRemove) {
      for (auto &inst : *bb) {
        if (!inst.use_empty())
          inst.replaceAllUsesWith(UndefValue::get(inst.getType()));
      }
      bb->replaceAllUsesWith(EmptyBB);
      bb->eraseFromParent();
    }

    EmptyBB->eraseFromParent();
  }

  // Delete the dead function
  _nf->eraseFromParent();

}

// In the cases where the global shared variable is used in a constant
// expression and then that CE is used in an Instruction, this implementation
// would not find it, however, we should have already converted all shared
// variable usages to instructions, so this works (hopefully)
void FunctionTransformer::findSharedVars() {
  DynamicSharedVar = nullptr;
  auto SK = 0;
  for (auto &BB : *SubkernelFs[SK]) {
    for (auto &I : BB) {
      for (Use &U : I.operands()) {
        Value *V = U.get();
        GlobalVariable *UseG = dyn_cast<GlobalVariable>(V);
        if (UseG && isStaticSharedVar(UseG) && !in_vector(CombinedSharedVars, UseG))
          CombinedSharedVars.push_back(UseG);
        if (UseG && isDynamicSharedVar(UseG)) {
          if (!DynamicSharedVar)
            DynamicSharedVar = UseG;
          else
            assert(DynamicSharedVar == UseG && "There can only be one dynamic shared variable per kernel");
        }
      }
    }
  }

  TypeVector Types;
  for (auto G : CombinedSharedVars) {
    assert(isa<PointerType>(G->getType()));
    Types.push_back(dyn_cast<PointerType>(G->getType())->getPointerElementType());
  }
  SharedVarsDataType = StructType::get(M->getContext(), Types);

  int Index = 0;
  for (auto G : CombinedSharedVars) {
    IndexInCombinedSharedVarsDataType[G] = Index;
    Index++;
  }
}

void FunctionTransformer::replaceDim3Usages() {

  FunctionType *FT = F->getFunctionType();
  TypeVector ArgTypes;

  for (unsigned i = 0; i < FT->getNumParams(); ++i) {
    ArgTypes.push_back(FT->getParamType(i));
  }

  // gridDim, blockIdx, blockDim, threadIdx
  for (unsigned i = 0; i < Dim3Names.size(); ++i) {
    ArgTypes.push_back(Dim3Type);
  }

  auto NewFT = FunctionType::get(FT->getReturnType(), ArgTypes, false);
  auto NewF = Function::Create(NewFT, F->getLinkage(), F->getAddressSpace(),
                               F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  auto NewFArgIt = NewF->arg_begin();
  for (auto &Arg : F->args()) {
    auto ArgName = Arg.getName();
    NewFArgIt->setName(ArgName);
    VMap[&Arg] = &(*NewFArgIt);

    NewFArgIt++;
  }
  for (auto &name : Dim3Names) {
    (NewFArgIt++)->setName(name);
  }

  SmallVector<ReturnInst*, 8> Returns;
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly, Returns);

  OriginalF = F;
  F = NewF;

  unsigned Dim3ArgStartIndex = FT->getNumParams();

  // TODO is there another instruction type other than CallInst that might call
  // the dim3 functions?
  for (auto &bb : *F)
    for (auto It = bb.begin(); It != bb.end();) {
      bool erased = false;
      auto &instruction = *It;
      if (CallInst *callInst = dyn_cast<CallInst>(&instruction))
        if (Function *calledFunction = callInst->getCalledFunction())
          for (unsigned i = 0; i < Dim3Names.size(); ++i) {
            vector<Function *> GetterFs = {
              Dim3Fs.Getterx,
              Dim3Fs.Gettery,
              Dim3Fs.Getterz,
            };
            for (unsigned dim = 0; dim < 3; ++dim) {
              if (calledFunction->getName() == Dim3GetterIntrinsicNames.at(Dim3Names[i])[dim]) {
                auto Dim3Arg = F->getArg(i + Dim3ArgStartIndex);
                ValueVector Dim3Args = convertDim3ToArgs(Dim3Arg, callInst);
                CallInst *Dim3Dim = CallInst::Create(
                    GetterFs[dim]->getFunctionType(), GetterFs[dim], Dim3Args,
                    "single_dim");
                Dim3Dim->insertAfter(dyn_cast<Instruction>(*(Dim3Args.end() - 1)));
                callInst->replaceAllUsesWith(Dim3Dim);
                It = callInst->eraseFromParent();
                erased = true;

                InlineFunctionInfo IFI;
                InlineResult IR = InlineFunction(*Dim3Dim, IFI);
                assert(IR.isSuccess());

                break;
              }
            }
          }
      if (!erased)
        ++It;
    }
}

bool dependsOnState(Value *V) {
  auto I = dyn_cast<Instruction>(V);
  if (isa<CallBase>(V)) {
    // TODO This should be fine if the called function is pure
    return true;
  } else if (isa<PHINode>(V)) {
    return true;
  } else if (I) {
    for (Use &U : I->operands())
      if (dependsOnState(U.get()))
        return true;
  }
  return false;
}

InstructionCost instCostFromArgs(Instruction *I, TargetTransformInfo *TTI) {
  InstructionCost Cost = TTI->getInstructionCost(I, TARGET_COST_KIND);
  for (Use &U : I->operands()) {
    Value *V = U.get();
    if (auto UseI = dyn_cast<Instruction>(V))
      Cost += instCostFromArgs(UseI, TTI);
  }
  return Cost;
}

InstructionCost getStoreCost(Instruction *I, TargetTransformInfo *TTI) {
  auto Store = new StoreInst(
      I, ConstantPointerNull::get(PointerType::get(I->getType(), I->getParent()->getParent()->getAddressSpace())), I);
  InstructionCost IC = TTI->getInstructionCost(Store, TARGET_COST_KIND);
  Store->eraseFromParent();
  return IC;
}

InstructionCost getLoadCost(Instruction *I, TargetTransformInfo *TTI) {
  auto Load = new LoadInst(
      I->getType(),
      ConstantPointerNull::get(PointerType::get(I->getType(), I->getParent()->getParent()->getAddressSpace())),
      "",
      I);
  InstructionCost IC = TTI->getInstructionCost(Load, TARGET_COST_KIND);
  Load->eraseFromParent();
  return IC;
}

// Returns the recalculated instruction
Instruction *recalculateArgOnlyInstAfterBarrier(Instruction *I, Instruction *InsertBefore) {
  queue<Instruction *> ToClone;
  InstVector Cloned;
  ValueToValueMapTy VMap;
  ToClone.push(I);
  while (!ToClone.empty()) {
    auto I = ToClone.front();
    ToClone.pop();

    auto *NI = I->clone();
    NI->insertBefore(InsertBefore);
    InsertBefore = NI;
    NI->setName(I->getName());
    VMap[I] = NI;
    Cloned.push_back(NI);

    for (Use &U : I->operands()) {
      Value *V = U.get();
      auto UseI = dyn_cast<Instruction>(V);
      if (UseI) {
        ToClone.push(UseI);
      }
    }
  }

  for (auto I : Cloned) {
    RemapInstruction(I, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
  }

  return Cloned[0];
}

void FunctionTransformer::optimizeUsedVals() {
  //map<SubkernelIdType, map<int, InstructionCost>> RecalcBenefit;

  for (auto SK : SubkernelIds) {
    for (auto _SK : SubkernelIds) {
      ValueVector UsedValsToRemove;
      auto &UsedVals = SubkernelUsedVals[SK][_SK];
      for (auto V : UsedVals) {
        auto I = dyn_cast<Instruction>(V);
        // If it purely depends on the arguments and global variables
        if (!dependsOnState(I)) {
#ifdef COST_ANALYSIS
          auto RecalculationCost = instCostFromArgs(I, TTI);
          auto StoreCost = getStoreCost(I, TTI);
          auto LoadCost = getLoadCost(I, TTI);
          int ExpectedLoadCount = 1;
          int ExpectedStoreCount = 1;

          // TODO think about this more
          if (ExpectedStoreCount * StoreCost + ExpectedLoadCount * LoadCost >= ExpectedLoadCount * RecalculationCost) {
            recalculateArgOnlyInstAfterBarrier(I);
          }
#endif
          // Insert the recalculated values at the start of the entry BB of the
          // subkernel
          Instruction *InsertBefore = SubkernelBBs[SK][0]->getFirstNonPHI();
          auto RecalcdUsedVal = recalculateArgOnlyInstAfterBarrier(I, InsertBefore);
          vector<User *> Users(I->users().begin(), I->users().end());
          for (User *U : Users) {
            if (auto UserI = dyn_cast<Instruction>(U)) {
              ValueToValueMapTy VMap;
              VMap[I] = RecalcdUsedVal;
              RemapInstruction(UserI, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
            } else {
              assert(false && "What else could use a value?");
            }
          }
          UsedValsToRemove.push_back(V);
        }
      }
      for (auto V : UsedValsToRemove) {
        UsedVals.erase(std::find(UsedVals.begin(), UsedVals.end(), V));
      }
    }
    //assert(verifyFunction(*SubkernelFs[SK]));
  }
}

void FunctionTransformer::indexUsedVals() {
  for (auto SK : SubkernelIds) {

    ValueVector CombinedUsedInsts;
    for (auto _SK : SubkernelIds)
      for (auto I : SubkernelUsedVals[SK][_SK])
        if (!in_vector(CombinedUsedInsts, I))
          CombinedUsedInsts.push_back(I);

    map<Value *, int> IndexInCombinedDataType;
    sortValueVector(SK, CombinedUsedInsts, IndexInCombinedDataType);
    this->CombinedUsedVals[SK] = CombinedUsedInsts;
    this->IndexInCombinedDataType[SK] = IndexInCombinedDataType;
  }

  vector<StructType *> CombinedDataTypes;
  for (auto SK : SubkernelIds) {
    ValueVector CombinedUsedVals = this->CombinedUsedVals[SK];
    TypeVector Types;
    for (auto Val : CombinedUsedVals) {
      Types.push_back(Val->getType());
    }
    CombinedDataTypes.push_back(StructType::get(M->getContext(), Types));
  }
  for (auto SK : SubkernelIds) {
    assert(CombinedDataTypes[0] == CombinedDataTypes[SK]);
  }
  CombinedDataType = CombinedDataTypes[0];
}

// When there is an Alloca in the kernel, in the cases where the alloca variable
// lives across barriers, it has to be transformed into a malloc which has to be
// free'd either at the lifetime end of the alloca if it exists or at the return
// points of the kernel. We could perhaps implement our own malloc for this but
// for the time being the stdlib malloc should be good enough
void FunctionTransformer::handleAllocas(Function *F) {
  // Do an easy check for whether we actually have barriers, if we don't, there
  // is no need to transform allocas
  if (BlocksAfterBarriers.size() == 0)
    return;

  DataLayout *DL = new DataLayout(M);

  for (auto &bb : *F) {
    for (auto It = bb.begin(); It != bb.end(); ) {
      auto &I = *It;

      if (auto Alloca = dyn_cast<AllocaInst>(&I)) {
        Instruction *Malloc = CallInst::CreateMalloc(
            static_cast<Instruction *>(Alloca),
            IntegerType::getInt32Ty(M->getContext()),
            Alloca->getAllocatedType(),
            ConstantInt::get(IntegerType::getInt32Ty(M->getContext()), DL->getTypeAllocSize(Alloca->getAllocatedType())),
            nullptr, nullptr, "");
        Malloc->takeName(Alloca);
        Alloca->replaceAllUsesWith(Malloc);
        It = Alloca->eraseFromParent();
        // TODO we are leaking the malloc - need to find a lifetime.end to free
        // it at or if it doesnt exist, when exiting the function
      } else {
        ++It;
      }
    }
  }
}

void FunctionTransformer::createSubkernels() {
  replaceDim3Usages();
  splitBlocksAroundBarriers(*F);
  handleAllocas(F);
  findSubkernelBBs();
  createSubkernelFunctionClones();
  assignBBIds();
  findSharedVars();
  findSubkernelUsedValsDom();
  optimizeUsedVals();
  indexUsedVals();
  SubkernelReturnType = getSubkernelsReturnType();
  for (auto SK : SubkernelIds) {
    transformSubkernels(SK);
  }
}

// TODO we have to put the alloca in the entry block or put lifetimes for it
// because currently for each loop execution we allocate more memory
class ThreadIdxLoop {
public:
  FunctionTransformer *T;
  Function *F;

  BasicBlock *EntryBB, *CondBB, *IncrBB, *EndBB;
  AllocaInst *IdxPtr;
  LoadInst *Idx;
  ICmpInst *Cond;

  BranchInst *FromCond;

  ThreadIdxLoop(std::string IdxName, Value *LoopTo, Function *F,
                FunctionTransformer *T, Instruction *InsertAllocaBefore) {
    this->F = F;
    this->T = T;

    EntryBB = BasicBlock::Create(F->getContext(), "loop_entry" + IdxName, F);

    IdxPtr = new AllocaInst(T->Dim3FieldType, F->getAddressSpace(),
                            ConstantInt::get(T->Dim3FieldType, 1),
                            IdxName + "_ptr", InsertAllocaBefore);
    new StoreInst(ConstantInt::get(T->Dim3FieldType, 0), IdxPtr, EntryBB);

    CondBB = BasicBlock::Create(F->getContext(), "loop_cond" + IdxName, F);
    BranchInst::Create(CondBB, EntryBB);

    Idx = new LoadInst(T->Dim3FieldType, IdxPtr, IdxName, CondBB);
    Cond = new ICmpInst(*CondBB, CmpInst::Predicate::ICMP_EQ, Idx, LoopTo, "cond_eq");

    IncrBB = BasicBlock::Create(F->getContext(), "loop_incr" + IdxName, F);
    BinaryOperator *IncrIdx = BinaryOperator::Create(
        Instruction::BinaryOps::Add,
        ConstantInt::get(T->Dim3FieldType, 1),
        Idx,
        IdxName + "_incr",
        IncrBB);
    new StoreInst(IncrIdx, IdxPtr, IncrBB);
    BranchInst::Create(CondBB, IncrBB);

    EndBB = BasicBlock::Create(F->getContext(), "loop_end" + IdxName, F);
  }

  void hookUpBBs(BasicBlock *BodyEntryBB, BasicBlock *BodyEndBB) {
    BranchInst::Create(EndBB, BodyEntryBB, Cond, CondBB);
    BranchInst::Create(IncrBB, BodyEndBB);
  }

};

ValueVector FunctionTransformer::convertDim3ToArgs(Value *D, Instruction *After) {
  ValueToValueMapTy VMap;
  for (auto &BB : *Dim3Fs.Dim3ToArg) {
    for (auto &_I : BB) {
      Instruction *I = &_I;
      if (CallInst *Call = dyn_cast<CallInst>(I))
        if (Call->getCalledFunction()->getName() == "__cpucuda_declared_dim3_getter") {
          VMap[I] = D;
          continue;
        }
      if (CallInst *Call = dyn_cast<CallInst>(I))
        if (Call->getCalledFunction()->getName() == "__cpucuda_declared_dim3_user") {
          ValueVector Args;
          for (unsigned i = 0; i < Call->getNumArgOperands(); ++i) {
            Args.push_back(VMap[Call->getArgOperand(i)]);
          }
          return Args;
        }
      auto *NI = I->clone();
      NI->insertAfter(After);
      After = NI;
      NI->setName(I->getName());
      VMap[I] = NI;
      RemapInstruction(NI, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    }
  }
  assert(false && "Unreachable");
  __builtin_unreachable();
}

void FunctionTransformer::createDriverFunction() {
  Function *F = OriginalF;

  FunctionType *FT = F->getFunctionType();
  TypeVector ArgTypes;

  for (unsigned i = 0; i < FT->getNumParams(); ++i) {
    ArgTypes.push_back(FT->getParamType(i));
  }

  // gridDim, blockIdx, blockDim
  for (unsigned i = 0; i < Dim3Names.size() - 1; ++i) {
    ArgTypes.push_back(Dim3Type);
  }

  // shared mem
  ArgTypes.push_back(SizeTType);

  auto NewFT = FunctionType::get(FT->getReturnType(), ArgTypes, false);
  auto NewF = Function::Create(NewFT, F->getLinkage(), F->getAddressSpace(),
                               F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  auto NewFArgIt = NewF->arg_begin();
  for (auto &Arg : F->args()) {
    auto ArgName = Arg.getName();
    NewFArgIt->setName(ArgName);
    VMap[&Arg] = &(*NewFArgIt);

    NewFArgIt++;
  }
  for (unsigned i = 0; i < Dim3Names.size() - 1; ++i) {
    (NewFArgIt++)->setName(Dim3Names[i]);
  }

  vector<CallInst *> Dim3Calls;

  // Now we have an empty function
  DriverF = NewF;

  int Dim3ArgStartIndex = FT->getNumParams();
  int SharedMemSizeParamIndex = Dim3ArgStartIndex + 3;
  Argument *BlockDimArg = DriverF->getArg(Dim3ArgStartIndex + 2);

  ConstantInt *Zero = ConstantInt::get(GepIndexType, 0);
  ConstantInt *One = ConstantInt::get(GepIndexType, 1);
  ConstantInt *mOne = ConstantInt::get(GepIndexType, -1);
  DataLayout *DL = new DataLayout(M);

  BasicBlock *EntryBB = BasicBlock::Create(DriverF->getContext(), "entry", DriverF);

  AllocaInst *SubkernelRetPtr = new AllocaInst(SubkernelReturnType, DriverF->getAddressSpace(), One, "ret", EntryBB);

  ValueVector Dim3Args = convertDim3ToArgs(BlockDimArg, SubkernelRetPtr);
  CallInst *BlockDimx = CallInst::Create(
      Dim3Fs.Getterx->getFunctionType(), Dim3Fs.Getterx, Dim3Args,
      "blockDim_x", EntryBB);
  Dim3Calls.push_back(BlockDimx);
  CallInst *BlockDimy = CallInst::Create(
      Dim3Fs.Getterx->getFunctionType(), Dim3Fs.Gettery, Dim3Args,
      "blockDim_y", EntryBB);
  Dim3Calls.push_back(BlockDimy);
  CallInst *BlockDimz = CallInst::Create(
      Dim3Fs.Getterx->getFunctionType(), Dim3Fs.Getterz, Dim3Args,
      "blockDim_z", EntryBB);
  Dim3Calls.push_back(BlockDimz);

  Instruction *StaticSharedData = CallInst::CreateMalloc(
      static_cast<Instruction *>(SubkernelRetPtr),
      IntegerType::getInt32Ty(M->getContext()),
      SharedVarsDataType,
      ConstantInt::get(IntegerType::getInt32Ty(M->getContext()), DL->getTypeAllocSize(SharedVarsDataType)),
      nullptr, nullptr, "static_shared_data");
  // TODO Handle dynamic shared data
  Instruction *DynSharedData = CallInst::CreateMalloc(
      static_cast<Instruction *>(SubkernelRetPtr),
      SizeTType,
      IntegerType::getInt8Ty(M->getContext()),
      DriverF->getArg(SharedMemSizeParamIndex),
      nullptr, nullptr, "static_shared_data");

  BinaryOperator *BlockSize = BinaryOperator::Create(
      Instruction::BinaryOps::Mul, BlockDimx, BlockDimy, "blockDimMul", EntryBB);
  BlockSize = BinaryOperator::Create(
      Instruction::BinaryOps::Mul, BlockSize, BlockDimz, "blockSize", EntryBB);

  Instruction *PreservedData;
  if (Options.MallocPreservedDataArray) {
    Value *MallocSize;
    ConstantInt *StructSize = ConstantInt::get(GepIndexType, DL->getTypeAllocSize(CombinedDataType));
    if (!Options.DynamicPreservedDataArray) {
      ConstantInt *MaxCudaThreads = ConstantInt::get(GepIndexType, MAX_CUDA_THREADS);
      MallocSize = ConstantExpr::getMul(StructSize, MaxCudaThreads);
    } else {
      MallocSize = BinaryOperator::Create(
          Instruction::BinaryOps::Mul, BlockSize, StructSize, "blockSize", EntryBB);
    }
    Instruction *Malloc = CallInst::CreateMalloc(
        static_cast<Instruction *>(StaticSharedData),
        IntegerType::getInt32Ty(M->getContext()),
        CombinedDataType,
        MallocSize, nullptr, nullptr, "preserved_data");
    PreservedData = Malloc;
  } else {
    Value *Size = Options.DynamicPreservedDataArray ?
      static_cast<Value *>(ConstantInt::get(GepIndexType, MAX_CUDA_THREADS)) : static_cast<Value *>(BlockSize);
    if (!Options.DynamicPreservedDataArray) {
      Size = ConstantInt::get(GepIndexType, MAX_CUDA_THREADS);
    } else {
      Size = BlockSize;
    }
    PreservedData = new AllocaInst(CombinedDataType, DriverF->getAddressSpace(),
                                   Size, "preserved_data", EntryBB);
  }

  GetElementPtrInst *SubkernelRetFromPtr = GetElementPtrInst::Create(
      SubkernelReturnType, SubkernelRetPtr, {Zero, Zero}, "", EntryBB);
  SubkernelRetFromPtr->setName("from_ptr");
  new StoreInst(mOne, SubkernelRetFromPtr, EntryBB);

  ConstantInt *EntrySKConst = ConstantInt::get(I32Type, EntrySubkernel);
  GetElementPtrInst *SubkernelRetNextPtr = GetElementPtrInst::Create(
      SubkernelReturnType, SubkernelRetPtr, {Zero, One}, "", EntryBB);
  SubkernelRetNextPtr->setName("next_ptr");
  new StoreInst(EntrySKConst, SubkernelRetNextPtr, EntryBB);

  BasicBlock *WhileEntryBB = BasicBlock::Create(DriverF->getContext(), "while_entry", DriverF);
  auto EntryBBEndInst = BranchInst::Create(WhileEntryBB, EntryBB);
  LoadInst *Next = new LoadInst(LLVMBBIdType, SubkernelRetNextPtr, "next", WhileEntryBB);
  LoadInst *From = new LoadInst(LLVMBBIdType, SubkernelRetFromPtr, "from", WhileEntryBB);

  BasicBlock *WhileEndBB = BasicBlock::Create(DriverF->getContext(), "while_end", DriverF);
  SwitchInst *Switch = SwitchInst::Create(Next, WhileEndBB, 0, WhileEntryBB);

  for (auto SK : SubkernelIds) {
    ConstantInt *CaseConst = llvm::ConstantInt::get(LLVMBBIdType, SK, /* isSigned */ true);
    BasicBlock *SwitchCaseBB = BasicBlock::Create(DriverF->getContext(), "switch_case", DriverF);
    Switch->addCase(CaseConst, SwitchCaseBB);

    auto InsertSubkernelCall = [&](Value *PreservedDataIdx, BasicBlock *SubkernelCallBB,
                                   Value *ThreadIdxx, Value *ThreadIdxy, Value *ThreadIdxz) {
      GetElementPtrInst *ThreadPreservedData = GetElementPtrInst::Create(
          CombinedDataType, PreservedData, {PreservedDataIdx}, "threadPreservedData", SubkernelCallBB);

      ValueVector Args = {From, ThreadPreservedData, StaticSharedData, DynSharedData};
      // original args + gridDim, blockIdx, blockDim (we dont need the last arg
      // which is shared_mem_size)
      for (auto _it = DriverF->args().begin(); _it + 1!= DriverF->args().end(); ++_it) {
        auto &Arg = *_it;
        Args.push_back(&Arg);
      }
      // threadIdx
      CallInst *ThreadIdx = CallInst::Create(
          Dim3Fs.ConstructorF->getFunctionType(), Dim3Fs.ConstructorF,
          {ThreadIdxx, ThreadIdxy, ThreadIdxz}, "threadIdx", SubkernelCallBB);
      Dim3Calls.push_back(ThreadIdx);
      Args.push_back(ThreadIdx);
      CallInst *SubkernelCall = CallInst::Create(
          SubkernelFs[SK]->getFunctionType(), SubkernelFs[SK],
          Args, "local_ret", SubkernelCallBB);
      new StoreInst(SubkernelCall, SubkernelRetPtr, SubkernelCallBB);

      return SubkernelCall;
    };

    CallInst *SubkernelCall;

    if (Options.SingleDimThreadLoop) {
      ThreadIdxLoop LoopLin("threadIdx_linear_index_", BlockSize, DriverF, this, EntryBBEndInst);

      BasicBlock *SubkernelCallBB = BasicBlock::Create(DriverF->getContext(), "subkernel_call", DriverF);

      auto ThreadIdxx = BinaryOperator::Create(
          Instruction::BinaryOps::URem, LoopLin.Idx, BlockDimx, "threadIdx.x", SubkernelCallBB);
      auto Tmp = BinaryOperator::Create(
          Instruction::BinaryOps::UDiv, LoopLin.Idx, BlockDimx, "rest", SubkernelCallBB);
      auto ThreadIdxy = BinaryOperator::Create(
          Instruction::BinaryOps::URem, Tmp, BlockDimy, "threadIdx.y", SubkernelCallBB);
      Tmp = BinaryOperator::Create(
          Instruction::BinaryOps::UDiv, Tmp, BlockDimy, "rest", SubkernelCallBB);
      auto ThreadIdxz = BinaryOperator::Create(
          Instruction::BinaryOps::URem, Tmp, BlockDimy, "threadIdx.z", SubkernelCallBB);

      SubkernelCall = InsertSubkernelCall(LoopLin.Idx, SubkernelCallBB, ThreadIdxx, ThreadIdxy, ThreadIdxz);

      LoopLin.hookUpBBs(SubkernelCallBB, SubkernelCallBB);

      BranchInst::Create(LoopLin.EntryBB, SwitchCaseBB);
      BranchInst::Create(WhileEntryBB, LoopLin.EndBB);

    } else {
      ThreadIdxLoop Loopz("threadIdx_z_", BlockDimz, DriverF, this, EntryBBEndInst);
      ThreadIdxLoop Loopy("threadIdx_y_", BlockDimy, DriverF, this, EntryBBEndInst);
      ThreadIdxLoop Loopx("threadIdx_x_", BlockDimx, DriverF, this, EntryBBEndInst);
      Loopz.hookUpBBs(Loopy.EntryBB, Loopy.EndBB);
      Loopy.hookUpBBs(Loopx.EntryBB, Loopx.EndBB);

      BasicBlock *SubkernelCallBB = BasicBlock::Create(DriverF->getContext(), "subkernel_call", DriverF);

      auto PreservedDataIdx = BinaryOperator::Create(
          Instruction::BinaryOps::Mul, BlockDimy, Loopz.Idx, "threadPreservedDataIdx", SubkernelCallBB);
      PreservedDataIdx = BinaryOperator::Create(
          Instruction::BinaryOps::Add, Loopy.Idx, PreservedDataIdx, "threadPreservedDataIdx", SubkernelCallBB);
      PreservedDataIdx = BinaryOperator::Create(
          Instruction::BinaryOps::Mul, BlockDimx, PreservedDataIdx, "threadPreservedDataIdx", SubkernelCallBB);
      PreservedDataIdx = BinaryOperator::Create(
          Instruction::BinaryOps::Add, Loopx.Idx, PreservedDataIdx, "threadPreservedDataIdx", SubkernelCallBB);

      SubkernelCall = InsertSubkernelCall(PreservedDataIdx, SubkernelCallBB, Loopx.Idx, Loopy.Idx, Loopz.Idx);

      Loopx.hookUpBBs(SubkernelCallBB, SubkernelCallBB);

      BranchInst::Create(Loopz.EntryBB, SwitchCaseBB);
      BranchInst::Create(WhileEntryBB, Loopz.EndBB);

    }

    if (Options.InlineSubkernels) {
      InlineFunctionInfo IFI;
      InlineResult IR = InlineFunction(*SubkernelCall, IFI);
      assert(IR.isSuccess());
    }

  }

  if (Options.InlineDim3Fs) {
    for (auto Dim3Call : Dim3Calls) {
      InlineFunctionInfo IFI;
      InlineResult IR = InlineFunction(*Dim3Call, IFI);
      assert(IR.isSuccess());
    }
  }
  auto Ret = ReturnInst::Create(M->getContext(), WhileEndBB);
  if (Options.MallocPreservedDataArray) {
    CallInst::CreateFree(PreservedData, Ret);
  }
}

void FunctionTransformer::getDim3StructType() {
  Function *Tmp;
  assignFunctionWithNameTo(M, Tmp, "__cpucuda_coerced_args_to_dim3");
  Dim3Type = Tmp->getReturnType();
  return;

  // Will use below if we transition to using dim3 ptrs instead of coalesced
  // args
  assignFunctionWithNameTo(M, Tmp, "__cpucuda_dim3_ptr_ret");
  Dim3PtrType = Tmp->getReturnType();
  Dim3Type = dyn_cast<PointerType>(Dim3PtrType)->getElementType();
}

void FunctionTransformer::createSelfContainedFunction() {
  // void __cpucuda_call_kernel_self_contained(
  //     dim3 grid_dim,
  //     dim3 block_dim,
  //     void** args,
  //     size_t shared_mem);
  Function *CpucudaCallKernel;
  assignFunctionWithNameTo(M, CpucudaCallKernel, "__cpucuda_call_kernel_self_contained");

  Function *ArgsToDim3F;
  assignFunctionWithNameTo(M, ArgsToDim3F, "__cpucuda_coerced_args_to_dim3");


  SelfContainedF = Function::Create(CpucudaCallKernel->getFunctionType(), F->getLinkage(),
                                    F->getAddressSpace(), F->getName(), F->getParent());
  BasicBlock *EntryBB = BasicBlock::Create(SelfContainedF->getContext(), "entry", SelfContainedF);
  BasicBlock *ExitBB = BasicBlock::Create(SelfContainedF->getContext(), "exit", SelfContainedF);
  ReturnInst::Create(M->getContext(), ExitBB);

  ValueVector CallArgs;
  for (unsigned i = 0; i < OriginalF->getFunctionType()->getNumParams(); i++) {
    ConstantInt *ArgIdx = ConstantInt::get(IntegerType::getInt32Ty(M->getContext()), i);
    // TODO Arg position will change with platform ABI
    auto ArgsPtrArg = SelfContainedF->getArg(4);
    auto SinglePtrTy = dyn_cast<PointerType>(ArgsPtrArg->getType())->getElementType();
    auto ArgPtr = GetElementPtrInst::Create(
        SinglePtrTy,
        ArgsPtrArg, {ArgIdx}, "cur_ptr", EntryBB);
    auto ArgPtrLoad = new LoadInst(
        SinglePtrTy, ArgPtr, "cur_ptr", EntryBB);
    auto CastArgPtr = new BitCastInst(
        ArgPtrLoad,
        PointerType::get(OriginalF->getArg(i)->getType(), SelfContainedF->getAddressSpace()),
        "cast_cur_ptr", EntryBB);
    auto Arg = new LoadInst(OriginalF->getArg(i)->getType(), CastArgPtr, "arg", EntryBB);
    CallArgs.push_back(Arg);

  }
  // TODO This will change with platform ABI
  // The two dim3's get passed coalesced
  vector<CallInst *> CallsToInline;
  int ArgIdx = 0;
  for (int Dim3Idx = 0; Dim3Idx < 2; Dim3Idx++) {
    ValueVector Dim3FArgs;
    for (unsigned i = 0; i < ArgsToDim3F->getFunctionType()->getNumParams(); i++) {
      Dim3FArgs.push_back(SelfContainedF->getArg(ArgIdx++));
    }
    auto ToDim3 = CallInst::Create(ArgsToDim3F->getFunctionType(), ArgsToDim3F, Dim3FArgs, "dim3", EntryBB);
    CallArgs.push_back(ToDim3);
    CallsToInline.push_back(ToDim3);
  }

  // TODO Args change with ABI
  ValueVector GridDim3Args = {SelfContainedF->getArg(0), SelfContainedF->getArg(1)};
  CallInst *GridDimx = CallInst::Create(
      Dim3Fs.Getterx->getFunctionType(), Dim3Fs.Getterx, GridDim3Args,
      "gridDim_x", EntryBB);
  CallsToInline.push_back(GridDimx);
  CallInst *GridDimy = CallInst::Create(
      Dim3Fs.Getterx->getFunctionType(), Dim3Fs.Gettery, GridDim3Args,
      "gridDim_y", EntryBB);
  CallsToInline.push_back(GridDimy);
  CallInst *GridDimz = CallInst::Create(
      Dim3Fs.Getterx->getFunctionType(), Dim3Fs.Getterz, GridDim3Args,
      "gridDim_z", EntryBB);
  CallsToInline.push_back(GridDimz);

  // For blocks in grid triple loop
  ThreadIdxLoop Loopz("_blockIdx_z_", GridDimz, SelfContainedF, this, GridDimx);
  ThreadIdxLoop Loopy("_blockIdx_y_", GridDimy, SelfContainedF, this, GridDimx);
  ThreadIdxLoop Loopx("_blockIdx_x_", GridDimx, SelfContainedF, this, GridDimx);
  Loopz.hookUpBBs(Loopy.EntryBB, Loopy.EndBB);
  Loopy.hookUpBBs(Loopx.EntryBB, Loopx.EndBB);

  BasicBlock *DriverFCallBB = BasicBlock::Create(DriverF->getContext(), "subkernel_call", SelfContainedF);

  CallInst *BlockIdx = CallInst::Create(
      Dim3Fs.ConstructorF->getFunctionType(), Dim3Fs.ConstructorF,
      {Loopx.Idx, Loopy.Idx, Loopz.Idx}, "blockIdx", DriverFCallBB);
  CallsToInline.push_back(BlockIdx);
  CallArgs.insert(CallArgs.end() - 1, BlockIdx);

  // handle different ABIs...
  CallArgs.push_back(SelfContainedF->getArg(5));

  CallInst::Create(DriverF->getFunctionType(), DriverF, CallArgs, "", DriverFCallBB);

  Loopx.hookUpBBs(DriverFCallBB, DriverFCallBB);
  BranchInst::Create(Loopz.EntryBB, EntryBB);
  BranchInst::Create(ExitBB, Loopz.EndBB);

  for (auto &F : CallsToInline) {
    InlineFunctionInfo IFI;
    InlineResult IR = InlineFunction(*F, IFI);
    assert(IR.isSuccess() && "Has to be inlined");
  }

  if (Options.UseSelfContainedKernel)
    SelfContainedF->takeName(OriginalF);

  // TODO We are leaking the args malloced memory... free it

}

void FunctionTransformer::createWrapperFunction() {
  // void __cpucuda_call_kernel(
  //     dim3 grid_dim,
  //     dim3 block_idx,
  //     dim3 block_dim,
  //     void** args,
  //     size_t shared_mem);
  Function *CpucudaCallKernel;
  assignFunctionWithNameTo(M, CpucudaCallKernel, "__cpucuda_call_kernel");

  Function *ArgsToDim3F;
  assignFunctionWithNameTo(M, ArgsToDim3F, "__cpucuda_coerced_args_to_dim3");


  WrapperF = Function::Create(CpucudaCallKernel->getFunctionType(), F->getLinkage(),
                              F->getAddressSpace(), F->getName(), F->getParent());
  BasicBlock *EntryBB = BasicBlock::Create(WrapperF->getContext(), "entry", WrapperF);
  BasicBlock *ExitBB = BasicBlock::Create(WrapperF->getContext(), "exit", WrapperF);
  ReturnInst::Create(M->getContext(), ExitBB);

  ValueVector CallArgs;
  for (unsigned i = 0; i < OriginalF->getFunctionType()->getNumParams(); i++) {
    ConstantInt *ArgIdx = ConstantInt::get(IntegerType::getInt32Ty(M->getContext()), i);
    // TODO Arg position will change with platform ABI
    auto ArgsPtrArg = WrapperF->getArg(6);
    auto SinglePtrTy = dyn_cast<PointerType>(ArgsPtrArg->getType())->getElementType();
    auto ArgPtr = GetElementPtrInst::Create(
        SinglePtrTy,
        ArgsPtrArg, {ArgIdx}, "cur_ptr", EntryBB);
    auto ArgPtrLoad = new LoadInst(
        SinglePtrTy, ArgPtr, "cur_ptr", EntryBB);
    auto CastArgPtr = new BitCastInst(
        ArgPtrLoad,
        PointerType::get(OriginalF->getArg(i)->getType(), WrapperF->getAddressSpace()),
        "cast_cur_ptr", EntryBB);
    auto Arg = new LoadInst(OriginalF->getArg(i)->getType(), CastArgPtr, "arg", EntryBB);
    CallArgs.push_back(Arg);

  }
  // TODO This will change with platform ABI
  // The three dim3's get passed coalesced
  vector<CallInst *> ToDim3Calls;
  int ArgIdx = 0;
  for (int Dim3Idx = 0; Dim3Idx < 3; Dim3Idx++) {
    ValueVector Dim3FArgs;
    for (unsigned i = 0; i < ArgsToDim3F->getFunctionType()->getNumParams(); i++) {
      Dim3FArgs.push_back(WrapperF->getArg(ArgIdx++));
    }
    auto ToDim3 = CallInst::Create(ArgsToDim3F->getFunctionType(), ArgsToDim3F, Dim3FArgs, "dim3", EntryBB);
    CallArgs.push_back(ToDim3);
    ToDim3Calls.push_back(ToDim3);
  }

  // Below would be how to pass a dim3 it if it was byval
  /*
    auto LastDim3Arg = WrapperF->getArg(ArgIdx++);
    Function *PtrToDim3F;
    assignFunctionWithNameTo(M, PtrToDim3F, "__cpucuda_dim3ptr_to_dim3");
    auto ToDim3 = CallInst::Create(PtrToDim3F->getFunctionType(), PtrToDim3F, LastDim3Arg, "dim3", EntryBB);
    ToDim3Calls.push_back(ToDim3);
    CallArgs.push_back(ToDim3);
  */

  // dynamic shared mem arg, will change with ABI
  CallArgs.push_back(WrapperF->getArg(7));

  CallInst::Create(DriverF->getFunctionType(), DriverF, CallArgs, "", EntryBB);
  BranchInst::Create(ExitBB, EntryBB);

  for (auto &F : ToDim3Calls) {
    InlineFunctionInfo IFI;
    InlineResult IR = InlineFunction(*F, IFI);
    assert(IR.isSuccess() && "Has to be inlined");
  }

  if (!Options.UseSelfContainedKernel)
    WrapperF->takeName(OriginalF);

}

void FunctionTransformer::getDim3Fs() {
  assignFunctionWithNameTo(M, Dim3Fs.ConstructorF, "__cpucuda_construct_dim3");
  assignFunctionWithNameTo(M, Dim3Fs.Getterx, "__cpucuda_dim3_get_x");
  assignFunctionWithNameTo(M, Dim3Fs.Gettery, "__cpucuda_dim3_get_y");
  assignFunctionWithNameTo(M, Dim3Fs.Getterz, "__cpucuda_dim3_get_z");
  assignFunctionWithNameTo(M, Dim3Fs.Dim3ToArg, "__cpucuda_dim3_to_arg");
}

void replaceAllFunctionUsesWith(Function *F, Function *RF) {
  vector<User *> FUsers(F->users().begin(), F->users().end());
  for (User *U : FUsers) {
    if (auto BCI = dyn_cast<BitCastInst>(U)) {
      auto NBCI = new BitCastInst(RF, BCI->getType(), "", BCI);
      NBCI->takeName(BCI);
      BCI->replaceAllUsesWith(NBCI);
      BCI->eraseFromParent();
    } else {
      assert(false && "Unhandled function usage case");
    }
  }
}

void FunctionTransformer::cleanup() {
  if (Options.UseSelfContainedKernel)
    //OriginalF->replaceAllUsesWith(SelfContainedF);
    replaceAllFunctionUsesWith(OriginalF, SelfContainedF);
  else
    //OriginalF->replaceAllUsesWith(WrapperF);
    replaceAllFunctionUsesWith(OriginalF, WrapperF);
  OriginalF->eraseFromParent();
  F->eraseFromParent();

  // Clean up the global shared variables
  for (GlobalVariable *G : CombinedSharedVars)
    G->eraseFromParent();

  // Dynamic shared var cannot be erased here because multiple template
  // instantiations of the same function could use the same extern __shared__
  // variable, we erase them after we are done cleaning up all of the
  // FunctionTransformer's
}

FunctionTransformer::FunctionTransformer(Module *M, Function *F, TargetTransformInfo *TTI) {
  this->M = M;
  this->F = F;
  this->TTI = TTI;

  LLVMBBIdType = IntegerType::getInt32Ty(M->getContext());
  LLVMSubkernelIdType = IntegerType::getInt32Ty(M->getContext());
  GepIndexType = IntegerType::getInt32Ty(M->getContext());
  I32Type = IntegerType::getInt32Ty(M->getContext());
  Dim3FieldType = IntegerType::getInt32Ty(M->getContext());
  // Will change with platform ABI maybe?
  SizeTType = IntegerType::getInt64Ty(M->getContext());

  getDim3Fs();
  getDim3StructType();

  createSubkernels();

  createDriverFunction();

  createWrapperFunction();

  createSelfContainedFunction();
}


void CPUCudaPass::cleanup(Module *M) {

  for (auto &Pair : FunctionTransformers) {
    Pair.second->cleanup();
  }

  // After we are done cleaning up individual FunctionTransformers we are now
  // sure that all usages of extern __shared__ variables are dead, we can delete
  // them now
  GlobalVarSet DeletedVars = {nullptr};
  for (auto &Pair : FunctionTransformers) {
    auto Var = Pair.second->DynamicSharedVar;
    if (!in_set(DeletedVars, Var)) {
      Var->eraseFromParent();
      DeletedVars.insert(Var);
    }
  }

  for (auto &Pair : FunctionTransformers) {
    delete Pair.second;
  }


  FunctionTransformers = std::map<Function *, FunctionTransformer *>();


  for (auto Name : {"__cpucuda_construct_dim3",
                    "__cpucuda_dim3_get_x",
                    "__cpucuda_dim3_get_y",
                    "__cpucuda_dim3_get_z",
                    "__cpucuda_coerced_args_to_dim3",
                    "__cpucuda_dim3ptr_to_dim3",
                    "__cpucuda_dim3_to_arg",
                    "__cpucuda_declared_function_user"}) {
    Function *ToErase;
    maybeAssignFunctionWithNameTo(M, ToErase, Name);
    if (ToErase)
      ToErase->eraseFromParent();
  }
}

void CPUCudaPass::transformCallSites(FunctionTransformer *FT) {
  Function *PushF;
  Function *LaunchKernelF;
  assignFunctionWithNameTo(M, PushF, "__cudaPushCallConfiguration");
  // When the compiler does some loop optimisations it sometimes splits the push
  // function and kernel call in different BBs which complicates converting both
  // of them to a single cudaLaunchKernel call, keep them separate for now
  bool convertToLaunchKernel = false;
  if (convertToLaunchKernel) {
    assignFunctionWithNameTo(M, LaunchKernelF, "__cpucudaLaunchKernel");
  }
  if (Options.UseSelfContainedKernel)
    assignFunctionWithNameTo(M, LaunchKernelF, "__cpucudaLaunchKernelSelfContainedWithPushedConfiguration");
  else
    assignFunctionWithNameTo(M, LaunchKernelF, "__cpucudaLaunchKernelWithPushedConfiguration");

  DataLayout *DL = new DataLayout(M);

  // TODO can a user appear twice in the users() if for example it has two
  // operands using the function value
  vector<User *> Users;
  for (User *U : FT->OriginalF->users()) {
    Users.push_back(U);
  }
  for (User *U : Users) {
    if (auto KernelCall = dyn_cast<CallInst>(U)) {
      if (KernelCall->getCalledFunction() != FT->OriginalF) {
        assert(false && "Unsupported case");
        continue;
      }

      CallInst *PushCall;
      Instruction *PrevInst = KernelCall;
      if (convertToLaunchKernel) {
        while (true) {
          PrevInst = PrevInst->getPrevNonDebugInstruction();
          PushCall = dyn_cast<CallInst>(PrevInst);
          if (PushCall && PushCall->getCalledFunction() == PushF)
            break;
        }
      }

      auto AS = KernelCall->getParent()->getParent()->getAddressSpace();
      auto Int8Ty = IntegerType::getInt8Ty(M->getContext());
      auto Int8PtrTy = PointerType::get(Int8Ty, AS);
      auto Int32Ty = IntegerType::getInt32Ty(M->getContext());

      int MallocSize = DL->getMaxPointerSizeInBits() * KernelCall->getNumArgOperands();
      for (unsigned i = 0; i < KernelCall->getNumArgOperands(); ++i) {
        auto ArgVal = KernelCall->getArgOperand(i);
        MallocSize += DL->getTypeAllocSize(ArgVal->getType());
      }

      Instruction *ArgPtrArray = CallInst::CreateMalloc(
          KernelCall,
          Int32Ty,
          Int8PtrTy,
          ConstantInt::get(Int32Ty, MallocSize),
          nullptr, nullptr, "arg_ptr_array");
      Value *ArgArray = GetElementPtrInst::Create(
          Int8PtrTy, ArgPtrArray, {ConstantInt::get(Int32Ty, KernelCall->getNumArgOperands())},
          "arg_ptr", KernelCall);
      ArgArray = new BitCastInst(ArgArray, Int8PtrTy, "cast_arg_ptr", KernelCall);
      int ArgArrayIdx = 0;
      for (unsigned i = 0; i < KernelCall->getNumArgOperands(); ++i) {
        auto ArgVal = KernelCall->getArgOperand(i);
        auto ArgPtr = GetElementPtrInst::Create(
            Int8PtrTy, ArgPtrArray, {ConstantInt::get(Int32Ty, i)}, "arg_ptr", KernelCall);
        auto CastArgPtr = new BitCastInst(
            ArgPtr, PointerType::get(PointerType::get(ArgVal->getType(), AS), AS), "arg_ptr_bitcast", KernelCall);
        auto _ArgMalloc = GetElementPtrInst::Create(
            Int8Ty, ArgArray, {ConstantInt::get(Int32Ty, ArgArrayIdx)},
            "arg_malloc", KernelCall);
        auto ArgMalloc = new BitCastInst(
            _ArgMalloc, PointerType::get(ArgVal->getType(), AS),
            "arg_malloc_bitcast", KernelCall);
        new StoreInst(ArgVal, ArgMalloc, KernelCall);
        new StoreInst(ArgMalloc, CastArgPtr, KernelCall);
        ArgArrayIdx += DL->getTypeAllocSize(ArgVal->getType());
      }
      ValueVector Args;
      Function *KernelFunction;
      if (Options.UseSelfContainedKernel)
        KernelFunction = FT->SelfContainedF;
      else
        KernelFunction = FT->WrapperF;

      auto CastWrapperF = new BitCastInst(
          KernelFunction, LaunchKernelF->getArg(0)->getType(), "kernel_bitcast", KernelCall);
      Args.push_back(CastWrapperF);
      int PushCallArgIdx = 0;
      if (convertToLaunchKernel) {
        // grid dim
        Args.push_back(PushCall->getArgOperand(PushCallArgIdx++));
        Args.push_back(PushCall->getArgOperand(PushCallArgIdx++));
        // block dim
        Args.push_back(PushCall->getArgOperand(PushCallArgIdx++));
        Args.push_back(PushCall->getArgOperand(PushCallArgIdx++));
      }

      // void **args
      //LoadInst *ArgArrayL = new LoadInst(PointerType::get(Int8PtrTy, AS), ArgArray, "args", KernelCall);
      Args.push_back(ArgPtrArray);

      if (convertToLaunchKernel) {
        // share mem size
        Args.push_back(PushCall->getArgOperand(PushCallArgIdx++));
        // stream
        auto Stream = PushCall->getArgOperand(PushCallArgIdx++);
        // TODO the argument number will change depending on ABI I think
        auto StreamCast = new BitCastInst(
            Stream, LaunchKernelF->getArg(7)->getType(), "stream_bitcast", KernelCall);
        Args.push_back(StreamCast);
      }

      CallInst::Create(LaunchKernelF->getFunctionType(), LaunchKernelF, Args, "", KernelCall);

      if (convertToLaunchKernel) {
        PushCall->eraseFromParent();
      }
      KernelCall->eraseFromParent();
    } else {
      // As far as I know, there should be no other instruction type calling the
      // kernel itself, some might use it as a parameter, for example a call to
      // cudaFuncSetCacheConfig(kernel)
    }
  }
}

namespace {

typedef map<Value *, ValueSet> EdgesTy;

void findSharedVarDeps(Value *Val, ValueSet &Nodes, EdgesTy &Edges) {
  for (User *U : Val->users()) {
    if (isa<Instruction>(U)) {
      Edges[U].insert(Val);
      Nodes.insert(U);
      Nodes.insert(Val);
    } else if (isa<Constant>(U)) {
      Edges[U].insert(Val);
      Nodes.insert(U);
      Nodes.insert(Val);
      findSharedVarDeps(U, Nodes, Edges);
    }
  }
}

Instruction *_constsToInsts(Value *_V,
                            ValueSet &Nodes,
                            EdgesTy &Edges,
                            map<Constant*, Instruction *> &Converted,
                            Instruction *InsertBefore) {

  auto I = dyn_cast<Instruction>(_V);
  auto C = dyn_cast<Constant>(_V);

  Instruction *NI;

  if (C) {
    auto ConvertedCIt = Converted.find(C);
    if (ConvertedCIt != Converted.end())
      return ConvertedCIt->second;

    if (auto CE = dyn_cast<ConstantExpr>(C)) {
      auto Opcode = CE->getOpcode();

      switch (Opcode) {
      case Instruction::FPExt:
      case Instruction::FPToSI:
      case Instruction::FPToUI:
      case Instruction::FPTrunc:
      case Instruction::IntToPtr:
      case Instruction::PtrToInt:
      case Instruction::SExt:
      case Instruction::SIToFP:
      case Instruction::Trunc:
      case Instruction::UIToFP:
      case Instruction::ZExt:
      case Instruction::BitCast: {
        NI = CastInst::Create((CastInst::CastOps)Opcode, CE->getOperand(0), CE->getType(), CE->getName(), InsertBefore);
        break;
      }
      case Instruction::GetElementPtr: {
        ValueVector Idxs;
        for (unsigned i = 1; i < CE->getNumOperands(); ++i)
          Idxs.push_back(CE->getOperand(i));
        NI = GetElementPtrInst::Create(
            dyn_cast<PointerType>(CE->getOperand(0)->getType())->getElementType(),
            CE->getOperand(0), Idxs, CE->getName(), InsertBefore);
        break;
      }
      default: {
        assert(false && "TODO ConstantExpr case not handled yet");
        break;
      }
      }
    } else if (auto CA = dyn_cast<ConstantAggregate>(C)) {
      assert(false && "TODO ConstantAggregate not handled yet");
    } else {
      assert(false && "????");
    }

    Converted[C] = NI;
    InsertBefore = NI;

  } else if (I) {
    NI = I;
  } else {
    assert(false && "???");
  }

  ValueToValueMapTy VMap;
  for (auto V : Edges[_V]) {
    // GlobalValues and ConstantData do not need further processing
    if (isa<GlobalValue>(V))
      continue;
    if (isa<ConstantData>(V))
      continue;
    assert(isa<Constant>(V) && "?????");
    // Convert the used constant to an instruction
    auto DepI = _constsToInsts(V, Nodes, Edges, Converted, InsertBefore);
    VMap[V] = DepI;
  }
  // Remap the used constants to the converted instructions
  RemapInstruction(NI, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

  return NI;
}

void constsToInsts(ValueSet &Nodes, EdgesTy &Edges) {
  Function *F = nullptr;
  map<Constant *, Instruction *> Converted;
  for (auto Val : Nodes) {
    if (auto I = dyn_cast<Instruction>(Val)) {
      F = I->getParent()->getParent();
      auto FirstI = F->getEntryBlock().getFirstNonPHI();
      _constsToInsts(I, Nodes, Edges, Converted, FirstI);
    }
  }
}

void deleteBrokenConsts(ValueSet &Nodes, EdgesTy Edges) {
  ValueVector ToDelete;
  for (auto Val : Nodes)
    if (isa<Constant>(Val) && !isa<GlobalValue>(Val))
      ToDelete.push_back(Val);

  while (!ToDelete.empty()) {
    for (auto It = ToDelete.begin(); It != ToDelete.end(); ) {
      auto C = dyn_cast<Constant>(*It);
      assert(C);
      if (C->hasNUses(0)) {
        C->destroyConstant();
        It = ToDelete.erase(It);
      } else {
        ++It;
      }
    }
  }
}

void breakConstExprUsages(Constant *C) {
  ValueSet Nodes;
  EdgesTy Edges;
  findSharedVarDeps(cast<Value>(C), Nodes, Edges);
  constsToInsts(Nodes, Edges);

  // Now all constants should be dead, delete them
  deleteBrokenConsts(Nodes, Edges);
}

bool isGlobalFunction(Function *F) {
  return F->hasFnAttribute(Attribute::CPUCUDAGlobal);
}

// Replaces constants which depend on shared variables with instructions
void breakConstExprGlobalUsages(Module *M) {
  for (auto &G : M->globals()) {
    if (isSharedVar(&G)) {
      breakConstExprUsages(&G);
    }
  }
  for (auto &F : *M) {
    if (isGlobalFunction(&F)) {
      breakConstExprUsages(&F);
    }
  }
}

} // namespace

PreservedAnalyses CPUCudaPass::run(Module &M,
                                   AnalysisManager<Module> &AM) {
  this->M = &M;
#ifdef COST_ANALYSIS
  TTI = &AM.getResult<TargetIRAnalysis>(M);
#endif

  breakConstExprGlobalUsages(&M);

  vector<Function *> OriginalFs;

  // TODO Does this include function declarations without definitions? If so, we
  // have to treat them separately
  for (auto &F : M) {
    OriginalFs.push_back(&F);
  }
  // Transform global functions
  for (auto _F : OriginalFs) {

    Function &F = *_F;

    if (!isGlobalFunction(&F))
      continue;

    LLVM_DEBUG(errs() << "processing function " << F.getName() << "\n");

    FunctionTransformers[&F] = new FunctionTransformer(&M, &F, TTI);

  }

  for (auto &Pair : FunctionTransformers) {
    transformCallSites(Pair.second);
  }

  cleanup(&M);

  // TODO optimise the preserved sets, although preserving anything seems
  // unlikely
  return PreservedAnalyses::none();
}
