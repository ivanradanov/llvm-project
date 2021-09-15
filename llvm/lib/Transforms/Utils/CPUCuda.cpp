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

#include <queue>
#include <vector>
#include <string>
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "cpucudapass"

static const int MAX_CUDA_THREADS = 1024;

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
typedef vector<Type *> TypeVector;
typedef vector<Instruction *> InstVector;
typedef vector<Argument *> ArgVector;

typedef int SubkernelIdType;
typedef int BBIdType;

const vector<std::string> Dim3Names = {
  "gridDim",
  "blockIdx",
  "blockDim",
  "threadIdx"
};

struct UsedValVars {
  ValueSet usedVals;
  InstSet usedSharedVars;
};

class FunctionTransformer {
public:
	struct {
		bool DynamicPreservedDataArray = true;
	} Options;

public:
  Module *M;
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
  map<SubkernelIdType, map<SubkernelIdType, InstVector>> SubkernelUsedSharedVars;
  map<SubkernelIdType, map<BasicBlock *, BBIdType>> SubkernelBBIds;
  map<BBIdType, BasicBlock *> OriginalFunBBs;

  map<SubkernelIdType, map<Value *, int>> IndexInCombinedDataType;
  map<SubkernelIdType, ValueVector> CombinedUsedVals;
  StructType *CombinedDataType;

  map<SubkernelIdType, map<Instruction *, int>> IndexInCombinedSharedVarsDataType;
  map<SubkernelIdType, InstVector> CombinedSharedVars;
  StructType *SharedVarsDataType;

  Function *DriverF;

  // Label type for which BB id we should continue from after we return or we
  // have come from
  IntegerType *LLVMBBIdType;
  IntegerType *LLVMSubkernelIdType;
  StructType *SubkernelReturnType;
  IntegerType *GepIndexType;
  IntegerType *Dim3FieldType;
  Type *Dim3Type;

  void splitBlocksAroundBarriers(Function &F);
  bool blockIsAfterBarrier(BasicBlock *BB);
  bool blockIsAfterBarrier(SubkernelIdType SK, BasicBlock *BB);
  void _findSubkernelBBs(BasicBlock *BB, BBSet &visited);
  UsedValVars findUsedVals(SubkernelIdType SK, BasicBlock *BB, ValueSet definedVals, BBVector visited);
  void findSubkernelUsedVals();
  SubkernelIdType findSubkernelFromBB(BBIdType BB);
  void createSubkernelFunctionClones();
  set<SubkernelIdType> getSubkernelSuccessors(SubkernelIdType SK);
  Type *getSubkernelReturnDataFieldType(SubkernelIdType FromSK, SubkernelIdType SuccSK);
  StructType *getSubkernelsReturnType();
  void assignBBIds();
  TypeVector getSubkernelParams(SubkernelIdType SK);
  vector<StringRef> getSubkernelParamNames(SubkernelIdType SK);
  void transformSubkernels(SubkernelIdType SK);
  void findSubkernelBBs(Function &F);
  void findSharedVars();
  void createSubkernels();
  Type *getCombinedDataType();
  int getValIndexInCombinedDataType(SubkernelIdType SK, Value *Val);
  void sortValueVector(SubkernelIdType SK, ValueVector &VV, map<Value *, int> &Indices);
  void removeReferencesInPhi(const BBVector &BBsToRemove);
  bool isSharedVar(Instruction &I);
  void createDriverFunction();
  void replaceDim3Usages();
  Type *getDim3StructType();
  void getDim3Fs();
  ValueVector convertDim3ToArgs(Value *D, Instruction *After);
	void cleanupFunctions();

  FunctionTransformer(Module *M, Function *F);

};

// TODO use proper function name mangling
bool callIsBarrier(CallInst *callInst) {
  if (Function *calledFunction = callInst->getCalledFunction()) {
    return calledFunction->getName().endswith("__cpucuda_syncthreadsv");
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
      //for (auto &instruction : bb) {
      for (auto _begin = ++bb.begin(); _begin != bb.end(); ++_begin) {
        auto &instruction = *_begin;
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

    Value *DataStructPtr = F->getArg(1);

    ConstantInt *Zero = ConstantInt::get(Pass->GepIndexType, 0);

    for (auto &Val : Pass->CombinedUsedVals[SK]) {
      Instruction *Inst = dyn_cast<Instruction>(Val);
      if (!Inst) {
        // TODO in the case when the val is an argument to the function, we must
        // insert it into the data param prior to the first call to a subkernel
        assert(dyn_cast<Argument>(Val) && "Values in CombinedUsedVals must be instructions or arguments");
        continue;
      }
      BasicBlock *ValBB = Inst->getParent();
      // We are only interested in values which are defined in the current
      // subkernel
      if (!in_vector(Pass->SubkernelBBs[SK], ValBB))
        continue;

      ConstantInt *Index = ConstantInt::get(
        Pass->GepIndexType, Pass->getValIndexInCombinedDataType(SK, Val));
      Instruction *NextInst = Inst->getNextNonDebugInstruction();
      // If next inst is Phi, we have to get the first following non-phi
      // instruction because all Phi's must be bunched at the start of a BB
      if (dyn_cast<PHINode>(NextInst)) {
        NextInst = NextInst->getParent()->getFirstNonPHI();
      }
      GetElementPtrInst *Gep = GetElementPtrInst::Create(
        Pass->getCombinedDataType(), DataStructPtr, {Zero, Index}, "", NextInst);
      assert(NextInst && "The Inst must not be a terminator instruction so a next instruction has to exist");
      new StoreInst(Val, Gep, NextInst);
    }

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

  void visitSwitchInst(SwitchInst &I) {
    assert(false && "Not yet implemented");
  }

  void visitIndirectBrInst(IndirectBrInst &I) {
    assert(false && "Not yet implemented");
  }

  void visitResumeInst(ResumeInst &I) {
    assert(false && "Not yet implemented");
  }

  void visitUnreachableInst(UnreachableInst &I) {
    assert(false && "Not yet implemented");
  }

  void visitCleanupReturnInst(CleanupReturnInst &I) {
    assert(false && "Not yet implemented");
  }

  void visitCatchReturnInst(CatchReturnInst &I) {
    assert(false && "Not yet implemented");
  }

  void visitCatchSwitchInst(CatchSwitchInst &I) {
    assert(false && "Not yet implemented");
  }

};

void FunctionTransformer::findSubkernelBBs(Function &F) {
  std::set<BasicBlock *> visited;
  _findSubkernelBBs(&F.getEntryBlock(), visited);
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
  for (auto Val : VV) {
    if (Instruction *I = dyn_cast<Instruction>(Val))
      IV.push_back(I);
    else if (Argument *A = dyn_cast<Argument>(Val))
      AV.push_back(A);
    else
      assert(false && "Used vals must be only instructions or arguments");
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

bool FunctionTransformer::isSharedVar(Instruction &I) {
  auto *Metadata = I.getMetadata(LLVMContext::MD_annotation);
  if (Metadata) {
    auto *Tuple = cast<MDTuple>(Metadata);
    for (auto &N : Tuple->operands())
      if (cast<MDString>(N.get())->getString() == "cpucuda_shared")
        return true;
  }
  return false;
}

// TODO test this function

// I am not sure whether the order in which we visit the BBs might affect the
// result - PHINode related problems?
UsedValVars FunctionTransformer::findUsedVals(SubkernelIdType SK, BasicBlock *BB, ValueSet definedVals, BBVector visited) {
  UsedValVars usedVals;

  if (in_vector(visited, BB))
    return usedVals;
  visited.push_back(BB);

  // Find all values which were not yet defined but are used
  for (auto &I : *BB) {
    definedVals.insert(&I);
    for (Use &U : I.operands()) {
      Value *v = U.get();
      // TODO is that all the cases?
      // We don't want Constants or Undef values
      Instruction *UseI = dyn_cast<Instruction>(v);
      // Do not add arguments as we will keep them in the subkernels
      if (UseI) {
        if (isSharedVar(*UseI))
          usedVals.usedSharedVars.insert(UseI);
        else if (!in_set(definedVals, v))
          usedVals.usedVals.insert(v);
      }
    }
  }

  // Recursively do the same for all basic blocks in the same subkernel
  for (auto Succ : successors(BB)) {
    if (blockIsAfterBarrier(SK, Succ))
      continue;
    UsedValVars _usedVals = findUsedVals(SK, Succ, definedVals, visited);
    usedVals.usedVals.insert(_usedVals.usedVals.begin(), _usedVals.usedVals.end());
  }

  return usedVals;
}

// I don't like this implementation (that it needs sorting), find something
// better if possible

// Currently only tracks registers and not values written to memory
void FunctionTransformer::findSubkernelUsedVals() {
  for (auto SK : SubkernelIds) {
    // Find values which persist across executions of different subkernels
    ValueVector CombinedUsedVals;
    for (auto _SK : SubkernelIds) {
      // Convert references of basic blocks to the cloned function
      BasicBlock *_SKEntryBB = convertBasicBlock(SubkernelBBs[_SK][0], SubkernelFs[_SK], SubkernelFs[SK]);
      UsedValVars UsedVals = findUsedVals(SK, _SKEntryBB, ValueSet(), BBVector());
      SubkernelUsedVals[SK][_SK] = ValueVector(UsedVals.usedVals.begin(), UsedVals.usedVals.end());
      SubkernelUsedSharedVars[SK][_SK] = InstVector(UsedVals.usedSharedVars.begin(), UsedVals.usedSharedVars.end());

      for (auto Val : UsedVals.usedVals) {
        if (!in_vector(CombinedUsedVals, Val)) {
          CombinedUsedVals.push_back(Val);
        }
      }
    }

    map<Value *, int> IndexInCombinedDataType;
    sortValueVector(SK, CombinedUsedVals, IndexInCombinedDataType);
    this->CombinedUsedVals[SK] = CombinedUsedVals;
    this->IndexInCombinedDataType[SK] = IndexInCombinedDataType;

    ValueVector CombinedSharedVars;
    for (Instruction *I : this->CombinedSharedVars[SK])
      CombinedSharedVars.push_back(I);
    map<Value *, int> IndexInCombinedSharedVarsDataType;
    sortValueVector(SK, CombinedSharedVars, IndexInCombinedSharedVarsDataType);
    for (auto &pair : IndexInCombinedSharedVarsDataType) {
      auto Val = pair.first;
      auto Index = pair.second;
      Instruction *I = dyn_cast<Instruction>(Val);
      assert(I);
      this->IndexInCombinedSharedVarsDataType[SK][I] = Index;
    }
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

// Deprecated
/*Type *FunctionTransformer::getSubkernelReturnDataFieldType(SubkernelIdType FromSK, SubkernelIdType SuccSK) {
  auto UsedVals = SubkernelUsedVals[FromSK][SuccSK];
  vector<Type *> types;
  for (auto Val : UsedVals) {
  types.push_back(Val->getType());
  }
  return StructType::get(M->getContext(), ArrayRef<Type *>(types));
  }*/

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
  params.push_back(PointerType::get(M->getContext(), SubkernelFs[SK]->getAddressSpace()));

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

  // Add return from exiting blocks
  for (auto &bb : nfunc_bbs) {
    Instruction *term = bb->getTerminator();
    TransformTerminator transformer(SK, this);
    transformer.visit(term);
  }

  // Construct the entry block which sets up the usedVals params and handles phi
  // instructions
  {
    BasicBlock *EntryBB = BasicBlock::Create(nf->getContext(), "generated_entry_block", nf, &nf->getEntryBlock());

    ConstantInt *Zero = ConstantInt::get(Type::getInt32Ty(nf->getContext()), 0);

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
        Val->replaceAllUsesWith(UnpackedVal);
        UnpackedVal->takeName(Val);
      }
      // The index at which the original arguments start
      unsigned i = 4;
      for (auto &Arg : _nf->args()) {
	      Arg.replaceAllUsesWith(nf->getArg(i));
	      ++i;
      }
    }

    {
      auto usedSharedVars = SubkernelUsedSharedVars[SK][SK];
      // Transfer usages of the used shared vars to the arguments to the function
      auto It = usedSharedVars.begin(), E = usedSharedVars.end();
      // Unpack args from data struct param and replace usages with them
      for (unsigned i = 0; It != E; ++It, ++i) {
        Instruction *I = (*It);
        ConstantInt *Index = ConstantInt::get(
          GepIndexType, IndexInCombinedSharedVarsDataType[SK][I]);
        // The third argument of the function is the structure of shared variables
        GetElementPtrInst *Gep = GetElementPtrInst::Create(
          SharedVarsDataType, nf->getArg(2), {Zero, Index}, "", EntryBB);
        I->replaceAllUsesWith(Gep);
        Gep->takeName(I);
        I->eraseFromParent();
      }
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
  // At this point, the unused basic blocks in nf might still use the
  // arguments of _nf so we cannot delete it yet

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

void FunctionTransformer::findSharedVars() {
  // TODO find where variable Alloca's are getting optimised out and disable it
  // for shared vars
  for (auto SK : SubkernelIds) {
    for (auto &BB : *SubkernelFs[SK]) {
      for (auto &I : BB) {
        if (isSharedVar(I))
          CombinedSharedVars[SK].push_back(&I);
      }
    }
  }

  TypeVector Types;
  for (auto I : CombinedSharedVars[0]) {
    assert(isa<AllocaInst>(I));
    Types.push_back(dyn_cast<AllocaInst>(I)->getAllocatedType());
  }
  SharedVarsDataType = StructType::get(M->getContext(), Types);
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
          // TODO use proper function name mangling
          for (unsigned i = 0; i < Dim3Names.size(); ++i)
            if (calledFunction->getName().endswith("__cpucuda_" + Dim3Names[i] + "v")) {
              auto Arg = F->getArg(i + Dim3ArgStartIndex);
              callInst->replaceAllUsesWith(Arg);
              It = callInst->eraseFromParent();
              erased = true;
              break;
            }
      if (!erased)
        ++It;

    }
}

void FunctionTransformer::createSubkernels() {
  replaceDim3Usages();
  splitBlocksAroundBarriers(*F);
  findSubkernelBBs(*F);
  createSubkernelFunctionClones();
  assignBBIds();
  findSharedVars();
  findSubkernelUsedVals();
  SubkernelReturnType = getSubkernelsReturnType();
  for (auto SK : SubkernelIds) {
    transformSubkernels(SK);
  }
}

class ThreadIdxLoop {
public:
  FunctionTransformer *T;
  Function *F;

  BasicBlock *EntryBB, *CondBB, *IncrBB, *EndBB;
  AllocaInst *IdxPtr;
  LoadInst *Idx;
  ICmpInst *Cond;

  BranchInst *FromCond;

  ThreadIdxLoop(std::string IdxName, Value *LoopTo, Function *F, FunctionTransformer *T) {
    this->F = F;
    this->T = T;

    EntryBB = BasicBlock::Create(F->getContext(), "loop_entry" + IdxName, F);

    IdxPtr = new AllocaInst(T->Dim3FieldType, F->getAddressSpace(),
                            ConstantInt::get(T->Dim3FieldType, 1),
                            IdxName + "_ptr", EntryBB);
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
        if (Call->getCalledFunction()->getName().contains("__cpucuda_declared_dim3_getter")) {
          VMap[I] = D;
          continue;
        }
      if (CallInst *Call = dyn_cast<CallInst>(I))
        if (Call->getCalledFunction()->getName().contains("__cpucuda_declared_dim3_user")) {
          ValueVector Args;
          for (unsigned i = 0; i < Call->getNumArgOperands(); ++i) {
            Args.push_back(VMap[Call->getArgOperand(i)]);
          }
          return Args;
        }
      auto *NI = I->clone();
      NI->insertAfter(After);
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

  // Now we have an empty function
  DriverF = NewF;

  int Dim3ArgStartIndex = FT->getNumParams();
  Argument *BlockDimArg = DriverF->getArg(Dim3ArgStartIndex + 2);

  ConstantInt *Zero = ConstantInt::get(GepIndexType, 0);
  ConstantInt *One = ConstantInt::get(GepIndexType, 1);
  ConstantInt *mOne = ConstantInt::get(GepIndexType, -1);

  BasicBlock *EntryBB = BasicBlock::Create(DriverF->getContext(), "entry", DriverF);

  AllocaInst *StaticSharedData = new AllocaInst(SharedVarsDataType, DriverF->getAddressSpace(), One, "static_shared_data", EntryBB);
  // TODO Handle dynamic shared data
  UndefValue *DynSharedData = UndefValue::get(PointerType::get(M->getContext(), DriverF->getAddressSpace()));

  ValueVector Dim3Args = convertDim3ToArgs(BlockDimArg, StaticSharedData);
  CallInst *BlockDimx = CallInst::Create(
	  Dim3Fs.Getterx->getFunctionType(), Dim3Fs.Getterx, Dim3Args,
	  "blockDim_x", EntryBB);
  CallInst *BlockDimy = CallInst::Create(
	  Dim3Fs.Getterx->getFunctionType(), Dim3Fs.Gettery, Dim3Args,
	  "blockDim_y", EntryBB);
  CallInst *BlockDimz = CallInst::Create(
	  Dim3Fs.Getterx->getFunctionType(), Dim3Fs.Getterz, Dim3Args,
	  "blockDim_z", EntryBB);

  AllocaInst *PreservedData;
  if (!Options.DynamicPreservedDataArray) {
	  ConstantInt *MaxCudaThreads = ConstantInt::get(GepIndexType, MAX_CUDA_THREADS);
    PreservedData = new AllocaInst(CombinedDataType, DriverF->getAddressSpace(),
                                   MaxCudaThreads, "preserved_data", EntryBB);
  } else {
	  BinaryOperator *BlockSize = BinaryOperator::Create(
		  Instruction::BinaryOps::Mul, BlockDimx, BlockDimy, "blockDimMul", EntryBB);
	  BlockSize = BinaryOperator::Create(
		  Instruction::BinaryOps::Mul, BlockSize, BlockDimz, "blockSize", EntryBB);
	  PreservedData = new AllocaInst(CombinedDataType, DriverF->getAddressSpace(),
	                                 BlockSize, "preserved_data", EntryBB);
  }

  AllocaInst *SubkernelRetPtr = new AllocaInst(SubkernelReturnType, DriverF->getAddressSpace(), One, "ret", EntryBB);

  GetElementPtrInst *SubkernelRetFromPtr = GetElementPtrInst::Create(
    SubkernelReturnType, SubkernelRetPtr, {Zero, Zero}, "", EntryBB);
  SubkernelRetFromPtr->setName("from_ptr");
  new StoreInst(mOne, SubkernelRetFromPtr, EntryBB);

  GetElementPtrInst *SubkernelRetNextPtr = GetElementPtrInst::Create(
    SubkernelReturnType, SubkernelRetPtr, {Zero, One}, "", EntryBB);
  SubkernelRetNextPtr->setName("next_ptr");
  new StoreInst(Zero, SubkernelRetNextPtr, EntryBB);

  BasicBlock *WhileEntryBB = BasicBlock::Create(DriverF->getContext(), "while_entry", DriverF);
  BranchInst::Create(WhileEntryBB, EntryBB);
  LoadInst *Next = new LoadInst(LLVMBBIdType, SubkernelRetNextPtr, "next", WhileEntryBB);
  LoadInst *From = new LoadInst(LLVMBBIdType, SubkernelRetFromPtr, "from", WhileEntryBB);

  BasicBlock *WhileEndBB = BasicBlock::Create(DriverF->getContext(), "while_end", DriverF);
  SwitchInst *Switch = SwitchInst::Create(Next, WhileEndBB, 0, WhileEntryBB);

  for (auto SK : SubkernelIds) {
    ConstantInt *CaseConst = llvm::ConstantInt::get(LLVMBBIdType, SK, /* isSigned */ true);
    BasicBlock *SwitchCaseBB = BasicBlock::Create(DriverF->getContext(), "switch_case", DriverF);
    Switch->addCase(CaseConst, SwitchCaseBB);

    ThreadIdxLoop Loopz("threadIdx_z", BlockDimz, DriverF, this);
    ThreadIdxLoop Loopy("threadIdx_y", BlockDimy, DriverF, this);
    ThreadIdxLoop Loopx("threadIdx_x", BlockDimx, DriverF, this);
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

    GetElementPtrInst *ThreadPreservedData = GetElementPtrInst::Create(
	    CombinedDataType, PreservedData, {PreservedDataIdx}, "threadPreservedData", SubkernelCallBB);

    ValueVector Args = {From, ThreadPreservedData, StaticSharedData, DynSharedData};
    // original args + gridDim, blockIdx, blockDim
    for (auto &Arg : DriverF->args()) {
	    Args.push_back(&Arg);
    }
    // threadIdx
    CallInst *ThreadIdx = CallInst::Create(
	    Dim3Fs.ConstructorF->getFunctionType(), Dim3Fs.ConstructorF,
	    {Loopx.Idx, Loopy.Idx, Loopz.Idx}, "threadIdx", SubkernelCallBB);
    Args.push_back(ThreadIdx);
    CallInst *SubkernelCall = CallInst::Create(
      SubkernelFs[SK]->getFunctionType(), SubkernelFs[SK],
      Args, "local_ret", SubkernelCallBB);
    new StoreInst(SubkernelCall, SubkernelRetPtr, SubkernelCallBB);
    // TODO inline those CallInsts

    Loopx.hookUpBBs(SubkernelCallBB, SubkernelCallBB);

    BranchInst::Create(Loopz.EntryBB, SwitchCaseBB);
    BranchInst::Create(WhileEntryBB, Loopz.EndBB);
  }

  ReturnInst::Create(M->getContext(), WhileEndBB);

}

Type *FunctionTransformer::getDim3StructType() {
  for (auto &F : *M)
    for (auto &bb : F)
      for (auto &instruction : bb)
        if (CallInst *callInst = dyn_cast<CallInst>(&instruction))
          if (Function *calledFunction = callInst->getCalledFunction())
            // TODO use proper function name mangling
            for (auto &name : Dim3Names)
              if (calledFunction->getName().endswith("__cpucuda_" + name + "v"))
                return calledFunction->getReturnType();
  // Return any random type - this should never happen anyways because all
  // kernels should use some of the dim3 variables
  LLVM_DEBUG(dbgs() << "Could not find dim3 struct type from function " << F->getName() << "\n");
  return LLVMBBIdType;
}

// TODO fix this ugly hack.

// Since we do not know how the dim3 structure will be represented in LLVM IR
// (It might depend on architecture or OS? I am not sure) this is a function
// which takes 3 arguments x, y, z and returns a dim3 structure

// TODO I think we need to inline all usages after we have used it and delete it
// because otherwise we will get an error about multiple definitions of the same
// function when linking
void assignToIfContains(Module *M, Function *&Assign, std::string String) {
  Assign = nullptr;
  for (auto &F : *M)
    if (F.getName().contains(String)) {
      Assign = &F;
      break;
    }
  assert(Assign);
}

void FunctionTransformer::getDim3Fs() {
  assignToIfContains(M, Dim3Fs.ConstructorF, "__cpucuda_construct_dim3");
  assignToIfContains(M, Dim3Fs.Getterx, "__cpucuda_dim3_get_x");
  assignToIfContains(M, Dim3Fs.Gettery, "__cpucuda_dim3_get_y");
  assignToIfContains(M, Dim3Fs.Getterz, "__cpucuda_dim3_get_z");
  assignToIfContains(M, Dim3Fs.Dim3ToArg, "__cpucuda_dim3_to_arg");
}

void FunctionTransformer::cleanupFunctions() {
	std::string NewName = std::string(OriginalF->getName()) + "_original";
	DriverF->takeName(OriginalF);
	OriginalF->setName(NewName);
}

FunctionTransformer::FunctionTransformer(Module *M, Function *F) {
  this->M = M;
  this->F = F;

  LLVMBBIdType = IntegerType::getInt32Ty(M->getContext());
  LLVMSubkernelIdType = IntegerType::getInt32Ty(M->getContext());
  GepIndexType = IntegerType::getInt32Ty(M->getContext());
  Dim3FieldType = IntegerType::getInt32Ty(M->getContext());
  getDim3Fs();
  Dim3Type = getDim3StructType();

  createSubkernels();

  createDriverFunction();

  cleanupFunctions();

}

PreservedAnalyses CPUCudaPass::run(Module &M,
                                   AnalysisManager<Module> &AM) {
  vector<Function *> OriginalFs;

  // TODO Does this include function declarations without definitions? If so, we
  // have to treat them separately
  for (auto &F : M) {
    OriginalFs.push_back(&F);
  }
  // Transform global functions
  for (auto _F : OriginalFs) {

    Function &F = *_F;

    if (!F.hasFnAttribute(Attribute::CPUCUDAGlobal))
      continue;

    LLVM_DEBUG(errs() << "processing function " << F.getName() << "\n");

    FunctionTransformer(&M, &F);

  }

  // TODO cleanup dim3 helper functions

  // TODO optimise the preserved sets, although preserving anything seems
  // unlikely
  return PreservedAnalyses::none();
}
