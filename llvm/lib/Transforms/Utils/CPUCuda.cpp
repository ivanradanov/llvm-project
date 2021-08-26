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

void CPUCudaPass::splitBlocksAroundBarriers(Function &F) {
  while ([&]() -> bool {
    for (auto &bb : F) {
      //for (auto &instruction : bb) {
      for (auto _begin = ++bb.begin(); _begin != bb.end(); ++_begin) {
        auto &instruction = *_begin;
        if (instrIsBarrier(&instruction)) {
          auto newbb = SplitBlock(&bb, &instruction);
          BlocksAfterBarriers.insert(newbb);
          return true;
        }
      }
    }
    return false;
  }());
}

bool CPUCudaPass::blockIsAfterBarrier(BasicBlock *BB) {
  return BlocksAfterBarriers.find(BB) != BlocksAfterBarriers.end();
}

bool CPUCudaPass::blockIsAfterBarrier(SubkernelIdType SK, BasicBlock *BB) {
  BasicBlock *OriginalFunBB = OriginalFunBBs[SubkernelBBIds[SK][BB]];
  return BlocksAfterBarriers.find(OriginalFunBB) != BlocksAfterBarriers.end();
}

template <class T>
bool in_vector(std::vector<T> &v, T key) {
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
struct TransformTerminator : public InstVisitor<TransformTerminator> {

  SubkernelIdType SK;
  LLVMContext &C;
  CPUCudaPass *Pass;

  Type *StructIndexType;

  TransformTerminator(SubkernelIdType SK, CPUCudaPass *Pass):
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
      GetElementPtrInst *Gep = GetElementPtrInst::Create(
        Pass->getCombinedDataType(), DataStructPtr, {Zero, Index}, "", NewBB);
      Instruction *NextInst = Inst->getNextNonDebugInstruction();
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

void CPUCudaPass::findSubkernelBBs(Function &F) {
  std::set<BasicBlock *> visited;
  _findSubkernelBBs(&F.getEntryBlock(), visited);
}

void CPUCudaPass::_findSubkernelBBs(BasicBlock *BB, BBSet &visited) {
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

SubkernelIdType CPUCudaPass::findSubkernelFromBB(BBIdType BB) {
  for (auto SK : SubkernelIds) {
    if (SubkernelBBIds[SK][SubkernelBBs[SK][0]] == BB)
      return SK;
  }
  return -1;
}

void CPUCudaPass::createSubkernelFunctionClones() {
  for (auto SK : SubkernelIds) {
    LLVM_DEBUG(dbgs() << "CPUCudaPass - generating new subkernel " << SK << "\n");
    ValueToValueMapTy VMap;
    // Clone the function to get a clone of the basic blocks
    Function *NF = CloneFunction(F, VMap);
    SubkernelFs[SK] = NF;
    SubkernelBBs[SK] = convert_bb_vector(SubkernelBBs[SK], F, NF);
  }
}

void CPUCudaPass::sortValueVector(SubkernelIdType SK, ValueVector &VV, map<Value *, int> &Indices) {
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

// I don't like this implementation, find something better if possible
void CPUCudaPass::findSubkernelUsedVals() {
  for (auto SK : SubkernelIds) {
    ValueVector CombinedUsedVals;
    map<Value *, int> IndexInCombinedDataType;
    for (auto _SK : SubkernelIds) {
      // Convert references of basic blocks to the cloned function
      BBVector NFuncBBs = convert_bb_vector(SubkernelBBs[_SK], SubkernelFs[_SK], SubkernelFs[SK]);
      ValueVector UsedVals = findValuesUsedInAndDefinedOutsideBBs(SubkernelFs[SK], NFuncBBs);
      SubkernelUsedVals[SK][_SK] = UsedVals;

      for (auto Val : UsedVals) {
        if (!in_vector(CombinedUsedVals, Val)) {
          CombinedUsedVals.push_back(Val);
        }
      }
    }
    sortValueVector(SK, CombinedUsedVals, IndexInCombinedDataType);
    this->CombinedUsedVals[SK] = CombinedUsedVals;
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

set<SubkernelIdType> CPUCudaPass::getSubkernelSuccessors(SubkernelIdType SK) {
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

Type *CPUCudaPass::getSubkernelReturnDataFieldType(SubkernelIdType FromSK, SubkernelIdType SuccSK) {
  auto UsedVals = SubkernelUsedVals[FromSK][SuccSK];
  vector<Type *> types;
  for (auto Val : UsedVals) {
    types.push_back(Val->getType());
  }
  return StructType::get(M->getContext(), ArrayRef<Type *>(types));
}

Type *CPUCudaPass::getCombinedDataType() {
  return CombinedDataType;
  assert(false && "impl");
}

int CPUCudaPass::getValIndexInCombinedDataType(SubkernelIdType SK, Value *Val) {
  return IndexInCombinedDataType[SK][Val];
}

StructType *CPUCudaPass::getSubkernelsReturnType() {
  vector<Type *> types;
  // The BB id we are coming from (for phi instruction handling)
  types.push_back(LLVMBBIdType);
  // The next subkernel to call
  types.push_back(LLVMSubkernelIdType);
  // Resulting in { from: blockIdType, to: blockIdType }
  return StructType::get(M->getContext(), ArrayRef<Type *>(types));
}

void CPUCudaPass::assignBBIds() {
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

TypeVector CPUCudaPass::getSubkernelParams(SubkernelIdType SK) {
  // Construct the data struct param
  std::vector<Type *> valparams;

  TypeVector params;
  // The id of the BB we returned from in the previous subkernel (for phi instr)
  params.push_back(LLVMBBIdType);
  // Values to be preserved between subkernel calls
  params.push_back(PointerType::get(CombinedDataType, SubkernelFs[SK]->getAddressSpace()));

  return params;
}

void CPUCudaPass::transformSubkernels(SubkernelIdType SK) {
  Function *_nf = SubkernelFs[SK];

  BBVector nfunc_bbs = SubkernelBBs[SK];

  auto usedVals = SubkernelUsedVals[SK][SK];

  TypeVector params = getSubkernelParams(SK);

  // Make a new function which will be the subkernel
  FunctionType *nfty = FunctionType::get(
    /* return type */ SubkernelReturnType,
    /* params */ params,
    /* isVarArg */ false);
  Function *nf = Function::Create(nfty, F->getLinkage(), F->getAddressSpace(),
                                  F->getName(), F->getParent());
  added_functions.insert(nf);
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

    // Transfer usages of the usedVals to the arguments to the function
    auto I = usedVals.begin(), E = usedVals.end();
    // Unpack args from data struct param and replace usages with them
    for (unsigned i = 0; I != E; ++I, ++i) {
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

void CPUCudaPass::createSubkernels(Function &F) {
  splitBlocksAroundBarriers(F);
  findSubkernelBBs(F);
  createSubkernelFunctionClones();
  assignBBIds();
  findSubkernelUsedVals();
  SubkernelReturnType = getSubkernelsReturnType();
  for (auto SK : SubkernelIds) {
    transformSubkernels(SK);
  }
}

PreservedAnalyses CPUCudaPass::run(Module &M,
                                   AnalysisManager<Module> &AM) {
  this->M = &M;

  LLVMBBIdType = IntegerType::getInt32Ty(M.getContext());
  LLVMSubkernelIdType = IntegerType::getInt32Ty(M.getContext());
  GepIndexType = IntegerType::getInt32Ty(M.getContext());

  for (auto &F : M) {
    // TODO HAVE TO RESET CLASS MEMBERS BEFORE EACH ITERATION!!!

    this->F = &F;
    // TODO make a clang attribute for this
    if (!F.getName().contains("mat_mul")) {
      continue;
    }

    // Will be unneeded when we add the clang attrib
    if (added_functions.find(&F) != added_functions.end()) {
      continue;
    }

    LLVM_DEBUG(errs() << "processing function " << F.getName() << "\n");

    createSubkernels(F);

  }

  // TODO optimise the preserved sets
  return PreservedAnalyses::none();
}
