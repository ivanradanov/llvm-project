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

// Converts a list of bbs to the corresponding list of bbs in the newly cloned
// function
//
// TODO this depends on the representation of blocks in a function - is
// there a better way to do it?
BBVector convert_bb_vector(BBVector &vold, Function *fold, Function *fnew) {
	std::vector<int> bb_ids;
	int id = 0;
	for (auto it = fold->begin(); it != fold->end(); ++it, ++id) {
		BasicBlock *bb = &(*it);
		if (in_vector(vold, bb)) {
			bb_ids.push_back(id);
		}
	}
	BBVector vnew;
	id = 0;
	for (auto it = fnew->begin(); it != fnew->end(); ++it, ++id) {
		BasicBlock *bb = &(*it);
		if (in_vector(bb_ids, id)) {
			vnew.push_back(bb);
		}
	}
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

	BasicBlock *createNewRetBB(Function *F, BasicBlock *BBFrom, SubkernelIdType ContSK) {
		BasicBlock *NewBB = BasicBlock::Create(F->getContext(), "generated_ret_block", F);
		Constant *from_label = llvm::ConstantInt::get(Pass->LLVMBBIdType,
		                                              /* value */ Pass->SubkernelBBIds[SK][BBFrom],
		                                              /* isSigned */ true);
		Constant *cont_label = llvm::ConstantInt::get(Pass->LLVMSubkernelIdType,
		                                              /* value */ ContSK,
		                                              /* isSigned */ true);
		// Allocate return type
		auto SKReturnType = Pass->SubkernelReturnType;
		AllocaInst *ReturnValPtr = new AllocaInst(
			SKReturnType,
      Pass->M->getDataLayout().getAllocaAddrSpace(),
      "returnvalalloca",
      NewBB);

    Constant *zero = llvm::ConstantInt::get(StructIndexType, 0);
		Constant *one = llvm::ConstantInt::get(StructIndexType, 1);
		Constant *two = llvm::ConstantInt::get(StructIndexType, 2);

		// Store from and to
    auto FromFieldPtr = GetElementPtrInst::Create(SKReturnType,
                                                  ReturnValPtr,
                                                  ArrayRef<Value *>(vector<Value *>({zero, zero})),
                                                  "", NewBB);
    new StoreInst(from_label, FromFieldPtr, NewBB);
    auto ToFieldPtr = GetElementPtrInst::Create(SKReturnType,
                                                ReturnValPtr,
                                                ArrayRef<Value *>(vector<Value *>({zero, one})),
                                                "", NewBB);
    new StoreInst(cont_label, ToFieldPtr, NewBB);

    // Get pointer to data
    auto DataFieldPtr = GetElementPtrInst::Create(SKReturnType,
                                                  ReturnValPtr,
                                                  ArrayRef<Value *>(vector<Value *>({zero, two})),
                                                  "", NewBB);
		// Get matching struct type and bitcast the pointer to it
		auto DataStructType = Pass->getSubkernelReturnDataFieldType(SK, ContSK);
    BitCastInst *DataStructPtr = new BitCastInst(DataFieldPtr,
                                                 DataStructType,
                                                 "", NewBB);

    // Populate data struct
    Value *DataStructVal = static_cast<Value *>(UndefValue::get(DataStructType));
    {
      auto UsedVals = Pass->SubkernelUsedVals[SK][ContSK];
      unsigned i = 0;
      for (auto &Val : UsedVals) {
	      DataStructVal = InsertValueInst::Create(DataStructVal, Val, ArrayRef<unsigned>(vector<unsigned>({i})), "", NewBB);
      }
    }

    // Store the struct at the pointer
    new StoreInst(DataStructVal, DataStructPtr, NewBB);

		// Load the ptr to the return val
    auto ReturnStruct = new LoadInst(SKReturnType, ReturnValPtr, "", NewBB);

		// return it
    ReturnInst::Create(C, ReturnStruct, NewBB);

    return NewBB;
	}

	void visitReturnInst(ReturnInst &I) {
		LLVM_DEBUG(dbgs() << "Transforming ReturnInst " << I << "\n");
		Value *ReturnStructVal = static_cast<Value *>(UndefValue::get(Pass->SubkernelReturnType));
    Constant *ContLabel = llvm::ConstantInt::get(Pass->LLVMSubkernelIdType,
                                                 /* value */ -1,
                                                 /* isSigned */ true);
    ReturnStructVal = InsertValueInst::Create(ReturnStructVal, ContLabel, ArrayRef<unsigned>(1), "", I.getParent());
		ReturnInst::Create(I.getContext(), ReturnStructVal, I.getParent());
		I.eraseFromParent();
	}

	void visitBranchInst(BranchInst &I) {
		LLVM_DEBUG(dbgs() << "Transforming BranchInst " << I << "\n");
		Function *F = I.getFunction();
		for (unsigned i = 0; i < I.getNumSuccessors(); i++) {
			BasicBlock *succ = I.getSuccessor(i);
			if (Pass->blockIsAfterBarrier(SK, succ)) {
				// If the succesor block is after a barrier, the branch instruction that
				// jumps to it should be an unconditional one generated when we split
				// the blocks around the barriers
				assert(I.getNumSuccessors() == 1);
				BasicBlock *RetBB = createNewRetBB(
					F, I.getParent(),
          Pass->findSubkernelFromBB(Pass->SubkernelBBIds[SK][succ]));
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
	func_bbs.push_back(BB);

	std::queue<BasicBlock *> to_walk;
	for (auto bb : successors(BB)) {
		to_walk.push(bb);
	}

	while (!to_walk.empty()) {
		auto bb = to_walk.front();
		to_walk.pop();

		func_bbs.push_back(bb);

		if (blockIsAfterBarrier(bb)) {
			_findSubkernelBBs(bb, visited);
		} else {
			func_bbs.push_back(bb);
			for (auto bbb : successors(bb)) {
				if (!in_vector(func_bbs, bbb))
					to_walk.push(bbb);
			}
		}
	}

	SubkernelIdType SK = SubkernelIds.size();
	SubkernelIds.insert(SK);
	SubkernelBBs[SK] = func_bbs;

}

SubkernelIdType CPUCudaPass::findSubkernelFromBB(BBIdType BB) {
	for (auto SK : SubkernelIds) {
		if (SubkernelBBIds[SK][SubkernelBBs[SK][0]] == BB)
			return SK;
	}
	assert(false && "There should always be a subkernel from a BB");
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

void CPUCudaPass::findSubkernelUsedVals() {
	for (auto SK : SubkernelIds) {
		for (auto _SK : SubkernelIds) {
			// Convert references of basic blocks to the cloned function
			BBVector NFuncBBs = convert_bb_vector(SubkernelBBs[_SK], SubkernelFs[_SK], SubkernelFs[SK]);
			ValueVector UsedVals = findValuesUsedInAndDefinedOutsideBBs(SubkernelFs[SK], NFuncBBs);
			SubkernelUsedVals[SK][_SK] = UsedVals;
		}
  }
}

set<SubkernelIdType> CPUCudaPass::getSubkernelSuccessors(SubkernelIdType SK) {
	set<SubkernelIdType> Successors;
	for (auto BB : SubkernelBBs[SK]) {
    for (auto bb : successors(BB)) {
	    bool added = false;
	    for (auto _SK : SubkernelIds) {
		    if (in_vector(SubkernelBBs[_SK], bb)) {
			    Successors.insert(_SK);
			    assert(!added && "There should be only one successor for a single BB");
			    added = true;
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

Type *CPUCudaPass::getSubkernelsReturnType() {
	TypeSize maxSize = TypeSize::Fixed(0);
	for (auto SK : SubkernelIds) {
    set<SubkernelIdType> SKSuccs = getSubkernelSuccessors(SK);
    for (auto SuccSK : SKSuccs) {
      Type *DataFieldType = getSubkernelReturnDataFieldType(SK, SuccSK);
      TypeSize size = M->getDataLayout().getTypeAllocSize(DataFieldType);
      if (size > maxSize)
        maxSize = size;
    }
  }
  vector<Type *> types;
	// The BB id we are coming from (for phi instruction handling)
	types.push_back(LLVMBBIdType);
	// The next subkernel to call
	types.push_back(LLVMSubkernelIdType);
	// Memory for the data struct which will be cast to the appropriate struct
	// type for the from/to subkernel pair
	types.push_back(ArrayType::get(Type::getInt8Ty(M->getContext()), maxSize));
	// Resulting in { from: blockIdType, to: blockIdType, data: [int8] }
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
	// Get values used in and defined outside the BBs, i.e. the ones that should
	// be used as an input
	auto usedVals = SubkernelUsedVals[SK][SK];

	// Use a struct
	if (true) {
    // Construct the data struct param
    std::vector<Type *> valparams;
    for (auto val : usedVals) {
      valparams.push_back(val->getType());
    }
    StructType *ValArgs = StructType::get(M->getContext(), ArrayRef<Type *>(valparams));

    TypeVector params;
    // The id of the BB we retuned from in the previous subkernel (for phi instr)
    params.push_back(LLVMBBIdType);
    // Values to be preserved between subkernel calls
    params.push_back(ValArgs);

    return params;
	} else {
		std::vector<Type *> params;
		params.push_back(LLVMBBIdType);
		for (auto val : usedVals) {
			params.push_back(val->getType());
		}
		return params;
	}
}

void CPUCudaPass::transformSubkernels(SubkernelIdType SK) {
	Function *_nf = SubkernelFs[SK];

	std::vector<BasicBlock *> nfunc_bbs = SubkernelBBs[SK];

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

	// Construct the entry block which sets up the usedVals params and handles phi
	// instructions
	{
		BBVector OriginalBBs;
		for (auto &BB : *nf) {
      OriginalBBs.push_back(&BB);
    }

		BasicBlock *EntryBB = BasicBlock::Create(nf->getContext(), "generated_entry_block", nf, &nf->getEntryBlock());

		// Transfer usages of the usedVals to the arguments to the function
		auto I = usedVals.begin(), E = usedVals.end();
		// Unpack args from data struct param and replace usages with them
		for (unsigned i = 0; I != E; ++I, ++i) {
			// The second argument of the function is the structure of usedVals
			Value *UnpackedArg = ExtractValueInst::Create(nf->getArg(1), i, "", EntryBB);
			(*I)->replaceAllUsesWith(UnpackedArg);
			UnpackedArg->takeName(*I);
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


	// Add return from exiting blocks
	for (auto &bb : nfunc_bbs) {
		Instruction *term = bb->getTerminator();
		TransformTerminator transformer(SK, this);
		transformer.visit(term);
	}

	LLVM_DEBUG(M->dump());
	// Erase unneeded basic blocks
	{
		BBVector to_remove;
		for (auto &bb : *nf) {
			if (!in_vector(nfunc_bbs, &bb))
				to_remove.insert(to_remove.begin(), &bb);
		}
		// Empty placeholder BB to replace BB usages
		BasicBlock *EmptyBB = BasicBlock::Create(M->getContext(), "empty_block", F);
		for (auto &bb : to_remove) {
			for (auto &inst : *bb) {
				if (!inst.use_empty())
					inst.replaceAllUsesWith(UndefValue::get(inst.getType()));
			}
			bb->replaceAllUsesWith(EmptyBB);
			bb->eraseFromParent();
		}
		EmptyBB->eraseFromParent();
	}
	LLVM_DEBUG(M->dump());

	// Delete the dead function
	_nf->eraseFromParent();
	LLVM_DEBUG(M->dump());

	// Add jump to starting block


}

void CPUCudaPass::createSubkernels(Function &F) {
	findSubkernelBBs(F);
	createSubkernelFunctionClones();
	findSubkernelUsedVals();
	SubkernelReturnType = getSubkernelsReturnType();
	assignBBIds();
	for (auto SK : SubkernelIds) {
		transformSubkernels(SK);
	}
}

PreservedAnalyses CPUCudaPass::run(Module &M,
								   AnalysisManager<Module> &AM) {
	this->M = &M;
	// TODO is it needed to reset class members? does this class get newly created
	// for each module?

	LLVMBBIdType = llvm::IntegerType::getInt32Ty(M.getContext());
	LLVMSubkernelIdType = llvm::IntegerType::getInt32Ty(M.getContext());

	for (auto &F : M) {
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
	LLVM_DEBUG(errs() << "final module dump:" << "\n");
	LLVM_DEBUG(M.dump());
	// TODO optimise the preserved sets
	return PreservedAnalyses::none();
}
