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
					blocks_after_barriers.insert(newbb);
					return true;
				}
			}
		}
		return false;
	}());
}

bool CPUCudaPass::blockIsAfterBarrier(BasicBlock *BB) {
	return blocks_after_barriers.find(BB) != blocks_after_barriers.end();
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

	BBVector &nfunc_bbs;
	LLVMContext &C;

	// Label type for which BB id we should continue from after we return or we
	// have come from
	Type *BBIdType;
	CPUCudaPass *Pass;

	TransformTerminator(BBVector &nfunc_bbs, CPUCudaPass *Pass):
		nfunc_bbs(nfunc_bbs),
		C(Pass->M->getContext()),
		Pass(Pass) {
		BBIdType = Pass->BBIdType;
	}

	BasicBlock *createNewRetBB(Function *F, int RetLabel) {
		BasicBlock *NewBB = BasicBlock::Create(F->getContext(), "generated_ret_block", F);
		ReturnInst::Create(F->getContext(), getReturnValue(RetLabel), NewBB);
		return NewBB;
	}

	BasicBlock *createNewRetBB(Function *F, Value *RetValue) {
		BasicBlock *NewBB = BasicBlock::Create(F->getContext(), "generated_ret_block", F);
		ReturnInst::Create(F->getContext(), RetValue, NewBB);
		return NewBB;
	}

	Value *getReturnValue(int ContLabel) {
		Constant *return_val = llvm::ConstantInt::get(BBIdType,
		                                              /* value */ ContLabel,
		                                              /* isSigned */ true);
		return static_cast<Value *>(return_val);
	}

	Value *getReturnValue(int ContLabel, int FromLabel) {
		Constant *cont_label = llvm::ConstantInt::get(BBIdType,
													  /* value */ ContLabel,
													  /* isSigned */ true);
		Constant *from_label = llvm::ConstantInt::get(BBIdType,
													  /* value */ FromLabel,
													  /* isSigned */ true);
		ArrayRef<Constant *> fields(std::vector<Constant *>({cont_label, from_label}));
		Constant *ReturnVal = ConstantStruct::getAnon(C, fields);
		return static_cast<Value *>(ReturnVal);
	}

	// get a unique positive ID of the BB (in the original kernel)
	int getId(BasicBlock *BB) {
		// TODO: implement
		return 1;
	}

	void visitReturnInst(ReturnInst &I) {
		LLVM_DEBUG(dbgs() << "Transforming ReturnInst " << I << "\n");
		// -1 stands for return
		Value *RetVal = getReturnValue(-1, getId(I.getParent()));
		ReturnInst::Create(I.getContext(), RetVal, I.getParent());
		I.eraseFromParent();
	}

	void visitBranchInst(BranchInst &I) {
		LLVM_DEBUG(dbgs() << "Transforming BranchInst " << I << "\n");
		Function *F = I.getFunction();
		for (unsigned i = 0; i < I.getNumSuccessors(); i++) {
			BasicBlock *succ = I.getSuccessor(i);
			if (!in_vector(nfunc_bbs, succ)) {
				BasicBlock *RetBB = createNewRetBB(F, getReturnValue(getId(succ), getId(I.getParent())));
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
			_splitFunctionAtBarriers(bb, visited);
		} else {
			func_bbs.push_back(bb);
			for (auto bbb : successors(bb)) {
				if (!in_vector(func_bbs, bbb))
					to_walk.push(bbb);
			}
		}
	}

	SubkernelIdType SK = SubkernelIds.size();
	SubkernelIds.push_back(SK);
	SubkernelBBs[SK] = func_bbs;

}

void CPUCudaPass::createSubkernelFunctionClones() {
	for (auto SK : SubkernelIds) {
		LLVM_DEBUG(dbgs() << "CPUCudaPass - generating new subkernel " << SK << "\n");
		ValueToValueMapTy VMap;
		// Clone the function to get a clone of the basic blocks
		Function *NF = CloneFunction(F, VMap);
		SubkernelFs[SK] = NF;
	}
}

void CPUCudaPass::findSubkernelUsedVals() {
	for (auto SK : SubkernelIds) {
		for (auto _SK : SubkernelIds) {
			// Convert references of basic blocks to the cloned function
			BBVector NFuncBBs = convert_bb_vector(SubkernelBBs[_SK], F, SubkernelFs[SK]);
			ValueVector UsedVals = findValuesUsedInAndDefinedOutsideBBs(_nf, NFuncBBs);
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



	// Convert references of basic blocks to the cloned function
	std::vector<BasicBlock *> nfunc_bbs = convert_bb_vector(func_bbs, F, _nf);

	// Find values used in and defined outside the BBs
	auto usedVals = findValuesUsedInAndDefinedOutsideBBs(_nf, nfunc_bbs);
	print_container(usedVals);

	// Make them the arguments to the function
	std::vector<Type *> valparams;
	for (auto val : usedVals) {
		LLVM_DEBUG(dbgs() << "value " << val->getName()
		           << " with type " << val->getType()
		           << " is live, add it as a param\n");
		valparams.push_back(val->getType());
	}
	StructType *ValArgs = StructType::get(M->getContext(), ArrayRef<Type *>(valparams));
	std::vector<Type *> params;
	// Values to be preserved between subkernel calls
	params.push_back(ValArgs);
	// The id of the BB we retuned from in the previous subkernel
	params.push_back(BBIdType);

	// Make a new fucntion which will be the subkernel
	FunctionType *nfty = FunctionType::get(Type::getInt32Ty(M->getContext()),
										   params, /* isVarArg */ false);
	Function *nf = Function::Create(nfty, F->getLinkage(), F->getAddressSpace(),
	                                F->getName(), F->getParent());
	added_functions.insert(nf);
	// Insert the cloned basic blocks
	nf->getBasicBlockList().splice(nf->begin(), _nf->getBasicBlockList());

	nf->takeName(_nf);

	// Transfer usages of the usedVals to the arguments to the function
	{
		Function::arg_iterator I = nf->arg_begin(), E = nf->arg_end();
		auto I2 = usedVals.begin(), E2 = usedVals.end();
		for (; I != E; ++I, ++I2) {
			(*I2)->replaceAllUsesWith(&*I);
			I->takeName(*I2);
		}
		assert(I2 == E2);
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
		TransformTerminator transformer(nfunc_bbs, this);
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
		for (auto &bb : to_remove) {
			// Entry BB for jumping from for phi instruction
			BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry_block", F);
			for (auto &inst : *bb) {
				if (!inst.use_empty())
					inst.replaceAllUsesWith(UndefValue::get(inst.getType()));
			}
			bb->replaceAllUsesWith(EntryBB);
			bb->eraseFromParent();
		}
	}
	LLVM_DEBUG(M->dump());

	// Delete the dead function
	_nf->eraseFromParent();
	LLVM_DEBUG(M->dump());

	// Add jump to starting block


}

void CPUCudaPass::findSubkernelBBs(Function &F) {
	std::set<BasicBlock *> visited;
	_findSubkernelBBs(&F.getEntryBlock(), visited);
}

PreservedAnalyses CPUCudaPass::run(Module &M,
								   AnalysisManager<Module> &AM) {
	this->M = &M;
	// TODO is it needed to reset class members? does this class get newly created
	// for each module?

	BBIdType = llvm::IntegerType::getInt32Ty(M.getContext());

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

		errs() << "processing function " << F.getName() << "\n";

		findSubkernelBBs(F);

		splitBlocksAroundBarriers(F);
		splitFunctionAtBarriers(F);

	}
	// TODO optimise the preserved sets
	return PreservedAnalyses::none();
}
