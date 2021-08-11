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

std::vector<Value *> findValuesUsedInAndDefinedOutsideBBs(Function *f, std::vector<BasicBlock *> bbs) {
	std::vector<Value *> defined_outside;
	std::vector<Value *> used_inside;
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
	std::vector<Value *> intersection;
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

void CPUCudaPass::_splitFunctionAtBarriers(BasicBlock *BB, BBSet &visited) {
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

	//StringRef nfunc_name = F->getName().str() + std::to_string(visited.size());
	//Type *nfunc_result_type = Type::getInt32Ty(M->getContext());
	//ArrayRef<Type *> nfunc_params_types = ArrayRef<Type *>();
	//FunctionType * nfunc_type = FunctionType::get(nfunc_result_type, nfunc_params_types, /* isVarArg */ false);
	//Function *nfunc = dyn_cast<Function>(M->getOrInsertFunction(nfunc_name, nfunc_type).getCallee());
	//assert(nfunc);

	LLVM_DEBUG(dbgs() << "CPUCudaPass - generating new subkernel " << visited.size() << " for " << F->getName() << "\n");

	// Clone the function to get a clone of the basic blocks
	ValueToValueMapTy VMap;
	Function *_nf = CloneFunction(F, VMap);

	// Convert references of basic blocks to the cloned function
	std::vector<BasicBlock *> nfunc_bbs = convert_bb_vector(func_bbs, F, _nf);

	// Find values used in and defined outside the BBs
	auto usedVals = findValuesUsedInAndDefinedOutsideBBs(_nf, nfunc_bbs);
	print_container(usedVals);

	// Make them the arguments to the function
	std::vector<Type *> params;
	for (auto val : usedVals) {
		LLVM_DEBUG(dbgs() << "value " << val->getName()
		           << " with type " << val->getType()
		           << " is live, add it as a param\n");
		params.push_back(val->getType());
	}

	// Make a new fucntion which will be the subkernel
	FunctionType *nfty = FunctionType::get(Type::getInt32Ty(M->getContext()),
										   params, /* isVarArg */ false);
	Function *nf = Function::Create(nfty, F->getLinkage(), F->getAddressSpace(),
	                                F->getName(), F->getParent());
	//F->getParent()->getFunctionList().insert(F->getIterator(), nf);
	added_functions.insert(nf);
	// Insert the clnoe basic blocks
	nf->getBasicBlockList().splice(nf->begin(), _nf->getBasicBlockList());
	_nf->eraseFromParent();

	for (auto val : usedVals) {
		
	}

	// Erase unneeded basic blocks
	/*
	for (auto &bb : *nfunc) {
		if (!in_vector(func_bbs, &bb))
			bb.eraseFromParent();
	}
	*/

	// Add jump to starting block

	// Add return from exiting blocks

}

void CPUCudaPass::splitFunctionAtBarriers(Function &F) {
	std::set<BasicBlock *> visited;
	_splitFunctionAtBarriers(&F.getEntryBlock(), visited);
}

PreservedAnalyses CPUCudaPass::run(Module &M,
								   AnalysisManager<Module> &AM) {
	this->M = &M;
	for (auto &F : M) {
		this->F = &F;
		/* temp */
		if (!F.getName().contains("mat_mul")) {
			continue;
		}

		if (added_functions.find(&F) != added_functions.end()) {
			continue;
		}

		errs() << "processing function " << F.getName() << "\n";

		splitBlocksAroundBarriers(F);
		splitFunctionAtBarriers(F);

	}
	// TODO optimise the preserved sets
	return PreservedAnalyses::none();
}
