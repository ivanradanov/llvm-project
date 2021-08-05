//===-- CPUCuda.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CPUCuda.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/CFG.h"

#include <queue>
#include <vector>
#include <string>

using namespace llvm;

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

void splitBlocksAroundBarriers(Function &F) {
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

void CPUCudaPass::blockIsAfterBarrier(BasicBlock *BB) {
	return blocks_after_barriers.find(BB) != blocks_after_barriers.end();
}

void CPUCudaPass::_splitFunctionAtBarriers(BasicBlock *BB, std::set<BasicBlock *> &visited) {
	if (visited.find(BB) == visited.end())
		return;
	visited.insert(BB);

	StringRef nfunc_name = F->getName().str() + std::to_string(visited.size());
	Type *nfunc_result_type = Type::getInt32Ty(M->getContext());
	ArrayRef<Type *> nfunc_params_types = ArrayRef<Type *>();
	FunctionType * nfunc_type = FunctionType::get(nfunc_result_type, nfunc_params_types, /* isVarArg */ false);
	Function *nfunc = dyn_cast<Function>(M->getOrInsertFunction(nfunc_name, nfunc_type).getCallee());
	assert(nfunc);

	errs() << "generating new subkernel " << nfunc_name << "\n";

	std::vector<BasicBlock *> func_bbs;
	func_bbs.push_back(BB);

	std::queue<BasicBlock *> to_walk;
	for (auto bb : successors(BB)) {
		to_walk.push(bb);
	}

	while (!to_walk.empty()) {
		auto bb = to_walk.front();
		to_walk.pop();

		auto &first_instruction = *bb->begin();
		if (instrIsBarrier(&first_instruction)) {
			_splitFunctionAtBarriers(bb, visited);
		} else {
			func_bbs.push_back(bb);
			for (auto bbb : successors(bb)) {
				if (func_bbs.end() == std::find(func_bbs.begin(), func_bbs.end(), bbb))
					to_walk.push(bbb);
			}
		}
	}


}

void CPUCudaPass::splitFunctionAtBarriers(Function &F) {
	std::set<BasicBlock *> visited;
	_splitFunctionAtBarriers(&F.getEntryBlock(), visited);
}

void findValsUsedAcrossBarrier(Instruction &I) {

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

		errs() << "processing function " << F.getName() << "\n";

		splitBlocksAroundBarriers(F);
		splitFunctionAtBarriers(F);

	}
	// TODO optimise the preserved sets
	return PreservedAnalyses::none();
}
