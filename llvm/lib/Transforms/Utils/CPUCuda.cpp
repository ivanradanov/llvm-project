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

using namespace llvm;

bool callIsBarrier(CallInst *callInst) {
	if (Function *calledFunction = callInst->getCalledFunction()) {
		return calledFunction->getName().endswith("__cpucuda_syncthreadsv");
	} else {
		return false;
	}
}

bool instrIsBarrier(Intruction *I) {
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
					SplitBlock(&bb, &instruction);
					return true;
				}
			}
		}
		return false;
	}());
}

void splitFunctionAtBarriers(Function &F) {
	_splitFunctionAtBarriers(F.getEntryBlock());
}

void _splitFunctionAtBarriers(BasicBlock *BB) {
	auto first_instruction = *BB.begin();
	if (instrIsBarrier(first_instruction))
}

void findValsUsedAcrossBarrier(Instruction &I) {

}

PreservedAnalyses CPUCudaPass::run(Function &F,
								   FunctionAnalysisManager &AM) {
	errs() << "processing function " << F.getName() << "\n";
	std::set<StringRef> done;
	splitBlocksAroundBarriers(F);
	splitFunctionAtBarriers(F);

	// TODO optimise the preserved sets
	return PreservedAnalyses::none();
}
