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

using namespace llvm;

bool callIsBarrier(CallInst *callInst) {
	if (Function *calledFunction = callInst->getCalledFunction()) {
		return calledFunction->getName().endswith("__cpucuda_syncthreadsv");
	} else {
		return false;
	}
}

void splitBlocksAround

PreservedAnalyses CPUCudaPass::run(Function &F,
                                   FunctionAnalysisManager &AM) {
	errs() << "processing function " << F.getName() << "\n";
	bool complete = false;
  while (!complete) {
    for (auto &bb : F) {
      for (auto &instruction : bb) {
        if (CallInst *callInst = dyn_cast<CallInst>(&instruction)) {
          if (callIsBarrier(callInst))
            SplitBlock(&bb, &instruction);
        }
      }
    }
  }
  return PreservedAnalyses::all();
}
