//===-- CPUCuda.h - Example Transformations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CPUCUDA_H
#define LLVM_TRANSFORMS_UTILS_CPUCUDA_H

#include "llvm/IR/PassManager.h"

#include <set>

namespace llvm {

	class CPUCudaPass : public PassInfoMixin<CPUCudaPass> {
	private:
		Module *M;
		Function *F;

		std::set<BasicBlock *> blocks_after_barriers;

		void _splitFunctionAtBarriers(BasicBlock *BB, std::set<BasicBlock *> &visited);
		void splitFunctionAtBarriers(Function &F);
		void splitBlocksAroundBarriers(Function &F);
		bool blockIsAfterBarrier(BasicBlock *BB);

	public:
		PreservedAnalyses run(Module &M, AnalysisManager<Module> &AM);

	};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_HELLOWORLD_H
