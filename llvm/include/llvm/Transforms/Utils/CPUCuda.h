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
#include <queue>
#include <vector>

namespace llvm {

	using std::vector;
	using std::set;
	using std::map;

	typedef vector<BasicBlock *> BBVector;
	typedef queue<BasicBlock *> BBQueue;
	typedef set<BasicBlock *> BBSet;

	typedef vector<Value *> ValueVector;

	typedef SubkernelIdType int;


	class CPUCudaPass : public PassInfoMixin<CPUCudaPass> {
	public:
		Module *M;
		Function *F;

		std::set<BasicBlock *> blocks_after_barriers;
		std::set<Function *> added_functions;

		set<SubkernelIdType> SubkernelIds;
		map<SubkernelIdType, BBVector> SubkernelBBs;
		map<SubkernelIdType, BBVector> SubkernelFs;
		map<SubkernelIdType, map<SubkernelId, ValueVector>> SubkernelUsedVals;


		// Label type for which BB id we should continue from after we return or we
		// have come from
		Type *BBIdType;
		StructType *SubkernelReturnType;

		void _splitFunctionAtBarriers(BasicBlock *BB, std::set<BasicBlock *> &visited);
		void splitFunctionAtBarriers(Function &F);
		void splitBlocksAroundBarriers(Function &F);
		bool blockIsAfterBarrier(BasicBlock *BB);

		PreservedAnalyses run(Module &M, AnalysisManager<Module> &AM);

	};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CPUCUDA_H
