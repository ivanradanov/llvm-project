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
#include <map>

namespace llvm {

	class FunctionTransformer;

	class CPUCudaPass : public PassInfoMixin<CPUCudaPass> {
	private:
		Module *M;

		std::map<Function *, FunctionTransformer *> FunctionTransformers;

		void transformCallSites(FunctionTransformer * FT);
		void createCpucudaCallFunction();

	public:
		void cleanup(Module *M);
		PreservedAnalyses run(Module &M, AnalysisManager<Module> &AM);

	};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CPUCUDA_H
