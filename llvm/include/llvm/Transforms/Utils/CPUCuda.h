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

	using std::vector;
	using std::set;
	using std::map;
	using std::queue;

	typedef vector<BasicBlock *> BBVector;
	typedef queue<BasicBlock *> BBQueue;
	typedef set<BasicBlock *> BBSet;

	typedef vector<Value *> ValueVector;
	typedef vector<Type *> TypeVector;
	typedef vector<Instruction *> InstVector;
	typedef vector<Argument *> ArgVector;

	typedef int SubkernelIdType;
	typedef int BBIdType;


	class CPUCudaPass : public PassInfoMixin<CPUCudaPass> {
	public:
		Module *M;
		Function *F;

		std::set<BasicBlock *> BlocksAfterBarriers;
		std::set<Function *> added_functions;

		set<SubkernelIdType> SubkernelIds;
		map<SubkernelIdType, BBVector> SubkernelBBs;
		map<SubkernelIdType, Function *> SubkernelFs;
		map<SubkernelIdType, map<SubkernelIdType, ValueVector>> SubkernelUsedVals;
		map<SubkernelIdType, map<BasicBlock *, BBIdType>> SubkernelBBIds;
		map<BBIdType, BasicBlock *> OriginalFunBBs;

		map<SubkernelIdType, map<Value *, int>> IndexInCombinedDataType;
		map<SubkernelIdType, ValueVector> CombinedUsedVals;
		StructType *CombinedDataType;


		// Label type for which BB id we should continue from after we return or we
		// have come from
		IntegerType *LLVMBBIdType;
		IntegerType *LLVMSubkernelIdType;
		StructType *SubkernelReturnType;
		IntegerType *GepIndexType;

		void splitBlocksAroundBarriers(Function &F);
		bool blockIsAfterBarrier(BasicBlock *BB);
		bool blockIsAfterBarrier(SubkernelIdType SK, BasicBlock *BB);
		void _findSubkernelBBs(BasicBlock *BB, BBSet &visited);
		void findSubkernelUsedVals();
		SubkernelIdType findSubkernelFromBB(BBIdType BB);
		void createSubkernelFunctionClones();
		set<SubkernelIdType> getSubkernelSuccessors(SubkernelIdType SK);
		Type *getSubkernelReturnDataFieldType(SubkernelIdType FromSK, SubkernelIdType SuccSK);
		StructType *getSubkernelsReturnType();
		void assignBBIds();
		TypeVector getSubkernelParams(SubkernelIdType SK);
		void transformSubkernels(SubkernelIdType SK);
		void findSubkernelBBs(Function &F);
		void createSubkernels(Function &F);
		Type *getCombinedDataType();
		int getValIndexInCombinedDataType(SubkernelIdType SK, Value *Val);
		void sortValueVector(SubkernelIdType SK, ValueVector &VV, map<Value *, int> Indices);

		PreservedAnalyses run(Module &M, AnalysisManager<Module> &AM);

	};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CPUCUDA_H
