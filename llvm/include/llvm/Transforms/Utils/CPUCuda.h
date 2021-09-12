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
	typedef set<Value *> ValueSet;
	typedef set<Instruction *> InstSet;
	typedef vector<Type *> TypeVector;
	typedef vector<Instruction *> InstVector;
	typedef vector<Argument *> ArgVector;

	typedef int SubkernelIdType;
	typedef int BBIdType;

	struct UsedValVars {
		ValueSet usedVals;
		InstSet usedSharedVars;
	};

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
		map<SubkernelIdType, map<SubkernelIdType, InstVector>> SubkernelUsedSharedVars;
		map<SubkernelIdType, map<BasicBlock *, BBIdType>> SubkernelBBIds;
		map<BBIdType, BasicBlock *> OriginalFunBBs;

		map<SubkernelIdType, map<Value *, int>> IndexInCombinedDataType;
		map<SubkernelIdType, ValueVector> CombinedUsedVals;
		StructType *CombinedDataType;

		map<SubkernelIdType, map<Instruction *, int>> IndexInCombinedSharedVarsDataType;
		map<SubkernelIdType, InstVector> CombinedSharedVars;
		StructType *SharedVarsDataType;

		Function *DriverFunction;

		// Label type for which BB id we should continue from after we return or we
		// have come from
		IntegerType *LLVMBBIdType;
		IntegerType *LLVMSubkernelIdType;
		StructType *SubkernelReturnType;
		IntegerType *GepIndexType;
		Type *Dim3Type;

		void splitBlocksAroundBarriers(Function &F);
		bool blockIsAfterBarrier(BasicBlock *BB);
		bool blockIsAfterBarrier(SubkernelIdType SK, BasicBlock *BB);
		void _findSubkernelBBs(BasicBlock *BB, BBSet &visited);
		UsedValVars findUsedVals(SubkernelIdType SK, BasicBlock *BB, ValueSet definedVals, BBVector visited);
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
		void findSharedVars();
		void createSubkernels(Function &F);
		Type *getCombinedDataType();
		int getValIndexInCombinedDataType(SubkernelIdType SK, Value *Val);
		void sortValueVector(SubkernelIdType SK, ValueVector &VV, map<Value *, int> &Indices);
		void removeReferencesInPhi(const BBVector &BBsToRemove);
		bool isSharedVar(Instruction &I);
		void createDriverFunction();
		void replaceDim3Usages(SubkernelIdType SK);
		Type getDim3StructType();

		PreservedAnalyses run(Module &M, AnalysisManager<Module> &AM);

	};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CPUCUDA_H
