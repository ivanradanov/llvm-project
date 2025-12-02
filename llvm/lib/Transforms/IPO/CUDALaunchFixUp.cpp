
#include "llvm/Transforms/IPO/CUDALaunchFixUp.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "cuda-launch-fixup"

// TODO Force inline all kernel stubs and delete their bodies

namespace {

constexpr char cudaLaunchSymbolName[] = "cudaLaunchKernel";
constexpr char kernelPrefix[] = "__mlir_launch_kernel_";
constexpr char kernelCoercedPrefix[] = "__mlir_launch_coerced_kernel_";

constexpr char cudaPushConfigName[] = "__cudaPushCallConfiguration";
constexpr char cudaPopConfigName[] = "__cudaPopCallConfiguration";

SmallVector<CallInst *> gatherCallers(Function *F) {
  if (!F)
    return {};
  SmallVector<CallInst *> ToHandle;
  for (auto User : F->users())
    if (auto CI = dyn_cast<CallInst>(User))
      if (CI->getCalledFunction() == F)
        ToHandle.push_back(CI);
  return ToHandle;
}

void fixup(Module &M) {
  auto LaunchKernelFunc = M.getFunction(cudaLaunchSymbolName);
  if (!LaunchKernelFunc)
    return;

  SmallPtrSet<CallInst *, 8> CoercedKernels;
  for (CallInst *CI : gatherCallers(LaunchKernelFunc)) {
    IRBuilder<> Builder(CI);
    auto FuncPtr = CI->getArgOperand(0);
    auto GridDim1 = CI->getArgOperand(1);
    auto GridDim2 = CI->getArgOperand(2);
    auto BlockDim1 = CI->getArgOperand(3);
    auto BlockDim2 = CI->getArgOperand(4);
    auto ArgPtr = CI->getArgOperand(5);
    auto SharedMemSize = CI->getArgOperand(6);
    auto StreamPtr = CI->getArgOperand(7);
    SmallVector<Value *> Args = {
        FuncPtr,   GridDim1,      GridDim2,  BlockDim1,
        BlockDim2, SharedMemSize, StreamPtr,
    };
    auto StubFunc = cast<Function>(CI->getArgOperand(0));

    AllocaInst *ArgPtrAlloca = cast<AllocaInst>(ArgPtr);
    assert(ArgPtrAlloca->getAllocatedType()->isPointerTy());
    unsigned ArgsOffset = Args.size();
    unsigned NumArgs = cast<ConstantInt>(ArgPtrAlloca->getArraySize())->getZExtValue();
    for (int I = 0; I < NumArgs; I++)
      Args.push_back(nullptr);

    for (Use &ArgPtrUse : ArgPtr->uses()) {
      if (ArgPtrUse.getUser() == CI)
        continue;

      Value *ThisArgPtr;
      int ArgIdx;
      if (auto Gep = dyn_cast<GetElementPtrInst>(ArgPtrUse.getUser())) {
        assert(Gep->getPointerOperand() == ArgPtr);
        assert(Gep->getNumIndices() == 1);
        Value *GepIdx = Gep->idx_begin()->get();
        ArgIdx = cast<ConstantInt>(GepIdx)->getSExtValue();
        assert(ArgIdx > 0);
        assert(Gep->getPointerOperandType()->isIntegerTy(8));
        ThisArgPtr = Gep->getPointerOperand();
      } else {
        ArgIdx = 0;
        ThisArgPtr = ArgPtr;
      }

      for (Use &ThisArgPtrUse : ThisArgPtr->uses()) {
        if (StoreInst* SI = dyn_cast<StoreInst>(ThisArgPtrUse.getUser())) {
          assert(SI->getPointerOperand() == ThisArgPtr);
          assert(Args[ArgsOffset + ArgIdx] == nullptr);
          Args[ArgsOffset + ArgIdx] = SI->getValueOperand();
        } else {
          assert(ThisArgPtrUse.getUser() == CI);
        }
      }
    }
    assert(all_of(Args, [](Value *V) {return V != nullptr;}));

    SmallVector<Type *> ArgTypes;
    for (Value *V : Args)
      ArgTypes.push_back(V->getType());
    Function *MlirLaunchFunc = Function::Create(
        FunctionType::get(Type::getVoidTy(M.getContext()), ArgTypes,
                          /*isVarAtg=*/false),
        llvm::GlobalValue::ExternalLinkage,
        kernelCoercedPrefix + StubFunc->getName(), M);

    CoercedKernels.insert(Builder.CreateCall(MlirLaunchFunc, Args));
    CI->eraseFromParent();
  }

  SmallVector<Function *> InlinedStubs;
  for (CallInst *CI : CoercedKernels) {
    Function *StubFunc = cast<Function>(CI->getArgOperand(0));
    for (User *callee : StubFunc->users()) {
      if (auto *CI = dyn_cast<CallInst>(callee)) {
        if (CI->getCalledFunction() == StubFunc) {
          InlineFunctionInfo IFI;
          InlineResult Res =
              InlineFunction(*CI, IFI, /*MergeAttributes=*/false);
          assert(Res.isSuccess());
          InlinedStubs.push_back(StubFunc);
          continue;
        }
      }
    }
  }
  for (Function *F : InlinedStubs) {
    F->erase(F->begin(), F->end());
    BasicBlock *BB = BasicBlock::Create(F->getContext(), "entry", F);
    ReturnInst::Create(F->getContext(), nullptr, BB->begin());
  }

  CoercedKernels.clear();
  DenseMap<Function *, SmallVector<AllocaInst *, 6>> FuncAllocas;
  auto PushConfigFunc = M.getFunction(cudaPushConfigName);
  for (CallInst *CI : gatherCallers(PushConfigFunc)) {
    Function *TheFunc = CI->getFunction();
    IRBuilder<> IRB(&TheFunc->getEntryBlock(),
                    TheFunc->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
    auto Allocas = FuncAllocas.lookup(TheFunc);
    if (Allocas.empty()) {
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "griddim64"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt32Ty(), nullptr, "griddim32"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "blockdim64"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt32Ty(), nullptr, "blockdim32"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "shmem_size"));
      Allocas.push_back(IRB.CreateAlloca(IRB.getPtrTy(), nullptr, "stream"));
      FuncAllocas.insert_or_assign(TheFunc, Allocas);
    }
    IRB.SetInsertPoint(CI);
    for (auto [Arg, Alloca] :
         llvm::zip_equal(llvm::drop_end(CI->operand_values()), Allocas))
      IRB.CreateStore(Arg, Alloca);
  }
  auto PopConfigFunc = M.getFunction(cudaPopConfigName);
  for (CallInst *PopCall : gatherCallers(PopConfigFunc)) {
    Function *TheFunc = PopCall->getFunction();
    auto Allocas = FuncAllocas.lookup(TheFunc);
    if (Allocas.empty()) {
      continue;
    }

    CallInst *KernelLaunch = PopCall;
    Instruction *It = PopCall;
    do {
      It = It->getNextNonDebugInstruction();
      KernelLaunch = dyn_cast<CallInst>(It);
    } while (!It->isTerminator() &&
             !(KernelLaunch && KernelLaunch->getCalledFunction() &&
               KernelLaunch->getCalledFunction()->getName().starts_with(
                   kernelCoercedPrefix)));

    assert(!It->isTerminator());

    IRBuilder<> IRB(PopCall);

    for (auto [Arg, Alloca] : llvm::zip(
             llvm::drop_begin(KernelLaunch->operand_values(), 1), Allocas)) {
      auto Load = cast<LoadInst>(Arg);
      LoadInst *NewLoad = IRB.CreateLoad(Arg->getType(), Alloca);
      Load->replaceAllUsesWith(NewLoad);
    }
    CoercedKernels.insert(KernelLaunch);

    It = PopCall->getParent()->getPrevNode()->getFirstNonPHIOrDbg();
    CallInst *PushCall = dyn_cast<CallInst>(It);
    while (!It->isTerminator() &&
           !(PushCall && PushCall->getCalledFunction() &&
             PushCall->getCalledFunction()->getName() == cudaPushConfigName)) {
      It = It->getNextNonDebugInstruction();
      PushCall = dyn_cast<CallInst>(It);
    }

    assert(!It->isTerminator());

    // Replace with success
    PushCall->replaceAllUsesWith(IRB.getInt32(0));
    PushCall->eraseFromParent();
    PopCall->replaceAllUsesWith(IRB.getInt32(0));
    PopCall->eraseFromParent();
  }
  for (CallInst *CI : CoercedKernels) {
    IRBuilder<> Builder(CI);
    auto FuncPtr = CI->getArgOperand(0);
    auto GridDim1 = CI->getArgOperand(1);
    auto GridDim2 = CI->getArgOperand(2);
    auto GridDimX = Builder.CreateTrunc(GridDim1, Builder.getInt32Ty());
    auto GridDimY = Builder.CreateLShr(
        GridDim1, ConstantInt::get(Builder.getInt64Ty(), 32));
    GridDimY = Builder.CreateTrunc(GridDimY, Builder.getInt32Ty());
    auto GridDimZ = GridDim2;
    auto BlockDim1 = CI->getArgOperand(3);
    auto BlockDim2 = CI->getArgOperand(4);
    auto BlockDimX = Builder.CreateTrunc(BlockDim1, Builder.getInt32Ty());
    auto BlockDimY = Builder.CreateLShr(
        BlockDim1, ConstantInt::get(Builder.getInt64Ty(), 32));
    BlockDimY = Builder.CreateTrunc(BlockDimY, Builder.getInt32Ty());
    auto BlockDimZ = BlockDim2;
    auto SharedMemSize = CI->getArgOperand(5);
    auto StreamPtr = CI->getArgOperand(6);
    SmallVector<Value *> Args = {
        FuncPtr,   GridDimX,  GridDimY,      GridDimZ,  BlockDimX,
        BlockDimY, BlockDimZ, SharedMemSize, StreamPtr,
    };
    auto StubFunc = cast<Function>(CI->getArgOperand(0));
    for (unsigned I = 7; I < CI->getNumOperands() - 1; I++)
      Args.push_back(CI->getArgOperand(I));
    SmallVector<Type *> ArgTypes;
    for (Value *V : Args)
      ArgTypes.push_back(V->getType());
    auto MlirLaunchFunc = Function::Create(
        FunctionType::get(Type::getVoidTy(M.getContext()), ArgTypes,
                          /*isVarAtg=*/false),
        llvm::GlobalValue::ExternalLinkage, kernelPrefix + StubFunc->getName(),
        M);

    Builder.CreateCall(MlirLaunchFunc, Args);
    CI->eraseFromParent();
  }
}

} // namespace

PreservedAnalyses llvm::CUDALaunchFixUp::run(Module &M,
                                             ModuleAnalysisManager &) {
  fixup(M);
  return PreservedAnalyses::none();
}
