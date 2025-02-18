
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>

#define DEBUG_TYPE "flang-enzyme-utils"

mlir::LogicalResult enzymePreprocessMLIRModule(mlir::ModuleOp _mlirModule) {
  mlir::ModuleOp *mlirModule = &_mlirModule;
  mlir::Operation *symbolTable =
      mlir::SymbolTable::getNearestSymbolTable(*mlirModule);
  mlir::IRRewriter builder(mlirModule->getContext());
  for (auto &op : mlirModule->getOps()) {
    LLVM_DEBUG(op.dump());
    if (auto f = mlir::dyn_cast<mlir::func::FuncOp>(&op)) {
      if (f.getName().contains("__enzyme_")) {
        LLVM_DEBUG(llvm::errs() << "ENZYME\n");

        std::optional<mlir::SymbolRefAttr> symbol;
        for (auto &block : f.getRegion()) {
          for (auto &op : block) {
            if (auto callOp = mlir::dyn_cast<fir::CallOp>(&op)) {
              auto thisSymbol = callOp.getCallee();
              if (!thisSymbol) {
                llvm::errs()
                    << "Found indirect call in enzyme wrapper, aborting.\n";
                return mlir::failure();
              }
              if (symbol) {
                llvm::errs()
                    << "Found two calls in enzyme wrapper, aborting.\n";
                return mlir::failure();
              }
              symbol = *thisSymbol;
            }
          }
        }
        if (!symbol) {
          llvm::errs() << "Did not find call in enzyme wrapper, aborting.\n";
          return mlir::failure();
        }

        auto callee = mlir::dyn_cast<mlir::func::FuncOp>(
            mlir::SymbolTable::lookupSymbolIn(symbolTable, *symbol));
        if (!callee) {
          llvm::errs() << "Callee is not func op, aborting.\n";
          return mlir::failure();
        }

        mlir::func::FuncOp newWrapper = nullptr;
        unsigned unwrapNum = 0;
        auto genNewWrapperFunc = [&](mlir::Type newArgTy) -> bool {
          if (!newWrapper) {
            auto wrapperFty = f.getFunctionType();
            mlir::SmallVector<mlir::Type> newInputs(wrapperFty.getInputs());
            unsigned wrappedArgNum = callee.getFunctionType().getNumInputs();
            unsigned wrapperArgNum = wrapperFty.getNumInputs();
            unwrapNum = wrapperArgNum - wrappedArgNum;
            for (unsigned i = 0; i < unwrapNum; i++) {
              fir::ReferenceType refTy =
                  mlir::dyn_cast<fir::ReferenceType>(newInputs[i]);
              if (!refTy) {
                llvm::errs() << "Wrapper argument not ref ty, aborting\n";
                return false;
              }
              mlir::Type realTy = refTy.getEleTy();
              newInputs[i] = realTy;
            }
            newInputs.insert(newInputs.begin(), newArgTy);
            auto newWrapperFty = mlir::FunctionType::get(
                wrapperFty.getContext(), newInputs, wrapperFty.getResults());
            f.setFunctionType(newWrapperFty);
            auto argAttrs = f.getArgAttrs();
            if (argAttrs) {
              mlir::SmallVector<mlir::Attribute> newArgAttrs(
                  argAttrs->getValue());
              newArgAttrs.insert(
                  newArgAttrs.begin(),
                  mlir::DictionaryAttr::get(newArgTy.getContext()));
              f.setAllArgAttrs(newArgAttrs);
            }
            newWrapper = f;
          }
          return true;
        };

        auto useRange = mlir::SymbolTable::getSymbolUses(f, symbolTable);
        if (!useRange) {
          llvm::errs() << "COuld not find use range, aborting.\n";
          return mlir::failure();
        }

        for (auto user : *useRange) {
          auto call = mlir::dyn_cast<fir::CallOp>(user.getUser());
          if (!call) {
            llvm::errs()
                << "Found a non-call use of enzyme wrapper, aborting.\n";
            return mlir::failure();
          }
          builder.setInsertionPoint(call);
          auto addrOf = builder.create<fir::AddrOfOp>(
              call.getLoc(), callee.getFunctionType(), *symbol);
          if (!genNewWrapperFunc(addrOf.getResult().getType()))
            return mlir::failure();
          mlir::SmallVector<mlir::Value> newInputs(call.getArgOperands());
          for (unsigned i = 0; i < unwrapNum; i++)
            newInputs[i] =
                builder.create<fir::LoadOp>(call.getLoc(), newInputs[i]);
          newInputs.insert(newInputs.begin(), addrOf);
          LLVM_DEBUG(llvm::errs() << "Replace\n" << call << "\n");
          auto newCall = builder.replaceOpWithNewOp<fir::CallOp>(
              call, newWrapper, newInputs);
          LLVM_DEBUG(llvm::errs() << newCall << "\n");
          LLVM_DEBUG(llvm::errs() << *newCall->getBlock() << "\n");
        }

        f.getRegion().getBlocks().clear();
        f.setVisibility(mlir::SymbolTable::Visibility::Private);
        auto externalLinkage = mlir::LLVM::linkage::Linkage::External;
        auto linkage =
            mlir::LLVM::LinkageAttr::get(f->getContext(), externalLinkage);
        f->setAttr("llvm.linkage", linkage);
      }
    }
    LLVM_DEBUG(op.dump());
  }
  return mlir::success();
}

mlir::LogicalResult enzymePreprocessLLVMModule(llvm::Module *llvmModule) {
  llvm::SmallVector<llvm::Function *> toHandle;
  for (auto &wrapper : *llvmModule)
    if (wrapper.getName().contains("__enzyme_"))
      toHandle.push_back(&wrapper);
  for (auto *wrapperP : toHandle) {
    auto &wrapper = *wrapperP;
    for (auto *user : wrapper.users()) {
      llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(user);
      if (!ci) {
        llvm::errs() << "warning: found non-call use of enzyme wrapper\n";
        continue;
      }
      if (ci->getCalledFunction() != &wrapper) {
        llvm::errs() << "warning: non callee use of enzyme wrapper\n";
        continue;
      }
      llvm::Function *wrappedFunc =
          llvm::dyn_cast<llvm::Function>(ci->getArgOperand(0));
      if (!wrappedFunc) {
        llvm::errs() << "warning: wrapped func was not a func\n";
        continue;
      }
      unsigned wrappedNumArgs = wrappedFunc->arg_size();
      unsigned wrapperNumArgs = wrapper.arg_size();
      assert(ci->arg_size() == wrapperNumArgs);
      assert(wrappedNumArgs + 1 <= wrapperNumArgs);
      llvm::SmallVector<llvm::Value *> newWrapperArgs;
      llvm::SmallVector<llvm::Value *> wrappedArgs;
      llvm::SmallVector<llvm::Type *> newWrapperParamTypes;
      for (unsigned i = 0; i < wrapperNumArgs - wrappedNumArgs; i++) {
        newWrapperArgs.push_back(ci->getArgOperand(i));
        newWrapperParamTypes.push_back(ci->getArgOperand(i)->getType());
      }
      for (unsigned i = newWrapperArgs.size(); i < wrapperNumArgs; i++)
        wrappedArgs.push_back(ci->getArgOperand(i));

      auto *oldFty = wrapper.getFunctionType();
      auto *newFty = llvm::FunctionType::get(
          llvm::PointerType::get(oldFty->getContext(), 0), newWrapperParamTypes,
          false);
      auto *newWrapper = llvm::Function::Create(newFty, wrapper.getLinkage(),
                                                wrapper.getName(), *llvmModule);

      auto *newWrapperCi = llvm::CallInst::Create(newWrapper, newWrapperArgs,
                                                  {}, "", ci->getIterator());
      auto *newCi = llvm::CallInst::Create(
          llvm::FunctionCallee(wrappedFunc->getFunctionType(), newWrapperCi),
          wrappedArgs, {}, "", ci->getIterator());
      ci->replaceAllUsesWith(newCi);
      ci->eraseFromParent();
    }
    wrapper.setName(wrapper.getName() + "_invalid");
  }
  return mlir::success();
}
