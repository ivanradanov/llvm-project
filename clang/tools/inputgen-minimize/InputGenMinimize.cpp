//===--- tools/clang-check/ClangCheck.cpp - Clang check tool --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements a clang-check tool that runs clang based on the info
//  stored in a compilation database.
//
//  This tool uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDumperUtils.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExternalASTMerger.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Types.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "clang/Tooling/Syntax/TokenBufferTokenManager.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "clang/Tooling/Syntax/Tree.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <type_traits>

using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory
    InputGenMinimizeCat("inputgen-minimize options");

namespace {
using namespace clang;

#define TRY_TO(CALL_EXPR)                                                      \
  do {                                                                         \
    if (!CALL_EXPR)                                                            \
      return false;                                                            \
  } while (false)

static constexpr bool EnableDebugging = false;
#define INPUTGEN_DEBUG(CODE)                                                   \
  do {                                                                         \
    if (EnableDebugging) {                                                     \
      CODE;                                                                    \
    }                                                                          \
  } while (0)

// FIXME need to check if the following works in c++
//
// namespace one {
//   foo();
// }
// namespace two {
//   using namespace one; // <-- does this retain effect in the second namespace
//   two
// }
// namespace two {
//   bar() {
//     foo();
//   }
// }
class ContextSplitter : public ASTConsumer,
                        public RecursiveASTVisitor<ContextSplitter> {
  typedef RecursiveASTVisitor<ContextSplitter> base;

public:
  ContextSplitter() {}

  void HandleTranslationUnit(ASTContext &Context) override {
    TranslationUnitDecl *D = Context.getTranslationUnitDecl();
    TraverseDecl(D);
  }

  bool shouldWalkTypesOfTypeLocs() const { return false; }

private:
  void SplitNested(Decl *D) {
    DeclContext *DC = dyn_cast<DeclContext>(D);
    if (!DC)
      return;

    DeclContext::decl_iterator It = DC->decls_begin();
    if (It == DC->decls_end())
      return;
    SmallVector<Decl *> NestedDs;
    for (auto *NestedD : DC->decls())
      NestedDs.push_back(NestedD);
    for (auto *NestedD : NestedDs) {
      // TODO also need to do handle
      if (auto *ND = dyn_cast<NamespaceDecl>(NestedD)) {
        Split(ND);
      } else if (auto *LD = dyn_cast<LinkageSpecDecl>(NestedD)) {
        Split(LD);
      }
    }
  }

  template <typename T> void Split(T *D) {
    SmallVector<Decl *> NestedDs;
    for (auto *NestedD : D->decls())
      NestedDs.push_back(NestedD);
    if (NestedDs.empty())
      return;
    for (auto *NestedD : llvm::drop_begin(NestedDs)) {
      DeclContext *NewDC;
      Decl *NewD;
      if constexpr (std::is_same<T, NamespaceDecl>::value) {
        auto *NewND = NamespaceDecl::Create(
            D->getASTContext(), D->getDeclContext(), D->isInlineNamespace(),
            D->getBeginLoc(), D->getLocation(), D->getIdentifier(), D,
            D->isNested());
        NewD = NewND;
        NewDC = NewND;
      } else if constexpr (std::is_same<T, LinkageSpecDecl>::value) {
        LinkageSpecDecl *LD = D;
        auto *NewLD = LinkageSpecDecl::Create(
            D->getASTContext(), LD->getDeclContext(), LD->getExternLoc(),
            LD->getLocation(), LD->getLanguage(), LD->hasBraces());
        NewD = NewLD;
        NewDC = NewLD;
      } else {
        static_assert(0);
      }
      D->removeDecl(NestedD);
      NestedD->setLexicalDeclContext(NewDC);
      NewDC->addDeclInternal(NestedD);
      D->getDeclContext()->addDeclInternal(NewD);
    }
  }

public:
  bool TraverseDecl(Decl *D) {
    if (!base::TraverseDecl(D))
      return false;
    SplitNested(D);
    return true;
  }
};

// FIXME this approach does not work when the same static function is defined in
// multiple TUs
class ReorderDecls : public ASTConsumer,
                     public RecursiveASTVisitor<ReorderDecls> {
  typedef RecursiveASTVisitor<ReorderDecls> base;

  ExternalASTMerger &EAM;

public:
  ReorderDecls(ExternalASTMerger &EAM) : EAM(EAM) {}

  struct Reorderable {
    Decl *Top;
    Decl *Bottom;
  };
  bool isInSystemHeader(Decl *D) {
    SourceManager &SM = D->getASTContext().getSourceManager();
    bool Res = SM.isInSystemHeader(D->getBeginLoc());
    INPUTGEN_DEBUG(llvm::dbgs() << "isInSystemHeader " << Res << "\n";
                   D->dumpColor(); D->getBeginLoc().dump(SM););
    return Res;
  }
  void HandleTranslationUnit(ASTContext &Context) override {
    TranslationUnitDecl *TU = Context.getTranslationUnitDecl();
    std::vector<Reorderable> Rs;
    for (auto *D : TU->decls()) {
      Decl *Top = D;
      while (true) {
        auto *DC = dyn_cast<DeclContext>(D);
        if (!DC)
          break;
        if (!isa<NamespaceDecl, LinkageSpecDecl>(D))
          break;
        if (DC->decls_empty())
          break;
        assert(std::next(DC->decls_begin()) == DC->decls_end());
        D = *DC->decls_begin();
      }
      Decl *Bottom = D;
      Rs.push_back({Top, Bottom});
    }
    std::stable_sort(
        Rs.begin(), Rs.end(), [&](const Reorderable &A, const Reorderable &B) {
          auto getDeclAndCtx = [&](const Reorderable &R) {
            Decl *Decl = EAM.FindOriginalDecl(R.Bottom);
            ASTContext *Ctx = nullptr;
            if (Decl)
              Ctx = &Decl->getASTContext();
            return std::make_tuple(Decl, Ctx);
          };
          auto [ADecl, ACtx] = getDeclAndCtx(A);
          auto [BDecl, BCtx] = getDeclAndCtx(B);

          // Always put the functions in the generated entry TU last
          if (!ADecl && BDecl)
            return false;
          if (!BDecl && ADecl)
            return true;

          if (!ADecl || !BDecl)
            return true;

          if (ACtx < BCtx)
            return true;
          if (ACtx > BCtx)
            return false;

          SourceLocation ALoc = ADecl->getBeginLoc();
          SourceLocation BLoc = BDecl->getBeginLoc();
          if (ALoc.isInvalid() || BLoc.isInvalid())
            return ALoc.getPtrEncoding() < BLoc.getPtrEncoding();
          return ACtx->getSourceManager().isBeforeInTranslationUnit(ALoc, BLoc);
        });
    for (auto R : Rs)
      TU->removeDecl(R.Top);
    for (auto R : Rs) {
      // TODO need to generate the appropriate #includes for this
      if (!isInSystemHeader(R.Bottom))
        TU->addDeclInternal(R.Top);
      // TODO not sure if dropping these is bad?
    }
  }
};

std::string CleanPath(StringRef PathRef) {
  llvm::SmallString<128> Path(PathRef);
  llvm::sys::path::remove_dots(Path, /*remove_dot_dot=*/true);
  // FIXME: figure out why this is necessary.
  llvm::sys::path::native(Path);
  return std::string(Path);
}

// Make the Path absolute using the current working directory of the given
// SourceManager if the Path is not an absolute path.
//
// The Path can be a path relative to the build directory, or retrieved from
// the SourceManager.
std::string MakeAbsolutePath(const SourceManager &SM, StringRef Path) {
  llvm::SmallString<128> AbsolutePath(Path);
  if (std::error_code EC =
          SM.getFileManager().getVirtualFileSystem().makeAbsolute(AbsolutePath))
    llvm::errs() << "Warning: could not make absolute file: '" << EC.message()
                 << '\n';
  // Handle symbolic link path cases.
  // We are trying to get the real file path of the symlink.
  auto Dir = SM.getFileManager().getOptionalDirectoryRef(
      llvm::sys::path::parent_path(AbsolutePath.str()));
  if (Dir) {
    StringRef DirName = SM.getFileManager().getCanonicalName(*Dir);
    // FIXME: getCanonicalName might fail to get real path on VFS.
    if (llvm::sys::path::is_absolute(DirName)) {
      SmallString<128> AbsoluteFilename;
      llvm::sys::path::append(AbsoluteFilename, DirName,
                              llvm::sys::path::filename(AbsolutePath.str()));
      return CleanPath(AbsoluteFilename);
    }
  }
  return CleanPath(AbsolutePath);
}
struct InputGenMinimizeTool {
  void addIncludes(llvm::StringRef IncludeHeader, bool IsAngled,
                   llvm::StringRef SearchPath, llvm::StringRef FileName,
                   CharSourceRange IncludeFilenameRange,
                   const SourceManager &SM) {
    SmallString<128> HeaderWithSearchPath;
    llvm::sys::path::append(HeaderWithSearchPath, SearchPath, IncludeHeader);
    std::string AbsoluteIncludeHeader =
        MakeAbsolutePath(SM, HeaderWithSearchPath);
    std::string IncludeLine =
        IsAngled ? ("#include <" + IncludeHeader + ">\n").str()
                 : ("#include \"" + IncludeHeader + "\"\n").str();

    INPUTGEN_DEBUG(llvm::dbgs() << "INCLUDE" << AbsoluteIncludeHeader << "\n"
                                << IncludeLine);

    // std::string AbsoluteOldHeader =
    // makeAbsolutePath(Context->Spec.OldHeader); std::string
    // AbsoluteCurrentFile = MakeAbsolutePath(SM, FileName); if
    // (AbsoluteOldHeader == AbsoluteCurrentFile) {
    //   // Find old.h includes "old.h".
    //   if (AbsoluteOldHeader == AbsoluteIncludeHeader) {
    //     OldHeaderIncludeRangeInHeader = IncludeFilenameRange;
    //     return;
    //   }
    //   HeaderIncludes.push_back(IncludeLine);
    // } else if (makeAbsolutePath(Context->Spec.OldCC) == AbsoluteCurrentFile)
    // {
    //   // Find old.cc includes "old.h".
    //   if (AbsoluteOldHeader == AbsoluteIncludeHeader) {
    //     OldHeaderIncludeRangeInCC = IncludeFilenameRange;
    //     return;
    //   }
    //   CCIncludes.push_back(IncludeLine);
    // }
  }
};

class FindAllIncludes : public PPCallbacks {
public:
  explicit FindAllIncludes(SourceManager *SM, InputGenMinimizeTool &IGMT)
      : SM(*SM), IGMT(IGMT) {}

  void InclusionDirective(SourceLocation HashLoc, const Token & /*IncludeTok*/,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef /*File*/, StringRef SearchPath,
                          StringRef /*RelativePath*/,
                          const Module * /*SuggestedModule*/,
                          bool /*ModuleImported*/,
                          SrcMgr::CharacteristicKind /*FileType*/) override {
    if (auto FileEntry = SM.getFileEntryRefForID(SM.getFileID(HashLoc)))
      IGMT.addIncludes(FileName, IsAngled, SearchPath, FileEntry->getName(),
                       FilenameRange, SM);
  }

private:
  const SourceManager &SM;
  InputGenMinimizeTool &IGMT;
};

class DepGatherer : public ASTConsumer,
                    public RecursiveASTVisitor<DepGatherer> {
  typedef RecursiveASTVisitor<DepGatherer> base;

public:
  std::set<const Decl *> &Deps;
  DepGatherer(std::set<const Decl *> &Deps) : Deps(Deps) {}

  // We should not be traversing the contents of namespace or translation units
  // as we are trying to minimize their contents and we do not depend on their
  // entirety. Notably it does not include RecordDecl (structs, classes).
  bool ShouldTraverse(const Decl *D) {
    return !isa<NamespaceDecl, LinkageSpecDecl, TranslationUnitDecl>(D);
  }

  template <typename T> void InsertParents(T *Node, ASTContext &C) {
    const auto &Parents = C.template getParents<T>(*Node);
    for (auto &Parent : Parents) {
      if (const Decl *D = Parent.template get<Decl>()) {
        INPUTGEN_DEBUG(llvm::dbgs() << "Parent Decl\n"; D->dumpColor());
        if (ShouldTraverse(D)) {
          INPUTGEN_DEBUG(llvm::dbgs() << "Should Traverse\n");
          GatherDeps(const_cast<Decl *>(D));
        } else {
          INPUTGEN_DEBUG(llvm::dbgs() << "No Traverse\n");
          Deps.insert(D);
        }
        InsertParents(D, C);
      }
    }
  }

  bool InsertDep(Decl *D) {
    if (Deps.count(D))
      return false;
    Deps.insert(D);
    INPUTGEN_DEBUG(llvm::dbgs() << "Inserted\n"; D->dumpColor());
    InsertParents(D, D->getASTContext());
    return true;
  }

  bool GatherDepsFromThisDecl(Decl *D) {
    if (!InsertDep(D))
      return true;

    INPUTGEN_DEBUG(llvm::dbgs() << "Traversing\n"; D->dumpColor(););

    return base::TraverseDecl(D);
  }

  bool VisitMemberExpr(MemberExpr *E) {
    GatherDeps(E->getMemberDecl());
    return true;
  }

  bool VisitTypedefDecl(TypedefDecl *D) {
    TRY_TO(TraverseType(D->getUnderlyingType()));
    return true;
  }

  bool VisitTypeAliasDecl(TypeAliasDecl *D) {
    TRY_TO(TraverseType(D->getUnderlyingType()));
    return true;
  }

  bool VisitDecl(Decl *D) {
    TRY_TO(GatherDeps(D));
    return true;
  }

  bool ShouldGatherDeps(Decl *D) {
    SourceLocation Loc = D->getLocation();
    auto &SM = D->getASTContext().getSourceManager();
    auto FID = SM.getFileID(Loc);
    if (FID.isInvalid())
      return true;
    const FileEntry *FE = SM.getFileEntryForID(FID);
    if (!FE)
      return true;
    StringRef Name = FE->tryGetRealPathName();
    if (Name.starts_with("/usr/include"))
      return false;
    return true;
  }

  bool GatherDeps(Decl *D) {
    if (!ShouldGatherDeps(D))
      return true;

    for (Decl *RD : D->redecls()) {
      INPUTGEN_DEBUG(llvm::dbgs() << "Redecl\n"; RD->dumpColor(););
      TRY_TO(GatherDepsFromThisDecl(RD));
    }
    return true;
  }

  std::set<const Type *> HandledTypes;
  void GatherDeps(QualType QT) { GatherDeps(QT.getTypePtr()); }
  void GatherDeps(const Type *T) {
    if (HandledTypes.count(T))
      return;
    HandledTypes.insert(T);
    INPUTGEN_DEBUG(llvm::dbgs() << "Used Type\n"; T->dump(););

    // TODO is that all types that can have decls?
    if (const TypedefType *TT = T->getAs<TypedefType>()) {
      GatherDeps(TT->getDecl());
    } else if (const RecordType *TT = T->getAs<RecordType>()) {
      GatherDeps(TT->getDecl());
    } else if (const EnumType *TT = T->getAs<EnumType>()) {
      GatherDeps(TT->getDecl());
    }
  }

  bool VisitType(Type *T) {
    INPUTGEN_DEBUG(llvm::dbgs() << "Traverse Type\n");
    GatherDeps(T);
    return true;
  }

  bool VisitValueDecl(ValueDecl *D) {
    TRY_TO(TraverseType(D->getType()));
    return base::VisitValueDecl(D);
  }

  bool VisitStmt(Stmt *S) {
    INPUTGEN_DEBUG(llvm::dbgs() << "Visit\n"; S->dumpColor(););
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) {
    INPUTGEN_DEBUG(llvm::dbgs() << "Visit DeclRefExpr\n");
    GatherDeps(E->getDecl());
    return true;
  }

  bool VisitCallExpr(CallExpr *E) {
    // TODO we would like to take care of indirect calls. we should probably
    // either gather all functions matching the signature or instrument indirect
    // calls and use a recording run to see which functions were used.

    //  TODO if we don't have access to the body there are a couple of options:
    //
    //    1. The definition is in another file? We would like integration with
    //    compile_commands.json and to be able to track those down.
    //
    //    3. The definition is in an external library - in that case we would
    //    like to generate the correct #include for it or just print the
    //    declaration.
    //
    //    Looks like there is a way to merge AST's together:
    //    https://clang.llvm.org/docs/LibASTImporter.html can we merge in all
    //    ASTs and then print out the extracted c/c++ from there?
    return true;
  }

private:
};

class ASTMinimizer : public ASTConsumer,
                     public RecursiveASTVisitor<ASTMinimizer> {
  typedef RecursiveASTVisitor<ASTMinimizer> base;

  bool &Done;

public:
  ASTMinimizer(raw_ostream &Out, bool &Done) : Done(Done), Out(Out) {}

  TranslationUnitDecl *TUD;
  void HandleTranslationUnit(ASTContext &Context) override {
    TUD = Context.getTranslationUnitDecl();
    TraverseDecl(TUD);
  }

  bool shouldWalkTypesOfTypeLocs() const { return false; }

  void generateEntryPoint(FunctionDecl *FD) {
    PrintingPolicy Policy(FD->getASTContext().getLangOpts());
    std::string Indent = "  ";

    Out << "void __inputrun_entry(char *Args) {\n";

    Out << Indent << "char *CurArg = Args;\n";
    for (unsigned I = 0; I < FD->getNumParams(); I++) {
      ParmVarDecl *Param = FD->getParamDecl(I);

      QualType T = Param->getType();
      std::string TypeStr = T.getAsString(Policy);
      Out << Indent << TypeStr << " Arg" << I << " = *(" << TypeStr
          << " *) CurArg;\n";
      Out << Indent << "CurArg += 16;\n";
    }

    Out << Indent << FD->getNameAsString() << "(";
    for (unsigned I = 0; I < FD->getNumParams(); I++) {
      if (I != 0)
        Out << ", ";
      Out << "Arg" << I;
    }
    Out << ");\n";

    Out << "}\n";
  }

  void PrintWithDeps(Decl *D) {
    std::set<const Decl *> Deps;
    DepGatherer DG(Deps);
    DG.GatherDeps(D);

    // TODO if possible try to omit the inputgen attr
    PrintingPolicy Policy(D->getASTContext().getLangOpts());
    std::function<bool(const Decl *)> Filter = [&Deps](const Decl *D) -> bool {
      return Deps.count(D);
    };
    TUD->print(Out, Policy, /*Indentation=*/0, /*PrintInstantiation=*/false,
               &Filter);
  }

  bool TraverseDecl(Decl *D) {
    if (FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D)) {
      if (FD->hasAttr<InputGenEntryAttr>()) {
        if (Done) {
          llvm::errs() << "Multiple inputgen entry points found, ignoring "
                       << FD->getName() << "\n";
          return true;
        }
#if 0
        Out << "\n";
        Out << "// =========== RECORDED FUNCTION BEGIN ===========\n";
        PrintWithDeps(D);
        Out << "// =========== RECORDED FUNCTION END ===========\n";
        Out << "\n";
#endif
        Out << "// =========== GENERATED ENTRY POINT BEGIN ===========\n";
        generateEntryPoint(FD);
        Out << "// =========== GENERATED ENTRY POINT END ===========\n";
        Done = true;
        // Don't traverse child nodes to avoid output duplication.
        return true;
      }
    }
    return base::TraverseDecl(D);
  }

private:
  std::string getName(Decl *D) {
    if (isa<NamedDecl>(D))
      return cast<NamedDecl>(D)->getQualifiedNameAsString();
    return "";
  }
  void print(Decl *D) {
    // TODO if possible try to omit the inputgen attr
    PrintingPolicy Policy(D->getASTContext().getLangOpts());
    D->print(Out, Policy, /*Indentation=*/0, /*PrintInstantiation=*/true);
  }

  raw_ostream &Out;
};

std::unique_ptr<ASTConsumer> CreateASTMinimizer(raw_ostream &Out, bool &Done) {
  return std::make_unique<ASTMinimizer>(Out, Done);
}

namespace init_convenience {
class TestDiagnosticConsumer : public DiagnosticConsumer {
private:
  std::unique_ptr<TextDiagnosticBuffer> Passthrough;
  const LangOptions *LangOpts = nullptr;

public:
  TestDiagnosticConsumer()
      : Passthrough(std::make_unique<TextDiagnosticBuffer>()) {}

  void BeginSourceFile(const LangOptions &LangOpts,
                       const Preprocessor *PP = nullptr) override {
    this->LangOpts = &LangOpts;
    return Passthrough->BeginSourceFile(LangOpts, PP);
  }

  void EndSourceFile() override {
    this->LangOpts = nullptr;
    Passthrough->EndSourceFile();
  }

  bool IncludeInDiagnosticCounts() const override {
    return Passthrough->IncludeInDiagnosticCounts();
  }

private:
  static void PrintSourceForLocation(const SourceLocation &Loc,
                                     SourceManager &SM) {
    const char *LocData = SM.getCharacterData(Loc, /*Invalid=*/nullptr);
    unsigned LocColumn =
        SM.getSpellingColumnNumber(Loc, /*Invalid=*/nullptr) - 1;
    FileID FID = SM.getFileID(Loc);
    llvm::MemoryBufferRef Buffer = SM.getBufferOrFake(FID, Loc);

    assert(LocData >= Buffer.getBufferStart() &&
           LocData < Buffer.getBufferEnd());

    const char *LineBegin = LocData - LocColumn;

    assert(LineBegin >= Buffer.getBufferStart());

    const char *LineEnd = nullptr;

    for (LineEnd = LineBegin; *LineEnd != '\n' && *LineEnd != '\r' &&
                              LineEnd < Buffer.getBufferEnd();
         ++LineEnd)
      ;

    llvm::StringRef LineString(LineBegin, LineEnd - LineBegin);

    llvm::errs() << LineString << '\n';
    llvm::errs().indent(LocColumn);
    llvm::errs() << '^';
    llvm::errs() << '\n';
  }

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override {
    if (Info.hasSourceManager() && LangOpts) {
      SourceManager &SM = Info.getSourceManager();

      if (Info.getLocation().isValid()) {
        Info.getLocation().print(llvm::errs(), SM);
        llvm::errs() << ": ";
      }

      SmallString<16> DiagText;
      Info.FormatDiagnostic(DiagText);
      llvm::errs() << DiagText << '\n';

      if (Info.getLocation().isValid()) {
        PrintSourceForLocation(Info.getLocation(), SM);
      }

      for (const CharSourceRange &Range : Info.getRanges()) {
        bool Invalid = true;
        StringRef Ref = Lexer::getSourceText(Range, SM, *LangOpts, &Invalid);
        if (!Invalid) {
          llvm::errs() << Ref << '\n';
        }
      }
    }
    DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);
  }
};

static llvm::cl::list<std::string>
    ClangArgs("Xcc",
              llvm::cl::desc("Argument to pass to the CompilerInvocation"),
              llvm::cl::CommaSeparated);

std::unique_ptr<CompilerInstance> BuildCompilerInstance() {
  auto Ins = std::make_unique<CompilerInstance>();
  auto DC = std::make_unique<TestDiagnosticConsumer>();
  const bool ShouldOwnClient = true;
  Ins->createDiagnostics(*llvm::vfs::getRealFileSystem(), DC.release(),
                         ShouldOwnClient);

  auto Inv = std::make_unique<CompilerInvocation>();

  std::vector<const char *> ClangArgv(ClangArgs.size());
  std::transform(ClangArgs.begin(), ClangArgs.end(), ClangArgv.begin(),
                 [](const std::string &s) -> const char * { return s.data(); });
  CompilerInvocation::CreateFromArgs(*Inv, ClangArgv, Ins->getDiagnostics());

  std::string Input = "c++";
  {
    using namespace driver::types;
    ID Id = lookupTypeForTypeSpecifier(Input.c_str());
    assert(Id != TY_INVALID);
    if (isCXX(Id)) {
      Inv->getLangOpts().CPlusPlus = true;
      Inv->getLangOpts().CPlusPlus11 = true;
      Inv->getHeaderSearchOpts().UseLibcxx = true;
    }
    if (isObjC(Id)) {
      Inv->getLangOpts().ObjC = 1;
    }
  }
  Inv->getLangOpts().ObjCAutoRefCount = false;

  // TODO we should collect these from the compilation for the file that
  // contains the entry function
  Inv->getLangOpts().Bool = true;
  Inv->getLangOpts().WChar = true;
  Inv->getLangOpts().Blocks = true;
  Inv->getLangOpts().DebuggerSupport = true;
  Inv->getLangOpts().SpellChecking = false;
  Inv->getLangOpts().ThreadsafeStatics = false;
  Inv->getLangOpts().AccessControl = false;
  Inv->getLangOpts().DollarIdents = true;
  Inv->getLangOpts().Exceptions = true;
  Inv->getLangOpts().CXXExceptions = true;
  // Needed for testing dynamic_cast.
  Inv->getLangOpts().RTTI = true;
  Inv->getCodeGenOpts().setDebugInfo(llvm::codegenoptions::FullDebugInfo);
  Inv->getTargetOpts().Triple = llvm::sys::getDefaultTargetTriple();

  Ins->setInvocation(std::move(Inv));

  TargetInfo *TI = TargetInfo::CreateTargetInfo(
      Ins->getDiagnostics(), Ins->getInvocation().TargetOpts);
  Ins->setTarget(TI);
  Ins->getTarget().adjust(Ins->getDiagnostics(), Ins->getLangOpts());
  Ins->createFileManager();
  Ins->createSourceManager(Ins->getFileManager());
  Ins->createPreprocessor(TU_Complete);

  return Ins;
}

std::unique_ptr<ASTContext>
BuildASTContext(CompilerInstance &CI, SelectorTable &ST, Builtin::Context &BC) {
  auto &PP = CI.getPreprocessor();
  auto AST =
      std::make_unique<ASTContext>(CI.getLangOpts(), CI.getSourceManager(),
                                   PP.getIdentifierTable(), ST, BC, PP.TUKind);
  AST->InitBuiltinTypes(CI.getTarget());
  return AST;
}

} // namespace init_convenience

/// A container for a CompilerInstance (possibly with an ExternalASTMerger
/// attached to its ASTContext).
///
/// Provides an accessor for the DeclContext origins associated with the
/// ExternalASTMerger (or an empty list of origins if no ExternalASTMerger is
/// attached).
///
/// This is the main unit of parsed source code maintained by clang-import-test.
struct CIAndOrigins {
  using OriginMap = clang::ExternalASTMerger::OriginMap;
  std::unique_ptr<CompilerInstance> CI;

  ASTContext &getASTContext() { return CI->getASTContext(); }
  FileManager &getFileManager() { return CI->getFileManager(); }
  const OriginMap &getOriginMap() {
    static const OriginMap EmptyOriginMap{};
    if (ExternalASTSource *Source = CI->getASTContext().getExternalSource())
      return static_cast<ExternalASTMerger *>(Source)->GetOrigins();
    return EmptyOriginMap;
  }
  DiagnosticConsumer &getDiagnosticClient() {
    return CI->getDiagnosticClient();
  }
  CompilerInstance &getCompilerInstance() { return *CI; }
};

ExternalASTMerger *
AddExternalSource(CIAndOrigins &CI,
                  llvm::MutableArrayRef<CIAndOrigins> Imports) {
  ExternalASTMerger::ImporterTarget Target(
      {CI.getASTContext(), CI.getFileManager()});
  llvm::SmallVector<ExternalASTMerger::ImporterSource, 3> Sources;
  for (CIAndOrigins &Import : Imports)
    Sources.emplace_back(Import.getASTContext(), Import.getFileManager(),
                         Import.getOriginMap());
  auto ES = std::make_unique<ExternalASTMerger>(Target, Sources);
  ExternalASTMerger *ESPtr = ES.get();
  INPUTGEN_DEBUG(ES->SetLogStream(llvm::dbgs()));
  CI.getASTContext().setExternalSource(ES.release());
  CI.getASTContext().getTranslationUnitDecl()->setHasExternalVisibleStorage();
  return ESPtr;
}

llvm::Error ParseSource(const std::string &Path, CompilerInstance &CI,
                        ASTConsumer &Consumer) {
  SourceManager &SM = CI.getSourceManager();
  auto FE = CI.getFileManager().getFileRef(Path);
  if (!FE) {
    llvm::consumeError(FE.takeError());
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("No such file or directory: ", Path), std::error_code());
  }
  SM.setMainFileID(SM.createFileID(*FE, SourceLocation(), SrcMgr::C_User));
  ParseAST(CI.getPreprocessor(), &Consumer, CI.getASTContext());
  return llvm::Error::success();
}

llvm::Expected<CIAndOrigins> Parse(const std::string &Path,
                                   llvm::MutableArrayRef<CIAndOrigins> Imports,
                                   raw_ostream *Out) {
  CIAndOrigins CI{init_convenience::BuildCompilerInstance()};
  auto ST = std::make_unique<SelectorTable>();
  auto BC = std::make_unique<Builtin::Context>();
  std::unique_ptr<ASTContext> AST =
      init_convenience::BuildASTContext(CI.getCompilerInstance(), *ST, *BC);
  CI.getCompilerInstance().setASTContext(AST.release());
  ExternalASTMerger *EAM = nullptr;
  if (Imports.size())
    EAM = AddExternalSource(CI, Imports);

  std::vector<std::unique_ptr<ASTConsumer>> ASTConsumers;

  if (Out) {
    ASTConsumers.push_back(std::make_unique<ContextSplitter>());
    if (EAM) {
      ASTConsumers.push_back(std::make_unique<ReorderDecls>(*EAM));
    }
    ASTConsumers.push_back(CreateASTPrinter(nullptr, ""));
  }

  CI.getDiagnosticClient().BeginSourceFile(
      CI.getCompilerInstance().getLangOpts(),
      &CI.getCompilerInstance().getPreprocessor());
  MultiplexConsumer Consumers(std::move(ASTConsumers));
  Consumers.Initialize(CI.getASTContext());

  if (llvm::Error PE = ParseSource(Path, CI.getCompilerInstance(), Consumers))
    return std::move(PE);
  CI.getDiagnosticClient().EndSourceFile();
  if (CI.getDiagnosticClient().getNumErrors())
    return llvm::make_error<llvm::StringError>(
        "Errors occurred while parsing the expression.", std::error_code());
  return std::move(CI);
}

class MinimizeActionFactory {
private:
  std::string OutStr;
  llvm::raw_string_ostream Out;
  bool Done = false;

public:
  const std::string &getGeneratedEntry() { return Out.str(); };
  MinimizeActionFactory() : Out(OutStr) {}
  std::unique_ptr<clang::ASTConsumer> newASTConsumer() {
    return CreateASTMinimizer(Out, Done);
  }
};

class ExternalSourceFrontendAction : public ASTFrontendAction {
  InputGenMinimizeTool &IGMT;

public:
  ExternalSourceFrontendAction(InputGenMinimizeTool &IGMT) : IGMT(IGMT) {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    CI.getPreprocessor().addPPCallbacks(
        std::make_unique<FindAllIncludes>(&CI.getSourceManager(), IGMT));
    SourceManager &SM = CI.getSourceManager();
    auto FE = CI.getFileManager().getFileRef(InFile);
    if (!FE) {
      llvm::consumeError(FE.takeError());
      return nullptr;
    }
    SM.setMainFileID(SM.createFileID(*FE, SourceLocation(), SrcMgr::C_User));
    return std::make_unique<clang::ASTConsumer>();
  }
  void ExecuteAction() override {
    clang::CompilerInstance *Compiler = &getCompilerInstance();
    assert(!Compiler->hasSema() && "CI already has Sema");

    // Set up our hooks into sema and parse the AST.
    if (hasCodeCompletionSupport() &&
        !Compiler->getFrontendOpts().CodeCompletionAt.FileName.empty())
      Compiler->createCodeCompletionConsumer();

    clang::CodeCompleteConsumer *CompletionConsumer = nullptr;
    if (Compiler->hasCodeCompletionConsumer())
      CompletionConsumer = &Compiler->getCodeCompletionConsumer();

    Compiler->createSema(getTranslationUnitKind(), CompletionConsumer);

    clang::ParseAST(Compiler->getSema(), Compiler->getFrontendOpts().ShowStats,
                    Compiler->getFrontendOpts().SkipFunctionBodies);
  }
  void EndSourceFile() override {
    CompilerInstance &CI = getCompilerInstance();

    // Inform the diagnostic client we are done with this source file.
    CI.getDiagnosticClient().EndSourceFile();
  }
  ~ExternalSourceFrontendAction() = default;
};

class AddExternalSourceAction : public ToolAction {

  std::vector<CIAndOrigins> &ImportCIs;
  InputGenMinimizeTool &IGMT;

public:
  ~AddExternalSourceAction() override = default;
  AddExternalSourceAction(std::vector<CIAndOrigins> &ImportCIs,
                          InputGenMinimizeTool &IGMT)
      : ImportCIs(ImportCIs), IGMT(IGMT) {}
  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    // Create a compiler instance to handle the actual work.
    std::unique_ptr<CompilerInstance> Compiler =
        std::make_unique<CompilerInstance>(std::move(PCHContainerOps));
    Compiler->setInvocation(std::move(Invocation));
    Compiler->setFileManager(Files);

    std::unique_ptr<FrontendAction> ScopedToolAction =
        std::make_unique<ExternalSourceFrontendAction>(IGMT);

    // Create the compiler's actual diagnostics engine.
    Compiler->createDiagnostics(Files->getVirtualFileSystem(), DiagConsumer,
                                /*ShouldOwnClient=*/false);
    if (!Compiler->hasDiagnostics())
      return false;

    Compiler->createSourceManager(*Files);

    const bool Success = Compiler->ExecuteAction(*ScopedToolAction);

    Files->clearStatCache();

    CIAndOrigins CI{std::move(Compiler)};
    ImportCIs.push_back(std::move(CI));

    return Success;
  }
};

} // namespace

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  // Initialize targets for clang module support.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  static llvm::cl::opt<std::string> BuildPath(
      "p", llvm::cl::desc("Build path"), llvm::cl::Required,
      llvm::cl::cat(InputGenMinimizeCat),
      llvm::cl::sub(llvm::cl::SubCommand::getAll()));
  llvm::cl::ResetAllOptionOccurrences();

  llvm::cl::HideUnrelatedOptions(InputGenMinimizeCat);

  std::string ErrorMessage;
  llvm::raw_string_ostream OS(ErrorMessage);
  if (!llvm::cl::ParseCommandLineOptions(argc, argv, "InputGenMinimize", &OS)) {
    auto Err = llvm::make_error<llvm::StringError>(
        ErrorMessage, llvm::inconvertibleErrorCode());
    if (Err) {
      llvm::report_fatal_error(
          Twine(
              "CommonOptionsParser: failed to parse command-line arguments. ") +
          llvm::toString(std::move(Err)));
    }
  }

  if (BuildPath.empty()) {
    llvm::errs() << "Build path needs to be specified.\n";
    return 1;
  }

  std::unique_ptr<CompilationDatabase> Compilations =
      CompilationDatabase::autoDetectFromDirectory(BuildPath, ErrorMessage);
  if (!Compilations) {
    llvm::errs() << "Error while trying to load a compilation database:\n"
                 << ErrorMessage << "Running without flags.\n";
    return 1;
  }

  auto Files = Compilations->getAllFiles();
  for (auto F : Files)
    llvm::errs() << F << "\n";
  ClangTool Tool(*Compilations, Files);

  MinimizeActionFactory MinimizeFactory;
  std::unique_ptr<FrontendActionFactory> FrontendFactory;

  FrontendFactory = newFrontendActionFactory(&MinimizeFactory);

  std::vector<CIAndOrigins> ImportCIs;
  InputGenMinimizeTool IGMT;
  AddExternalSourceAction AESA(ImportCIs, IGMT);

  int res = Tool.run(FrontendFactory.get());
  if (res)
    return res;

  res = Tool.run(&AESA);
  if (res)
    return res;

  int FD;
  SmallString<128> Filename;
  if (std::error_code EC = llvm::sys::fs::createTemporaryFile(
          "entry-temp", "cpp", FD, Filename)) {
    llvm::errs() << "Error: " << EC.message() << "\n";
    return 1;
  }

  llvm::raw_fd_ostream O(FD, true);
  O << MinimizeFactory.getGeneratedEntry();
  O.close();

  llvm::Expected<CIAndOrigins> ExpressionCI =
      Parse(Filename.c_str(), ImportCIs, &llvm::outs());
  if (auto E = ExpressionCI.takeError()) {
    llvm::errs() << "error: " << llvm::toString(std::move(E)) << "\n";
    exit(-1);
  }

  return 0;
}
