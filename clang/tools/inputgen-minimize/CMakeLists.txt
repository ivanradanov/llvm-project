set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
  )

add_clang_tool(inputgen-minimize
  InputGenMinimize.cpp
  )

clang_target_link_libraries(inputgen-minimize
  PRIVATE
  clangLex
  clangSema
  clangParse
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangRewriteFrontend
  clangSerialization
  clangStaticAnalyzerFrontend
  clangTooling
  clangToolingSyntax
  LLVMTargetParser
  )
