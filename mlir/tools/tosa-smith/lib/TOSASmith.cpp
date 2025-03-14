#include "TOSASmith.h"
#include "Module.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/FileUtilities.h"

static llvm::cl::opt<std::string> outputFileName(
    "o",
    llvm::cl::desc("Output filename"),
    llvm::cl::init("-")
);

llvm::LogicalResult tosas::runTOSASmith() {

    // Register the required dialects
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);

    context.loadDialect<mlir::tosa::TosaDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::tensor::TensorDialect>();
    context.loadDialect<mlir::index::IndexDialect>();

    // Init the module builder
    tosas::Module theModule(context);
    mlir::Operation *moduleOp = theModule.build().getOperation();

    // Generate the TOSA MLIR
    if (outputFileName.getValue() == "-") {
        moduleOp->dump();
    } else {
        std::string errorMsg;
        if (auto output = mlir::openOutputFile(outputFileName.getValue(), &errorMsg)) {
            moduleOp->print(output->os());
            output->keep();
        } else {
            llvm::errs() << errorMsg << "\n";
            return llvm::failure();
        }
    }
    
    return llvm::success();
}

llvm::LogicalResult tosas::TOSASmithMain(int argc, char **argv) {

    llvm::cl::ParseCommandLineOptions(argc, argv, "TOSA Smith");
    llvm::InitLLVM(argc, argv);

    return runTOSASmith();
}