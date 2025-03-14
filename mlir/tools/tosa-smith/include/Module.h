#ifndef TOSA_SMITH_MODULE_H
#define TOSA_SMITH_MODULE_H

#include "Function.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace tosas {

static llvm::cl::opt<unsigned> numFunc{
    "f",
    llvm::cl::desc("The number of functions to create [defualt: 1]"),
    llvm::cl::init(1)
};

class Module {
public:
    Module(mlir::MLIRContext &ctx) : builder(&ctx), context(&ctx) {}

    mlir::ModuleOp build() {
        mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToStart(&theModule.getBodyRegion().back());

        // Setup the print utils in the runner library
        for (auto width: SUPPORTED_INT_WIDTH) {
            Function::buildPrintMemrefIOfWidth(builder, width);
        }
        for (auto width: SUPPORTED_FLOAT_WIDTH) {
            Function::buildPrintMemrefFOfWidth(builder, width);
        }

        // Build other functions
        for (unsigned i = 1; i < numFunc; i++) {
            tosas::Function theFunction(context);
            mlir::Operation *f = theFunction.build(builder, /* funcIdx */ i);
            context.addDefinedFunc(f);
        }

        // Build the main function
        tosas::Function theMain(context, /* isMain */ true);
        theMain.build(builder);

        return theModule;
    }

private:
    mlir::OpBuilder builder;
    Context context;
};

} // namespace tosas

#endif