#ifndef TOSA_SMITH_FUNCTION_H
#define TOSA_SMITH_FUNCTION_H

#include "Utils.h"
#include "Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace tosas {

static const llvm::cl::opt<unsigned> maxConstNum{
    "max-func-const-num",
    llvm::cl::desc("The maximum number of constants defined [default: 10]"),
    llvm::cl::init(10)
};

static const llvm::cl::opt<unsigned> maxConstMin{
    "min-func-const-num",
    llvm::cl::desc("The minimum number of constants defined [default: 5]"),
    llvm::cl::init(5)
};

static const llvm::cl::opt<unsigned> maxBodyOpNum{
    "max-func-ops-num",
    llvm::cl::desc("The maximun number of operations in the function body [default: 60]"),
    llvm::cl::init(60)
};

static const llvm::cl::opt<unsigned> minBodyOpNum{
    "min-func-ops-num",
    llvm::cl::desc("The minimum number of operations in the function body [default: 20]"),
    llvm::cl::init(20)
};

static const llvm::cl::opt<unsigned> printValNum {
    "v",
    llvm::cl::desc("The number of values that are randomly selected to print [default: 1]"),
    llvm::cl::init(1)
};

class Function {
public:
    Function(
        Context &parentCtx,
        bool isMain = false
    ) : isMain(isMain), context(parentCtx) {}

    static mlir::func::FuncOp buildPrintMemrefIOfWidth(mlir::OpBuilder &builder,
                                                       unsigned width)
    {
        std::string funcName = "printMemrefI" + std::to_string(width);
        mlir::func::FuncOp printFunc = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), 
            funcName,
            builder.getFunctionType(
                {mlir::UnrankedTensorType::get(
                    mlir::IntegerType::get(builder.getContext(), width)
                )}, {}
            )
        );
        printFunc.setPrivate();
        return printFunc;
    }

    static mlir::func::FuncOp buildPrintMemrefFOfWidth(mlir::OpBuilder &builder,
                                                       unsigned width)
    {
        assert(
            (width == 32 || width == 64) 
            && "Only 32 and 64 width float numbers are supported"
        );

        std::string funcName = "printMemrefF" + std::to_string(width);
        mlir::FloatType fTy;
        if (width == 32) fTy = mlir::Float32Type::get(builder.getContext());

        mlir::func::FuncOp printFunc = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), 
            funcName,
            builder.getFunctionType({mlir::UnrankedTensorType::get(fTy)}, {})
        );
        printFunc.setPrivate();
        return printFunc;
    }

    mlir::Operation *build(mlir::OpBuilder &builder, unsigned funcIdx = 0) {
        // Create an empty function
        std::string fName = "func" + std::to_string(funcIdx);
        mlir::func::FuncOp theFunction = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), 
            isMain ? "main" : fName,
            isMain ? builder.getFunctionType({}, {}) : context.randomFunctionType()
        );

        for (auto val: theFunction.getArguments())
            context.addSingleVal(val);

        // Init the function
        theFunction.addEntryBlock();
        func = &theFunction;

        auto savedBuildPos = builder.saveInsertionPoint();
        builder.setInsertionPointToEnd(&theFunction.getBody().front());

        // Build operations in the function body
        buildTosaConstOps(builder);
        buildBody(builder);
        buildFuncReturnOp(builder);
        
        builder.restoreInsertionPoint(savedBuildPos);
        return theFunction.getOperation();
    }

private:
    bool isMain;
    mlir::func::FuncOp *func;
    
    Operation op;
    Context context;

    void buildTosaConstOps(mlir::OpBuilder &builder) {
        context.setTransposeAttr(builder.getDenseI32ArrayAttr({0, 2, 1}));

        mlir::Operation *indexOp = op.buildIndexConst(builder, 0);
        context.setIndexZeroConst(indexOp->getResult(0));

        for (unsigned i = 0; i < maxConstNum; i++) {
            op.buildRandomTensor(builder, context, context.getInitTensorTy());
        }
    }

    void buildBody(mlir::OpBuilder& builder) {
        op.buildOps(builder, context, {minBodyOpNum, maxBodyOpNum}, 0);

        // We select and print tensors in the main function
        if (isMain) {
            for (unsigned i = 0; i < printValNum; i++)
                op.buildValPrint(builder, context);
        }
    }

    void buildFuncReturnOp(mlir::OpBuilder &builder) {
        if (mlir::Operation *opToRet = op.buildFuncReturnOp(builder, context, isMain)) {
            func->setFunctionType(builder.getFunctionType(
                func->getArgumentTypes(),
                opToRet->getResultTypes()
            ));
        }
    }
};

} // namespace tosas

#endif