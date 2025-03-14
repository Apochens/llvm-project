#ifndef TOSA_SMITH_CONTEXT_H
#define TOSA_SMITH_CONTEXT_H

#include "Type.h"
#include "mlir/IR/BuiltinTypes.h"

namespace tosas {

class Context {
public:
    Context() {}

    Context(mlir::MLIRContext *ctx) : mlirCtx(ctx) {
        initTensorTy = randomTensorType();
    }

    inline mlir::MLIRContext *getMLIRContext() const { return mlirCtx; }

    template<typename T = mlir::Type> mlir::Value randomVal(unsigned width = 32);
    mlir::Value randomValCompatibleOf(mlir::Value &val);

    template<typename T = mlir::Type>
    std::pair<mlir::Value, mlir::Value> randomValPair(unsigned width = 32) {
        if (mlir::Value val1 = randomVal<T>(width)) {
            if (mlir::Value val2 = randomValCompatibleOf(val1)) {
                return { val1, val2 };
            }
        }

        return { nullptr, nullptr };
    }

    inline mlir::IntegerType randomIntegerType() const {
        return mlir::IntegerType::get(mlirCtx, random(SUPPORTED_INT_WIDTH));
    }

    mlir::FloatType randomFloatType() const {
        llvm::SmallVector<mlir::FloatType> supportedTypes;
        for (unsigned width: SUPPORTED_FLOAT_WIDTH) {
            if (width == 32) {
                supportedTypes.push_back(mlir::Float32Type::get(mlirCtx));
            }
            if (width == 64) {
                supportedTypes.push_back(mlir::Float64Type::get(mlirCtx));
            }
        }
        return random(supportedTypes);
    }

    // Because cast an integer to a float won't loss the accuracy,
    // we generate integer type tensor by default
    template<typename T = mlir::IntegerType>
    mlir::TensorType randomTensorType(unsigned rank = DEFAULT_TENSOR_RANK) {
        mlir::Type targetType;
        if (isInteger<T>())
            targetType = randomIntegerType();
        if (isFloat<T>())
            targetType = randomFloatType();

        if (rank == 0) {
            assert(false && "randomTensorType<0> unimplemented.");
            return mlir::UnrankedTensorType::get(targetType);
        }

        llvm::SmallVector<int64_t> tensorShape;
        tensorShape.push_back(1);
        for (unsigned i = 1; i < rank; i++) {
            tensorShape.push_back(random(tensorLenMax, tensorLenMin));
        }

        return  mlir::RankedTensorType::get(tensorShape, targetType);
    }

    mlir::FunctionType randomFunctionType() {
        unsigned funcArgNum = random(funcArgNumMax);
        llvm::SmallVector<mlir::Type> funcArg;
        for (unsigned i = 0; i < funcArgNum; i++) {
            funcArg.push_back(initTensorTy);
        }

        // The function returns void by default, and the return type range is
        // modified when the return values are decided in the buildFuncCallOp
        // builder.
        return mlir::FunctionType::get(mlirCtx, funcArg, {});
    }

    inline void addConstVals(mlir::Operation *op) {
        constVals.insert(op->getResults().begin(),
                         op->getResults().end());
    }

    inline void addResultVals(mlir::Operation *op) {
        resultVals.insert(op->getResults().begin(),
                          op->getResults().end());
    }

    inline void addSingleVal(mlir::Value val) { constVals.insert(val); }

    inline mlir::Operation *randomFunc() { return random(moduleFuncs); }
    inline void addDefinedFunc(mlir::Operation *f) { moduleFuncs.insert(f); }
    inline bool hasDefinedFunc() const { return moduleFuncs.size() > 0; }

    inline void setTransposeConst(mlir::Value val) { transposeConst = val; }
    inline mlir::Value getTransposeConst() const { return transposeConst; }

    inline void setIndexZeroConst(mlir::Value val) { indexZeroConst = val; }
    inline mlir::Value getIndexZeroConst() const { return indexZeroConst; }

    inline mlir::TensorType getInitTensorTy() const { return initTensorTy; }
private:
    mlir::MLIRContext *mlirCtx;

    llvm::DenseSet<mlir::Value> constVals;
    llvm::DenseSet<mlir::Value> resultVals;
    llvm::DenseSet<mlir::Operation *> moduleFuncs;

    mlir::Value transposeConst;
    mlir::Value indexZeroConst;
    mlir::TensorType initTensorTy;
};

template<typename T>
mlir::Value Context::randomVal(unsigned width) {
    llvm::DenseSet<mlir::Value> candVals;

    for (mlir::Value val: resultVals) {
        mlir::RankedTensorType valTy = mlir::cast<mlir::RankedTensorType>(val.getType());
        if (valTy.getElementTypeBitWidth() == width) {
            if (isInteger<T>()) {
                // For operations that only takes integers, float numbers are excluded
                if (valTy.getElementType().isInteger())
                    candVals.insert(val);
            } else {
                candVals.insert(val);
            }
        }
    }

    // If we can find any candidate from the result value pool,
    // a randomly selected value is returned
    if (!candVals.empty())
        return random(candVals);

    for (mlir::Value val: constVals) {
        mlir::RankedTensorType valTy = mlir::cast<mlir::RankedTensorType>(val.getType());
        if (valTy.getElementTypeBitWidth() == width) {
            if (isInteger<T>()) {
                // For operations that only takes integers, float numbers are excluded
                if (valTy.getElementType().isInteger())
                    candVals.insert(val);
            } else {
                candVals.insert(val);
            }
        }
    }

    if (candVals.empty())
        return nullptr;

    assert(!candVals.empty() && "No candidate values for building new operations!");
    return random(candVals);
}

mlir::Value Context::randomValCompatibleOf(mlir::Value &val) {
    mlir::TensorType valTy = mlir::cast<mlir::TensorType>(val.getType());
    unsigned width = valTy.getElementTypeBitWidth();
    auto shape = valTy.getShape();

    llvm::DenseSet<mlir::Value> candVals;
    for (mlir::Value resVal: resultVals) {
        mlir::TensorType resValTy = mlir::cast<mlir::TensorType>(resVal.getType());
        if (resValTy.getElementTypeBitWidth() == width && resValTy.getShape() == shape) {
            if (valTy.getElementType().isInteger()) {
                // For operations that only takes integers, float numbers are excluded
                if (resValTy.getElementType().isInteger())
                    candVals.insert(resVal);
            } else {
                candVals.insert(resVal);
            }
        }
    }

    // If we can find any candidate from the result value pool,
    // a randomly selected value is returned
    if (!candVals.empty())
        return random(candVals);

    for (mlir::Value val: constVals) {
        mlir::RankedTensorType valTy = mlir::cast<mlir::RankedTensorType>(val.getType());
        if (valTy.getElementTypeBitWidth() == width && valTy.getShape() == shape) {
            if (valTy.getElementType().isInteger()) {
                // For operations that only takes integers, float numbers are excluded
                if (valTy.getElementType().isInteger())
                    candVals.insert(val);
            } else {
                candVals.insert(val);
            }
        }
    }

    if (candVals.empty())
        return nullptr;

    assert(!candVals.empty() && "No candidate values for building new operations!");
    return random(candVals);
}

static Context EMPTY_CONTEXT;

} // namespace tosas

#endif