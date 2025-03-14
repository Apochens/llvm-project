#ifndef TOSA_SMITH_TYPE_H
#define TOSA_SMITH_TYPE_H

#include "Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include <typeinfo>

/* Constant definitions */
namespace tosas {

// Tensor dimension [1, 3]
const unsigned tensorDimMax =  3;
const unsigned tensorDimMin =  1;

const unsigned DEFAULT_TENSOR_RANK = 3;

// Tensor length [1, 32]
const unsigned tensorLenMax = 32;
const unsigned tensorLenMin =  1;

const unsigned funcArgNumMax = 3; // [0, 3]
const unsigned funcRetNumMax = 1; // [0, 1]

const llvm::SmallVector<unsigned> SUPPORTED_INT_WIDTH   = {32};
const llvm::SmallVector<unsigned> SUPPORTED_FLOAT_WIDTH = {32};

using ClampRegion = std::tuple<int64_t, int64_t, double, double>;
using NumRange = std::pair<unsigned, unsigned>;

enum class CtrlFlowTy {
    TosaIf,
    TosaWhile,
    None
};

} // namespace tosas

/* Type checking helpers */
namespace tosas {

template<typename T>
bool isElemOfTy(mlir::Value &val) {
    mlir::TensorType tensorTy = mlir::cast<mlir::TensorType>(val.getType());
    return mlir::isa<T>(tensorTy.getElementType());
}

bool isElemOfWidth(mlir::Value &val, unsigned width) {
    mlir::TensorType tensorTy = mlir::cast<mlir::TensorType>(val.getType());
    return tensorTy.getElementTypeBitWidth() == width;
}

template<typename T>
bool isInteger() { return false; }

template<>
bool isInteger<mlir::IntegerType>() { return true; }

template<typename T>
bool isFloat() { return false; }

template<>
bool isFloat<mlir::FloatType>() { return true; }

template<>
bool isFloat<mlir::Float32Type>() { return true; }

template<>
bool isFloat<mlir::Float64Type>() { return true; }

} // namespace tosas

#endif