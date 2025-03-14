#ifndef TOSA_SMITH_OPERATION_H
#define TOSA_SMITH_OPERATION_H

#include "Utils.h"
#include "Context.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

/* TOP DECLARATIONS */
namespace tosas {

static const llvm::cl::opt<bool> enableCtrlFlow{
    "enable-control-flow",
    llvm::cl::desc("Enable control flow constructs like conditional if and while loop"),
    llvm::cl::init(false)
};

static const llvm::cl::opt<unsigned> maxRegionOpNum{
    "max-region-ops-num",
    llvm::cl::desc("The maximum number of operations in a nested region [default: 5]"),
    llvm::cl::init(5)
};

static const llvm::cl::opt<unsigned> maxNestDepth{
    "max-nest-depth",
    llvm::cl::desc("The maximum depth of the nested regions [default: 2]"),
    llvm::cl::init(2)
};

static const llvm::cl::opt<unsigned> maxLoopIterTimes{
    "max-loop-iter-times",
    llvm::cl::desc("The maximum iteration times of a while loop [default: 5]"),
    llvm::cl::init(5)
};

using CmptGen = std::function<mlir::Operation *(mlir::OpBuilder &, Context &)>;
using CtrlGen = std::function<mlir::Operation *(mlir::OpBuilder &, Context &, unsigned)>;
using CallGen = std::function<mlir::Operation *(mlir::OpBuilder &, Context &)>;

}

/* CLASS DECLARATIONS */
namespace tosas {

class Operation {
public:
    Operation() {
        initTensorOpGens();             // 1
        initActivationFuncGens();       // 2
        initElementwiseBinOpGens();     // 3
        initElementwiseUnaryOpGens();   // 4
        initElementwiseTriOpGens();     // 5
        initCmpOpGens();                // 6
        initReductionOpGens();          // 7
        initDataLayoutOpGens();         // 8
        initCtrlGens();                 // 11
        funcCallGen = buildFuncCallOp();
    }

    inline CmptGen randomCmptGen() { return random(cmptGenList); }
    inline CtrlGen randomCtrlGen() { return random(ctrlGenList); }
    inline CtrlGen &getTosaIfGen()    { return ctrlGenList[0]; }
    inline CtrlGen &getTosaWhileGen() { return ctrlGenList[1]; }
    inline CallGen &getFuncCallGen()  { return funcCallGen; }

    // Return the returned operation, i.e., the argument of the func.return
    mlir::Operation *buildFuncReturnOp(
        mlir::OpBuilder &builder,
        Context &parentCtx,
        bool isVoid = false
    ) {
        if (isVoid) {
            builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
            return nullptr;
        } else {
            mlir::Operation *opToRet = parentCtx.randomVal().getDefiningOp();
            builder.create<mlir::func::ReturnOp>(
                builder.getUnknownLoc(), opToRet->getResults() 
            );
            return opToRet;
        }
    }

    // Main procedure for building operations in a region
    void buildOps(mlir::OpBuilder &builder, Context &parentCtx,
                  NumRange range, unsigned nestDepth, CtrlFlowTy cfTy = CtrlFlowTy::None)
    {
        unsigned numOps = random(range.second, range.first);

        for (unsigned i = 0; i < numOps; i++) {
            mlir::Operation *retOp = nullptr;
            if (parentCtx.hasDefinedFunc() && random(99, 0) < 10) {
                retOp = getFuncCallGen()(builder, parentCtx);
            } else {
                if (enableCtrlFlow && nestDepth < maxNestDepth && random(99, 0) < 10) {
                    switch (cfTy) {
                        case CtrlFlowTy::TosaIf:
                            retOp = getTosaWhileGen()(builder, parentCtx, nestDepth);
                            break;
                        case CtrlFlowTy::TosaWhile:
                            retOp = getTosaIfGen()(builder, parentCtx, nestDepth);
                            break;
                        default:
                            retOp = randomCtrlGen()(builder, parentCtx, nestDepth);
                            break;
                    }
                } else {
                    retOp = randomCmptGen()(builder, parentCtx);
                }
            }
            if (retOp == nullptr)
                i = i - 1;
        }
    }

    void buildValPrint(mlir::OpBuilder &builder, Context &parentCtx) {
        mlir::Value valToPrint = parentCtx.randomVal();
        auto oldTy = mlir::dyn_cast<mlir::RankedTensorType>(valToPrint.getType());
        auto newTy = mlir::UnrankedTensorType::get(oldTy.getElementType());
        mlir::Operation *tensorCastOp = builder.create<mlir::tensor::CastOp>(
            builder.getUnknownLoc(),
            newTy,
            valToPrint
        ).getOperation();

        // Then call the print helper function in runner utils
        std::string funcName;
        if (oldTy.getElementType().isInteger()) {
            funcName = "printMemrefI" + std::to_string(oldTy.getElementTypeBitWidth());
        } else {
            funcName = "printMemrefF" + std::to_string(oldTy.getElementTypeBitWidth());
        }
        builder.create<mlir::func::CallOp>(
            builder.getUnknownLoc(),
            funcName,
            mlir::TypeRange(),
            tensorCastOp->getResults()
        );
    }

    mlir::Operation *buildZeroTensor(mlir::OpBuilder &builder,
                                     mlir::TensorType tensorTy)
    {
        return buildConstTensor(builder, EMPTY_CONTEXT, 0, tensorTy, false);
    }

    mlir::Operation *buildOneTensor(mlir::OpBuilder &builder,
                                    mlir::TensorType tensorTy)
    {
        return buildConstTensor(builder, EMPTY_CONTEXT, 1, tensorTy, false);
    }

    mlir::Operation *buildRandomTensor(mlir::OpBuilder &builder,
                                       Context &parentCtx,
                                       mlir::TensorType tensorTy)
    {
        return buildConstTensor(builder, parentCtx, randomInteger(), tensorTy);
    }

    mlir::Operation *buildConstTensor(
        mlir::OpBuilder &builder,
        Context &parentCtx,
        unsigned value,
        mlir::TensorType tensorTy,
        bool addToPool = true
    ) {
        mlir::Attribute attr;
        if (tensorTy.getElementType().isInteger()) {
            attr = mlir::IntegerAttr::get(tensorTy.getElementType(), value);
        } else {
            attr = mlir::FloatAttr::get(tensorTy.getElementType(), value);
        }

        auto denseAttr = mlir::DenseElementsAttr::get(tensorTy, attr);
        mlir::Operation *constOp = builder.create<mlir::tosa::ConstOp>(
            builder.getUnknownLoc(),
            denseAttr.getType(),
            denseAttr
        ).getOperation();

        if (addToPool)
            parentCtx.addConstVals(constOp);

        return constOp;
    }

    mlir::Operation *buildIndexConst(mlir::OpBuilder &builder, int64_t attrVal) {
        return builder.create<mlir::index::ConstantOp>(
            builder.getUnknownLoc(),
            builder.getIndexAttr(attrVal)
        ).getOperation();
    }

    mlir::Operation *castToFloatByTosaCastOp(mlir::OpBuilder &builder,
                                             mlir::Value &val)
    {
        mlir::TensorType oldTy = mlir::cast<mlir::TensorType>(val.getType());
        mlir::TensorType newTy = mlir::RankedTensorType::get(
            oldTy.getShape(),
            oldTy.getElementTypeBitWidth() == 32 
                ? builder.getF32Type()
                : builder.getF64Type()
        );
        return builder.create<mlir::tosa::CastOp>(
            builder.getUnknownLoc(),
            newTy,
            val
        ).getOperation();
    }

    mlir::Value extractScalarTensor(mlir::OpBuilder &builder,
                                    mlir::Value &val,
                                    llvm::ArrayRef<mlir::Value> range)
    {
        mlir::Operation *extractOp = builder.create<mlir::tensor::ExtractOp>(
            builder.getUnknownLoc(),
            val,
            range
        ).getOperation();
        val = extractOp->getResult(0);

        // Build tensor<i1>
        mlir::Operation *fromOp = builder.create<mlir::tensor::FromElementsOp>(
            builder.getUnknownLoc(),
            mlir::RankedTensorType::get({}, val.getType()),
            mlir::ValueRange(val)
        ).getOperation();
        val = fromOp->getResult(0);

        return val;
    }

    mlir::Operation *clampValueByTosaClampOp(mlir::OpBuilder &builder,
                                             mlir::Value &val,
                                             FClampRegion region)
    {
        mlir::FloatAttr minAttr =
            builder.getF32FloatAttr(std::get<0>(region));
        mlir::FloatAttr maxAttr =
            builder.getF32FloatAttr(std::get<1>(region));
    
        return builder.create<mlir::tosa::ClampOp>(
            builder.getUnknownLoc(),
            val.getType(),
            val,
            minAttr,
            maxAttr
        ).getOperation();
    }

    mlir::Operation *clampValueByTosaClampOp(mlir::OpBuilder &builder,
                                             mlir::Value &val,
                                             IClampRegion region)
    {
        mlir::IntegerAttr minAttr =
            builder.getI64IntegerAttr(std::get<0>(region));
        mlir::IntegerAttr maxAttr =
            builder.getI64IntegerAttr(std::get<1>(region));

        return builder.create<mlir::tosa::ClampOp>(
            builder.getUnknownLoc(),
            val.getType(),
            val,
            minAttr,
            maxAttr
        ).getOperation();
    }

private:
    llvm::SmallVector<CmptGen> cmptGenList; // Store all the computation operation generator
    llvm::SmallVector<CtrlGen> ctrlGenList; // Store all the control flow operation generator
    CallGen funcCallGen; // Generator to build function calls

    template<typename O, typename T = mlir::Type, unsigned W = 32>
    CmptGen buildTosaBinaryOp() {
        return [this](mlir::OpBuilder &builder, Context &parentCtx) {
            auto [operand1, operand2] = parentCtx.randomValPair<T>(W);
            if (operand1 == nullptr || operand2 == nullptr)
                return (mlir::Operation *) nullptr;

            // If the operation targets float numbers only, we cast the randomly selected
            // integer tensor to the float one 
            if (isFloat<T>() && isElemOfTy<mlir::IntegerType>(operand1)) {
                mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand1);
                parentCtx.addResultVals(castOp);
                operand1 = castOp->getResult(0);
            }
            if (isFloat<T>() && isElemOfTy<mlir::IntegerType>(operand2)) {
                mlir::Operation *castOp = castToFloatByTosaCastOp(builder, operand2);
                parentCtx.addResultVals(castOp);
                operand2 = castOp->getResult(0);
            }

            // If the two oepration has different types, convert the integer one to float
            if (operand1.getType() != operand2.getType()) {
                if (isElemOfTy<mlir::IntegerType>(operand1)) {
                    mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand1);
                    parentCtx.addResultVals(castOp);
                    operand1 = castOp->getResult(0);
                }
                if (isElemOfTy<mlir::IntegerType>(operand2)) {
                    mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand2);
                    parentCtx.addResultVals(castOp);
                    operand2 = castOp->getResult(0);
                }
            }

            // Reconditioning for tosa.add
            if (std::is_same<O, mlir::tosa::AddOp>::value
                || std::is_same<O, mlir::tosa::SubOp>::value)
            {
                
                if (std::is_same<T, mlir::IntegerType>::value) {
                    IClampRegion region{ INT32_MIN / 2, INT32_MAX / 2 };
                    operand1 = this
                        ->clampValueByTosaClampOp(builder, operand1, region)
                        ->getResult(0);
    
                    operand2 = this
                        ->clampValueByTosaClampOp(builder, operand2, region)
                        ->getResult(0);
                }

                if (std::is_same<T, mlir::FloatType>::value) {
                    FClampRegion region{ (double) INT32_MIN / 2, (double) INT32_MAX / 2 };
                    operand1 = this
                        ->clampValueByTosaClampOp(builder, operand1, region)
                        ->getResult(0);
    
                    operand2 = this
                        ->clampValueByTosaClampOp(builder, operand2, region)
                        ->getResult(0);
                }
            }

            // Recondition inputs for tosa.int_div
            if (std::is_same<O, mlir::tosa::IntDivOp>::value) {
                mlir::TensorType oldTy = mlir::cast<mlir::TensorType>(operand2.getType());
                mlir::Value zeroVal = buildZeroTensor(builder, oldTy)->getResult(0);
                mlir::Value oneVal = buildOneTensor(builder, oldTy)->getResult(0);

                mlir::Type boolTy = mlir::RankedTensorType::get(
                    oldTy.getShape(),
                    builder.getIntegerType(1)
                );
                mlir::Value equalVal = builder.create<mlir::tosa::EqualOp>(
                    builder.getUnknownLoc(),
                    boolTy,
                    zeroVal,
                    operand2
                ).getOperation()->getResult(0);

                operand2 = builder.create<mlir::tosa::SelectOp>(
                    builder.getUnknownLoc(),
                    operand2.getType(),
                    equalVal,
                    oneVal,
                    operand2
                ).getOperation()->getResult(0);
            }

            if (std::is_same<O, mlir::tosa::LogicalLeftShiftOp>::value) {
                mlir::Operation *clzOp = builder.create<mlir::tosa::ClzOp>(
                    builder.getUnknownLoc(),
                    operand1.getType(),
                    operand1
                ).getOperation();

                mlir::Operation *absOp = builder.create<mlir::tosa::AbsOp>(
                    builder.getUnknownLoc(),
                    operand2.getType(),
                    operand2
                ).getOperation();
                operand2 = absOp->getResult(0);

                mlir::Operation *minOp = builder.create<mlir::tosa::MinimumOp>(
                    builder.getUnknownLoc(),
                    operand2.getType(),
                    operand2,
                    clzOp->getResult(0)
                ).getOperation();
                operand2 = minOp->getResult(0);
            }

            if (std::is_same<O, mlir::tosa::LogicalRightShiftOp>::value) {
                mlir::Operation *absOp = builder.create<mlir::tosa::AbsOp>(
                    builder.getUnknownLoc(),
                    operand2.getType(),
                    operand2
                ).getOperation();
                operand2 = absOp->getResult(0);

                if (std::is_same<T, mlir::IntegerType>::value) {
                    mlir::Operation *clampOp = this->clampValueByTosaClampOp(
                        builder, operand2, 
                        IClampRegion{0, 16}
                    );
                    operand2 = clampOp->getResult(0);
                } else {
                    mlir::Operation *clampOp = this->clampValueByTosaClampOp(
                        builder, operand2, 
                        FClampRegion{0, 16}
                    );
                    operand2 = clampOp->getResult(0);
                }
            }

            if (std::is_same<O, mlir::tosa::PowOp>::value) {
                // TODO:
            }

            mlir::Operation *newOp = builder.create<O>(
                builder.getUnknownLoc(),
                operand1.getType(),
                operand1,
                operand2
            ).getOperation();

            parentCtx.addResultVals(newOp);
            return newOp;
        };
    }

    template<typename O, typename T = mlir::Type, unsigned W = 32>
    CmptGen buildTosaUnaryOp() {
        return [this](mlir::OpBuilder &builder, Context &parentCtx) {
            mlir::Value operand = parentCtx.randomVal<T>(W);
            if (operand == nullptr) return (mlir::Operation *) nullptr;

            // If the target type is float, but the random value is integer,
            // cast it to the type float
            if (isFloat<T>() && isElemOfTy<mlir::IntegerType>(operand)) {
                mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand);
                parentCtx.addResultVals(castOp);
                operand = castOp->getResult(0);
            }

            // Value clamping for tosa.rsqrt and tosa.log
            if (std::is_same<O, mlir::tosa::RsqrtOp>::value
                || std::is_same<O, mlir::tosa::LogOp>::value
            ) {
                mlir::Operation *absOp = builder.create<mlir::tosa::AbsOp>(
                    builder.getUnknownLoc(),
                    operand.getType(),
                    operand
                ).getOperation();
                operand = absOp->getResult(0);

                if (std::is_same<T, mlir::IntegerType>::value) {
                    mlir::Operation *clampOp = this->clampValueByTosaClampOp(
                        builder, operand,
                        IClampRegion{1, INT32_MAX}
                    );
                    operand = clampOp->getResult(0);
                } else {
                    mlir::Operation *clampOp = this->clampValueByTosaClampOp(
                        builder, operand,
                        FClampRegion{(double) 0.1, (double) INT32_MAX}
                    );
                    operand = clampOp->getResult(0);
                }
            }

            if (std::is_same<O, mlir::tosa::ReciprocalOp>::value) {
                mlir::TensorType operandTy = mlir::cast<mlir::TensorType>(operand.getType());
                mlir::Value zeroVal = buildZeroTensor(builder, operandTy)->getResult(0);
                mlir::Value oneVal = buildOneTensor(builder, operandTy)->getResult(0);

                mlir::Type boolTy = mlir::RankedTensorType::get(
                    operandTy.getShape(),
                    builder.getIntegerType(1)
                );
                mlir::Value equalVal = builder.create<mlir::tosa::EqualOp>(
                    builder.getUnknownLoc(),
                    boolTy,
                    zeroVal,
                    operand
                ).getOperation()->getResult(0);

                operand = builder.create<mlir::tosa::SelectOp>(
                    builder.getUnknownLoc(),
                    operand.getType(),
                    equalVal,
                    oneVal,
                    operand
                ).getOperation()->getResult(0);
            }

            // Value clamping for tosa.exp [e^x < i32_max]
            if (std::is_same<O, mlir::tosa::ExpOp>::value) {
                if (std::is_same<T, mlir::IntegerType>::value) {
                    operand = this->clampValueByTosaClampOp(
                        builder, operand,
                        IClampRegion{INT32_MIN, log(INT32_MAX)}
                    )->getResult(0);
                } else {
                    operand = this->clampValueByTosaClampOp(
                        builder, operand,
                        FClampRegion{INT32_MIN, log(INT32_MAX)}
                    )->getResult(0);
                }
            }

            mlir::Operation *newOp = builder.create<O>(
                builder.getUnknownLoc(),
                operand.getType(),
                operand
            ).getOperation();

            parentCtx.addResultVals(newOp);
            return newOp;
        };
    }

//===----------------------------------------------------------------------===//
// 01. Tensor Operators
//===----------------------------------------------------------------------===//

    CmptGen buildTosaArgMaxOp();
    // CmptGen buildTosaMatMulOp();

    void initTensorOpGens() {
        cmptGenList.push_back(buildTosaArgMaxOp());
        // cmptGenList.push_back(buildTosaMatMulOp());
    }

//===----------------------------------------------------------------------===//
// 02. Activation Functions
//===----------------------------------------------------------------------===//

    CmptGen buildTosaErfOp()     { return buildTosaUnaryOp<mlir::tosa::ErfOp,     mlir::FloatType>(); }
    CmptGen buildTosaTanhOp()    { return buildTosaUnaryOp<mlir::tosa::TanhOp,    mlir::FloatType>(); }
    CmptGen buildTosaSigmoidOp() { return buildTosaUnaryOp<mlir::tosa::SigmoidOp, mlir::FloatType>(); }

    void initActivationFuncGens() {
        cmptGenList.push_back(buildTosaErfOp());
        cmptGenList.push_back(buildTosaTanhOp());
        cmptGenList.push_back(buildTosaSigmoidOp());
    }

//===----------------------------------------------------------------------===//
// 03. Elementwise Binary Operators
//===----------------------------------------------------------------------===//

    CmptGen buildTosaAddOp()    { return buildTosaBinaryOp<mlir::tosa::AddOp>(); }
    CmptGen buildTosaSubOp()    { return buildTosaBinaryOp<mlir::tosa::SubOp>(); }
    // CmptGen buildTosaMulOp();
    CmptGen buildTosaIntDivOp() { return buildTosaBinaryOp<mlir::tosa::IntDivOp, mlir::IntegerType>(); }
    CmptGen buildTosaPowOp()    { return buildTosaBinaryOp<mlir::tosa::PowOp,    mlir::FloatType>(); }

    CmptGen buildTosaBitwiseAndOp() { return buildTosaBinaryOp<mlir::tosa::BitwiseAndOp, mlir::IntegerType>(); }
    CmptGen buildTosaBitwiseOrOp()  { return buildTosaBinaryOp<mlir::tosa::BitwiseOrOp,  mlir::IntegerType>(); }
    CmptGen buildTosaBitwiseXorOp() { return buildTosaBinaryOp<mlir::tosa::BitwiseXorOp, mlir::IntegerType>(); }

    CmptGen buildTosaLogicalAndOp() { return buildTosaBinaryOp<mlir::tosa::LogicalAndOp, mlir::IntegerType, 1>();}
    CmptGen buildTosaLogicalOrOp()  { return buildTosaBinaryOp<mlir::tosa::LogicalOrOp,  mlir::IntegerType, 1>();}
    CmptGen buildTosaLogicalXorOp() { return buildTosaBinaryOp<mlir::tosa::LogicalXorOp, mlir::IntegerType, 1>();}

    CmptGen buildTosaLogicalLeftShiftOp()  { return buildTosaBinaryOp<mlir::tosa::LogicalLeftShiftOp,  mlir::IntegerType>(); }
    CmptGen buildTosaLogicalRightShiftOp() { return buildTosaBinaryOp<mlir::tosa::LogicalRightShiftOp, mlir::IntegerType>(); }
    CmptGen buildTosaArithmeticRightShiftOp();

    CmptGen buildTosaMaximumOp() { return buildTosaBinaryOp<mlir::tosa::MaximumOp>(); }
    CmptGen buildTosaMinimumOP() { return buildTosaBinaryOp<mlir::tosa::MinimumOp>(); }

    void initElementwiseBinOpGens() {
        cmptGenList.push_back(buildTosaAddOp());
        cmptGenList.push_back(buildTosaSubOp());
        //// cmptGenList.push_back(buildTosaMulOp());
        cmptGenList.push_back(buildTosaIntDivOp());
        cmptGenList.push_back(buildTosaPowOp());

        cmptGenList.push_back(buildTosaBitwiseAndOp());
        cmptGenList.push_back(buildTosaBitwiseOrOp());
        cmptGenList.push_back(buildTosaBitwiseXorOp());

        cmptGenList.push_back(buildTosaLogicalAndOp());
        cmptGenList.push_back(buildTosaLogicalOrOp());
        cmptGenList.push_back(buildTosaLogicalXorOp());

        cmptGenList.push_back(buildTosaLogicalLeftShiftOp());
        cmptGenList.push_back(buildTosaLogicalRightShiftOp());
        //// cmptGenList.push_back(buildTosaArithmeticRightShiftOp());

        cmptGenList.push_back(buildTosaMaximumOp());
        cmptGenList.push_back(buildTosaMinimumOP());
    }

//===----------------------------------------------------------------------===//
// 04. Elementwise Unary Operators
//===----------------------------------------------------------------------===//

    CmptGen buildTosaAbsOp()    { return buildTosaUnaryOp<mlir::tosa::AbsOp>(); }
    CmptGen buildTosaNegateOp() { return buildTosaUnaryOp<mlir::tosa::NegateOp>(); }

    CmptGen buildTosaClzOp()   { return buildTosaUnaryOp<mlir::tosa::ClzOp,   mlir::IntegerType>(); }
    CmptGen buildTosaSinOp()   { return buildTosaUnaryOp<mlir::tosa::SinOp,   mlir::FloatType>(); }
    CmptGen buildTosaCosOp()   { return buildTosaUnaryOp<mlir::tosa::CosOp,   mlir::FloatType>(); }
    CmptGen buildTosaExpOp()   { return buildTosaUnaryOp<mlir::tosa::ExpOp,   mlir::FloatType>(); }
    CmptGen buildTosaLogOp()   { return buildTosaUnaryOp<mlir::tosa::LogOp,   mlir::FloatType>(); }
    CmptGen buildTosaCeilOp()  { return buildTosaUnaryOp<mlir::tosa::CeilOp,  mlir::FloatType>(); }
    CmptGen buildTosaFloorOp() { return buildTosaUnaryOp<mlir::tosa::FloorOp, mlir::FloatType>(); }
    CmptGen buildTosaRsqrtOp() { return buildTosaUnaryOp<mlir::tosa::RsqrtOp, mlir::FloatType>(); }
    CmptGen buildTosaReciprocalOp() { return buildTosaUnaryOp<mlir::tosa::ReciprocalOp, mlir::FloatType>(); }

    CmptGen buildTosaBitwiseNotOp() { return buildTosaUnaryOp<mlir::tosa::BitwiseNotOp, mlir::IntegerType>(); }
    CmptGen buildTosaLogicalNotOp() { return buildTosaUnaryOp<mlir::tosa::LogicalNotOp, mlir::IntegerType, 1>(); }

    void initElementwiseUnaryOpGens() {
        cmptGenList.push_back(buildTosaAbsOp());
        cmptGenList.push_back(buildTosaNegateOp());
        cmptGenList.push_back(buildTosaClzOp());
        cmptGenList.push_back(buildTosaSinOp());
        cmptGenList.push_back(buildTosaCosOp());
        cmptGenList.push_back(buildTosaExpOp());
        cmptGenList.push_back(buildTosaLogOp());
        cmptGenList.push_back(buildTosaCeilOp());
        cmptGenList.push_back(buildTosaFloorOp());
        cmptGenList.push_back(buildTosaRsqrtOp());
        cmptGenList.push_back(buildTosaReciprocalOp());
        cmptGenList.push_back(buildTosaBitwiseNotOp());
        cmptGenList.push_back(buildTosaLogicalNotOp());
    }

//===----------------------------------------------------------------------===//
// 05. Elementwise Ternary Operators
//===----------------------------------------------------------------------===//

    CmptGen buildTosaSelectOp();

    void initElementwiseTriOpGens() {
        cmptGenList.push_back(buildTosaSelectOp());
    }

//===----------------------------------------------------------------------===//
// 06. Comparison Operators
//===----------------------------------------------------------------------===//

    template<typename O>
    CmptGen buildTosaCmpOp() {
        return [this](mlir::OpBuilder &builder, Context &parentCtx) {
            auto [operand1, operand2] = parentCtx.randomValPair();

            // If the two oepration has different types, convert the integer one to float
            if (operand1.getType() != operand2.getType()) {
                if (isElemOfTy<mlir::IntegerType>(operand1)) {
                    mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand1);
                    parentCtx.addResultVals(castOp);
                    operand1 = castOp->getResult(0);
                }
                if (isElemOfTy<mlir::IntegerType>(operand2)) {
                    mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand2);
                    parentCtx.addResultVals(castOp);
                    operand2 = castOp->getResult(0);
                }
            }

            // The comparison computation results tensors with i1 
            mlir::RankedTensorType inTy = mlir::cast<mlir::RankedTensorType>(operand1.getType());
            mlir::RankedTensorType retTy = mlir::RankedTensorType::get(
                inTy.getShape(),
                builder.getIntegerType(1)
            );

            mlir::Operation *newOp = builder.create<O>(
                builder.getUnknownLoc(),
                retTy,
                operand1,
                operand2
            ).getOperation();

            parentCtx.addResultVals(newOp);
            return newOp;
        };
    }

    CmptGen buildTosaEqualOp()        { return buildTosaCmpOp<mlir::tosa::EqualOp>(); }
    CmptGen buildTosaGreaterOp()      { return buildTosaCmpOp<mlir::tosa::GreaterOp>(); }
    CmptGen buildTosaGreaterEqualOp() { return buildTosaCmpOp<mlir::tosa::GreaterEqualOp>(); }

    void initCmpOpGens() {
        cmptGenList.push_back(buildTosaEqualOp());
        cmptGenList.push_back(buildTosaGreaterOp());
        cmptGenList.push_back(buildTosaGreaterEqualOp());
    }

//===----------------------------------------------------------------------===//
// 07. Reduction Operators
//===----------------------------------------------------------------------===//

    template<typename O, typename T = mlir::Type, unsigned W = 32>
    CmptGen buildTosaReduceOp() {
        return [](mlir::OpBuilder &builder, Context &parentCtx) {
            mlir::Value operand = parentCtx.randomVal<T>(W);
            if (operand == nullptr)
                return (mlir::Operation *) nullptr;
            
            mlir::TensorType oldTy = mlir::cast<mlir::RankedTensorType>(operand.getType());
            if (oldTy.getRank() == 0) {
                // Avoid reduce tensor<i1/i32>
                return (mlir::Operation *) nullptr;
            }

            llvm::SmallVector<int64_t> shape(oldTy.getShape());
            unsigned axisIdx = random(shape.size() - 1); // Reduce on this axis
            shape[axisIdx] = 1;
            mlir::IntegerAttr axisAttr = builder.getI32IntegerAttr(axisIdx);

            mlir::TensorType newTy = mlir::RankedTensorType::get(
                shape,
                oldTy.getElementType()
            );

            mlir::Operation *newOp = builder.create<O>(
                builder.getUnknownLoc(),
                newTy,
                operand,
                axisAttr
            ).getOperation();

            parentCtx.addResultVals(newOp);
            return newOp;
        };
    }

    CmptGen buildTosaReduceSumOp()  { return buildTosaReduceOp<mlir::tosa::ReduceSumOp>(); }
    CmptGen buildTosaReduceProdOp() { return buildTosaReduceOp<mlir::tosa::ReduceProductOp>(); }
    CmptGen buildTosaReduceMaxOp()  { return buildTosaReduceOp<mlir::tosa::ReduceMaxOp>(); }
    CmptGen buildTosaReduceMinOp()  { return buildTosaReduceOp<mlir::tosa::ReduceMinOp>(); }
    CmptGen buildTosaReduceAllOp()  { return buildTosaReduceOp<mlir::tosa::ReduceAllOp, mlir::IntegerType, 1>(); }
    CmptGen buildTosaReduceAnyOp()  { return buildTosaReduceOp<mlir::tosa::ReduceAnyOp, mlir::IntegerType, 1>(); }

    void initReductionOpGens() {
        cmptGenList.push_back(buildTosaReduceSumOp());
        cmptGenList.push_back(buildTosaReduceProdOp());
        cmptGenList.push_back(buildTosaReduceMaxOp());
        cmptGenList.push_back(buildTosaReduceMinOp());
        cmptGenList.push_back(buildTosaReduceAllOp());
        cmptGenList.push_back(buildTosaReduceAnyOp());
    }

//===----------------------------------------------------------------------===//
// 08. Data Layout Operators
//===----------------------------------------------------------------------===//

    // CmptGen buildTosaReshapeOp();
    CmptGen buildTosaReverseOp();
    // CmptGen buildTosaSliceOp();
    // CmptGen buildTosaTileOp();

    void initDataLayoutOpGens() {
        // cmptGenList.push_back(buildTosaReshapeOp());
        cmptGenList.push_back(buildTosaReverseOp());
        // cmptGenList.push_back(buildTosaSliceOp());
        // cmptGenList.push_back(buildTosaTileOp());
    }

//===----------------------------------------------------------------------===//
// 09. Scatter/Gather Operators
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 10. Image Operators
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  11. Control Flow Operators
//===----------------------------------------------------------------------===//

    CtrlGen buildTosaIfOp();
    CtrlGen buildTosaWhileOp();

    void initCtrlGens() {
        ctrlGenList.push_back(buildTosaIfOp());
        ctrlGenList.push_back(buildTosaWhileOp());
    }

//===----------------------------------------------------------------------===//
//  Other Operators
//===----------------------------------------------------------------------===//

    CallGen buildFuncCallOp();
};

} // namespace tosas

namespace tosas {

// CmptGen Operation::buildTosaReshapeOp() {
//     return [](mlir::OpBuilder &builder, Context &parentCtx) {
//         mlir::Value operand = parentCtx.randomVal();
//         auto operandTy = mlir::cast<mlir::TensorType>(operand.getType());

//         int64_t totalElemCount = 1;
//         for (unsigned i = 0; i < operandTy.getRank(); i++)
//             totalElemCount *= operandTy.getShape()[i];
        
//         llvm::SmallVector<int64_t> factors;
//         auto randomFactor = [&factors](int64_t num) -> int64_t {
//             factors.clear();

//             for (int64_t i = 1; i < sqrt((double) num); i++) {
//                 if (num % i == 0)
//                     factors.push_back(i);
//             }

//             return (factors.size() == 1 && factors[0] == 1) ? num : random(factors);
//         };

//         llvm::SmallVector<int64_t> newShape;
//         while (totalElemCount != 1) {
//             int64_t curElemCount = randomFactor(totalElemCount);
//             newShape.push_back(curElemCount);
//             totalElemCount /= curElemCount;
//         }

//         // The rank of the tensor is 0
//         if (newShape.empty()) {
//             newShape.push_back(1);
//         }

//         mlir::Operation *newOp = builder.create<mlir::tosa::ReshapeOp>(
//             builder.getUnknownLoc(),
//             mlir::RankedTensorType::get(
//                 newShape,
//                 operandTy.getElementType()
//             ),
//             operand,
//             builder.getDenseI64ArrayAttr(newShape)
//         ).getOperation();

//         parentCtx.addResultVals(newOp);
//         return newOp;
//     };
// }

CmptGen Operation::buildTosaReverseOp() {
    return [](mlir::OpBuilder &builder, Context &parentCtx) {
        mlir::Value operand = parentCtx.randomVal();
        mlir::TensorType operandTy = mlir::cast<mlir::TensorType>(operand.getType());
        mlir::IntegerAttr indexAttr = builder.getI32IntegerAttr(
            random(operandTy.getRank() - 1)
        );

        mlir::Operation *newOp = builder.create<mlir::tosa::ReverseOp>(
            builder.getUnknownLoc(),
            operandTy,
            operand,
            indexAttr
        ).getOperation();

        parentCtx.addResultVals(newOp);
        return newOp;
    };
}

// CmptGen Operation::buildTosaSliceOp() {
//     return [](mlir::OpBuilder &builder, Context &parentCtx) {
//         mlir::Value operand = parentCtx.randomVal();
//         auto operandTy = mlir::cast<mlir::TensorType>(operand.getType());

//         llvm::SmallVector<int64_t> startVals, sizeVals;
//         for (unsigned i = 0; i < operandTy.getRank(); i++) {
//             int64_t start = random(operandTy.getShape()[i] - 1);
//             int64_t size = random(operandTy.getShape()[i] - start, 1);

//             startVals.push_back(start);
//             sizeVals.push_back(size);
//         }

//         mlir::Operation *newOp = builder.create<mlir::tosa::SliceOp>(
//             builder.getUnknownLoc(),
//             mlir::RankedTensorType::get(
//                 sizeVals,
//                 operandTy.getElementType()
//             ),
//             operand,
//             builder.getDenseI64ArrayAttr(startVals),
//             builder.getDenseI64ArrayAttr(sizeVals)
//         ).getOperation();

//         parentCtx.addResultVals(newOp);
//         return newOp;
//     };
// }

// CmptGen Operation::buildTosaTileOp() {
//     return [](mlir::OpBuilder &builder, Context &parentCtx) {
//         mlir::Value operand = parentCtx.randomVal();
//         auto operandTy = mlir::cast<mlir::TensorType>(operand.getType());
        
//         llvm::SmallVector<int64_t> mulVals, newShape;
//         for (unsigned i = 0; i < operandTy.getRank(); i++) {
//             int64_t multiple = random(5, 1);
//             mulVals.push_back(multiple);
//             newShape.push_back(multiple * operandTy.getShape()[i]);
//         }
//         mlir::DenseI64ArrayAttr multiples = builder.getDenseI64ArrayAttr(mulVals);
//         auto newTy = mlir::RankedTensorType::get(newShape, operandTy.getElementType());

//         mlir::Operation *newOp = builder.create<mlir::tosa::TileOp>(
//             builder.getUnknownLoc(),
//             newTy,
//             operand,
//             multiples
//         ).getOperation();

//         parentCtx.addResultVals(newOp);
//         return newOp;
//     };
// }

// CmptGen Operation::buildTosaMulOp() {
//     return [this](mlir::OpBuilder &builder, Context &parentCtx) {
//         auto [operand1, operand2] = parentCtx.randomValPair();

//         // If the two oepration has different types, convert the integer one to float
//         if (operand1.getType() != operand2.getType()) {
//             if (isElemOfTy<mlir::IntegerType>(operand1)) {
//                 mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand1);
//                 parentCtx.addResultVals(castOp);
//                 operand1 = castOp->getResult(0);
//             }
//             if (isElemOfTy<mlir::IntegerType>(operand2)) {
//                 mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand2);
//                 parentCtx.addResultVals(castOp);
//                 operand2 = castOp->getResult(0);
//             }
//         }

//         // Value clamping
//         ClampRegion region = { - (1 << 16), 1 << 16, (double) - (1 << 16), (double) (1 << 16)};
//         operand1 = this->clampValueByTosaClampOp(builder, operand1, region)
//             ->getResult(0);
//         operand2 = this->clampValueByTosaClampOp(builder, operand2, region)
//             ->getResult(0);

//         mlir::Operation *newOp = builder.create<mlir::tosa::MulOp>(
//             builder.getUnknownLoc(),
//             operand1.getType(),
//             operand1,
//             operand2,
//             // 'tosa.mul' op require shift to be 0 for float type
//             operand1.getType().isInteger() ? random(4) : 0
//         ).getOperation();

//         parentCtx.addResultVals(newOp);
//         return newOp;
//     };
// }

CmptGen Operation::buildTosaArithmeticRightShiftOp() {
    return [this](mlir::OpBuilder &builder, Context &parentCtx) {
        auto [operand1, operand2] = parentCtx.randomValPair<mlir::IntegerType>();
        
        mlir::Operation *absOp = builder.create<mlir::tosa::AbsOp>(
            builder.getUnknownLoc(),
            operand2.getType(),
            operand2
        ).getOperation();
        operand2 = absOp->getResult(0);

        mlir::Operation *clampOp = this->clampValueByTosaClampOp(
            builder, operand2, 
            IClampRegion{0, 16}
        );
        operand2 = clampOp->getResult(0);

        mlir::Operation *newOp =
            builder.create<mlir::tosa::ArithmeticRightShiftOp>(
                builder.getUnknownLoc(),
                operand1.getType(),
                operand1,
                operand2,
                builder.getBoolAttr(true)
            ).getOperation();

        parentCtx.addResultVals(newOp);
        return newOp;
    };
}

CmptGen Operation::buildTosaArgMaxOp() {
    return [](mlir::OpBuilder &builder, Context &parentCtx) {
        // NOTE: The format of the generated MLIR becomes invalid when randomly
        // seleting float tensors.
        mlir::Value operand = parentCtx.randomVal<mlir::IntegerType>();
        if (operand == nullptr)
            return (mlir::Operation *) nullptr;

        mlir::RankedTensorType oldTy = mlir::cast<mlir::RankedTensorType>(operand.getType());
        if (oldTy.getRank() == 0)  {
            // Avoid argmax tensor<i32>
            return (mlir::Operation *) nullptr;
        }

        llvm::ArrayRef<int64_t> shape = oldTy.getShape();
        unsigned axis = random(shape.size() - 1);
        llvm::SmallVector<int64_t> newShape;
        for (unsigned i = 0; i < shape.size(); i++) {
            if (i != axis)
                newShape.push_back(shape[i]);
        }
        mlir::IntegerAttr axisAttr = builder.getI32IntegerAttr(axis);
        mlir::RankedTensorType newTy = mlir::RankedTensorType::get(newShape, oldTy.getElementType());

        mlir::Operation *newOp = builder.create<mlir::tosa::ArgMaxOp>(
            builder.getUnknownLoc(),
            newTy,
            operand,
            axisAttr
        ).getOperation();

        parentCtx.addResultVals(newOp);
        return newOp;
    };
}

CmptGen Operation::buildTosaSelectOp() {
    return [this](mlir::OpBuilder &builder, Context &parentCtx) {
        mlir::Value pred = parentCtx.randomVal(1);
        if (pred == nullptr)
            return (mlir::Operation *) nullptr;

        mlir::Value operand1 = parentCtx.randomValCompatibleOf(pred);
        if (operand1 == nullptr)
            return (mlir::Operation *) nullptr;
        
        mlir::Value operand2 = parentCtx.randomValCompatibleOf(pred);
        if (operand2 == nullptr)
            return (mlir::Operation *) nullptr;

        // If the two oepration has different types, convert the integer one to float
        if (operand1.getType() != operand2.getType()) {
            if (isElemOfTy<mlir::IntegerType>(operand1)) {
                mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand1);
                parentCtx.addResultVals(castOp);
                operand1 = castOp->getResult(0);
            }
            if (isElemOfTy<mlir::IntegerType>(operand2)) {
                mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand2);
                parentCtx.addResultVals(castOp);
                operand2 = castOp->getResult(0);
            }
        }

        mlir::Operation *newOp = builder.create<mlir::tosa::SelectOp>(
            builder.getUnknownLoc(),
            operand1.getType(),
            pred,
            operand1,
            operand2
        ).getOperation();

        parentCtx.addResultVals(newOp);
        return newOp;
    };
}

// CmptGen Operation::buildTosaMatMulOp() {
//     return [this](mlir::OpBuilder &builder, Context &parentCtx) {
//         auto [operand1, operand2] = parentCtx.randomValPair();
//         if (operand1 == nullptr)
//             return (mlir::Operation *) nullptr;

//         mlir::RankedTensorType op2Ty = mlir::cast<mlir::RankedTensorType>(operand2.getType());
//         if (op2Ty.getRank() != 3)
//             return (mlir::Operation *) nullptr;

//         // If the two oepration has different types, convert the integer one to float
//         if (operand1.getType() != operand2.getType()) {
//             if (isElemOfTy<mlir::IntegerType>(operand1)) {
//                 mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand1);
//                 parentCtx.addResultVals(castOp);
//                 operand1 = castOp->getResult(0);
//             }
//             if (isElemOfTy<mlir::IntegerType>(operand2)) {
//                 mlir::Operation *castOp = this->castToFloatByTosaCastOp(builder, operand2);
//                 parentCtx.addResultVals(castOp);
//                 operand2 = castOp->getResult(0);
//             }
//         }

//         // Transpose one of the operands, i.e., switch the 2th and 3th dimensions
//         llvm::SmallVector<int64_t> transposedShape = {
//             op2Ty.getShape()[0],
//             op2Ty.getShape()[2],
//             op2Ty.getShape()[1]
//         };

//         mlir::RankedTensorType transposedTy = mlir::RankedTensorType::get(
//             transposedShape,
//             op2Ty.getElementType()
//         );
//         mlir::Operation *transposeOp = builder.create<mlir::tosa::TransposeOp>(
//             builder.getUnknownLoc(),
//             transposedTy,
//             operand2,
//             parentCtx.getTransposeAttr()
//         ).getOperation();
//         operand2 = transposeOp->getResult(0);

//         mlir::RankedTensorType newTy = mlir::RankedTensorType::get(
//             {op2Ty.getShape()[0], op2Ty.getShape()[1], op2Ty.getShape()[1]},
//             op2Ty.getElementType()
//         );
//         mlir::Operation *newOp = builder.create<mlir::tosa::MatMulOp>(
//             builder.getUnknownLoc(),
//             newTy,
//             operand1,
//             operand2
//         ).getOperation();

//         parentCtx.addResultVals(newOp);
//         return transposeOp;
//     };
// }

CallGen Operation::buildFuncCallOp() {
    return [](mlir::OpBuilder &builder, Context &parentCtx) {
        if (parentCtx.hasDefinedFunc())
            return (mlir::Operation *) nullptr;

        mlir::Operation *funcOp = parentCtx.randomFunc();
        mlir::func::FuncOp func = mlir::cast<mlir::func::FuncOp>(funcOp);

        llvm::SmallVector<mlir::Value> args;
        for (unsigned i = 0; i < func.getNumArguments(); i++) {
            // Use the mlir::Value casted from mlir::BlockArgument to sample values
            mlir::Value param = func.getArgument(i);
            args.push_back(parentCtx.randomValCompatibleOf(param));
        }

        mlir::Operation *newOp = builder.create<mlir::func::CallOp>(
            builder.getUnknownLoc(),
            func,
            args
        ).getOperation();
        parentCtx.addResultVals(newOp);

        return newOp;
    };
}

CtrlGen Operation::buildTosaIfOp() {
    return [this](mlir::OpBuilder &builder, Context &parentCtx, unsigned depth) {
        mlir::Value cond = parentCtx.randomVal<mlir::IntegerType>(1);
        if (cond == nullptr)
            return (mlir::Operation *) nullptr;

        llvm::SmallVector<mlir::Value> range{
            (unsigned) mlir::cast<mlir::TensorType>(cond.getType()).getRank(),
            parentCtx.getIndexZeroConst()
        };
        cond = this->extractScalarTensor(builder, cond, range);

        mlir::Value refYieldVal = parentCtx.randomVal<>();
        mlir::tosa::IfOp condIf = builder.create<mlir::tosa::IfOp>(
            builder.getUnknownLoc(),
            mlir::TypeRange(refYieldVal.getType()),
            cond,
            mlir::ValueRange()
        );

        auto savedBuildPos = builder.saveInsertionPoint();

        auto buildBranch = [&](mlir::Region &region) {
            builder.createBlock(&region);
            Context branchCtx(parentCtx);
            this->buildOps(builder, branchCtx, {0, maxRegionOpNum}, depth + 1, CtrlFlowTy::TosaIf);

            mlir::Value yieldVal;
            if (isElemOfTy<mlir::IntegerType>(refYieldVal)) {
                yieldVal = branchCtx.randomValCompatibleOf(refYieldVal);
                if (yieldVal == nullptr)
                    return false;
            } else {
                yieldVal = branchCtx.randomValCompatibleOf(refYieldVal);
                if (yieldVal == nullptr)
                    return false;
                if (isElemOfTy<mlir::IntegerType>(yieldVal)) {
                    yieldVal = this->castToFloatByTosaCastOp(builder, yieldVal)->getResult(0);
                }
            }

            builder.create<mlir::tosa::YieldOp>(
                builder.getUnknownLoc(),
                mlir::ValueRange(yieldVal)
            );
            return true;
        };
        if (!buildBranch(condIf.getThenGraph())
            || !buildBranch(condIf.getElseGraph()))
            return (mlir::Operation *) nullptr;        
        
        builder.restoreInsertionPoint(savedBuildPos);

        parentCtx.addResultVals(condIf.getOperation());
        return condIf.getOperation();
    };
}

CtrlGen Operation::buildTosaWhileOp() {
    return [this](mlir::OpBuilder &builder, Context &parentCtx, unsigned depth) {
        mlir::TensorType refYieldTy = mlir::cast<mlir::TensorType>(
            parentCtx.randomVal<mlir::IntegerType>().getType()
        );
        mlir::Value indVar = this->buildZeroTensor(
            builder,
            refYieldTy
        )->getResult(0);

        mlir::tosa::WhileOp whileLoop = builder.create<mlir::tosa::WhileOp>(
            builder.getUnknownLoc(),
            mlir::TypeRange(refYieldTy),
            mlir::ValueRange(indVar)
        );

        auto savedBuildPos = builder.saveInsertionPoint();

        // Loop condition region
        mlir::Block *condBlock = builder.createBlock(&whileLoop.getCondGraph());
        condBlock->addArgument(indVar.getType(), indVar.getLoc());

        mlir::Value iterTimes = this->buildConstTensor(
            builder,
            parentCtx,
            maxLoopIterTimes,
            refYieldTy,
            false
        )->getResult(0);

        mlir::Operation *cmpOp = builder.create<mlir::tosa::GreaterOp>(
            builder.getUnknownLoc(),
            mlir::RankedTensorType::get(refYieldTy.getShape(), builder.getI1Type()),
            iterTimes,
            condBlock->getArgument(0)
        ).getOperation();

        mlir::Value condYieldVal = cmpOp->getResult(0);
        if (refYieldTy.getRank() != 0) {
            llvm::SmallVector<mlir::Value> range{
                (unsigned) refYieldTy.getRank(),
                parentCtx.getIndexZeroConst()
            };
            condYieldVal = this->extractScalarTensor(builder, condYieldVal, range);
        }
        builder.create<mlir::tosa::YieldOp>(builder.getUnknownLoc(),
                                            mlir::ValueRange(condYieldVal));

        // Loop body region
        mlir::Block *bodyBlock = builder.createBlock(whileLoop.getBody());
        bodyBlock->addArgument(indVar.getType(), indVar.getLoc());

        Context whileBodyCtx(parentCtx);
        whileBodyCtx.addSingleVal(bodyBlock->getArgument(0));
        this->buildOps(builder, whileBodyCtx, {0, maxRegionOpNum}, depth + 1, CtrlFlowTy::TosaWhile);

        mlir::Value step = this->buildOneTensor(
            builder,
            refYieldTy
        )->getResult(0);
        
        mlir::Operation *addOp = builder.create<mlir::tosa::AddOp>(
            builder.getUnknownLoc(),
            refYieldTy,
            bodyBlock->getArgument(0),
            step
        ).getOperation();
        builder.create<mlir::tosa::YieldOp>(builder.getUnknownLoc(),
                                            mlir::ValueRange(addOp->getResult(0)));

        builder.restoreInsertionPoint(savedBuildPos);

        parentCtx.addResultVals(whileLoop.getOperation());
        return whileLoop.getOperation();
    };
}

} // namespace tosas

#endif // TOSA_SMITH_OPERATION_H