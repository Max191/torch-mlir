// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "propagate-cat"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Takes a dim and a tensor, and returns true if the dim is the outermost
// dimension, ignoring all unit dims
static bool isOuterDimension(Value tensor, Value dim) {
  auto dimOp = dim.getDefiningOp<ConstantIntOp>();
  if (!dimOp) {
    return false;
  }
  auto tensorType =
      llvm::dyn_cast<ValueTensorType>(tensor.getType());
  if (!tensorType) {
    return false;
  }
  auto tensorShape = tensorType.getSizes();
  int64_t dimVal = dimOp.getValueAttr().getInt();
  for (int i = 0; i < dimVal; i++) {
    if (tensorShape[i] != 1) {
      return false;
    }
  }
  return true;
}

static ValueTensorType
getTransposedVTensorType(PatternRewriter &rewriter,
                         ValueTensorType inputType, int64_t tDim0,
                         int64_t tDim1) {
  SmallVector<int64_t> newShape(inputType.getSizes());
  int64_t tDimSize = newShape[tDim0];
  newShape[tDim0] = newShape[tDim1];
  newShape[tDim1] = tDimSize;
  return ValueTensorType::get(rewriter.getContext(), newShape,
                                            inputType.getOptionalDtype());
}

namespace {
class PropagateAtenCatOp : public OpRewritePattern<AtenCatOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCatOp catOp,
                                PatternRewriter &rewriter) const override {
    // Inputs are torch.list<vtensor>
    LDBG("catOp: " << catOp);
    Value tensorList = catOp.getTensors();
    SmallVector<Value> tensors;
    if (!getListConstructElements(tensorList, tensors)) {
      return catOp.emitError(
          "unimplemented: the tensor list is not from list construct");
    }
    LDBG("tensorList: " << tensorList);
    auto catResultType = llvm::dyn_cast<ValueTensorType>(
        catOp.getResult().getType());
    if (!catResultType) {
      return failure();
    }
    LDBG("catResultType: " << catResultType);

    Value catDim = catOp.getDim();
    int64_t catDimVal;
    auto dimConstantOp = catDim.getDefiningOp<ConstantIntOp>();
    if (!dimConstantOp) {
      return failure();
    }
    LDBG("dimConstantOp: " << dimConstantOp);
    catDimVal = dimConstantOp.getValueAttr().getInt();
    Value transposeDim;
    int64_t transposeDimVal;

    // torch.aten.transpose.int -> torch.prim.ListConstruct -> torch.aten.cat
    SmallVector<std::optional<AtenTransposeIntOp>>
        candidateTransposes(tensors.size(), std::nullopt);
    for (auto [i, tensor] : llvm::enumerate(tensors)) {
      if (isOuterDimension(tensor, catDim)) {
        return failure();
      }
      auto maybeTransposeOp =
          tensor.getDefiningOp<AtenTransposeIntOp>();
      if (maybeTransposeOp) {
        auto transposeOp =
            llvm::cast<AtenTransposeIntOp>(*maybeTransposeOp);
        Value tDim0 = transposeOp.getDim0();
        Value tDim1 = transposeOp.getDim1();
        if (isOuterDimension(transposeOp.getSelf(), tDim0) ||
            isOuterDimension(transposeOp.getSelf(), tDim1)) {
          auto tDimOp0 = tDim0.getDefiningOp<ConstantIntOp>();
          auto tDimOp1 = tDim1.getDefiningOp<ConstantIntOp>();
          if (tDimOp0 && tDimOp1) {
            int64_t tDim0Val = tDimOp0.getValueAttr().getInt();
            int64_t tDim1Val = tDimOp1.getValueAttr().getInt();
            if (tDim0Val == catDimVal) {
              if (!transposeDim || tDim1Val == transposeDimVal) {
                candidateTransposes[i] = transposeOp;
                transposeDim = tDim1;
                transposeDimVal = tDim1Val;
              }
            } else if (tDim1Val == catDimVal) {
              if (!transposeDim || tDim0Val == transposeDimVal) {
                candidateTransposes[i] = transposeOp;
                transposeDim = tDim0;
                transposeDimVal = tDim0Val;
              }
            }
          }
        }
      }
    }

    if (llvm::all_of(candidateTransposes,
                     [](auto op) -> bool { return op == std::nullopt; })) {
      return failure();
    }

    SmallVector<Value> newCatInputs;
    for (auto [i, transposeOp] : llvm::enumerate(candidateTransposes)) {
      if (transposeOp) {
        newCatInputs.push_back(transposeOp->getSelf());
      } else {
        Value tensor = tensors[i];
        auto tensorType =
            llvm::dyn_cast<ValueTensorType>(tensor.getType());
        if (!tensorType) {
          return failure();
        }
        ValueTensorType inputTransposeType =
            getTransposedVTensorType(rewriter, tensorType, catDimVal,
                                     transposeDimVal);
        Value inputTranspose =
            rewriter.create<AtenTransposeIntOp>(
                catOp.getLoc(), inputTransposeType, tensor, catDim,
                transposeDim);
        newCatInputs.push_back(inputTranspose);
      }
    }

    ValueTensorType newCatResultType = getTransposedVTensorType(
        rewriter, catResultType, catDimVal, transposeDimVal);
    Value newCatInputsList = rewriter.create<PrimListConstructOp>(
        catOp.getLoc(), tensorList.getType(), newCatInputs);
    Value newCatOp = rewriter.create<AtenCatOp>(
        catOp.getLoc(), newCatResultType, newCatInputsList, transposeDim);
    rewriter.replaceOpWithNewOp<AtenTransposeIntOp>(
        catOp, catResultType, newCatOp, catDim, transposeDim);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::
  populatePropagateCatPatterns(RewritePatternSet &patterns) {
    MLIRContext *context = patterns.getContext();
    patterns.add<PropagateAtenCatOp>(context);
}
