#ifndef TOSA_SMITH_H
#define TOSA_SMITH_H

#include "llvm/Support/LogicalResult.h"
#include <string>

namespace tosas {

llvm::LogicalResult runTOSASmith();
llvm::LogicalResult TOSASmithMain(int argc, char **argv);

}

#endif