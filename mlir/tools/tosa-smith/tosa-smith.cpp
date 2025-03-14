#include "TOSASmith.h"

int main(int argc, char **argv) {
    return llvm::failed(tosas::TOSASmithMain(argc, argv));
}