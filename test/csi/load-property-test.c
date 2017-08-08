// RUN: %clang_csi_toolc %tooldir/null-tool.c -o %t-null-tool.o
// RUN: %clang_csi_toolc %tooldir/load-property-test-tool.c -o %t-tool.o
// RUN: %link_csi %t-tool.o %t-null-tool.o -o %t-tool.o
// RUN: %clang_csi_c %s -o %t.o
// RUN: %clang_csi %t.o %t-tool.o %csirtlib -o %t
// RUN: %run %t | FileCheck %s

#include <stdio.h>

static int global = 0;

int main(int argc, char **argv) {
  int x = global + 1;               // Read-before-write on global
  printf("x is %d\n", x);           // Read-before-write on x
  global = 1;                       // Write on global
  x = global + 1;                   // Read on global; write on x
  printf("x is %d\n", x);           // Read on x
  printf("global is %d\n", global); // Read on global
  // CHECK: num_loads = 5
  // CHECK: num_read_before_writes = 2
  return 0;
}
