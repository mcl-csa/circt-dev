//RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK: func @check_arithmetic(%[[A:.*]]: i2449, %[[B:.*]]: i2449) {
func @check_arithmetic(%a : i2449, %b : i2449) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.neg %[[A]] : i2449
    %0 = "llhd.neg"(%a) : (i2449) -> i2449

    // CHECK-NEXT: %{{.*}} = llhd.smod %[[A]], %[[B]] : i2449
    %2 = "llhd.smod"(%a, %b) : (i2449, i2449) -> i2449

    return
}
