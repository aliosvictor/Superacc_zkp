use std::cell::RefCell;

use crate::types::{FloatType, OperationPrecision};
use crate::zkp::verifiers::common::{AddCounter, MulCounter};

#[derive(Debug, Clone, Default)]
pub struct OperationSnapshot {
    pub add: AddCounter,
    pub mul: MulCounter,
}

thread_local! {
    static OPERATION_COUNTERS: RefCell<OperationSnapshot> =
        RefCell::new(OperationSnapshot::default());
}

pub fn reset_operation_counters() {
    OPERATION_COUNTERS.with(|cell| {
        *cell.borrow_mut() = OperationSnapshot::default();
    });
}

pub fn record_add(precision: OperationPrecision, count: usize) {
    if count == 0 {
        return;
    }
    OPERATION_COUNTERS.with(|cell| {
        cell.borrow_mut().add.record(precision, count);
    });
}

pub fn record_mul(precision: OperationPrecision, count: usize) {
    if count == 0 {
        return;
    }
    OPERATION_COUNTERS.with(|cell| {
        cell.borrow_mut().mul.record(precision, count);
    });
}

pub fn snapshot_operation_counters() -> OperationSnapshot {
    OPERATION_COUNTERS.with(|cell| cell.borrow().clone())
}

pub fn take_operation_snapshot() -> OperationSnapshot {
    OPERATION_COUNTERS.with(|cell| {
        let snapshot = {
            let counters_ref = cell.borrow();
            counters_ref.clone()
        };
        *cell.borrow_mut() = OperationSnapshot::default();
        snapshot
    })
}

pub fn add_op<T: FloatType>(lhs: T, rhs: T) -> T {
    record_add(T::operation_precision(), 1);
    lhs + rhs
}

pub fn mul_op<T: FloatType>(lhs: T, rhs: T) -> T {
    record_mul(T::operation_precision(), 1);
    lhs * rhs
}
