use std::cell::RefCell;

use crate::types::{FloatType, OperationPrecision};
use crate::zkp::verifiers::common::{AddCounter, MulCounter};

/// Snapshot of the floating point addition/multiplication counters broken
/// down by precision. This feeds the documentation in verification reports.
#[derive(Debug, Clone, Default)]
pub struct OperationSnapshot {
    pub add: AddCounter,
    pub mul: MulCounter,
}

thread_local! {
    static OPERATION_COUNTERS: RefCell<OperationSnapshot> =
        RefCell::new(OperationSnapshot::default());
}

/// Resets the thread-local counters so a new proof run starts from zero.
pub fn reset_operation_counters() {
    OPERATION_COUNTERS.with(|cell| {
        *cell.borrow_mut() = OperationSnapshot::default();
    });
}

/// Adds `count` additions executed at the given precision to the tracker.
pub fn record_add(precision: OperationPrecision, count: usize) {
    if count == 0 {
        return;
    }
    OPERATION_COUNTERS.with(|cell| {
        cell.borrow_mut().add.record(precision, count);
    });
}

/// Adds `count` multiplications executed at the given precision to the tracker.
pub fn record_mul(precision: OperationPrecision, count: usize) {
    if count == 0 {
        return;
    }
    OPERATION_COUNTERS.with(|cell| {
        cell.borrow_mut().mul.record(precision, count);
    });
}

/// Returns the current counters without clearing them.
pub fn snapshot_operation_counters() -> OperationSnapshot {
    OPERATION_COUNTERS.with(|cell| cell.borrow().clone())
}

/// Returns the current counters and resets them immediately after.
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

/// Adds two floating point numbers while recording a precision-aware cost.
pub fn add_op<T: FloatType>(lhs: T, rhs: T) -> T {
    record_add(T::operation_precision(), 1);
    lhs + rhs
}

/// Multiplies two floating point numbers while recording the cost.
pub fn mul_op<T: FloatType>(lhs: T, rhs: T) -> T {
    record_mul(T::operation_precision(), 1);
    lhs * rhs
}
