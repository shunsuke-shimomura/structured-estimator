//! # structured-estimator
//!
//! Structured Unscented Kalman Filter (UKF) library.
//!
//! Supports structured state types with manifold operations:
//! - Quaternion (3D error manifold for 4D quaternion)
//! - Direction (2D error manifold on 3D unit sphere)
//! - Standard vectors (Vector1–Vector6)
//!
//! State, observation, and input types are composed from these components
//! using derive macros (`EstimationState`, `EstimationOutputStruct`,
//! `EstimationGaussianInput`).

pub mod components;
pub mod ekf;
pub mod ukf;

#[cfg(feature = "autodiff")]
pub mod autodiff;
pub mod value_structs;
pub use structured_estimator_macro::{EstimationGaussianInput, EstimationOutputStruct, EstimationState};

#[cfg(feature = "sunpou")]
pub mod sunpou_integration;
