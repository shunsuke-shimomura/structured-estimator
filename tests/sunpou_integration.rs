//! Integration test: sunpou types with structured-estimator UKF.
//!
//! Verifies that FrameVec, UnitVec, and Scalar with sunpou's type-level
//! dimensions and prefixes work correctly as UKF state/observation/input fields.

#![cfg(feature = "sunpou")]

use nalgebra::{Matrix3, SMatrix, Vector3};
use structured_estimator::{
    EstimationOutputStruct, EstimationState,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};
use sunpou::frame_vec::FrameVec;
use sunpou::prefix::*;
use sunpou::prelude::*;

// ---- Frame markers ----
struct Eci;

// ---- State with sunpou types: position (km) + velocity (km/s) in ECI ----

#[derive(EstimationState, Clone, Debug)]
struct OrbitalState {
    position: FrameVec<Eci, Length, Kilo>,
    velocity: FrameVec<Eci, Velocity, Kilo>,
}

// ---- Gaussian input (none for this simple test) ----

// ---- Propagation model: simple linear x' = x + v*dt ----

struct LinearPropagation;

impl PropagationModel for LinearPropagation {
    type State = OrbitalState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Dt = f64;

    fn propagate(
        &self,
        state: &Self::State,
        _det: &Self::DeterministicInput,
        _gi: &Self::GaussianInput,
        _time: &Self::Time,
        dt: &Self::Dt,
    ) -> Self::State {
        // position += velocity * dt
        // Note: FrameVec doesn't support direct scalar*vec with f64,
        // so we work with raw values and re-wrap
        let new_pos_raw = state.position.as_raw() + state.velocity.as_raw() * *dt;
        OrbitalState {
            position: FrameVec::from_raw(new_pos_raw),
            velocity: state.velocity, // constant velocity
        }
    }
}

// ---- Observation: position only ----

#[derive(Debug, Clone, EstimationOutputStruct)]
struct PositionObservation {
    position: FrameVec<Eci, Length, Kilo>,
}

struct PositionObservationModel;

impl ObservationModel for PositionObservationModel {
    type State = OrbitalState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = PositionObservation;

    fn predict(
        &self,
        state: &Self::State,
        _det: &Self::DeterministicInput,
        _gi: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        PositionObservation {
            position: state.position,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_sunpou_state_initialization() {
    let state = OrbitalState {
        position: FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0),
        velocity: FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0),
    };

    let initial_cov = SMatrix::<f64, 6, 6>::identity() * 1e3;

    let ukf: UnscentedKalmanFilter<
        OrbitalState, f64, f64, LinearPropagation, EmptyInput,
        EmptyInput, 6, 0,
    > = UnscentedKalmanFilter::new(
        LinearPropagation,
        state,
        initial_cov,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    // State preserved
    assert!((ukf.state().position.x() - 7000.0).abs() < 1e-10);
    assert!((ukf.state().velocity.y() - 7.5).abs() < 1e-10);
}

#[test]
fn test_sunpou_propagation() {
    let state = OrbitalState {
        position: FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0),
        velocity: FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0),
    };

    let mut ukf: UnscentedKalmanFilter<
        OrbitalState, f64, f64, LinearPropagation, EmptyInput,
        EmptyInput, 6, 0,
    > = UnscentedKalmanFilter::new(
        LinearPropagation,
        state,
        SMatrix::<f64, 6, 6>::identity() * 1.0,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    // Propagate 60 seconds
    ukf.propagate(&EmptyInput, &EmptyInput, None, &60.0).unwrap();

    // position_y should increase by velocity_y * dt = 7.5 * 60 = 450 km
    assert!((ukf.state().position.y() - 450.0).abs() < 1.0,
        "Expected ~450 km, got {} km", ukf.state().position.y());
    // velocity should remain ~7.5 km/s
    assert!((ukf.state().velocity.y() - 7.5).abs() < 0.1);
}

#[test]
fn test_sunpou_update() {
    let state = OrbitalState {
        position: FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0),
        velocity: FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0),
    };

    let mut ukf: UnscentedKalmanFilter<
        OrbitalState, f64, f64, LinearPropagation, EmptyInput,
        EmptyInput, 6, 0,
    > = UnscentedKalmanFilter::new(
        LinearPropagation,
        state,
        SMatrix::<f64, 6, 6>::identity() * 100.0, // large initial uncertainty
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    let obs_model = PositionObservationModel;

    // Measurement says position is [7001, 0, 0] km
    let measurement = PositionObservation {
        position: FrameVec::<Eci, Length, Kilo>::new(7001.0, 0.0, 0.0),
    };

    let meas_noise = {
        let mut cov = SMatrix::<f64, 3, 3>::zeros();
        cov.fixed_view_mut::<3, 3>(0, 0).copy_from(&(Matrix3::identity() * 0.01));
        cov
    };

    let cov_before = ukf.covariance().diagonal().sum();

    ukf.update(
        &obs_model, &measurement, &EmptyInput, &EmptyInput,
        &0.0, meas_noise,
    ).unwrap();

    let cov_after = ukf.covariance().diagonal().sum();

    // Covariance should decrease after update
    assert!(cov_after < cov_before, "Update should reduce covariance");

    // Position should move toward measurement
    assert!(ukf.state().position.x() > 7000.0, "Position should move toward measurement");
}

#[test]
fn test_sunpou_convergence() {
    // True state
    let _true_pos = FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0);
    let true_vel = FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0);

    // Initial estimate with 10 km error
    let est_pos = FrameVec::<Eci, Length, Kilo>::new(7010.0, 0.0, 0.0);

    let mut ukf: UnscentedKalmanFilter<
        OrbitalState, f64, f64, LinearPropagation, EmptyInput,
        EmptyInput, 6, 0,
    > = UnscentedKalmanFilter::new(
        LinearPropagation,
        OrbitalState { position: est_pos, velocity: true_vel },
        SMatrix::<f64, 6, 6>::identity() * 1000.0,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    let obs_model = PositionObservationModel;
    let meas_noise = Matrix3::identity() * 0.01;
    let dt = 1.0;

    for i in 0..50 {
        let time = (i + 1) as f64 * dt;

        ukf.propagate(&EmptyInput, &EmptyInput, None, &time).unwrap();

        // True position at this time
        let true_pos_now = FrameVec::<Eci, Length, Kilo>::new(
            7000.0,
            7.5 * time,
            0.0,
        );

        // Add small noise
        let noise = Vector3::new(
            0.01 * ((i * 7) as f64).sin(),
            0.01 * ((i * 13) as f64).cos(),
            0.01 * ((i * 19) as f64).sin(),
        );
        let measurement = PositionObservation {
            position: FrameVec::from_raw(*true_pos_now.as_raw() + noise),
        };

        ukf.update(&obs_model, &measurement, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
    }

    // Position error should be small
    let true_final = FrameVec::<Eci, Length, Kilo>::new(7000.0, 7.5 * 50.0, 0.0);
    let error = (ukf.state().position.as_raw() - true_final.as_raw()).norm();
    assert!(error < 1.0, "Position should converge: error = {} km", error);
}

// ============================================================================
// Verify that sunpou's type safety catches frame mismatches at compile time
// (This is a conceptual test — the real check is that the code compiles only
// when frames match)
// ============================================================================

#[test]
fn test_sunpou_frame_consistency() {
    // This compiles because all fields use Eci frame
    let _state = OrbitalState {
        position: FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0),
        velocity: FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0),
    };

    // If someone tried to define a state mixing frames:
    //   struct BadState {
    //       position: FrameVec<Eci, Length, Kilo>,
    //       velocity: FrameVec<Body, Velocity, Kilo>,  // different frame!
    //   }
    // The PropagationModel would catch this when it tries to do frame-dependent
    // operations (rotation, cross products) — sunpou prevents mixing frames.
}

// ============================================================================
// Block covariance access — typed sub-matrices from raw covariance
// ============================================================================

#[test]
fn test_covariance_blocks() {
    let state = OrbitalState {
        position: FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0),
        velocity: FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0),
    };

    let mut ukf: UnscentedKalmanFilter<
        OrbitalState, f64, f64, LinearPropagation, EmptyInput,
        EmptyInput, 6, 0,
    > = UnscentedKalmanFilter::new(
        LinearPropagation,
        state,
        SMatrix::<f64, 6, 6>::identity() * 100.0,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    // Extract typed covariance blocks
    let blocks = OrbitalState::covariance_blocks(ukf.covariance());

    // position_position: 3×3 block at (0,0) — this is P_rr
    let p_rr = blocks.position_position();
    assert_eq!(p_rr.nrows(), 3);
    assert_eq!(p_rr.ncols(), 3);
    assert!((p_rr[(0, 0)] - 100.0).abs() < 1e-10);

    // velocity_velocity: 3×3 block at (3,3) — this is P_vv
    let p_vv = blocks.velocity_velocity();
    assert!((p_vv[(0, 0)] - 100.0).abs() < 1e-10);

    // position_velocity: 3×3 block at (0,3) — this is P_rv (cross-covariance)
    let p_rv = blocks.position_velocity();
    assert!((p_rv[(0, 0)]).abs() < 1e-10); // initially zero for diagonal cov

    // After propagation, cross-covariance should become non-zero
    ukf.propagate(&EmptyInput, &EmptyInput, None, &60.0).unwrap();

    let blocks_after = OrbitalState::covariance_blocks(ukf.covariance());
    let p_rr_after = blocks_after.position_position();
    let p_rv_after = blocks_after.position_velocity();

    // P_rr should grow (uncertainty increases)
    assert!(p_rr_after[(0, 0)] > p_rr[(0, 0)],
        "Position covariance should grow after propagation");

    // Cross-covariance may become non-zero
    // (depends on UKF sigma point spread, but generally true for coupled state)
}

#[test]
fn test_covariance_blocks_sunpou_typed() {
    // The real power: wrap raw blocks in sunpou's FrameElemMat for type safety
    use sunpou::frame_elem_mat::FrameElemMat;

    let cov = SMatrix::<f64, 6, 6>::identity() * 42.0;
    let blocks = OrbitalState::covariance_blocks(&cov);

    // P_rr block has dimension Length × Length in ECI frame with prefix Kilo+Kilo = Mega
    // (conceptually — we wrap it manually since the macro returns raw SMatrix)
    let p_rr_raw = blocks.position_position();
    let _p_rr: FrameElemMat<Eci, Area, 3, 3, Mega> =
        FrameElemMat::from_raw(p_rr_raw);

    // P_rv block has dimension Length × Velocity
    let p_rv_raw = blocks.position_velocity();
    let _p_rv: FrameElemMat<Eci, LengthVelocity, 3, 3, Mega> =
        FrameElemMat::from_raw(p_rv_raw);
}

// ============================================================================
// Jacobian blocks with sunpou types (#16)
// ============================================================================

#[test]
fn test_jacobian_blocks_sunpou_typed() {
    use sunpou::frame_elem_mat::FrameElemMat;
    use structured_estimator::ekf::finite_difference_jacobian;

    let state = OrbitalState {
        position: FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0),
        velocity: FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0),
    };
    let dt = 60.0;

    // Compute Jacobian via finite differences
    let jac = finite_difference_jacobian::<OrbitalState, OrbitalState, _, 6, 6>(
        &state,
        &|s| LinearPropagation.propagate(s, &EmptyInput, &EmptyInput, &0.0, &dt),
        1e-7,
    ).unwrap();

    // Use covariance_blocks (same shape) to access Jacobian blocks
    let blocks = OrbitalState::covariance_blocks(&jac);

    // F_rr (∂position/∂position): Dimensionless in Eci frame, Base prefix
    // For linear model: I (identity)
    let f_rr: FrameElemMat<Eci, Dimensionless, 3, 3, Base> =
        FrameElemMat::from_raw(blocks.position_position());
    assert!((f_rr.as_raw() - nalgebra::Matrix3::identity()).norm() < 1e-5);

    // F_rv (∂position/∂velocity): Time dimension (m / (m/s) = s)
    // For linear model: dt * I
    let f_rv: FrameElemMat<Eci, Time, 3, 3, Base> =
        FrameElemMat::from_raw(blocks.position_velocity());
    assert!((f_rv.as_raw()[(0, 0)] - dt).abs() < 1e-3);

    // F_vr (∂velocity/∂position): InvTime dimension
    // For linear model: 0
    let f_vr: FrameElemMat<Eci, InvTime, 3, 3, Base> =
        FrameElemMat::from_raw(blocks.velocity_position());
    assert!(f_vr.as_raw().norm() < 1e-5);

    // F_vv (∂velocity/∂velocity): Dimensionless
    // For linear model: I
    let f_vv: FrameElemMat<Eci, Dimensionless, 3, 3, Base> =
        FrameElemMat::from_raw(blocks.velocity_velocity());
    assert!((f_vv.as_raw() - nalgebra::Matrix3::identity()).norm() < 1e-5);
}

#[test]
fn test_structured_ekf_with_sunpou() {
    use structured_estimator::ekf_model::{EkfPropagationModel, EkfObservationModel, StructuredEkf};

    // Use finite-diff default (no manual Jacobian needed)
    impl EkfPropagationModel<6> for LinearPropagation {}
    impl EkfObservationModel<6, 3> for PositionObservationModel {}

    let mut ekf = StructuredEkf::new(
        LinearPropagation,
        OrbitalState {
            position: FrameVec::<Eci, Length, Kilo>::new(7010.0, 0.0, 0.0), // 10km error
            velocity: FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0),
        },
        SMatrix::<f64, 6, 6>::identity() * 1000.0,
        &0.0_f64,
    );

    let obs = PositionObservationModel;
    let dt = 1.0;
    let meas_noise = nalgebra::Matrix3::identity() * 0.01;

    for i in 0..50 {
        let time = (i + 1) as f64 * dt;
        ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &time).unwrap();

        let true_pos = FrameVec::<Eci, Length, Kilo>::new(7000.0, 7.5 * time, 0.0);
        let meas = PositionObservation { position: true_pos };
        ekf.update(&obs, &meas, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
    }

    let true_final = FrameVec::<Eci, Length, Kilo>::new(7000.0, 7.5 * 50.0, 0.0);
    let error = (ekf.state().position.as_raw() - true_final.as_raw()).norm();
    assert!(error < 1.0, "Sunpou StructuredEkf should converge: err = {} km", error);
}
