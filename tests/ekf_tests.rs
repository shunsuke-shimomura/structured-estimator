//! EKF tests — uses the SAME models as UKF tests, verifies convergence,
//! and cross-validates EKF vs UKF estimates.

use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use structured_estimator::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    components::GaussianValueType,
    ekf::ExtendedKalmanFilter,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};

// ---- Simple linear model (position + velocity) ----

#[derive(EstimationState, Clone, Debug)]
struct LinearState {
    position: SVector<f64, 3>,
    velocity: SVector<f64, 3>,
}

struct LinearPropagation;

impl PropagationModel for LinearPropagation {
    type State = LinearState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, state: &Self::State, _d: &EmptyInput, _g: &EmptyInput, _t: &f64, dt: &f64) -> Self::State {
        LinearState {
            position: state.position + state.velocity * *dt,
            velocity: state.velocity,
        }
    }
}

#[derive(EstimationOutputStruct, Debug, Clone)]
struct PositionObs {
    position: SVector<f64, 3>,
}

struct PositionObsModel;

impl ObservationModel for PositionObsModel {
    type State = LinearState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = PositionObs;
    fn predict(&self, state: &Self::State, _d: &EmptyInput, _g: &EmptyInput, _t: &f64) -> PositionObs {
        PositionObs { position: state.position }
    }
}

// ---- Tests ----

#[test]
fn test_ekf_initialization() {
    let ekf: ExtendedKalmanFilter<LinearState, f64, f64, LinearPropagation, EmptyInput, 6> =
        ExtendedKalmanFilter::new(
            LinearPropagation,
            LinearState { position: Vector3::new(1.0, 0.0, 0.0), velocity: Vector3::new(0.0, 1.0, 0.0) },
            SMatrix::<f64, 6, 6>::identity() * 100.0,
            &0.0_f64,
            1e-7,
        );
    assert!((ekf.state().position[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_ekf_propagation_linear() {
    let mut ekf: ExtendedKalmanFilter<LinearState, f64, f64, LinearPropagation, EmptyInput, 6> =
        ExtendedKalmanFilter::new(
            LinearPropagation,
            LinearState { position: Vector3::new(0.0, 0.0, 0.0), velocity: Vector3::new(1.0, 0.0, 0.0) },
            SMatrix::<f64, 6, 6>::identity(),
            &0.0_f64,
            1e-7,
        );

    ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &10.0).unwrap();

    // After 10s at 1 m/s, position should be [10, 0, 0]
    assert!((ekf.state().position[0] - 10.0).abs() < 1e-6);
    assert!((ekf.state().velocity[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_ekf_finite_diff_jacobian_linear() {
    // For a linear system, the finite-diff Jacobian should be exact (up to ε²)
    use structured_estimator::ekf::finite_difference_jacobian;

    let state = LinearState {
        position: Vector3::new(7000.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 7.5, 0.0),
    };
    let dt = 60.0;

    let jac = finite_difference_jacobian::<LinearState, LinearState, _, 6, 6>(
        &state,
        &|s| LinearPropagation.propagate(s, &EmptyInput, &EmptyInput, &0.0, &dt),
        1e-7,
    ).unwrap();

    // Expected Jacobian for linear system x' = x + v*dt:
    // | I  dt*I |
    // | 0  I    |
    let expected_diag = 1.0;
    let expected_off = dt;

    // Check diagonal blocks
    for i in 0..3 {
        assert!((jac[(i, i)] - expected_diag).abs() < 1e-5, "diag[{}] = {}", i, jac[(i, i)]);
        assert!((jac[(i+3, i+3)] - expected_diag).abs() < 1e-5);
    }
    // Check off-diagonal (∂position/∂velocity = dt*I)
    for i in 0..3 {
        assert!((jac[(i, i+3)] - expected_off).abs() < 1e-3,
            "off-diag[{},{}] = {}, expected {}", i, i+3, jac[(i, i+3)], expected_off);
    }
}

#[test]
fn test_ekf_update_reduces_covariance() {
    let mut ekf: ExtendedKalmanFilter<LinearState, f64, f64, LinearPropagation, EmptyInput, 6> =
        ExtendedKalmanFilter::new(
            LinearPropagation,
            LinearState { position: Vector3::zeros(), velocity: Vector3::zeros() },
            SMatrix::<f64, 6, 6>::identity() * 100.0,
            &0.0_f64,
            1e-7,
        );

    let obs = PositionObsModel;
    let meas = PositionObs { position: Vector3::new(1.0, 0.0, 0.0) };
    let cov_before = ekf.covariance().diagonal().sum();

    ekf.update(&obs, &meas, &EmptyInput, &EmptyInput, &0.0, Matrix3::identity() * 0.01).unwrap();

    let cov_after = ekf.covariance().diagonal().sum();
    assert!(cov_after < cov_before, "Update should reduce covariance");
    assert!(ekf.state().position[0] > 0.5, "State should move toward measurement");
}

#[test]
fn test_ekf_convergence() {
    let true_pos = Vector3::new(100.0, 0.0, 0.0);
    let true_vel = Vector3::new(0.0, 1.0, 0.0);

    let mut ekf: ExtendedKalmanFilter<LinearState, f64, f64, LinearPropagation, EmptyInput, 6> =
        ExtendedKalmanFilter::new(
            LinearPropagation,
            LinearState {
                position: true_pos + Vector3::new(10.0, 0.0, 0.0),
                velocity: true_vel,
            },
            SMatrix::<f64, 6, 6>::identity() * 1000.0,
            &0.0_f64,
            1e-7,
        );

    let obs = PositionObsModel;
    let dt = 1.0;

    for i in 0..50 {
        let time = (i + 1) as f64 * dt;
        ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &time).unwrap();

        let true_pos_now = true_pos + true_vel * time;
        let noise = Vector3::new(
            0.1 * ((i * 7) as f64).sin(),
            0.1 * ((i * 13) as f64).cos(),
            0.1 * ((i * 19) as f64).sin(),
        );
        let meas = PositionObs { position: true_pos_now + noise };
        ekf.update(&obs, &meas, &EmptyInput, &EmptyInput, &time, Matrix3::identity() * 0.01).unwrap();
    }

    let true_final = true_pos + true_vel * 50.0;
    let err = (ekf.state().position - true_final).norm();
    assert!(err < 1.0, "EKF should converge: position error = {} m", err);
}

#[test]
fn test_ekf_vs_ukf_consistency() {
    // Run both EKF and UKF on the same linear problem and verify similar estimates
    let initial_state = LinearState {
        position: Vector3::new(100.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 1.0, 0.0),
    };
    let initial_cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

    let mut ekf: ExtendedKalmanFilter<LinearState, f64, f64, LinearPropagation, EmptyInput, 6> =
        ExtendedKalmanFilter::new(
            LinearPropagation,
            initial_state.clone(),
            initial_cov,
            &0.0_f64,
            1e-7,
        );

    let mut ukf: UnscentedKalmanFilter<LinearState, f64, f64, LinearPropagation, EmptyInput, EmptyInput, 6, 0> =
        UnscentedKalmanFilter::new(
            LinearPropagation,
            initial_state,
            initial_cov,
            &0.0_f64,
            UKFParameters::new(1e-3, 2.0, 0.0),
        );

    let obs = PositionObsModel;
    let dt = 1.0;

    for i in 0..20 {
        let time = (i + 1) as f64 * dt;
        let true_pos = Vector3::new(100.0, time, 0.0);
        let meas = PositionObs { position: true_pos };

        ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &time).unwrap();
        ukf.propagate(&EmptyInput, &EmptyInput, None, &time).unwrap();

        ekf.update(&obs, &meas, &EmptyInput, &EmptyInput, &time, Matrix3::identity() * 1.0).unwrap();
        ukf.update(&obs, &meas, &EmptyInput, &EmptyInput, &time, Matrix3::identity() * 1.0).unwrap();
    }

    // For a LINEAR system, EKF and UKF should give very similar results
    let ekf_pos = ekf.state().position;
    let ukf_pos = ukf.state().position;
    let diff = (ekf_pos - ukf_pos).norm();
    assert!(diff < 1.0, "EKF and UKF should agree for linear system: diff = {}", diff);
}

// ---- Quaternion model test (nonlinear, manifold) ----

#[derive(EstimationState, Clone, Debug)]
struct AttState {
    attitude: UnitQuaternion<f64>,
    gyro_bias: SVector<f64, 3>,
}

struct AttProp;

impl PropagationModel for AttProp {
    type State = AttState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, state: &Self::State, _d: &EmptyInput, _g: &EmptyInput, _t: &f64, dt: &f64) -> Self::State {
        let omega = Vector3::new(0.01, 0.0, 0.0) - state.gyro_bias;
        AttState {
            attitude: UnitQuaternion::new(omega * *dt) * state.attitude,
            gyro_bias: state.gyro_bias,
        }
    }
}

#[test]
fn test_ekf_quaternion_propagation() {
    let mut ekf: ExtendedKalmanFilter<AttState, f64, f64, AttProp, EmptyInput, 6> =
        ExtendedKalmanFilter::new(
            AttProp,
            AttState {
                attitude: UnitQuaternion::identity(),
                gyro_bias: SVector::zeros(),
            },
            SMatrix::<f64, 6, 6>::identity() * 0.1,
            &0.0_f64,
            1e-7,
        );

    for i in 0..10 {
        let time = (i + 1) as f64 * 0.1;
        ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &time).unwrap();
    }

    // Quaternion should remain unit
    assert!((ekf.state().attitude.norm() - 1.0).abs() < 1e-10);
    // Should have rotated
    assert!(ekf.state().attitude.angle() > 0.0);
}
