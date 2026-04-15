//! Attitude determination tests using UKF with quaternion state.
//!
//! Based on s5e/c5a attitude determination. State: [quaternion (3D error), gyro_bias (3D)] = 6 dims.
//! Ported from c5a with crate::Time → f64, external deps removed.

use nalgebra::{Matrix3, Quaternion, SVector, UnitQuaternion, Vector3};
use structured_estimator::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    components::{Direction, GaussianValueType},
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};

// ---- Model definitions (from c5a/estimation/attitude_determination.rs) ----

#[derive(EstimationGaussianInput, Clone, Debug)]
struct AttitudePropagationInput {
    angular_velocity: SVector<f64, 3>,
}

#[derive(EstimationState, Clone, Debug)]
struct AttitudeState {
    attitude: UnitQuaternion<f64>,
    gyro_bias: SVector<f64, 3>,
}

struct AttitudePropagationModel;

impl PropagationModel for AttitudePropagationModel {
    type State = AttitudeState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = AttitudePropagationInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(
        &self,
        state: &Self::State,
        _det: &Self::DeterministicInput,
        gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
        dt: &Self::Dt,
    ) -> Self::State {
        let omega_est = gaussian_input.angular_velocity - state.gyro_bias;
        let delta_q = UnitQuaternion::new(omega_est * *dt);
        AttitudeState {
            attitude: delta_q * state.attitude,
            gyro_bias: state.gyro_bias,
        }
    }
}

#[derive(EstimationOutputStruct, Debug)]
struct AttitudeObservation {
    attitude: UnitQuaternion<f64>,
}

struct AttitudeObservationModel;

impl ObservationModel for AttitudeObservationModel {
    type State = AttitudeState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = AttitudeObservation;
    fn predict(&self, state: &Self::State, _det: &Self::DeterministicInput, _gi: &Self::GaussianInput, _t: &Self::Time) -> Self::Observation {
        AttitudeObservation { attitude: state.attitude }
    }
}

fn process_noise_covariance() -> nalgebra::SMatrix<f64, 6, 6> {
    let mut matrix = nalgebra::SMatrix::<f64, 6, 6>::zeros();
    matrix.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(Matrix3::identity() * 1e-8)); // gyro bias drift
    matrix
}

fn create_estimation(initial_q: UnitQuaternion<f64>) -> UnscentedKalmanFilter<
    AttitudeState, f64, f64, AttitudePropagationModel, EmptyInput,
    AttitudePropagationInputGaussian, 6, 3,
> {
    let initial_covariance = {
        let mut mat = nalgebra::SMatrix::<f64, 6, 6>::zeros();
        mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&(Matrix3::identity() * 0.1));
        mat.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * 0.01));
        mat
    };
    UnscentedKalmanFilter::new(
        AttitudePropagationModel,
        AttitudeState { attitude: initial_q, gyro_bias: SVector::zeros() },
        initial_covariance,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    )
}

// ---- Tests ----

#[test]
fn test_initialization() {
    let q = UnitQuaternion::identity();
    let ukf = create_estimation(q);
    assert!((ukf.state().attitude.angle() - q.angle()).abs() < 1e-10);
    assert!(ukf.state().gyro_bias.norm() < 1e-10);
    let eigenvalues = ukf.covariance().symmetric_eigenvalues();
    for eig in eigenvalues.iter() {
        assert!(eig > &0.0, "Covariance should be positive definite");
    }
}

#[test]
fn test_propagation_preserves_unit_quaternion() {
    let initial_q = UnitQuaternion::from_quaternion(Quaternion::new(0.5, 0.5, 0.5, 0.5));
    let mut ukf = create_estimation(initial_q);
    let omega = Vector3::new(0.01, 0.0, 0.0);
    let gyro_input = AttitudePropagationInputGaussian {
        angular_velocity: omega,
        angular_velocity_covariance: Matrix3::identity() * 1e-6,
    };
    for i in 0..100 {
        let time = (i + 1) as f64 * 0.1;
        ukf.propagate(&EmptyInput, &gyro_input, Some(process_noise_covariance()), &time).unwrap();
        // Quaternion should remain unit
        let q = ukf.state().attitude;
        assert!((q.norm() - 1.0).abs() < 1e-10, "Quaternion should be unit at step {}", i);
    }
}

#[test]
fn test_propagation_pos_and_neg_symmetry() {
    let model = AttitudePropagationModel;
    let q0 = UnitQuaternion::from_quaternion(Quaternion::new(-0.18, 0.73, -0.55, 0.36));
    let state0 = AttitudeState { attitude: q0, gyro_bias: SVector::zeros() };
    let omega = Vector3::new(0.0, 0.0, 1.0);
    let dt = 0.01;

    let g_nominal = AttitudePropagationInput { angular_velocity: omega };
    let state_nom = model.propagate(&state0, &EmptyInput, &g_nominal, &0.0, &dt);
    let _nom_err = state_nom.attitude.error(&state0.attitude);

    let g_pos = AttitudePropagationInput { angular_velocity: omega + Vector3::new(1.0, 0.0, 0.0) };
    let state_pos = model.propagate(&state0, &EmptyInput, &g_pos, &0.0, &dt);
    let pos_err = state_pos.attitude.error(&state_nom.attitude);

    let g_neg = AttitudePropagationInput { angular_velocity: omega - Vector3::new(1.0, 0.0, 0.0) };
    let state_neg = model.propagate(&state0, &EmptyInput, &g_neg, &0.0, &dt);
    let neg_err = state_neg.attitude.error(&state_nom.attitude);

    // Positive and negative deviations should be approximately antisymmetric
    let sum = pos_err + neg_err;
    assert!(sum.norm() < 1e-2, "pos + neg errors should nearly cancel: {:?}", sum);
}

#[test]
fn test_update_reduces_covariance() {
    let true_q = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    let mut ukf = create_estimation(UnitQuaternion::identity()); // wrong initial

    let obs_model = AttitudeObservationModel;
    let measurement = AttitudeObservation { attitude: true_q };
    let measurement_noise = Matrix3::identity() * 0.01;

    let cov_before = ukf.covariance().diagonal().sum();

    ukf.update(
        &obs_model,
        &measurement,
        &EmptyInput,
        &EmptyInput,
        &0.0_f64,
        measurement_noise,
    ).unwrap();

    let cov_after = ukf.covariance().diagonal().sum();
    assert!(cov_after < cov_before, "Update should reduce covariance");
}

#[test]
fn test_convergence_with_attitude_observations() {
    let true_q = UnitQuaternion::from_euler_angles(0.3, -0.2, 0.5);
    let true_bias = Vector3::new(0.005, -0.003, 0.001);
    let true_omega = Vector3::new(0.0, 0.0, 0.05);

    let mut ukf = create_estimation(UnitQuaternion::identity());
    let obs_model = AttitudeObservationModel;
    let dt = 0.1;
    let mut true_attitude = true_q;

    for i in 0..200 {
        let time = (i + 1) as f64 * dt;

        // True dynamics
        let delta_q = UnitQuaternion::new((true_omega - true_bias) * dt);
        true_attitude = delta_q * true_attitude;

        // Propagate
        let gyro_input = AttitudePropagationInputGaussian {
            angular_velocity: true_omega + Vector3::new(
                0.0005 * ((i * 7) as f64).sin(),
                0.0005 * ((i * 13) as f64).cos(),
                0.0005 * ((i * 19) as f64).sin(),
            ),
            angular_velocity_covariance: Matrix3::identity() * 1e-6,
        };
        ukf.propagate(&EmptyInput, &gyro_input, Some(process_noise_covariance()), &time).unwrap();

        // Update every 5 steps (simulating 2Hz star tracker at 10Hz gyro)
        if i % 5 == 0 {
            let noisy_q = UnitQuaternion::new(Vector3::new(
                0.001 * ((i * 3) as f64).sin(),
                0.001 * ((i * 5) as f64).cos(),
                0.001 * ((i * 11) as f64).sin(),
            )) * true_attitude;
            let measurement = AttitudeObservation { attitude: noisy_q };
            ukf.update(&obs_model, &measurement, &EmptyInput, &EmptyInput, &time, Matrix3::identity() * 0.001).unwrap();
        }
    }

    let angle_error = (ukf.state().attitude * true_attitude.inverse()).angle();
    assert!(angle_error < 0.05, "Attitude should converge: error = {} rad", angle_error);
}
