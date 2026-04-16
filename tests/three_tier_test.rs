//! Three-tier Jacobian consistency test (#15):
//! 1. Manual analytical Jacobian (via manifold helpers)
//! 2. Finite-difference Jacobian (default)
//! 3. UKF (sigma-point, no Jacobian)
//!
//! All three should converge to similar estimates on the same problem.

use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use structured_estimator::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    components::GaussianValueType,
    ekf_model::{EkfObservationModel, EkfPropagationModel, StructuredEkf},
    manifold_jacobian,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};

// ---- Attitude + gyro bias model ----

#[derive(EstimationState, Clone, Debug)]
struct AttState {
    attitude: UnitQuaternion<f64>,
    gyro_bias: SVector<f64, 3>,
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct GyroInput {
    angular_velocity: SVector<f64, 3>,
}

struct AttProp;

impl PropagationModel for AttProp {
    type State = AttState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = GyroInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, s: &AttState, _d: &EmptyInput, gi: &GyroInput, _t: &f64, dt: &f64) -> AttState {
        let omega_corrected = gi.angular_velocity - s.gyro_bias;
        AttState {
            attitude: UnitQuaternion::new(omega_corrected * *dt) * s.attitude,
            gyro_bias: s.gyro_bias,
        }
    }
}

// --- Finite-diff EKF (Tier 2): empty impl, uses default ---
struct AttPropFD;
impl PropagationModel for AttPropFD {
    type State = AttState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = GyroInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, s: &AttState, _d: &EmptyInput, gi: &GyroInput, _t: &f64, dt: &f64) -> AttState {
        let omega_corrected = gi.angular_velocity - s.gyro_bias;
        AttState {
            attitude: UnitQuaternion::new(omega_corrected * *dt) * s.attitude,
            gyro_bias: s.gyro_bias,
        }
    }
}
impl EkfPropagationModel<6> for AttPropFD {} // finite-diff default

// --- Manual Jacobian EKF (Tier 1) ---
struct AttPropManual;
impl PropagationModel for AttPropManual {
    type State = AttState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = GyroInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, s: &AttState, _d: &EmptyInput, gi: &GyroInput, _t: &f64, dt: &f64) -> AttState {
        let omega_corrected = gi.angular_velocity - s.gyro_bias;
        AttState {
            attitude: UnitQuaternion::new(omega_corrected * *dt) * s.attitude,
            gyro_bias: s.gyro_bias,
        }
    }
}
impl EkfPropagationModel<6> for AttPropManual {
    fn state_jacobian(
        &self, state: &AttState, _d: &EmptyInput, gi: &GyroInput, _t: &f64, dt: &f64,
    ) -> Result<SMatrix<f64, 6, 6>, structured_estimator::components::KalmanFilterError> {
        let omega_corrected = gi.angular_velocity - state.gyro_bias;
        let mut j = SMatrix::<f64, 6, 6>::zeros();
        j.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&manifold_jacobian::quaternion_propagation_jacobian(&omega_corrected, *dt));
        j.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&manifold_jacobian::quaternion_bias_jacobian(*dt));
        j.fixed_view_mut::<3, 3>(3, 3).copy_from(&Matrix3::identity());
        Ok(j)
    }
}

// --- Observation: attitude (star tracker) ---
#[derive(EstimationOutputStruct, Debug)]
struct AttObs {
    attitude: UnitQuaternion<f64>,
}

struct AttObsMod;
impl ObservationModel for AttObsMod {
    type State = AttState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = AttObs;
    fn predict(&self, s: &AttState, _d: &EmptyInput, _g: &EmptyInput, _t: &f64) -> AttObs {
        AttObs { attitude: s.attitude }
    }
}
impl EkfObservationModel<6, 3> for AttObsMod {}

// ============================================================================
// Test: all three tiers converge similarly
// ============================================================================

#[test]
fn test_three_tier_attitude_convergence() {
    let true_bias = Vector3::new(0.005, -0.003, 0.001);
    let true_omega = Vector3::new(0.0, 0.0, 0.05);
    let init_state = AttState {
        attitude: UnitQuaternion::identity(),
        gyro_bias: SVector::zeros(),
    };
    let init_cov = {
        let mut c = SMatrix::<f64, 6, 6>::zeros();
        c.fixed_view_mut::<3, 3>(0, 0).copy_from(&(Matrix3::identity() * 0.1));
        c.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * 0.01));
        c
    };
    let proc_noise = {
        let mut q = SMatrix::<f64, 6, 6>::zeros();
        q.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * 1e-8));
        q
    };

    // Create all three estimators
    let mut ekf_manual = StructuredEkf::new(AttPropManual, init_state.clone(), init_cov, &0.0);
    let mut ekf_fd = StructuredEkf::new(AttPropFD, init_state.clone(), init_cov, &0.0);
    let mut ukf: UnscentedKalmanFilter<AttState, f64, f64, AttProp, EmptyInput, GyroInputGaussian, 6, 3> =
        UnscentedKalmanFilter::new(AttProp, init_state, init_cov, &0.0, UKFParameters::new(1e-3, 2.0, 0.0));

    let obs = AttObsMod;
    let dt = 0.1;
    let mut true_q = UnitQuaternion::identity();

    for i in 0..100 {
        let time = (i + 1) as f64 * dt;

        // True dynamics
        let delta_q = UnitQuaternion::new((true_omega - true_bias) * dt);
        true_q = delta_q * true_q;

        // Gyro measurement
        let gyro = GyroInputGaussian {
            angular_velocity: true_omega + Vector3::new(
                0.0005 * ((i * 7) as f64).sin(),
                0.0005 * ((i * 13) as f64).cos(),
                0.0005 * ((i * 19) as f64).sin(),
            ),
            angular_velocity_covariance: Matrix3::identity() * 1e-6,
        };

        // Propagate all three
        ekf_manual.propagate(&EmptyInput, &gyro, Some(proc_noise), &time).unwrap();
        ekf_fd.propagate(&EmptyInput, &gyro, Some(proc_noise), &time).unwrap();
        ukf.propagate(&EmptyInput, &gyro, Some(proc_noise), &time).unwrap();

        // Update every 5 steps
        if i % 5 == 0 {
            let noisy_q = UnitQuaternion::new(Vector3::new(
                0.001 * ((i * 3) as f64).sin(),
                0.001 * ((i * 5) as f64).cos(),
                0.001 * ((i * 11) as f64).sin(),
            )) * true_q;
            let meas = AttObs { attitude: noisy_q };

            ekf_manual.update(&obs, &meas, &EmptyInput, &EmptyInput, &time, Matrix3::identity() * 0.001).unwrap();
            ekf_fd.update(&obs, &meas, &EmptyInput, &EmptyInput, &time, Matrix3::identity() * 0.001).unwrap();
            ukf.update(&obs, &meas, &EmptyInput, &EmptyInput, &time, Matrix3::identity() * 0.001).unwrap();
        }
    }

    // All three should have converged
    let err_manual = (ekf_manual.state().attitude * true_q.inverse()).angle();
    let err_fd = (ekf_fd.state().attitude * true_q.inverse()).angle();
    let err_ukf = (ukf.state().attitude * true_q.inverse()).angle();

    println!("Attitude errors (rad): manual={:.4}, fd={:.4}, ukf={:.4}", err_manual, err_fd, err_ukf);

    assert!(err_manual < 0.1, "Manual EKF should converge: {}", err_manual);
    assert!(err_fd < 0.1, "FD EKF should converge: {}", err_fd);
    assert!(err_ukf < 0.1, "UKF should converge: {}", err_ukf);

    // All three should give similar results
    let diff_manual_fd = (ekf_manual.state().attitude * ekf_fd.state().attitude.inverse()).angle();
    let diff_manual_ukf = (ekf_manual.state().attitude * ukf.state().attitude.inverse()).angle();
    let diff_fd_ukf = (ekf_fd.state().attitude * ukf.state().attitude.inverse()).angle();

    println!("Pairwise diffs (rad): manual-fd={:.4}, manual-ukf={:.4}, fd-ukf={:.4}",
        diff_manual_fd, diff_manual_ukf, diff_fd_ukf);

    assert!(diff_manual_fd < 0.05, "Manual and FD should agree: {}", diff_manual_fd);
    assert!(diff_fd_ukf < 0.05, "FD and UKF should agree: {}", diff_fd_ukf);
}
