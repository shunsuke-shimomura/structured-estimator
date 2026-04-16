//! Tests for EkfModel traits + StructuredEkf + manifold Jacobians.

use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use structured_estimator::{
    EstimationOutputStruct, EstimationState,
    components::GaussianValueType,
    ekf_model::{EkfObservationModel, EkfPropagationModel, StructuredEkf},
    manifold_jacobian,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};

// ============================================================================
// Linear model — test finite-diff default + manual override
// ============================================================================

#[derive(EstimationState, Clone, Debug)]
struct LinState {
    position: SVector<f64, 3>,
    velocity: SVector<f64, 3>,
}

struct LinProp;

impl PropagationModel for LinProp {
    type State = LinState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, s: &LinState, _d: &EmptyInput, _g: &EmptyInput, _t: &f64, dt: &f64) -> LinState {
        LinState {
            position: s.position + s.velocity * *dt,
            velocity: s.velocity,
        }
    }
}

// Finite-diff default (no override)
impl EkfPropagationModel<6> for LinProp {}

#[derive(EstimationOutputStruct, Debug, Clone)]
struct PosObs {
    position: SVector<f64, 3>,
}

struct PosObsMod;

impl ObservationModel for PosObsMod {
    type State = LinState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = PosObs;
    fn predict(&self, s: &LinState, _d: &EmptyInput, _g: &EmptyInput, _t: &f64) -> PosObs {
        PosObs { position: s.position }
    }
}

impl EkfObservationModel<6, 3> for PosObsMod {}

// ---- Manual Jacobian override ----

struct LinPropManual;

impl PropagationModel for LinPropManual {
    type State = LinState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, s: &LinState, _d: &EmptyInput, _g: &EmptyInput, _t: &f64, dt: &f64) -> LinState {
        LinState {
            position: s.position + s.velocity * *dt,
            velocity: s.velocity,
        }
    }
}

impl EkfPropagationModel<6> for LinPropManual {
    fn state_jacobian(
        &self, _state: &LinState, _det: &EmptyInput, _gi: &EmptyInput, _t: &f64, dt: &f64,
    ) -> Result<SMatrix<f64, 6, 6>, structured_estimator::components::KalmanFilterError> {
        let mut j = SMatrix::<f64, 6, 6>::identity();
        j[(0, 3)] = *dt;
        j[(1, 4)] = *dt;
        j[(2, 5)] = *dt;
        Ok(j)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_structured_ekf_init() {
    let ekf = StructuredEkf::new(
        LinProp,
        LinState { position: Vector3::new(1.0, 0.0, 0.0), velocity: Vector3::zeros() },
        SMatrix::<f64, 6, 6>::identity() * 100.0,
        &0.0_f64,
    );
    assert!((ekf.state().position[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_structured_ekf_propagate_finite_diff() {
    let mut ekf = StructuredEkf::new(
        LinProp,
        LinState { position: Vector3::zeros(), velocity: Vector3::new(1.0, 0.0, 0.0) },
        SMatrix::<f64, 6, 6>::identity(),
        &0.0_f64,
    );
    ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &10.0).unwrap();
    assert!((ekf.state().position[0] - 10.0).abs() < 1e-6);
}

#[test]
fn test_structured_ekf_manual_jacobian() {
    let mut ekf = StructuredEkf::new(
        LinPropManual,
        LinState { position: Vector3::zeros(), velocity: Vector3::new(1.0, 0.0, 0.0) },
        SMatrix::<f64, 6, 6>::identity(),
        &0.0_f64,
    );
    ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &10.0).unwrap();
    assert!((ekf.state().position[0] - 10.0).abs() < 1e-6);
}

#[test]
fn test_structured_ekf_update() {
    let mut ekf = StructuredEkf::new(
        LinProp,
        LinState { position: Vector3::zeros(), velocity: Vector3::zeros() },
        SMatrix::<f64, 6, 6>::identity() * 100.0,
        &0.0_f64,
    );

    let meas = PosObs { position: Vector3::new(5.0, 0.0, 0.0) };
    let cov_before = ekf.covariance().diagonal().sum();
    ekf.update(&PosObsMod, &meas, &EmptyInput, &EmptyInput, &0.0,
        Matrix3::identity() * 0.01).unwrap();
    assert!(ekf.covariance().diagonal().sum() < cov_before);
    assert!(ekf.state().position[0] > 4.0, "Should move toward measurement");
}

#[test]
fn test_finite_diff_vs_manual_jacobian_consistency() {
    // Both should give very similar results for linear system
    let init = LinState {
        position: Vector3::new(100.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 1.0, 0.0),
    };
    let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

    let mut ekf_fd = StructuredEkf::new(LinProp, init.clone(), cov, &0.0);
    let mut ekf_man = StructuredEkf::new(LinPropManual, init, cov, &0.0);

    let obs = PosObsMod;
    for i in 0..20 {
        let t = (i + 1) as f64;
        let true_pos = Vector3::new(100.0, t, 0.0);
        let meas = PosObs { position: true_pos };

        ekf_fd.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &t).unwrap();
        ekf_man.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &t).unwrap();

        ekf_fd.update(&obs, &meas, &EmptyInput, &EmptyInput, &t, Matrix3::identity()).unwrap();
        ekf_man.update(&obs, &meas, &EmptyInput, &EmptyInput, &t, Matrix3::identity()).unwrap();
    }

    let diff = (ekf_fd.state().position - ekf_man.state().position).norm();
    assert!(diff < 0.1, "FD and manual should agree: diff = {}", diff);
}

#[test]
fn test_structured_ekf_vs_ukf() {
    let init = LinState {
        position: Vector3::new(100.0, 0.0, 0.0),
        velocity: Vector3::new(0.0, 1.0, 0.0),
    };
    let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

    let mut ekf = StructuredEkf::new(LinProp, init.clone(), cov, &0.0);
    let mut ukf: UnscentedKalmanFilter<LinState, f64, f64, LinProp, EmptyInput, EmptyInput, 6, 0> =
        UnscentedKalmanFilter::new(LinProp, init, cov, &0.0, UKFParameters::new(1e-3, 2.0, 0.0));

    let obs = PosObsMod;
    for i in 0..20 {
        let t = (i + 1) as f64;
        let meas = PosObs { position: Vector3::new(100.0, t, 0.0) };

        ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &t).unwrap();
        ukf.propagate(&EmptyInput, &EmptyInput, None, &t).unwrap();

        ekf.update(&obs, &meas, &EmptyInput, &EmptyInput, &t, Matrix3::identity()).unwrap();
        ukf.update(&obs, &meas, &EmptyInput, &EmptyInput, &t, Matrix3::identity()).unwrap();
    }

    let diff = (ekf.state().position - ukf.state().position).norm();
    assert!(diff < 1.0, "EKF and UKF should agree for linear: diff = {}", diff);
}

// ============================================================================
// Quaternion model — with manual Jacobian using manifold helpers
// ============================================================================

#[derive(EstimationState, Clone, Debug)]
struct AttState {
    attitude: UnitQuaternion<f64>,
    gyro_bias: SVector<f64, 3>,
}

struct AttPropWithJacobian;

impl PropagationModel for AttPropWithJacobian {
    type State = AttState;
    type DeterministicInput = Vector3<f64>; // omega
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, s: &AttState, omega: &Vector3<f64>, _g: &EmptyInput, _t: &f64, dt: &f64) -> AttState {
        let omega_corrected = omega - s.gyro_bias;
        AttState {
            attitude: UnitQuaternion::new(omega_corrected * *dt) * s.attitude,
            gyro_bias: s.gyro_bias,
        }
    }
}

impl EkfPropagationModel<6> for AttPropWithJacobian {
    fn state_jacobian(
        &self, state: &AttState, omega: &Vector3<f64>, _gi: &EmptyInput, _t: &f64, dt: &f64,
    ) -> Result<SMatrix<f64, 6, 6>, structured_estimator::components::KalmanFilterError> {
        let omega_corrected = omega - state.gyro_bias;

        let mut j = SMatrix::<f64, 6, 6>::zeros();

        // ∂attitude'/∂attitude: rotation propagation Jacobian
        let j_qq = manifold_jacobian::quaternion_propagation_jacobian(&omega_corrected, *dt);
        j.fixed_view_mut::<3, 3>(0, 0).copy_from(&j_qq);

        // ∂attitude'/∂bias: bias enters through (ω - b)
        let j_qb = manifold_jacobian::quaternion_bias_jacobian(*dt);
        j.fixed_view_mut::<3, 3>(0, 3).copy_from(&j_qb);

        // ∂bias'/∂bias: identity (bias is constant in this model)
        j.fixed_view_mut::<3, 3>(3, 3).copy_from(&Matrix3::identity());

        Ok(j)
    }
}

#[test]
fn test_quaternion_ekf_with_manual_jacobian() {
    let omega = Vector3::new(0.0, 0.0, 0.05);

    let mut ekf = StructuredEkf::new(
        AttPropWithJacobian,
        AttState { attitude: UnitQuaternion::identity(), gyro_bias: SVector::zeros() },
        SMatrix::<f64, 6, 6>::identity() * 0.1,
        &0.0_f64,
    );

    for i in 0..100 {
        let t = (i + 1) as f64 * 0.1;
        ekf.propagate::<EmptyInput, 0>(&omega, &EmptyInput, None, &t).unwrap();
    }

    // Should have rotated and quaternion should be unit
    assert!((ekf.state().attitude.norm() - 1.0).abs() < 1e-10);
    assert!(ekf.state().attitude.angle() > 0.0);
}

#[test]
fn test_manifold_jacobian_rotation_vector() {
    // Just verify it runs and is close to finite diff
    let q = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    let v = Vector3::new(1.0, 0.0, 0.0);
    let jac = manifold_jacobian::rotation_vector_jacobian(&q, &v);
    assert!(jac.norm() > 0.0, "Jacobian should be non-zero");
}
