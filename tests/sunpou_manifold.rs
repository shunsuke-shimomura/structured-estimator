//! Tests for sunpou manifold types in UKF:
//! - Rotation<F1, F2> as attitude state (3D axis-angle error)
//! - FrameDirection<F> as direction state (2D tangent error)

#![cfg(feature = "sunpou")]

use nalgebra::{Matrix2, Matrix3, SMatrix, UnitQuaternion, Unit, Vector3};
use structured_estimator::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};
use sunpou::frame_direction::FrameDirection;
use sunpou::frame_vec::FrameVec;
use sunpou::prelude::*;
use sunpou::prefix::*;
use sunpou::rotation::Rotation as SunRotation;

// Frame markers
struct Eci;
struct Body;

// ============================================================================
// Attitude estimation: Rotation<Body, Eci> + gyro bias
// ============================================================================

#[derive(EstimationState, Clone, Debug)]
struct AttitudeState {
    attitude: SunRotation<Body, Eci>,
    gyro_bias: FrameVec<Body, AngularVelocity>,
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct GyroInput {
    angular_velocity: FrameVec<Body, AngularVelocity>,
}

struct AttitudePropagation;

impl PropagationModel for AttitudePropagation {
    type State = AttitudeState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = GyroInput;
    type Time = f64;
    type Dt = f64;

    fn propagate(
        &self, state: &Self::State, _det: &Self::DeterministicInput,
        gi: &Self::GaussianInput, _t: &Self::Time, dt: &Self::Dt,
    ) -> Self::State {
        let omega = *gi.angular_velocity.as_raw() - *state.gyro_bias.as_raw();
        let delta_q = UnitQuaternion::new(omega * *dt);
        AttitudeState {
            attitude: SunRotation::from_raw(delta_q * state.attitude.into_raw()),
            gyro_bias: state.gyro_bias,
        }
    }
}

#[derive(EstimationOutputStruct, Debug)]
struct AttitudeObs {
    attitude: SunRotation<Body, Eci>,
}

struct AttitudeObsModel;

impl ObservationModel for AttitudeObsModel {
    type State = AttitudeState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = AttitudeObs;

    fn predict(&self, state: &Self::State, _det: &Self::DeterministicInput,
               _gi: &Self::GaussianInput, _t: &Self::Time) -> Self::Observation {
        AttitudeObs { attitude: state.attitude }
    }
}

#[test]
fn test_rotation_in_ukf_state() {
    let initial_att = SunRotation::<Body, Eci>::identity();

    let mut ukf: UnscentedKalmanFilter<
        AttitudeState, f64, f64, AttitudePropagation, EmptyInput,
        GyroInputGaussian, 6, 3,
    > = UnscentedKalmanFilter::new(
        AttitudePropagation,
        AttitudeState {
            attitude: initial_att,
            gyro_bias: FrameVec::new(0.0, 0.0, 0.0),
        },
        SMatrix::<f64, 6, 6>::identity() * 0.1,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    // Propagate
    let gyro_input = GyroInputGaussian {
        angular_velocity: FrameVec::new(0.01, 0.0, 0.0),
        angular_velocity_covariance: Matrix3::identity() * 1e-6,
    };
    ukf.propagate(&EmptyInput, &gyro_input, None, &1.0).unwrap();

    // State should have rotated slightly
    let q = ukf.state().attitude.into_raw();
    assert!(q.angle() > 0.0, "Should have rotated");
}

#[test]
fn test_rotation_update_convergence() {
    let true_att = SunRotation::<Body, Eci>::from_angle_z(0.3);
    let true_bias = FrameVec::<Body, AngularVelocity>::new(0.005, -0.003, 0.001);

    let mut ukf: UnscentedKalmanFilter<
        AttitudeState, f64, f64, AttitudePropagation, EmptyInput,
        GyroInputGaussian, 6, 3,
    > = UnscentedKalmanFilter::new(
        AttitudePropagation,
        AttitudeState {
            attitude: SunRotation::identity(),
            gyro_bias: FrameVec::new(0.0, 0.0, 0.0),
        },
        SMatrix::<f64, 6, 6>::identity() * 1.0,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    let obs_model = AttitudeObsModel;
    let omega = FrameVec::<Body, AngularVelocity>::new(0.0, 0.0, 0.05);
    let mut true_q = true_att.into_raw();
    let dt = 0.1;

    for i in 0..100 {
        let time = (i + 1) as f64 * dt;
        let delta = UnitQuaternion::new((*omega.as_raw() - *true_bias.as_raw()) * dt);
        true_q = delta * true_q;

        let gyro_input = GyroInputGaussian {
            angular_velocity: omega,
            angular_velocity_covariance: Matrix3::identity() * 1e-6,
        };
        ukf.propagate(&EmptyInput, &gyro_input, None, &time).unwrap();

        if i % 5 == 0 {
            let meas = AttitudeObs {
                attitude: SunRotation::from_raw(true_q),
            };
            ukf.update(&obs_model, &meas, &EmptyInput, &EmptyInput, &time,
                       Matrix3::identity() * 0.001).unwrap();
        }
    }

    let error = (ukf.state().attitude.into_raw() * true_q.inverse()).angle();
    assert!(error < 0.1, "Attitude should converge: error = {} rad", error);
}

// ============================================================================
// Direction estimation: FrameDirection<Body> + gyro bias
// ============================================================================

#[derive(EstimationState, Clone, Debug)]
struct DirectionState {
    mag_dir: FrameDirection<Body>,
    gyro_bias: FrameVec<Body, AngularVelocity>,
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct DirGyroInput {
    angular_velocity: FrameVec<Body, AngularVelocity>,
}

struct DirPropagation;

impl PropagationModel for DirPropagation {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = DirGyroInput;
    type Time = f64;
    type Dt = f64;

    fn propagate(
        &self, state: &Self::State, _det: &Self::DeterministicInput,
        gi: &Self::GaussianInput, _t: &Self::Time, dt: &Self::Dt,
    ) -> Self::State {
        let omega = *gi.angular_velocity.as_raw() - *state.gyro_bias.as_raw();
        let rot = nalgebra::Rotation3::new(omega * *dt);
        DirectionState {
            mag_dir: rot * state.mag_dir.clone(),
            gyro_bias: state.gyro_bias,
        }
    }
}

#[derive(EstimationOutputStruct, Debug, Clone)]
struct MagObs {
    mag_field: FrameVec<Body, Force>,  // using Force as proxy for magnetic field dim
}

struct MagObsModel {
    mag_norm: f64,
}

impl ObservationModel for MagObsModel {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = MagObs;

    fn predict(&self, state: &Self::State, _det: &Self::DeterministicInput,
               _gi: &Self::GaussianInput, _t: &Self::Time) -> Self::Observation {
        MagObs {
            mag_field: FrameVec::from_raw(state.mag_dir.dir().into_inner() * self.mag_norm),
        }
    }
}

#[test]
fn test_frame_direction_in_ukf_state() {
    let initial_dir = FrameDirection::<Body>::from_dir(
        Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)),
    );

    let mut ukf: UnscentedKalmanFilter<
        DirectionState, f64, f64, DirPropagation, EmptyInput,
        DirGyroInputGaussian, 5, 3,
    > = UnscentedKalmanFilter::new(
        DirPropagation,
        DirectionState {
            mag_dir: initial_dir,
            gyro_bias: FrameVec::new(0.0, 0.0, 0.0),
        },
        {
            let mut cov = SMatrix::<f64, 5, 5>::zeros();
            cov.fixed_view_mut::<2, 2>(0, 0).copy_from(&(Matrix2::identity() * 0.03));
            cov.fixed_view_mut::<3, 3>(2, 2).copy_from(&(Matrix3::identity() * 0.01));
            cov
        },
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    let gyro = DirGyroInputGaussian {
        angular_velocity: FrameVec::new(0.0, 0.0, 0.1),
        angular_velocity_covariance: Matrix3::identity() * 1e-6,
    };
    ukf.propagate(&EmptyInput, &gyro, None, &0.1).unwrap();

    // Direction should have rotated
    let dir = ukf.state().mag_dir.dir().into_inner();
    assert!(dir[1].abs() > 0.001, "Direction should rotate about z");
}

#[test]
fn test_frame_direction_convergence() {
    let true_dir = FrameDirection::<Body>::from_dir(
        Unit::new_normalize(Vector3::new(1.0, 0.5, 0.3)),
    );
    let true_bias = Vector3::new(0.01, -0.005, 0.002);
    let true_omega = Vector3::new(0.0, 0.0, 0.1);
    let mag_norm = 50.0e-6;

    let mut ukf: UnscentedKalmanFilter<
        DirectionState, f64, f64, DirPropagation, EmptyInput,
        DirGyroInputGaussian, 5, 3,
    > = UnscentedKalmanFilter::new(
        DirPropagation,
        DirectionState {
            mag_dir: FrameDirection::from_dir(Unit::new_normalize(Vector3::new(1.0, 0.4, 0.2))),
            gyro_bias: FrameVec::new(0.0, 0.0, 0.0),
        },
        {
            let mut cov = SMatrix::<f64, 5, 5>::zeros();
            cov.fixed_view_mut::<2, 2>(0, 0).copy_from(&(Matrix2::identity() * 0.03));
            cov.fixed_view_mut::<3, 3>(2, 2).copy_from(&(Matrix3::identity() * 4e-4));
            cov
        },
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    let obs_model = MagObsModel { mag_norm };
    let process_noise = {
        let mut q = SMatrix::<f64, 5, 5>::zeros();
        q.fixed_view_mut::<2, 2>(0, 0).copy_from(&(Matrix2::identity() * 1e-6));
        q.fixed_view_mut::<3, 3>(2, 2).copy_from(&(Matrix3::identity() * 1e-8));
        q
    };
    let meas_noise = Matrix3::identity() * (2.0e-6 * 2.0e-6);
    let dt = 0.1;
    let mut true_d = true_dir.clone();

    for i in 0..100 {
        let time = (i + 1) as f64 * dt;
        let rot = nalgebra::Rotation3::new((true_omega - true_bias) * dt);
        true_d = rot * true_d;

        let gyro = DirGyroInputGaussian {
            angular_velocity: FrameVec::from_raw(nalgebra::Vector3::new(
                true_omega[0] + 0.001 * ((i * 7) as f64).sin(),
                true_omega[1] + 0.001 * ((i * 13) as f64).cos(),
                true_omega[2] + 0.001 * ((i * 19) as f64).sin(),
            )),
            angular_velocity_covariance: Matrix3::identity() * 1e-6,
        };
        ukf.propagate(&EmptyInput, &gyro, Some(process_noise), &time).unwrap();

        let true_mag = true_d.dir().into_inner() * mag_norm;
        let meas = MagObs {
            mag_field: FrameVec::from_raw(nalgebra::Vector3::new(
                true_mag[0] + 1e-6 * ((i * 3) as f64).sin(),
                true_mag[1] + 1e-6 * ((i * 5) as f64).cos(),
                true_mag[2] + 1e-6 * ((i * 11) as f64).sin(),
            )),
        };
        ukf.update(&obs_model, &meas, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
    }

    let est = ukf.state().mag_dir.dir().into_inner();
    let tru = true_d.dir().into_inner();
    let angle_err = est.dot(&tru).clamp(-1.0, 1.0).acos();
    assert!(angle_err < 0.05, "Direction should converge: err = {:.2}°", angle_err.to_degrees());
}
