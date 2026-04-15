//! Direction estimation tests using UKF with Direction manifold state.
//!
//! Based on s5e/c5a direction estimation. Tests the full UKF cycle with
//! Direction (2D manifold on unit sphere) and gyro bias estimation.

use nalgebra::{Matrix2, Matrix3, Rotation3, SMatrix, SVector, Unit, Vector3};
use structured_estimator::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    components::Direction,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter, input_shift},
    value_structs::EmptyInput,
};

// ---- Model definitions ----

#[derive(Debug, Clone, EstimationState)]
struct DirectionState {
    mag_dir: Direction,
    gyro_bias: SVector<f64, 3>,
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct GyroInput {
    angular_velocity: Vector3<f64>,
}

struct DirectionPropagation;

impl PropagationModel for DirectionPropagation {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = GyroInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, state: &Self::State, _det: &Self::DeterministicInput, gi: &Self::GaussianInput, _t: &Self::Time, dt: &Self::Dt) -> Self::State {
        let omega = gi.angular_velocity - state.gyro_bias;
        let rot = Rotation3::new(omega * *dt);
        DirectionState { mag_dir: rot * state.mag_dir.clone(), gyro_bias: state.gyro_bias }
    }
}

#[derive(Debug, Clone, EstimationOutputStruct)]
struct MagFieldObs {
    mag_field: Vector3<f64>,
}

struct MagFieldObsModel { mag_norm: f64 }

impl ObservationModel for MagFieldObsModel {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = MagFieldObs;
    fn predict(&self, state: &Self::State, _det: &Self::DeterministicInput, _gi: &Self::GaussianInput, _t: &Self::Time) -> Self::Observation {
        MagFieldObs { mag_field: state.mag_dir.dir().into_inner() * self.mag_norm }
    }
}

// Helper: initialization via input_shift (magnetometer → direction)

#[derive(EstimationGaussianInput, Clone, Debug)]
struct MagInitInput {
    mag_field: Vector3<f64>,
}

#[derive(EstimationOutputStruct, Clone, Debug)]
struct MagDirOutput {
    mag_direction: Direction,
}

fn mag_dir_from_field(input: &MagInitInput) -> MagDirOutput {
    MagDirOutput {
        mag_direction: Direction::from_dir(Unit::new_normalize(input.mag_field)),
    }
}

// ---- Tests ----

#[test]
fn test_direction_initialization_via_input_shift() {
    let mag_field = Vector3::new(30.0e-6, 20.0e-6, 10.0e-6);
    let input = MagInitInputGaussian {
        mag_field: mag_field,
        mag_field_covariance: Matrix3::identity() * (2.0e-6 * 2.0e-6),
    };

    let params = UKFParameters::new(1e-3, 2.0, 0.0);
    let (output, covariance): (MagDirOutput, SMatrix<f64, 2, 2>) =
        input_shift(&input, mag_dir_from_field, &params).unwrap();

    // Direction should point in the same direction as mag_field
    let expected_dir = mag_field.normalize();
    let actual_dir = output.mag_direction.dir().into_inner();
    assert!((actual_dir - expected_dir).norm() < 1e-5, "Direction should match: {:?} vs {:?}", actual_dir, expected_dir);

    // Covariance should be positive definite and small
    let eigenvalues = covariance.symmetric_eigenvalues();
    for eig in eigenvalues.iter() {
        assert!(eig > &0.0, "Covariance should be positive definite");
    }
}

#[test]
fn test_direction_propagation() {
    let initial_dir = Direction::from_dir(Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)));
    let state = DirectionState { mag_dir: initial_dir.clone(), gyro_bias: SVector::zeros() };
    let model = DirectionPropagation;

    // Rotate 90° about z
    let omega = Vector3::new(0.0, 0.0, core::f64::consts::FRAC_PI_2);
    let gi = GyroInput { angular_velocity: omega };
    let new_state = model.propagate(&state, &EmptyInput, &gi, &0.0, &1.0);

    // Direction should now point in y
    let dir = new_state.mag_dir.dir().into_inner();
    assert!((dir[0]).abs() < 1e-10, "x should be ~0: {}", dir[0]);
    assert!((dir[1] - 1.0).abs() < 1e-10, "y should be ~1: {}", dir[1]);
}

#[test]
fn test_ukf_direction_convergence() {
    let true_dir = Direction::from_dir(Unit::new_normalize(Vector3::new(1.0, 0.5, 0.3)));
    let true_bias = Vector3::new(0.01, -0.005, 0.002);
    let true_omega = Vector3::new(0.0, 0.0, 0.1);
    let mag_norm = 50.0e-6;

    let initial_dir = Direction::from_dir(Unit::new_normalize(Vector3::new(1.0, 0.4, 0.2)));
    let initial_cov = {
        let mut cov = SMatrix::<f64, 5, 5>::zeros();
        cov.fixed_view_mut::<2, 2>(0, 0).copy_from(&(Matrix2::identity() * 0.03));
        cov.fixed_view_mut::<3, 3>(2, 2).copy_from(&(Matrix3::identity() * 4e-4));
        cov
    };

    let mut ukf = UnscentedKalmanFilter::new(
        DirectionPropagation,
        DirectionState { mag_dir: initial_dir, gyro_bias: Vector3::zeros() },
        initial_cov,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    let obs_model = MagFieldObsModel { mag_norm };
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
        let true_rot = Rotation3::new((true_omega - true_bias) * dt);
        true_d = true_rot * true_d;

        let gyro = GyroInputGaussian {
            angular_velocity: true_omega + Vector3::new(
                0.001 * ((i * 7) as f64).sin(),
                0.001 * ((i * 13) as f64).cos(),
                0.001 * ((i * 19) as f64).sin(),
            ),
            angular_velocity_covariance: Matrix3::identity() * 1e-6,
        };
        ukf.propagate(&EmptyInput, &gyro, Some(process_noise), &time).unwrap();

        let true_mag = true_d.dir().into_inner() * mag_norm;
        let measurement = MagFieldObs {
            mag_field: true_mag + Vector3::new(
                1e-6 * ((i * 3) as f64).sin(),
                1e-6 * ((i * 5) as f64).cos(),
                1e-6 * ((i * 11) as f64).sin(),
            ),
        };
        ukf.update(&obs_model, &measurement, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
    }

    let est_dir = ukf.state().mag_dir.dir().into_inner();
    let true_dir_vec = true_d.dir().into_inner();
    let angle_error = est_dir.dot(&true_dir_vec).clamp(-1.0, 1.0).acos();
    assert!(angle_error < 0.05, "Direction should converge: error = {:.2}°", angle_error.to_degrees());

    let bias_error = (ukf.state().gyro_bias - true_bias).norm();
    assert!(bias_error < 0.01, "Bias should converge: error = {:.4} rad/s", bias_error);
}
