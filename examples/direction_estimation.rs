//! Direction estimation example using UKF with structured state.
//!
//! Estimates magnetic field direction and gyro bias from magnetometer
//! and gyroscope measurements, based on the s5e/c5a direction estimator.
//!
//! State: [mag_dir (Direction, 2D), gyro_bias (Vector3, 3D)] → 5 total dims
//! Observation: mag_field (Vector3, 3D)
//! Input: angular_velocity (Vector3, 3D) as Gaussian input

use nalgebra::{Matrix2, Matrix3, Rotation3, SMatrix, SVector, Unit, Vector3};
use structured_estimator::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    components::Direction,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};

// ---- State: magnetic direction + gyro bias ----

#[derive(Debug, Clone, EstimationState)]
struct DirectionState {
    mag_dir: Direction,          // 2D manifold on unit sphere
    gyro_bias: SVector<f64, 3>,  // 3D bias vector
}

// ---- Gaussian input: angular velocity from gyro ----

#[derive(EstimationGaussianInput, Clone, Debug)]
struct GyroInput {
    angular_velocity: Vector3<f64>,
}

// ---- Propagation model: rotate direction by (omega - bias) * dt ----

struct DirectionPropagationModel;

impl PropagationModel for DirectionPropagationModel {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = GyroInput;
    type Time = f64;
    type Dt = f64;

    fn propagate(
        &self,
        state: &Self::State,
        _det_input: &Self::DeterministicInput,
        gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
        dt: &Self::Dt,
    ) -> Self::State {
        let omega_corrected = gaussian_input.angular_velocity - state.gyro_bias;
        let rotation = Rotation3::new(omega_corrected * *dt);
        DirectionState {
            mag_dir: rotation * state.mag_dir.clone(),
            gyro_bias: state.gyro_bias,
        }
    }
}

// ---- Observation model: predict mag field vector from direction ----

#[derive(Debug, Clone, EstimationOutputStruct)]
struct MagFieldObservation {
    mag_field: Vector3<f64>,
}

struct MagFieldObservationModel {
    mag_norm: f64, // assumed known magnetic field magnitude
}

impl ObservationModel for MagFieldObservationModel {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = MagFieldObservation;

    fn predict(
        &self,
        state: &Self::State,
        _det_input: &Self::DeterministicInput,
        _gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        MagFieldObservation {
            mag_field: state.mag_dir.dir().into_inner() * self.mag_norm,
        }
    }
}

fn main() {
    // ---- True state ----
    let true_mag_dir = Direction::from_dir(Unit::new_normalize(Vector3::new(1.0, 0.5, 0.3)));
    let true_gyro_bias = Vector3::new(0.01, -0.005, 0.002); // rad/s
    let true_omega = Vector3::new(0.0, 0.0, 0.1); // rad/s rotation about z
    let mag_norm = 50.0e-6; // 50 μT

    // ---- Initial estimate (with error) ----
    let initial_dir = Direction::from_dir(Unit::new_normalize(Vector3::new(1.0, 0.4, 0.2)));
    let initial_state = DirectionState {
        mag_dir: initial_dir,
        gyro_bias: Vector3::zeros(), // unknown bias
    };

    let initial_covariance = {
        let mut cov = SMatrix::<f64, 5, 5>::zeros();
        // Direction uncertainty: ~10 deg
        cov.fixed_view_mut::<2, 2>(0, 0)
            .copy_from(&(Matrix2::identity() * 0.03));
        // Gyro bias uncertainty: ~0.02 rad/s
        cov.fixed_view_mut::<3, 3>(2, 2)
            .copy_from(&(Matrix3::identity() * 0.02 * 0.02));
        cov
    };

    // ---- UKF setup ----
    let params = UKFParameters::new(1e-3, 2.0, 0.0);
    let mut ukf = UnscentedKalmanFilter::new(
        DirectionPropagationModel,
        initial_state,
        initial_covariance,
        &0.0_f64,
        params,
    );

    // ---- Process noise ----
    let process_noise = {
        let mut q = SMatrix::<f64, 5, 5>::zeros();
        q.fixed_view_mut::<2, 2>(0, 0)
            .copy_from(&(Matrix2::identity() * 1e-6));
        q.fixed_view_mut::<3, 3>(2, 2)
            .copy_from(&(Matrix3::identity() * 1e-8));
        q
    };

    // ---- Measurement noise ----
    let measurement_noise = Matrix3::identity() * (2.0e-6 * 2.0e-6); // 2 μT std

    // ---- Observation model ----
    let obs_model = MagFieldObservationModel { mag_norm };

    // ---- Simulation loop ----
    let dt = 0.1; // 10 Hz
    let mut true_dir = true_mag_dir.clone();
    let mut time = 0.0_f64;

    println!("=== Direction Estimation with Structured UKF ===\n");
    println!("State: [mag_dir (2D Direction), gyro_bias (3D Vector)] → 5 dims total\n");

    for step in 0..100 {
        time += dt;

        // ---- True dynamics: rotate direction ----
        let true_rotation = Rotation3::new((true_omega - true_gyro_bias) * dt);
        true_dir = true_rotation * true_dir;

        // ---- Gyro measurement (with bias and noise) ----
        let gyro_noise = Vector3::new(
            0.001 * ((step * 7) as f64).sin(),
            0.001 * ((step * 13) as f64).cos(),
            0.001 * ((step * 19) as f64).sin(),
        );
        let gyro_measurement = true_omega + gyro_noise;

        // ---- Propagate UKF ----
        let gyro_input = GyroInputGaussian {
            angular_velocity: gyro_measurement,
            angular_velocity_covariance: Matrix3::identity() * (0.001 * 0.001),
        };
        ukf.propagate(
            &EmptyInput,
            &gyro_input,
            Some(process_noise),
            &time,
        )
        .unwrap();

        // ---- Magnetometer measurement ----
        let true_mag = true_dir.dir().into_inner() * mag_norm;
        let mag_noise = Vector3::new(
            1.0e-6 * ((step * 3) as f64).sin(),
            1.0e-6 * ((step * 5) as f64).cos(),
            1.0e-6 * ((step * 11) as f64).sin(),
        );
        let mag_measurement = MagFieldObservation {
            mag_field: true_mag + mag_noise,
        };

        // ---- Update UKF ----
        ukf.update(
            &obs_model,
            &mag_measurement,
            &EmptyInput,
            &EmptyInput,
            &time,
            measurement_noise,
        )
        .unwrap();

        // ---- Print progress ----
        if step % 20 == 0 || step == 99 {
            let est_dir = ukf.state().mag_dir.dir().into_inner();
            let true_dir_vec = true_dir.dir().into_inner();
            let angle_error = est_dir.dot(&true_dir_vec).clamp(-1.0, 1.0).acos();
            let bias_error = (ukf.state().gyro_bias - true_gyro_bias).norm();
            println!(
                "t={:>5.1}s  dir_err={:>6.2}°  bias_err={:.4} rad/s  P_diag=[{:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}]",
                time,
                angle_error.to_degrees(),
                bias_error,
                ukf.covariance()[(0, 0)],
                ukf.covariance()[(1, 1)],
                ukf.covariance()[(2, 2)],
                ukf.covariance()[(3, 3)],
                ukf.covariance()[(4, 4)],
            );
        }
    }

    println!("\n=== Estimation complete ===");
    let final_dir = ukf.state().mag_dir.dir().into_inner();
    let true_dir_vec = true_dir.dir().into_inner();
    let final_angle_error = final_dir.dot(&true_dir_vec).clamp(-1.0, 1.0).acos();
    println!("Final direction error: {:.2}°", final_angle_error.to_degrees());
    println!("Final bias estimate: {:?}", ukf.state().gyro_bias);
    println!("True bias:           {:?}", true_gyro_bias);
}
