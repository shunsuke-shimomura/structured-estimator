//! # 方向推定 (Direction Estimation) — UKF vs EKF 比較
//!
//! 状態: [mag_dir: Direction (2D S²多様体), gyro_bias: Vector3 (3D)]
//! 観測: 磁場ベクトル (3D)
//! 入力: ジャイロ角速度 (ガウスノイズ付き)
//!
//! Direction は 2D 接空間（球面の接平面）を持つ多様体。
//! UKF では merge_sigma/error_from が自動的に球面上の摂動を処理。
//! EKF (有限差分) も同じインフラを使うため、そのまま動作。
//!
//! ## 実行
//! ```
//! cargo run --example direction_estimation_comparison
//! ```

use nalgebra::{Matrix2, Matrix3, Rotation3, SMatrix, SVector, Unit, Vector3};
use structured_estimator::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    components::Direction,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    ekf_model::{EkfObservationModel, EkfPropagationModel, StructuredEkf},
    value_structs::EmptyInput,
};

// ---- 状態定義: 磁場方向 (2D) + ジャイロバイアス (3D) = 5D ----

#[derive(EstimationState, Clone, Debug)]
struct DirectionState {
    /// 磁場方向 (body座標系, 2D接空間)
    mag_dir: Direction,
    /// ジャイロバイアス (body座標系, rad/s)
    gyro_bias: SVector<f64, 3>,
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct GyroInput {
    angular_velocity: Vector3<f64>,
}

// ---- 伝搬モデル ----

struct DirectionPropagation;

impl PropagationModel for DirectionPropagation {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = GyroInput;
    type Time = f64;
    type Dt = f64;

    fn propagate(
        &self, state: &Self::State, _det: &EmptyInput,
        gyro: &GyroInput, _time: &f64, dt: &f64,
    ) -> Self::State {
        let omega = gyro.angular_velocity - state.gyro_bias;
        let rot = Rotation3::new(omega * *dt);
        DirectionState {
            mag_dir: rot * state.mag_dir.clone(), // 方向を回転
            gyro_bias: state.gyro_bias,
        }
    }
}

// 有限差分EKF: 空の impl で自動対応
impl EkfPropagationModel<5> for DirectionPropagation {}

// ---- 観測モデル: 磁場ベクトル = 方向 × 磁場強度 ----

#[derive(Debug, Clone, EstimationOutputStruct)]
struct MagFieldObservation {
    mag_field: Vector3<f64>,
}

struct MagFieldObservationModel {
    mag_norm: f64,
}

impl ObservationModel for MagFieldObservationModel {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = MagFieldObservation;

    fn predict(
        &self, state: &Self::State, _det: &EmptyInput, _gi: &EmptyInput, _t: &f64,
    ) -> MagFieldObservation {
        MagFieldObservation {
            mag_field: state.mag_dir.dir().into_inner() * self.mag_norm,
        }
    }
}

impl EkfObservationModel<5, 3> for MagFieldObservationModel {}

fn main() {
    println!("=== 方向推定: UKF vs 有限差分EKF ===\n");
    println!("状態: [mag_dir (Direction, 2D多様体), gyro_bias (3D)] = 5次元\n");

    let true_dir = Direction::from_dir(Unit::new_normalize(Vector3::new(1.0, 0.5, 0.3)));
    let true_bias = Vector3::new(0.01, -0.005, 0.002);
    let true_omega = Vector3::new(0.0, 0.0, 0.1);
    let mag_norm = 50.0e-6;

    let init_state = DirectionState {
        mag_dir: Direction::from_dir(Unit::new_normalize(Vector3::new(1.0, 0.4, 0.2))),
        gyro_bias: Vector3::zeros(),
    };
    let init_cov = {
        let mut c = SMatrix::<f64, 5, 5>::zeros();
        c.fixed_view_mut::<2, 2>(0, 0).copy_from(&(Matrix2::identity() * 0.03));
        c.fixed_view_mut::<3, 3>(2, 2).copy_from(&(Matrix3::identity() * 4e-4));
        c
    };
    let process_noise = {
        let mut q = SMatrix::<f64, 5, 5>::zeros();
        q.fixed_view_mut::<2, 2>(0, 0).copy_from(&(Matrix2::identity() * 1e-6));
        q.fixed_view_mut::<3, 3>(2, 2).copy_from(&(Matrix3::identity() * 1e-8));
        q
    };
    let meas_noise = Matrix3::identity() * (2.0e-6 * 2.0e-6);

    // UKF
    let mut ukf: UnscentedKalmanFilter<
        DirectionState, f64, f64, DirectionPropagation, EmptyInput,
        GyroInputGaussian, 5, 3,
    > = UnscentedKalmanFilter::new(
        DirectionPropagation, init_state.clone(), init_cov, &0.0, UKFParameters::new(1e-3, 2.0, 0.0),
    );

    // EKF (有限差分)
    let mut ekf = StructuredEkf::new(DirectionPropagation, init_state, init_cov, &0.0);

    let obs_model_ukf = MagFieldObservationModel { mag_norm };
    let obs_model_ekf = MagFieldObservationModel { mag_norm };
    let dt = 0.1;
    let mut true_d = true_dir.clone();

    for i in 0..100 {
        let time = (i + 1) as f64 * dt;
        let rot = Rotation3::new((true_omega - true_bias) * dt);
        true_d = rot * true_d;

        let gyro = GyroInputGaussian {
            angular_velocity: true_omega + Vector3::new(
                0.001 * ((i * 7) as f64).sin(),
                0.001 * ((i * 13) as f64).cos(),
                0.001 * ((i * 19) as f64).sin(),
            ),
            angular_velocity_covariance: Matrix3::identity() * 1e-6,
        };

        ukf.propagate(&EmptyInput, &gyro, Some(process_noise), &time).unwrap();
        ekf.propagate(&EmptyInput, &gyro, Some(process_noise), &time).unwrap();

        let true_mag = true_d.dir().into_inner() * mag_norm;
        let mag_noise = Vector3::new(
            1e-6 * ((i * 3) as f64).sin(),
            1e-6 * ((i * 5) as f64).cos(),
            1e-6 * ((i * 11) as f64).sin(),
        );
        let meas = MagFieldObservation { mag_field: true_mag + mag_noise };

        ukf.update(&obs_model_ukf, &meas, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
        ekf.update(&obs_model_ekf, &meas, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();

        if i % 25 == 24 || i == 0 {
            let ukf_err = ukf.state().mag_dir.dir().dot(&true_d.dir()).clamp(-1.0, 1.0).acos();
            let ekf_err = ekf.state().mag_dir.dir().dot(&true_d.dir()).clamp(-1.0, 1.0).acos();
            println!(
                "t={:>5.1}s | UKF: {:.3}° | EKF(FD): {:.3}° | bias_ukf={:.4} | bias_ekf={:.4}",
                time, ukf_err.to_degrees(), ekf_err.to_degrees(),
                (ukf.state().gyro_bias - true_bias).norm(),
                (ekf.state().gyro_bias - true_bias).norm(),
            );
        }
    }

    println!("\n=== 最終結果 ===");
    let ukf_err = ukf.state().mag_dir.dir().dot(&true_d.dir()).clamp(-1.0, 1.0).acos();
    let ekf_err = ekf.state().mag_dir.dir().dot(&true_d.dir()).clamp(-1.0, 1.0).acos();
    println!("方向誤差: UKF={:.3}°, EKF(FD)={:.3}°", ukf_err.to_degrees(), ekf_err.to_degrees());
    println!("バイアス誤差: UKF={:.5}, EKF={:.5}",
        (ukf.state().gyro_bias - true_bias).norm(),
        (ekf.state().gyro_bias - true_bias).norm());
}
