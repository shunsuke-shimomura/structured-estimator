//! # 軌道推定 (Orbit Determination) — UKF vs EKF (手動ヤコビアン)
//!
//! 状態: [position (3D), velocity (3D)] = 6D (全てユークリッド)
//! 伝搬: ケプラー力学 (Modified Equinoctial Elements による解析解)
//! 観測: GNSS位置・速度
//!
//! 軌道推定はユークリッド状態なので、ヤコビアンの解析的導出が可能。
//! codegen で build.rs から生成することも可能（examples/direction_estimation_comparison.rs参照）。
//!
//! ## 実行
//! ```
//! cargo run --example orbit_estimation
//! ```

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use structured_estimator::{
    EstimationOutputStruct, EstimationState,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    ekf_model::{EkfObservationModel, EkfPropagationModel, StructuredEkf},
    value_structs::EmptyInput,
};

const GME: f64 = 3.986004418e14; // m³/s²

// ============================================================================
// 状態定義: 位置 + 速度 (ECI, m, m/s)
// ============================================================================

#[derive(EstimationState, Clone, Debug)]
struct OrbitalState {
    /// ECI位置 [m]
    position: SVector<f64, 3>,
    /// ECI速度 [m/s]
    velocity: SVector<f64, 3>,
}

// ============================================================================
// 伝搬モデル: 2体ケプラー (Euler step)
//
// 実運用では MEE (Modified Equinoctial Elements) による解析解を使うが、
// ここでは簡略化のため Euler step で示す。
// ============================================================================

struct KeplerianPropagation;

impl PropagationModel for KeplerianPropagation {
    type State = OrbitalState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Dt = f64;

    fn propagate(
        &self, state: &Self::State, _det: &EmptyInput, _gi: &EmptyInput, _time: &f64, dt: &f64,
    ) -> Self::State {
        let r = state.position;
        let v = state.velocity;
        let r_norm = r.norm();

        // 2体加速度: a = -μ/|r|³ · r
        let accel = -r * (GME / (r_norm * r_norm * r_norm));

        // Euler step (簡略化。実運用では RK4 や解析解を推奨)
        OrbitalState {
            position: r + v * *dt,
            velocity: v + accel * *dt,
        }
    }
}

// 有限差分EKF
impl EkfPropagationModel<6> for KeplerianPropagation {}

// ============================================================================
// 手動ヤコビアンEKF: 2体力学のヤコビアンは解析的に導出可能
//
// F = | I       dt·I           |
//     | G·dt    I + (dG/dv)·dt |  ← ただし dG/dv = 0 (加速度は速度に依存しない)
//
// ここで G = ∂a/∂r (gravity gradient tensor):
//   G = -μ/|r|³ (I - 3 r̂ r̂ᵀ)
// ============================================================================

struct KeplerianPropagationManualJacobian;

impl PropagationModel for KeplerianPropagationManualJacobian {
    type State = OrbitalState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Dt = f64;

    fn propagate(
        &self, state: &Self::State, _det: &EmptyInput, _gi: &EmptyInput, _time: &f64, dt: &f64,
    ) -> Self::State {
        let r = state.position;
        let v = state.velocity;
        let r_norm = r.norm();
        let accel = -r * (GME / (r_norm * r_norm * r_norm));
        OrbitalState {
            position: r + v * *dt,
            velocity: v + accel * *dt,
        }
    }
}

impl EkfPropagationModel<6> for KeplerianPropagationManualJacobian {
    fn state_jacobian(
        &self, state: &OrbitalState, _det: &EmptyInput, _gi: &EmptyInput, _time: &f64, dt: &f64,
    ) -> Result<SMatrix<f64, 6, 6>, structured_estimator::components::KalmanFilterError> {
        let r = state.position;
        let r_norm = r.norm();
        let r3 = r_norm * r_norm * r_norm;
        let r5 = r3 * r_norm * r_norm;

        // Gravity gradient: G = ∂a/∂r = -μ/r³ · (I - 3 r̂ r̂ᵀ)
        // = -μ/r³ · I + 3μ/r⁵ · r rᵀ
        let gravity_gradient = -Matrix3::identity() * (GME / r3)
            + (r * r.transpose()) * (3.0 * GME / r5);

        let mut f = SMatrix::<f64, 6, 6>::identity();
        // ∂position'/∂velocity = dt · I
        f.fixed_view_mut::<3, 3>(0, 3).copy_from(&(Matrix3::identity() * *dt));
        // ∂velocity'/∂position = G · dt
        f.fixed_view_mut::<3, 3>(3, 0).copy_from(&(gravity_gradient * *dt));

        Ok(f)
    }
}

// ============================================================================
// 観測モデル: GNSS位置・速度直接観測
// ============================================================================

#[derive(Debug, Clone, EstimationOutputStruct)]
struct GnssObservation {
    position: Vector3<f64>,
    velocity: Vector3<f64>,
}

struct GnssObservationModel;

impl ObservationModel for GnssObservationModel {
    type State = OrbitalState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = GnssObservation;

    fn predict(&self, state: &Self::State, _det: &EmptyInput, _gi: &EmptyInput, _t: &f64) -> GnssObservation {
        GnssObservation {
            position: state.position,
            velocity: state.velocity,
        }
    }
}

impl EkfObservationModel<6, 6> for GnssObservationModel {}

fn main() {
    println!("=== 軌道推定: UKF vs 有限差分EKF vs 手動ヤコビアンEKF ===\n");

    // LEO円軌道 (高度 ~630 km)
    let r0 = SVector::<f64, 3>::new(7000e3, 0.0, 0.0);
    let v_circ = (GME / 7000e3).sqrt();
    let v0 = SVector::<f64, 3>::new(0.0, v_circ, 0.0);
    println!("初期軌道: r={:.0} km, v={:.3} km/s (円軌道)", r0[0] / 1e3, v_circ / 1e3);

    // 1 km の初期位置誤差
    let r0_est = r0 + SVector::<f64, 3>::new(1000.0, 0.0, 0.0);
    let v0_est = v0 + SVector::<f64, 3>::new(0.0, 1.0, 0.0);

    let init_state = OrbitalState { position: r0_est, velocity: v0_est };
    let init_cov = SMatrix::<f64, 6, 6>::identity() * 1e6;

    // UKF
    let mut ukf: UnscentedKalmanFilter<
        OrbitalState, f64, f64, KeplerianPropagation, EmptyInput, EmptyInput, 6, 0,
    > = UnscentedKalmanFilter::new(
        KeplerianPropagation, init_state.clone(), init_cov, &0.0, UKFParameters::new(1e-3, 2.0, 0.0),
    );

    // EKF (有限差分)
    let mut ekf_fd = StructuredEkf::new(KeplerianPropagation, init_state.clone(), init_cov, &0.0);

    // EKF (手動ヤコビアン)
    let mut ekf_manual = StructuredEkf::new(KeplerianPropagationManualJacobian, init_state, init_cov, &0.0);

    let obs_model_fd = GnssObservationModel;
    let obs_model_manual = GnssObservationModel;
    let dt = 60.0;

    // 真の軌道を伝搬 (Euler step)
    let mut true_r = r0;
    let mut true_v = v0;

    for i in 0..20 {
        let time = (i + 1) as f64 * dt;

        // 真値伝搬
        let r_norm = true_r.norm();
        let accel = -true_r * (GME / (r_norm * r_norm * r_norm));
        true_r = true_r + true_v * dt;
        true_v = true_v + accel * dt;

        // 推定器伝搬
        ukf.propagate(&EmptyInput, &EmptyInput, None, &time).unwrap();
        ekf_fd.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &time).unwrap();
        ekf_manual.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &time).unwrap();

        // GNSS観測 (10m ノイズ)
        let noise_pos = Vector3::new(
            10.0 * ((i * 7) as f64).sin(),
            10.0 * ((i * 13) as f64).cos(),
            10.0 * ((i * 19) as f64).sin(),
        );
        let meas = GnssObservation { position: true_r + noise_pos, velocity: true_v };
        let meas_noise = {
            let mut cov = SMatrix::<f64, 6, 6>::zeros();
            cov.fixed_view_mut::<3, 3>(0, 0).copy_from(&(Matrix3::identity() * 100.0));
            cov.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * 0.01));
            cov
        };

        ukf.update(&GnssObservationModel, &meas, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
        ekf_fd.update(&obs_model_fd, &meas, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
        ekf_manual.update(&obs_model_manual, &meas, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();

        let ukf_err = (ukf.state().position - true_r).norm();
        let fd_err = (ekf_fd.state().position - true_r).norm();
        let man_err = (ekf_manual.state().position - true_r).norm();
        println!(
            "t={:>5.0}s | 位置誤差: UKF={:>8.1}m, EKF(FD)={:>8.1}m, EKF(手動)={:>8.1}m",
            time, ukf_err, fd_err, man_err,
        );
    }

    // 共分散ブロック
    let cov = OrbitalState::covariance_blocks(ukf.covariance());
    println!("\n共分散 (UKF, 最終):");
    println!("  σ_pos = [{:.1}, {:.1}, {:.1}] m",
        cov.position_position()[(0,0)].sqrt(),
        cov.position_position()[(1,1)].sqrt(),
        cov.position_position()[(2,2)].sqrt());
    println!("  σ_vel = [{:.4}, {:.4}, {:.4}] m/s",
        cov.velocity_velocity()[(0,0)].sqrt(),
        cov.velocity_velocity()[(1,1)].sqrt(),
        cov.velocity_velocity()[(2,2)].sqrt());
}
