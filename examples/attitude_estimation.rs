//! # 姿勢推定 (Attitude Estimation)
//!
//! 状態: [attitude: UnitQuaternion (3D SO(3)多様体), gyro_bias: Vector3 (3D)]
//! 観測: スタートラッカーによる姿勢直接観測
//! 入力: ジャイロ角速度 (ガウスノイズ付き)
//!
//! 3つの推定手法を同じモデルで比較:
//!   1. UKF (Unscented Kalman Filter) — シグマ点でヤコビアン不要
//!   2. EKF (有限差分) — 既存モデルの propagate/predict をそのまま流用
//!   3. EKF (手動ヤコビアン) — manifold_jacobian ヘルパーで解析的に記述
//!
//! ## 実行
//! ```
//! cargo run --example attitude_estimation
//! ```

use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use structured_estimator::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    // UKF
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    // EKF (有限差分デフォルト + 手動ヤコビアンオーバーライド)
    ekf_model::{EkfObservationModel, EkfPropagationModel, StructuredEkf},
    // 多様体ヤコビアンヘルパー (手動EKF用)
    manifold_jacobian,
    value_structs::EmptyInput,
};

// ============================================================================
// Step 1: 状態・入力・観測の構造体を定義
//
// #[derive(EstimationState)] が以下を自動生成:
//   - AttitudeStateSigmaPoint (接空間表現: attitude→3D axis-angle, bias→3D = 計6D)
//   - AttitudeStateNominal (基準点: attitude→UnitQuaternion, bias→nominal)
//   - ValueStructTrait, StateStructTrait 等の trait impl
//   - CovarianceBlocks アクセサ (attitude_attitude, attitude_gyro_bias, 等)
//   - SVector<f64, 6> ↔ SigmaPoint の変換
// ============================================================================

#[derive(EstimationState, Clone, Debug)]
struct AttitudeState {
    /// 姿勢クォータニオン (Body→ECI)
    /// 多様体: SO(3), 接空間は 3D axis-angle (rad)
    attitude: UnitQuaternion<f64>,

    /// ジャイロバイアス (Body座標系, rad/s)
    /// ユークリッド空間: 3D
    gyro_bias: SVector<f64, 3>,
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct GyroInput {
    /// ジャイロ測定値 (角速度 + ノイズ, Body座標系)
    angular_velocity: SVector<f64, 3>,
}

#[derive(EstimationOutputStruct, Debug)]
struct AttitudeObservation {
    /// スタートラッカーによる姿勢観測
    attitude: UnitQuaternion<f64>,
}

// ============================================================================
// Step 2: 伝搬モデルを定義
//
// PropagationModel trait を実装する。これだけで UKF と 有限差分EKF の両方で使える。
// ============================================================================

struct AttitudePropagationModel;

impl PropagationModel for AttitudePropagationModel {
    type State = AttitudeState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = GyroInput; // ジャイロはガウスノイズ付き
    type Time = f64;
    type Dt = f64;

    fn propagate(
        &self,
        state: &Self::State,
        _det: &Self::DeterministicInput,
        gyro: &Self::GaussianInput,
        _time: &Self::Time,
        dt: &Self::Dt,
    ) -> Self::State {
        // バイアス補正した角速度
        let omega_corrected = gyro.angular_velocity - state.gyro_bias;
        // 小角度回転: q' = exp(ω*dt) ⊗ q
        let delta_q = UnitQuaternion::new(omega_corrected * *dt);

        AttitudeState {
            attitude: delta_q * state.attitude,
            gyro_bias: state.gyro_bias, // バイアスはランダムウォーク（process noiseで表現）
        }
    }
}

// ============================================================================
// Step 3: 観測モデルを定義
// ============================================================================

struct StarTrackerObservationModel;

impl ObservationModel for StarTrackerObservationModel {
    type State = AttitudeState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = AttitudeObservation;

    fn predict(
        &self,
        state: &Self::State,
        _det: &Self::DeterministicInput,
        _gi: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        // スタートラッカーは姿勢を直接観測
        AttitudeObservation {
            attitude: state.attitude,
        }
    }
}

// ============================================================================
// Step 4a: 有限差分EKF — 空の impl だけで OK
//
// state_jacobian() のデフォルト実装が自動的に有限差分を使う。
// 多様体 (quaternion) の接空間も merge_sigma/error_from で正しく処理される。
// ============================================================================

impl EkfPropagationModel<6> for AttitudePropagationModel {}
impl EkfObservationModel<6, 3> for StarTrackerObservationModel {}

// ============================================================================
// Step 4b: 手動ヤコビアンEKF — manifold_jacobian ヘルパーを使用
//
// 解析的ヤコビアンを提供することで、有限差分の 12回モデル評価を 1回に削減。
// ヤコビアンの各ブロックを構造体フィールド名で考える:
//
//   F = | ∂attitude'/∂attitude   ∂attitude'/∂gyro_bias |
//       | ∂gyro_bias'/∂attitude  ∂gyro_bias'/∂gyro_bias|
//
//   = | R(δq)ᵀ              -dt·I    |
//     | 0                    I        |
// ============================================================================

struct AttitudePropagationManualJacobian;

// 同じ propagate を再実装（trait の制約上、別の struct が必要）
impl PropagationModel for AttitudePropagationManualJacobian {
    type State = AttitudeState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = GyroInput;
    type Time = f64;
    type Dt = f64;

    fn propagate(
        &self, state: &Self::State, _det: &Self::DeterministicInput,
        gyro: &Self::GaussianInput, _time: &Self::Time, dt: &Self::Dt,
    ) -> Self::State {
        let omega_corrected = gyro.angular_velocity - state.gyro_bias;
        let delta_q = UnitQuaternion::new(omega_corrected * *dt);
        AttitudeState {
            attitude: delta_q * state.attitude,
            gyro_bias: state.gyro_bias,
        }
    }
}

impl EkfPropagationModel<6> for AttitudePropagationManualJacobian {
    fn state_jacobian(
        &self,
        state: &AttitudeState,
        _det: &EmptyInput,
        gyro: &GyroInput,
        _time: &f64,
        dt: &f64,
    ) -> Result<SMatrix<f64, 6, 6>, structured_estimator::components::KalmanFilterError> {
        let omega_corrected = gyro.angular_velocity - state.gyro_bias;

        let mut f = SMatrix::<f64, 6, 6>::zeros();

        // --- ∂attitude'/∂attitude (3×3): 回転伝搬の接空間ヤコビアン ---
        // exp(ω·dt)⊗q の q に対する微分 = R(exp(ω·dt))ᵀ
        f.fixed_view_mut::<3, 3>(0, 0).copy_from(
            &manifold_jacobian::quaternion_propagation_jacobian(&omega_corrected, *dt),
        );

        // --- ∂attitude'/∂gyro_bias (3×3): バイアスが姿勢に与える影響 ---
        // ω_corrected = ω_meas - bias なので ∂/∂bias = -∂/∂ω ≈ -dt·I
        f.fixed_view_mut::<3, 3>(0, 3).copy_from(
            &manifold_jacobian::quaternion_bias_jacobian(*dt),
        );

        // --- ∂gyro_bias'/∂attitude (3×3): 0 (バイアスは姿勢に依存しない) ---
        // (すでに zeros)

        // --- ∂gyro_bias'/∂gyro_bias (3×3): I (バイアスは不変) ---
        f.fixed_view_mut::<3, 3>(3, 3).copy_from(&Matrix3::identity());

        Ok(f)
    }
}

// ============================================================================
// Step 5: シミュレーション実行
// ============================================================================

fn main() {
    println!("=== 姿勢推定: UKF vs 有限差分EKF vs 手動ヤコビアンEKF ===\n");

    // --- 真値 ---
    let true_bias = Vector3::new(0.005, -0.003, 0.001); // rad/s
    let true_omega = Vector3::new(0.0, 0.0, 0.05); // z軸周り 0.05 rad/s

    // --- 初期推定 ---
    let init_state = AttitudeState {
        attitude: UnitQuaternion::identity(),
        gyro_bias: SVector::zeros(), // バイアス未知
    };
    let init_cov = {
        let mut c = SMatrix::<f64, 6, 6>::zeros();
        c.fixed_view_mut::<3, 3>(0, 0).copy_from(&(Matrix3::identity() * 0.1)); // 姿勢不確かさ
        c.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * 0.01)); // バイアス不確かさ
        c
    };
    let process_noise = {
        let mut q = SMatrix::<f64, 6, 6>::zeros();
        q.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * 1e-8)); // バイアスドリフト
        q
    };

    // --- 3つの推定器を生成 ---

    // (1) UKF — ヤコビアン不要、シグマ点ベース
    let mut ukf: UnscentedKalmanFilter<
        AttitudeState, f64, f64,
        AttitudePropagationModel, EmptyInput,
        GyroInputGaussian, 6, 3,
    > = UnscentedKalmanFilter::new(
        AttitudePropagationModel,
        init_state.clone(),
        init_cov,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    // (2) EKF (有限差分) — PropagationModel だけで動く、空の EkfPropagationModel impl
    let mut ekf_fd = StructuredEkf::new(
        AttitudePropagationModel,
        init_state.clone(),
        init_cov,
        &0.0_f64,
    );

    // (3) EKF (手動ヤコビアン) — manifold_jacobian ヘルパーで解析的に記述
    let mut ekf_manual = StructuredEkf::new(
        AttitudePropagationManualJacobian,
        init_state,
        init_cov,
        &0.0_f64,
    );

    // --- シミュレーションループ ---
    let obs_model_fd = StarTrackerObservationModel;
    let obs_model_manual = StarTrackerObservationModel;
    let dt = 0.1;
    let mut true_q = UnitQuaternion::identity();

    for i in 0..200 {
        let time = (i + 1) as f64 * dt;

        // 真値の伝搬
        let delta_q = UnitQuaternion::new((true_omega - true_bias) * dt);
        true_q = delta_q * true_q;

        // ジャイロ測定（バイアス + ノイズ込み）
        let gyro_noise = Vector3::new(
            0.0005 * ((i * 7) as f64).sin(),
            0.0005 * ((i * 13) as f64).cos(),
            0.0005 * ((i * 19) as f64).sin(),
        );
        let gyro = GyroInputGaussian {
            angular_velocity: true_omega + gyro_noise,
            angular_velocity_covariance: Matrix3::identity() * 1e-6,
        };

        // 伝搬
        ukf.propagate(&EmptyInput, &gyro, Some(process_noise), &time).unwrap();
        ekf_fd.propagate(&EmptyInput, &gyro, Some(process_noise), &time).unwrap();
        ekf_manual.propagate(&EmptyInput, &gyro, Some(process_noise), &time).unwrap();

        // 観測更新（5ステップごと = 2Hz スタートラッカー）
        if i % 5 == 0 {
            let stt_noise = UnitQuaternion::new(Vector3::new(
                0.001 * ((i * 3) as f64).sin(),
                0.001 * ((i * 5) as f64).cos(),
                0.001 * ((i * 11) as f64).sin(),
            ));
            let measurement = AttitudeObservation {
                attitude: stt_noise * true_q,
            };
            let meas_noise = Matrix3::identity() * 0.001;

            ukf.update(&StarTrackerObservationModel, &measurement, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
            ekf_fd.update(&obs_model_fd, &measurement, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
            ekf_manual.update(&obs_model_manual, &measurement, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
        }

        // 10ステップごとに進捗表示
        if i % 50 == 49 || i == 0 {
            let err_ukf = (ukf.state().attitude * true_q.inverse()).angle();
            let err_fd = (ekf_fd.state().attitude * true_q.inverse()).angle();
            let err_man = (ekf_manual.state().attitude * true_q.inverse()).angle();
            println!(
                "t={:>5.1}s | UKF: {:.4}° | EKF(FD): {:.4}° | EKF(手動): {:.4}°",
                time,
                err_ukf.to_degrees(),
                err_fd.to_degrees(),
                err_man.to_degrees(),
            );
        }
    }

    // --- 最終結果 ---
    println!("\n=== 最終結果 (t=20s) ===");
    let q_ukf = ukf.state().attitude;
    let q_fd = ekf_fd.state().attitude;
    let q_man = ekf_manual.state().attitude;

    println!("姿勢誤差:");
    println!("  UKF:      {:.4}°", (q_ukf * true_q.inverse()).angle().to_degrees());
    println!("  EKF(FD):  {:.4}°", (q_fd * true_q.inverse()).angle().to_degrees());
    println!("  EKF(手動): {:.4}°", (q_man * true_q.inverse()).angle().to_degrees());

    println!("\nバイアス推定 (真値: [{:.4}, {:.4}, {:.4}] rad/s):",
        true_bias[0], true_bias[1], true_bias[2]);
    println!("  UKF:      {:?}", ukf.state().gyro_bias);
    println!("  EKF(FD):  {:?}", ekf_fd.state().gyro_bias);
    println!("  EKF(手動): {:?}", ekf_manual.state().gyro_bias);

    // 共分散ブロックアクセス
    let cov_blocks = AttitudeState::covariance_blocks(ukf.covariance());
    println!("\n共分散 (UKF):");
    println!("  σ_attitude = [{:.2e}, {:.2e}, {:.2e}]",
        cov_blocks.attitude_attitude()[(0,0)].sqrt(),
        cov_blocks.attitude_attitude()[(1,1)].sqrt(),
        cov_blocks.attitude_attitude()[(2,2)].sqrt());
    println!("  σ_bias     = [{:.2e}, {:.2e}, {:.2e}]",
        cov_blocks.gyro_bias_gyro_bias()[(0,0)].sqrt(),
        cov_blocks.gyro_bias_gyro_bias()[(1,1)].sqrt(),
        cov_blocks.gyro_bias_gyro_bias()[(2,2)].sqrt());
}
