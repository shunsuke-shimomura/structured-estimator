# sunpou 多様体型の UKF 統合 — 設計判断

## 判断内容

sunpou の `Rotation<F1, F2>` と `FrameDirection<F>` に `GaussianValueType` を
実装し、UKF 状態フィールドとして直接使えるようにする。

## 接空間の定義

| 型 | 多様体の次元 | Sigma (接空間) | merge_sigma | error |
|---|---|---|---|---|
| `Rotation<F1, F2>` | 3 (SO(3)) | `Vector3<f64>` (axis-angle) | `δq * q_nominal` | `(q * q_criteria⁻¹).scaled_axis()` |
| `FrameDirection<F>` | 2 (S²) | `Vector2<f64>` (接平面) | ONB basis × σ → 回転適用 | 球面距離を接平面に射影 |
| `FrameVec<F, D, P>` | 3 (ℝ³) | `Vector3<f64>` | `nominal + sigma` | `self - criteria` |
| `Scalar<D, P>` | 1 (ℝ¹) | `Vector1<f64>` | `nominal + sigma` | `self - criteria` |

## フレーム安全性

`Rotation<Body, Eci>` を状態に持つ場合、PropagationModel の型シグネチャで
フレームが強制される:

```rust
#[derive(EstimationState)]
struct AttitudeState {
    attitude: Rotation<Body, Eci>,     // Body→ECI rotation
    gyro_bias: FrameVec<Body, AngularVelocity>,  // Body frame bias
}
```

`Rotation<Body, Ecef>` と `Rotation<Body, Eci>` は異なる型なので、
取り違えはコンパイルエラーになる。

## 選択肢

### A. sunpou 型に GaussianValueType を実装 (採用)

structured-estimator の sunpou feature 内に impl を配置。

- **Pros**: sunpou 型がそのまま UKF 状態になる。フレーム安全性が伝搬。
  derive マクロの変更不要。
- **Cons**: structured-estimator が sunpou の内部 API (as_raw, from_raw) に依存。

### B. sunpou 側に GaussianValueType を実装

- **Pros**: 各型の近くに impl がある。
- **Cons**: sunpou が structured-estimator に依存 → 循環依存。

### C. ラッパー型を作る

- **Cons**: 冗長。sunpou 型の利点が薄れる。

## Angle<Rad/Deg> について

sunpou に `Angle<Rad>` / `Angle<Deg>` 型を追加。rad と deg の混在は
コンパイルエラー。UKF の sigma (axis-angle) は常に rad なので、
`Angle<Rad>` として型安全に扱える。

## 結論

A を採用。derive マクロ変更なしで、sunpou の多様体型がフレーム安全な
UKF 状態として使える。
