# EKF 設計 — Jacobian 計算の3層アプローチ

## 概要

UKF に加えて EKF (Extended Kalman Filter) を実装する。EKF には伝搬・観測モデルの
Jacobian が必要。3つの Jacobian 計算手法を提供し、ユーザーが選択可能にする。

## Jacobian 計算手法

### 1. 有限差分 (Finite Difference)

既存の `PropagationModel::propagate()` をそのまま利用。接空間の各次元に ε 摂動を
加え、出力の接空間差分から Jacobian を数値近似する。

```
J[i][j] ≈ (f(x + εe_j) - f(x - εe_j)) / (2ε)
```

**利点**: 既存モデルがそのまま使える。sunpou 型完全対応。no_std。
**欠点**: O(ε²) の数値誤差。2n 回のモデル評価が必要（n = 状態次元）。

実装: 既存の `merge_sigma()` と `error_from()` で接空間の摂動/射影を行う。
UKF のシグマ点と同じインフラを再利用。

### 2. num-dual 自動微分 (Forward-mode AD)

`std` feature で有効化。ユーザーは `SVector<T: RealField, DIM>` で
数値コアを記述。`num_dual::jacobian()` で機械精度の Jacobian を自動計算。

```rust
trait AutoDiffModel {
    const STATE_DIM: usize;
    fn propagate_flat<T: nalgebra::RealField + Copy>(
        &self, state: &SVector<T, { Self::STATE_DIM }>, dt: T,
    ) -> SVector<T, { Self::STATE_DIM }>;
}
```

**利点**: 機械精度。手動 Jacobian 不要。
**欠点**: std 必要。sunpou 型は直接使えない（flat vector で記述）。
  ただし EKF framework が structured state ↔ flat vector の変換を行う。

### 3. 手動 Jacobian

ユーザーが Jacobian 関数を直接提供。

```rust
trait ManualJacobianModel {
    fn propagate(&self, state: &State, ...) -> State;
    fn jacobian(&self, state: &State, ...) -> SMatrix<f64, SDIM, SDIM>;
}
```

**利点**: 最高性能。数値誤差なし。
**欠点**: ユーザーの実装負担。数式の手動微分が必要。

## EKF の predict/update アルゴリズム

```
Predict:
  x̂⁻ = f(x̂⁺, u)
  F = ∂f/∂x |_{x̂⁺}          ← Jacobian (3手法のいずれか)
  P⁻ = F P⁺ Fᵀ + Q

Update:
  ŷ = h(x̂⁻)
  H = ∂h/∂x |_{x̂⁻}          ← Jacobian
  S = H P⁻ Hᵀ + R
  K = P⁻ Hᵀ S⁻¹
  δx = K (z - ŷ)
  x̂⁺ = x̂⁻ ⊕ δx              ← 接空間更新 (merge_sigma)
  P⁺ = (I - K H) P⁻
```

`⊕` は接空間での加算（多様体対応: quaternion は exp map、direction は球面摂動）。
既存の `merge_sigma()` / `error_from()` がこの役割を果たす。

## 接空間 Jacobian の扱い

状態が多様体成分（quaternion, direction）を含む場合、Jacobian は接空間に対して
定義される。有限差分では自動的に接空間 Jacobian が得られる
（merge_sigma で摂動、error_from で射影するため）。

## 実装順序

1. 有限差分 Jacobian ユーティリティ（接空間ベース）
2. EKF struct (predict + update)
3. num-dual 自動微分統合 (feature-gated)
4. テスト: UKF と EKF の推定結果の一致検証
