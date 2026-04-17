# `cargo expand` ガイド — マクロ展開コードの確認方法

## これは何？

`#[derive(EstimationState)]` は、コンパイル時に以下のコードを自動生成します:

- `{Name}SigmaPoint` 構造体 — 接空間の表現
- `{Name}Nominal` 構造体 — 基準点の表現
- `{Name}CovarianceBlocks` 構造体 — 共分散ブロックアクセサ
- `ValueStructTrait`, `StateStructTrait` 等の trait 実装
- `SVector<f64, DIM>` との相互変換

これらの生成コードは通常のソースファイルには現れないため、IDE で「定義に飛ぶ」と
マクロ定義に飛ばされます。`cargo expand` を使えば、展開後の完全な Rust コードを
確認できます。

## インストール

```bash
cargo install cargo-expand
```

## 基本的な使い方

### ライブラリ全体の展開

```bash
cargo expand --lib
```

### 特定の example の展開

```bash
cargo expand --example attitude_estimation
```

### 特定のモジュールだけ展開

```bash
cargo expand --lib ukf          # ukf モジュールのみ
cargo expand --lib components   # components モジュールのみ
```

### ファイルに保存

```bash
cargo expand --example attitude_estimation > target/expanded_attitude.rs
```

## 具体例: AttitudeState の展開結果

以下のユーザーコード:

```rust
#[derive(EstimationState, Clone, Debug)]
struct AttitudeState {
    attitude: UnitQuaternion<f64>,
    gyro_bias: SVector<f64, 3>,
}
```

は `cargo expand` で以下に展開されます:

### 1. SigmaPoint 構造体（接空間表現）

```rust
struct AttitudeStateSigmaPoint {
    // UnitQuaternion の接空間は Vector3<f64> (axis-angle)
    attitude: <UnitQuaternion<f64> as GaussianValueType>::Sigma,  // = Vector3<f64>
    // SVector<f64, 3> の接空間はそのまま Vector3<f64>
    gyro_bias: <SVector<f64, 3> as GaussianValueType>::Sigma,     // = Vector3<f64>
}
// → 合計 6 次元 (3 + 3)
```

### 2. Nominal 構造体（基準点）

```rust
struct AttitudeStateNominal {
    // UnitQuaternion の Nominal は自身 (多様体の基準点)
    attitude: <UnitQuaternion<f64> as GaussianValueType>::Nominal,  // = UnitQuaternion<f64>
    // SVector の Nominal は空マーカー
    gyro_bias: <SVector<f64, 3> as GaussianValueType>::Nominal,     // = Vector3EmptyNominal
}
```

### 3. ValueStructTrait 実装

```rust
impl ValueStructTrait for AttitudeState {
    type SigmaStruct = AttitudeStateSigmaPoint;
    type NominalStruct = AttitudeStateNominal;

    fn algebraize(&self) -> (Self::NominalStruct, Self::SigmaStruct) {
        // 各フィールドを (Nominal, Sigma) に分解
        // attitude: (UnitQuaternion, Vector3::zeros())  ← 多様体は基準点を保持
        // gyro_bias: (EmptyNominal, gyro_bias_value)    ← ユークリッドは値がそのままSigma
    }
}
```

### 4. NominalStructTrait 実装

```rust
impl NominalStructTrait for AttitudeStateNominal {
    fn merge_sigma(&self, sigma: &AttitudeStateSigmaPoint) -> AttitudeState {
        AttitudeState {
            // attitude: exp(sigma.attitude) ⊗ self.attitude  ← 多様体の加算
            attitude: self.attitude.merge_sigma(&sigma.attitude),
            // gyro_bias: EmptyNominal + sigma.gyro_bias      ← ユークリッドの加算
            gyro_bias: self.gyro_bias.merge_sigma(&sigma.gyro_bias),
        }
    }
}
```

### 5. CovarianceBlocks

```rust
struct AttitudeStateCovarianceBlocks<'a> {
    raw: &'a SMatrix<f64, 6, 6>,
}

impl AttitudeStateCovarianceBlocks<'_> {
    fn attitude_attitude(&self) -> SMatrix<f64, 3, 3> {
        self.raw.fixed_view::<3, 3>(0, 0).clone_owned()
    }
    fn attitude_gyro_bias(&self) -> SMatrix<f64, 3, 3> {
        self.raw.fixed_view::<3, 3>(0, 3).clone_owned()
    }
    fn gyro_bias_attitude(&self) -> SMatrix<f64, 3, 3> {
        self.raw.fixed_view::<3, 3>(3, 0).clone_owned()
    }
    fn gyro_bias_gyro_bias(&self) -> SMatrix<f64, 3, 3> {
        self.raw.fixed_view::<3, 3>(3, 3).clone_owned()
    }
}
```

### 6. SVector 変換

```rust
// SigmaPoint → SVector<f64, 6>: flatten
impl From<AttitudeStateSigmaPoint> for SVector<f64, 6> {
    fn from(value: AttitudeStateSigmaPoint) -> Self {
        let mut data = [0.0_f64; 6];
        // attitude (3D) → data[0..3]
        // gyro_bias (3D) → data[3..6]
        SVector::from_row_slice(&data)
    }
}

// SVector<f64, 6> → SigmaPoint: unflatten
impl From<SVector<f64, 6>> for AttitudeStateSigmaPoint {
    fn from(vec: SVector<f64, 6>) -> Self {
        // data[0..3] → attitude (Vector3)
        // data[3..6] → gyro_bias (Vector3)
    }
}
```

## VSCode での確認方法

1. `#[derive(EstimationState)]` にカーソルを置く
2. `Ctrl+Shift+P` (コマンドパレット)
3. `rust-analyzer: Expand Macro Recursively` を選択
4. 読み取り専用バッファに展開結果が表示される

## フィールドの次元の確認

各フィールドの接空間次元は型から自動決定されます:

| フィールド型 | 接空間型 (Sigma) | 次元 |
|-------------|-----------------|------|
| `UnitQuaternion<f64>` | `Vector3<f64>` | 3 (axis-angle) |
| `Direction` | `Vector2<f64>` | 2 (接平面) |
| `SVector<f64, N>` | `SVector<f64, N>` | N |
| `Vector3<f64>` | `Vector3<f64>` | 3 |
| `f64` | `Vector1<f64>` | 1 |
| `FrameVec<F, D, P>` (sunpou) | `Vector3<f64>` | 3 |
| `Rotation<F1, F2>` (sunpou) | `Vector3<f64>` | 3 |
| `FrameDirection<F>` (sunpou) | `Vector2<f64>` | 2 |
| `Scalar<D, P>` (sunpou) | `Vector1<f64>` | 1 |

## トラブルシューティング

### `cargo expand` が動かない

```bash
# nightly toolchain が必要
rustup install nightly
# cargo-expand は nightly の -Zunpretty=expanded を使用
```

### 展開結果が長すぎる

特定の型だけ見たい場合:

```bash
# SigmaPoint 構造体だけ
cargo expand --example attitude_estimation 2>&1 | grep -A 20 "struct AttitudeStateSigmaPoint"

# CovarianceBlocks だけ
cargo expand --example attitude_estimation 2>&1 | grep -A 30 "CovarianceBlocks"

# 特定の trait impl だけ
cargo expand --example attitude_estimation 2>&1 | grep -A 30 "impl.*ValueStructTrait"
```

### rust-analyzer で補完が効かない

proc-macro-srv が停止している可能性:

1. VSCode: `Ctrl+Shift+P` → `rust-analyzer: Restart server`
2. 設定確認: `rust-analyzer.procMacro.enable` が `true` であること
