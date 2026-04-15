# 型付き共分散ブロック — 設計判断

## 判断内容

`EstimationState` derive マクロが `{Name}CovarianceBlocks` 構造体を自動生成し、
UKF の生の `SMatrix<f64, DIM, DIM>` から各フィールドペアの部分行列を型安全に
取り出せるようにする。

## 背景

UKF の共分散行列は `SMatrix<f64, 6, 6>` のようなフラットな行列だが、物理的には
ブロック構造を持つ:

```
State = [position(km), velocity(km/s)]
P = | P_rr [km²]        P_rv [km·(km/s)]    |
    | P_vr [(km/s)·km]  P_vv [(km/s)²]      |
```

ユーザーが手動で `fixed_view::<3,3>(0,0)` のようなインデックスを書くのは
エラーが起きやすい。

## 設計

### マクロが生成するもの

```rust
#[derive(EstimationState)]
struct OrbitalState {
    position: FrameVec<Eci, Length, Kilo>,  // dim=3
    velocity: FrameVec<Eci, Velocity, Kilo>, // dim=3
}
// マクロが自動生成 ↓
struct OrbitalStateCovarianceBlocks<'a> {
    raw: &'a SMatrix<f64, 6, 6>,
}
impl OrbitalStateCovarianceBlocks<'_> {
    fn position_position(&self) -> SMatrix<f64, 3, 3> { /* (0,0) */ }
    fn position_velocity(&self) -> SMatrix<f64, 3, 3> { /* (0,3) */ }
    fn velocity_position(&self) -> SMatrix<f64, 3, 3> { /* (3,0) */ }
    fn velocity_velocity(&self) -> SMatrix<f64, 3, 3> { /* (3,3) */ }
}
impl OrbitalState {
    fn covariance_blocks(cov: &SMatrix<f64, 6, 6>) -> OrbitalStateCovarianceBlocks<'_> { ... }
}
```

### sunpou 型への変換

ブロックアクセサは `SMatrix<f64, R, C>` を返す。sunpou 型が欲しい場合は
ユーザーが `FrameElemMat::from_raw(block)` で包む:

```rust
let blocks = OrbitalState::covariance_blocks(ukf.covariance());
let p_rr: FrameElemMat<Eci, Area, 3, 3, Mega> =
    FrameElemMat::from_raw(blocks.position_position());
```

## 選択肢

### A. マクロで SMatrix ブロックアクセサを生成 (採用)

- **Pros**: sunpou 非依存。マクロが sunpou を知る必要がない。
  sunpou を使わないユーザーも恩恵を受ける。
  sunpou 型への変換は `from_raw` 一行。
- **Cons**: 完全に型推論で sunpou 型が得られるわけではない（from_raw が必要）。

### B. マクロで sunpou 型のブロックを直接返す

- **Pros**: ブロックの型が自動的に `FrameElemMat<Eci, Area, 3, 3, Mega>` になる。
- **Cons**: マクロが sunpou の型システム（Dim, Prefix）を理解する必要がある。
  フィールド型 `FrameVec<Eci, Length, Kilo>` からジェネリクスを抽出するパーサーが必要。
  非常に複雑で脆弱。sunpou を使わないユーザーには不要。

### C. マクロ生成なし（手動アクセス）

- **Pros**: 実装コストゼロ。
- **Cons**: ユーザーが `fixed_view::<3,3>(0,0)` を手書き。オフセットの間違いが多発。

## 結論

A を採用。sunpou 非依存でブロックアクセサを生成し、sunpou 型への変換は
ユーザー側の `from_raw` に委ねる。将来、マクロが sunpou 型を解析できるように
なれば B に移行可能。
