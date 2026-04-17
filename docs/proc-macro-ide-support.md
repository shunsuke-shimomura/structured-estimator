# proc macro 生成コードの IDE 可視性 — 調査結果

## 問題

`#[derive(EstimationState)]` が生成する構造体・trait impl に対して、
IDE の「定義に飛ぶ」がマクロ定義に飛ばされ、生成コードの確認が困難。

## 現状のデファクト

### 1. rust-analyzer の proc macro 展開（最も一般的）

rust-analyzer は `proc-macro-srv` を介して proc macro をバックグラウンドで
展開している。**serde, tokio, bevy 等の主要クレートはこの仕組みに依存**。

- 2024-2025年で安定性が大幅に改善
- derive macro で生成された型・メソッドへの **補完・ホバー・定義ジャンプ** が
  基本的に動作する
- ただし **ジャンプ先がマクロ定義になる** 問題は依然として存在
  （展開後のコードに対応するソース位置がマクロ自体になるため）

**対処法**: VSCode のコマンドパレットで
`rust-analyzer: Expand Macro Recursively` を実行すると、
カーソル位置のマクロ展開結果が読み取り専用バッファに表示される。

### 2. `cargo expand`（デバッグ用標準ツール）

```bash
cargo install cargo-expand
cargo expand --lib  # ライブラリ全体の展開
cargo expand my_module  # 特定モジュール
```

nightly の `-Zunpretty=expanded` を使用。**マクロ展開後の完全な Rust コード**
が標準出力に出る。ファイルにリダイレクトして読める:

```bash
cargo expand --lib > target/expanded.rs
```

### 3. `build.rs` + `include!` パターン（IDE 最良体験）

**tonic/prost** (protobuf コード生成) が代表例:

```rust
// build.rs
fn main() {
    tonic_build::compile_protos("proto/service.proto").unwrap();
    // → target/build/.../out/service.rs が生成される
}

// src/lib.rs
pub mod service {
    tonic::include_proto!("service");
    // = include!(concat!(env!("OUT_DIR"), "/service.rs"));
}
```

**生成された .rs ファイルが実体として存在する**ため、rust-analyzer が
通常のソースファイルとしてインデックスし、定義ジャンプ・補完・ホバーが
完全に動作する。**IDE 体験として最良**。

### 4. 各クレートの対応状況

| クレート | 方式 | IDE 定義ジャンプ |
|---------|------|---------------|
| serde | derive macro | ○ (rust-analyzer 展開) |
| tokio | attribute macro | ○ |
| diesel | declarative macro (`table!`) | △ (macro 内 DSL は弱い) |
| tonic/prost | **build.rs + include!** | **◎ (完全動作)** |
| sqlx | proc macro (compile-time SQL) | △ (部分的) |
| bevy | derive macro | ○ (複雑な型推論で弱い場合あり) |

## 推奨

### structured-estimator の場合

**現状維持 (derive macro) + ドキュメント改善** が最もバランスが良い:

1. **rust-analyzer がほぼ動作する** — 生成された SigmaPoint 型、
   CovarianceBlocks メソッド等は補完・ホバーが効く
2. **「定義に飛ぶ」がマクロに飛ぶ問題** — `Expand Macro Recursively` で対処可能
3. **`cargo expand` ガイド** — ユーザーに展開コードの確認方法を案内

### build.rs 移行が正当化されるケース

- マクロ生成コードが非常に大きい（100行+）
- ユーザーが生成コードを頻繁に読む/デバッグする必要がある
- IDE 体験が最優先（tonic/prost のユースケース）

structured-estimator の生成コードは ~50行/struct で、頻繁に読む必要はないため、
derive macro のままで十分。ただし `cargo expand` の使い方をドキュメントすべき。

## 参考: ユーザー向けガイド

```bash
# EstimationState マクロの展開結果を確認
cargo expand --lib | grep -A 100 "struct AttitudeStateSigmaPoint"

# VSCode: カーソルを #[derive(EstimationState)] に置いて
# Ctrl+Shift+P → "rust-analyzer: Expand Macro Recursively"
```
