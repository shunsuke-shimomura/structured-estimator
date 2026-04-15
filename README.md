# structured-estimator

Structured Unscented Kalman Filter (UKF) library for Rust.

## Features

- **Structured state types**: Compose state from heterogeneous components (vectors, quaternions, directions) using derive macros
- **Manifold-aware**: UnitQuaternion uses 3D error manifold, Direction uses 2D error on unit sphere
- **Generic UKF**: Configurable sigma point parameters (α, β, κ)
- **Propagation + Update**: Full predict-update cycle with cross-covariance
- **Gaussian input support**: Stochastic inputs in both propagation and observation models

## Quick Example

```rust
use structured_estimator::{EstimationState, EstimationOutputStruct, EstimationGaussianInput};
use structured_estimator::ukf::{PropagationModel, ObservationModel, UnscentedKalmanFilter, UKFParameters};
use structured_estimator::components::Direction;
use nalgebra::{Vector3, SVector};

#[derive(EstimationState, Clone, Debug)]
struct MyState {
    mag_dir: Direction,       // 2D error manifold
    gyro_bias: SVector<f64, 3>, // 3D vector
}

// Define propagation and observation models, then run UKF...
```

## License

MIT
