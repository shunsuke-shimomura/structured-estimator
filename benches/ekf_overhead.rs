//! Benchmark: EKF (finite-diff Jacobian) vs UKF vs raw model evaluation.
//!
//! Measures the overhead of Jacobian computation for EKF relative to
//! a single model evaluation and UKF's sigma-point approach.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use structured_estimator::{
    EstimationOutputStruct, EstimationState,
    ekf::ExtendedKalmanFilter,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};

// ---- 6D orbital state (same as test models) ----

#[derive(EstimationState, Clone, Debug)]
struct OrbState {
    position: SVector<f64, 3>,
    velocity: SVector<f64, 3>,
}

struct OrbProp;

impl PropagationModel for OrbProp {
    type State = OrbState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, s: &OrbState, _d: &EmptyInput, _g: &EmptyInput, _t: &f64, dt: &f64) -> OrbState {
        OrbState {
            position: s.position + s.velocity * *dt,
            velocity: s.velocity,
        }
    }
}

#[derive(EstimationOutputStruct, Debug, Clone)]
struct PosObs {
    position: SVector<f64, 3>,
}

struct PosObsModel;

impl ObservationModel for PosObsModel {
    type State = OrbState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = f64;
    type Observation = PosObs;
    fn predict(&self, s: &OrbState, _d: &EmptyInput, _g: &EmptyInput, _t: &f64) -> PosObs {
        PosObs { position: s.position }
    }
}

fn make_state() -> OrbState {
    OrbState {
        position: Vector3::new(7000e3, 0.0, 0.0),
        velocity: Vector3::new(0.0, 7.5e3, 0.0),
    }
}

fn bench_raw_propagation(c: &mut Criterion) {
    let state = make_state();
    c.bench_function("raw_propagation", |b| {
        b.iter(|| {
            OrbProp.propagate(black_box(&state), &EmptyInput, &EmptyInput, &0.0, &60.0)
        })
    });
}

fn bench_ekf_propagate(c: &mut Criterion) {
    c.bench_function("ekf_propagate_6d", |b| {
        b.iter_batched(
            || {
                ExtendedKalmanFilter::<OrbState, f64, f64, OrbProp, EmptyInput, 6>::new(
                    OrbProp, make_state(),
                    SMatrix::<f64, 6, 6>::identity() * 100.0,
                    &0.0, 1e-7,
                )
            },
            |mut ekf| {
                ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &60.0).unwrap();
                ekf
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_ukf_propagate(c: &mut Criterion) {
    c.bench_function("ukf_propagate_6d", |b| {
        b.iter_batched(
            || {
                UnscentedKalmanFilter::<OrbState, f64, f64, OrbProp, EmptyInput, EmptyInput, 6, 0>::new(
                    OrbProp, make_state(),
                    SMatrix::<f64, 6, 6>::identity() * 100.0,
                    &0.0, UKFParameters::new(1e-3, 2.0, 0.0),
                )
            },
            |mut ukf| {
                ukf.propagate(&EmptyInput, &EmptyInput, None, &60.0).unwrap();
                ukf
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_ekf_update(c: &mut Criterion) {
    c.bench_function("ekf_update_6d", |b| {
        b.iter_batched(
            || {
                let mut ekf = ExtendedKalmanFilter::<OrbState, f64, f64, OrbProp, EmptyInput, 6>::new(
                    OrbProp, make_state(),
                    SMatrix::<f64, 6, 6>::identity() * 100.0,
                    &0.0, 1e-7,
                );
                ekf.propagate::<EmptyInput, 0>(&EmptyInput, &EmptyInput, None, &60.0).unwrap();
                ekf
            },
            |mut ekf| {
                let meas = PosObs { position: Vector3::new(7000e3, 7.5e3 * 60.0, 0.0) };
                ekf.update(&PosObsModel, &meas, &EmptyInput, &EmptyInput, &60.0,
                    Matrix3::identity() * 100.0).unwrap();
                ekf
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_ukf_update(c: &mut Criterion) {
    c.bench_function("ukf_update_6d", |b| {
        b.iter_batched(
            || {
                let mut ukf = UnscentedKalmanFilter::<OrbState, f64, f64, OrbProp, EmptyInput, EmptyInput, 6, 0>::new(
                    OrbProp, make_state(),
                    SMatrix::<f64, 6, 6>::identity() * 100.0,
                    &0.0, UKFParameters::new(1e-3, 2.0, 0.0),
                );
                ukf.propagate(&EmptyInput, &EmptyInput, None, &60.0).unwrap();
                ukf
            },
            |mut ukf| {
                let meas = PosObs { position: Vector3::new(7000e3, 7.5e3 * 60.0, 0.0) };
                ukf.update(&PosObsModel, &meas, &EmptyInput, &EmptyInput, &60.0,
                    Matrix3::identity() * 100.0).unwrap();
                ukf
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_finite_diff_jacobian_standalone(c: &mut Criterion) {
    use structured_estimator::ekf::finite_difference_jacobian;
    let state = make_state();

    c.bench_function("finite_diff_jacobian_6d", |b| {
        b.iter(|| {
            finite_difference_jacobian::<OrbState, OrbState, _, 6, 6>(
                black_box(&state),
                &|s| OrbProp.propagate(s, &EmptyInput, &EmptyInput, &0.0, &60.0),
                1e-7,
            )
        })
    });
}

fn bench_hand_coded_jacobian(c: &mut Criterion) {
    // What a hand-coded Jacobian looks like: just fill in the matrix directly
    c.bench_function("hand_coded_jacobian_6d", |b| {
        let dt = black_box(60.0);
        b.iter(|| {
            let mut j = SMatrix::<f64, 6, 6>::identity();
            j[(0, 3)] = dt;
            j[(1, 4)] = dt;
            j[(2, 5)] = dt;
            j
        })
    });
}

criterion_group!(
    benches,
    bench_raw_propagation,
    bench_ekf_propagate,
    bench_ukf_propagate,
    bench_ekf_update,
    bench_ukf_update,
    bench_finite_diff_jacobian_standalone,
    bench_hand_coded_jacobian,
);
criterion_main!(benches);
