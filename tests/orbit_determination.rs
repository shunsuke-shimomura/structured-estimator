//! Orbit determination tests using UKF with ECI position+velocity state.
//!
//! Based on s5e/c5a orbit determination with Keplerian propagation.
//! State: [position (3D), velocity (3D)] = 6 dims.

use nalgebra::{ComplexField, Matrix3, SVector, Vector3, Vector6};
use structured_estimator::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};

const GME: f64 = 3.986004418e14; // m³/s²

// ---- Kepler solver (from c5a) ----

fn kepler_eccentric_anomaly(m: f64, e: f64) -> f64 {
    let mut large_e = if e < 0.8 { m } else { core::f64::consts::PI };
    for _ in 0..30 {
        let f = large_e - e * large_e.sin() - m;
        let fp = 1.0 - e * large_e.cos();
        let d = f / fp;
        large_e -= d;
        if d.abs() < 1e-14 { break; }
    }
    large_e
}

fn true_anomaly_from_e(large_e: f64, e: f64) -> f64 {
    let s = ((1.0 + e).sqrt()) * (large_e * 0.5).sin();
    let c = ((1.0 - e).sqrt()) * (large_e * 0.5).cos();
    2.0 * s.atan2(c)
}

fn eccentric_anomaly_from_true(nu: f64, e: f64) -> f64 {
    let s = ((1.0 - e).sqrt()) * (0.5 * nu).sin();
    let c = ((1.0 + e).sqrt()) * (0.5 * nu).cos();
    2.0 * s.atan2(c)
}

// ---- Equinoctial orbit (from c5a) ----

#[derive(Debug, Clone)]
struct EquinoctialOrbit {
    p: f64, f: f64, g: f64, h: f64, k: f64, l: f64,
}

impl EquinoctialOrbit {
    fn propagate_kepler(&self, dt: f64) -> Self {
        let e = (self.f * self.f + self.g * self.g).try_sqrt().unwrap();
        let psi = self.g.atan2(self.f);
        let nu = (self.l - psi).rem_euclid(core::f64::consts::TAU);

        let a = self.p / (1.0 - e * e);
        let n = (GME / (a * a * a)).try_sqrt().unwrap();
        let large_e = eccentric_anomaly_from_true(nu, e);
        let large_m = (large_e - e * large_e.sin()).rem_euclid(core::f64::consts::TAU);
        let propagated_m = (large_m + n * dt).rem_euclid(core::f64::consts::TAU);
        let propagated_e = kepler_eccentric_anomaly(propagated_m, e);
        let propagated_nu = true_anomaly_from_e(propagated_e, e);
        let propagated_l = (psi + propagated_nu).rem_euclid(core::f64::consts::TAU);

        EquinoctialOrbit { p: self.p, f: self.f, g: self.g, h: self.h, k: self.k, l: propagated_l }
    }

    fn to_eci(&self) -> (SVector<f64, 3>, SVector<f64, 3>) {
        let (p, f, g, h, k, l) = (self.p, self.f, self.g, self.h, self.k, self.l);
        let (s_l, c_l) = (l.sin(), l.cos());
        let w = 1.0 + f * c_l + g * s_l;
        let r = p / w;
        let smp = (GME / p).sqrt();
        let (kk, hh) = (k * k, h * h);
        let s2 = 1.0 + hh + kk;
        let tkh = 2.0 * k * h;
        let fhat = SVector::<f64, 3>::new(1.0 - kk + hh, tkh, -2.0 * k) / s2;
        let ghat = SVector::<f64, 3>::new(tkh, 1.0 + kk - hh, 2.0 * h) / s2;
        let position = r * c_l * fhat + r * s_l * ghat;
        let velocity = -smp * (g + s_l) * fhat + smp * (f + c_l) * ghat;
        (position, velocity)
    }

    fn from_eci(r: SVector<f64, 3>, v: SVector<f64, 3>) -> Self {
        let rmag = r.norm();
        let rdv = r.dot(&v);
        let rhat = r / rmag;
        let hvec = r.cross(&v);
        let hmag = hvec.norm();
        let hhat = hvec / hmag;
        let p = hmag * hmag / GME;
        let denom = 1.0 + hhat[2];
        let k = hhat[0] / denom;
        let h = -hhat[1] / denom;
        let vhat = (rmag * v - rdv * rhat) / hmag;
        let ecc = v.cross(&hvec) / GME - rhat;
        let (kk, hh) = (k * k, h * h);
        let s2 = 1.0 + hh + kk;
        let tkh = 2.0 * k * h;
        let fhat = SVector::<f64, 3>::new(1.0 - kk + hh, tkh, -2.0 * k) / s2;
        let ghat = SVector::<f64, 3>::new(tkh, 1.0 + kk - hh, 2.0 * h) / s2;
        let f = ecc.dot(&fhat);
        let g = ecc.dot(&ghat);
        let l = (rhat[1] - vhat[0]).atan2(rhat[0] + vhat[1]).rem_euclid(core::f64::consts::TAU);
        EquinoctialOrbit { p, f, g, h, k, l }
    }
}

// ---- UKF models ----

#[derive(Debug, Clone, EstimationState)]
struct OrbitalState {
    position: SVector<f64, 3>,
    velocity: SVector<f64, 3>,
}

#[derive(Debug, Clone, EstimationGaussianInput)]
struct OrbitProcessNoise {
    process_noise: SVector<f64, 6>,
}

struct KeplerianPropagation;

impl PropagationModel for KeplerianPropagation {
    type State = OrbitalState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = OrbitProcessNoise;
    type Time = f64;
    type Dt = f64;
    fn propagate(&self, state: &Self::State, _det: &Self::DeterministicInput, gi: &Self::GaussianInput, _t: &Self::Time, dt: &Self::Dt) -> Self::State {
        let orb = EquinoctialOrbit::from_eci(state.position, state.velocity);
        let prop = orb.propagate_kepler(*dt);
        let prop_noisy = EquinoctialOrbit {
            p: prop.p + gi.process_noise[0],
            f: prop.f + gi.process_noise[1],
            g: prop.g + gi.process_noise[2],
            h: prop.h + gi.process_noise[3],
            k: prop.k + gi.process_noise[4],
            l: prop.l + gi.process_noise[5],
        };
        let (pos, vel) = prop_noisy.to_eci();
        OrbitalState { position: pos, velocity: vel }
    }
}

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
    fn predict(&self, state: &Self::State, _det: &Self::DeterministicInput, _gi: &Self::GaussianInput, _t: &Self::Time) -> Self::Observation {
        GnssObservation { position: state.position, velocity: state.velocity }
    }
}

// ---- Tests ----

#[test]
fn test_keplerian_propagation_circular() {
    // LEO circular orbit: 7000 km altitude
    let r0 = SVector::<f64, 3>::new(7000e3, 0.0, 0.0);
    let v_circ = (GME / 7000e3).sqrt();
    let v0 = SVector::<f64, 3>::new(0.0, v_circ, 0.0);

    let orb = EquinoctialOrbit::from_eci(r0, v0);
    let period = 2.0 * core::f64::consts::PI * (7000e3_f64.powi(3) / GME).sqrt();

    // After one full orbit, should return to start
    let prop = orb.propagate_kepler(period);
    let (r1, v1) = prop.to_eci();
    assert!((r1 - r0).norm() < 1.0, "Position error after full orbit: {} m", (r1 - r0).norm());
    assert!((v1 - v0).norm() < 0.01, "Velocity error after full orbit: {} m/s", (v1 - v0).norm());
}

#[test]
fn test_ukf_orbit_initialization() {
    let r0 = SVector::<f64, 3>::new(7000e3, 0.0, 0.0);
    let v0 = SVector::<f64, 3>::new(0.0, (GME / 7000e3).sqrt(), 0.0);

    let ukf: UnscentedKalmanFilter<
        OrbitalState, f64, f64, KeplerianPropagation, EmptyInput,
        OrbitProcessNoiseGaussian, 6, 6,
    > = UnscentedKalmanFilter::new(
        KeplerianPropagation,
        OrbitalState { position: r0, velocity: v0 },
        nalgebra::SMatrix::<f64, 6, 6>::identity() * 1e3,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    assert!((ukf.state().position - r0).norm() < 1e-10);
    assert!((ukf.state().velocity - v0).norm() < 1e-10);
}

#[test]
fn test_ukf_orbit_convergence() {
    let r0_true = SVector::<f64, 3>::new(7000e3, 0.0, 0.0);
    let v_circ = (GME / 7000e3).sqrt();
    let v0_true = SVector::<f64, 3>::new(0.0, v_circ, 0.0);

    // Initial estimate with 1 km position error
    let r0_est = r0_true + SVector::<f64, 3>::new(1000.0, 0.0, 0.0);
    let v0_est = v0_true + SVector::<f64, 3>::new(0.0, 1.0, 0.0);

    let mut ukf: UnscentedKalmanFilter<
        OrbitalState, f64, f64, KeplerianPropagation, EmptyInput,
        OrbitProcessNoiseGaussian, 6, 6,
    > = UnscentedKalmanFilter::new(
        KeplerianPropagation,
        OrbitalState { position: r0_est, velocity: v0_est },
        nalgebra::SMatrix::<f64, 6, 6>::identity() * 1e6,
        &0.0_f64,
        UKFParameters::new(1e-3, 2.0, 0.0),
    );

    let obs_model = GnssObservationModel;
    let true_orb = EquinoctialOrbit::from_eci(r0_true, v0_true);
    let dt = 60.0; // 1-minute intervals

    for i in 0..20 {
        let time = (i + 1) as f64 * dt;

        // Propagate
        let proc_noise = OrbitProcessNoiseGaussian {
            process_noise: Vector6::zeros(),
            process_noise_covariance: nalgebra::SMatrix::<f64, 6, 6>::identity() * 1e-4,
        };
        ukf.propagate(&EmptyInput, &proc_noise, None, &time).unwrap();

        // True state
        let true_prop = true_orb.propagate_kepler(time);
        let (true_pos, true_vel) = true_prop.to_eci();

        // GNSS measurement with 10m noise
        let noise_pos = Vector3::new(
            10.0 * ((i * 7) as f64).sin(),
            10.0 * ((i * 13) as f64).cos(),
            10.0 * ((i * 19) as f64).sin(),
        );
        let measurement = GnssObservation {
            position: true_pos + noise_pos,
            velocity: true_vel,
        };

        let meas_noise = {
            let mut cov = nalgebra::SMatrix::<f64, 6, 6>::zeros();
            cov.fixed_view_mut::<3, 3>(0, 0).copy_from(&(Matrix3::identity() * 100.0)); // 10m std
            cov.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * 0.01));
            cov
        };
        ukf.update(&obs_model, &measurement, &EmptyInput, &EmptyInput, &time, meas_noise).unwrap();
    }

    let final_true = true_orb.propagate_kepler(20.0 * dt);
    let (true_pos_final, _) = final_true.to_eci();
    let pos_error = (ukf.state().position - true_pos_final).norm();
    assert!(pos_error < 100.0, "Position should converge: error = {} m", pos_error);
}
