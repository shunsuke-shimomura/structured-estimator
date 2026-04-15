//! Integration with sunpou — compile-time dimensional analysis for UKF state types.
//!
//! Enables using `FrameVec<F, D, P>`, `UnitVec<D, N, P>`, and `Scalar<D, P>`
//! as fields in `#[derive(EstimationState)]` structs, providing compile-time
//! unit and frame checking for estimation models.
//!
//! # Example
//!
//! ```rust,ignore
//! use sunpou::prelude::*;
//! use sunpou::prefix::*;
//! use structured_estimator::EstimationState;
//!
//! #[derive(EstimationState, Clone, Debug)]
//! struct OrbitalState {
//!     position: FrameVec<Eci, Length, Kilo>,     // km, ECI frame
//!     velocity: FrameVec<Eci, Velocity, Kilo>,   // km/s, ECI frame
//! }
//! ```
//!
//! The propagation and observation models then get frame + unit checking:
//! - Returning `FrameVec<Body, ...>` where `FrameVec<Eci, ...>` is expected → compile error
//! - Mixing km and m in state → compile error (different prefix)

use crate::components::{GaussianNominalType, GaussianValueType};
use nalgebra::{Vector1, Vector3};

// ============================================================================
// FrameVec<F, D, P> — 3D frame-tagged vector (Euclidean, same as Vector3)
// ============================================================================

/// Nominal marker for FrameVec (zero-sized, carries frame+dim+prefix info).
pub struct FrameVecNominal<F, D, P> {
    _marker: core::marker::PhantomData<(F, D, P)>,
}

impl<F, D, P> Clone for FrameVecNominal<F, D, P> {
    fn clone(&self) -> Self { Self { _marker: core::marker::PhantomData } }
}

impl<F, D, P> Default for FrameVecNominal<F, D, P> {
    fn default() -> Self {
        Self { _marker: core::marker::PhantomData }
    }
}

impl<F, D, P> GaussianNominalType for FrameVecNominal<F, D, P> {
    type Value = sunpou::frame_vec::FrameVec<F, D, P>;
    type Sigma = Vector3<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value {
        sunpou::frame_vec::FrameVec::from_raw(nalgebra::Vector3::new(sigma[0], sigma[1], sigma[2]))
    }
}

impl<F, D, P> GaussianValueType for sunpou::frame_vec::FrameVec<F, D, P> {
    type Nominal = FrameVecNominal<F, D, P>;
    type Sigma = Vector3<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma) {
        (FrameVecNominal::default(), *self.as_raw())
    }
    fn error(&self, criteria: &Self) -> Self::Sigma {
        self.as_raw() - criteria.as_raw()
    }
}

// ============================================================================
// UnitVec<D, 3, P> — 3D frame-less vector (Euclidean, same as Vector3)
// ============================================================================

/// Nominal marker for UnitVec<D, 3, P>.
pub struct UnitVec3Nominal<D, P> {
    _marker: core::marker::PhantomData<(D, P)>,
}

impl<D, P> Clone for UnitVec3Nominal<D, P> {
    fn clone(&self) -> Self { Self { _marker: core::marker::PhantomData } }
}

impl<D, P> Default for UnitVec3Nominal<D, P> {
    fn default() -> Self {
        Self { _marker: core::marker::PhantomData }
    }
}

impl<D, P> GaussianNominalType for UnitVec3Nominal<D, P> {
    type Value = sunpou::unit_vec::UnitVec<D, 3, P>;
    type Sigma = Vector3<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value {
        sunpou::unit_vec::UnitVec::from_raw(nalgebra::SVector::from([sigma[0], sigma[1], sigma[2]]))
    }
}

impl<D, P> GaussianValueType for sunpou::unit_vec::UnitVec<D, 3, P> {
    type Nominal = UnitVec3Nominal<D, P>;
    type Sigma = Vector3<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma) {
        let raw = self.as_raw();
        (
            UnitVec3Nominal::default(),
            Vector3::new(raw[0], raw[1], raw[2]),
        )
    }
    fn error(&self, criteria: &Self) -> Self::Sigma {
        let s = self.as_raw();
        let c = criteria.as_raw();
        Vector3::new(s[0] - c[0], s[1] - c[1], s[2] - c[2])
    }
}

// ============================================================================
// Scalar<D, P> — unit-tagged scalar (Euclidean, same as f64)
// ============================================================================

/// Nominal marker for Scalar<D, P>.
pub struct ScalarNominal<D, P> {
    _marker: core::marker::PhantomData<(D, P)>,
}

impl<D, P> Clone for ScalarNominal<D, P> {
    fn clone(&self) -> Self { Self { _marker: core::marker::PhantomData } }
}

impl<D, P> Default for ScalarNominal<D, P> {
    fn default() -> Self {
        Self { _marker: core::marker::PhantomData }
    }
}

impl<D, P> GaussianNominalType for ScalarNominal<D, P> {
    type Value = sunpou::scalar::Scalar<D, P>;
    type Sigma = Vector1<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value {
        sunpou::scalar::Scalar::from_raw(sigma[0])
    }
}

impl<D, P> GaussianValueType for sunpou::scalar::Scalar<D, P> {
    type Nominal = ScalarNominal<D, P>;
    type Sigma = Vector1<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma) {
        (ScalarNominal::default(), Vector1::new(self.into_raw()))
    }
    fn error(&self, criteria: &Self) -> Self::Sigma {
        Vector1::new(self.into_raw() - criteria.into_raw())
    }
}

// ============================================================================
// Rotation<F1, F2> — frame-tagged quaternion (3D axis-angle error manifold)
// ============================================================================

/// Nominal type for Rotation<F1, F2>: stores the quaternion reference point.
pub struct RotationNominal<F1, F2> {
    quat: nalgebra::UnitQuaternion<f64>,
    _marker: core::marker::PhantomData<(F1, F2)>,
}

impl<F1, F2> Clone for RotationNominal<F1, F2> {
    fn clone(&self) -> Self { Self { quat: self.quat, _marker: core::marker::PhantomData } }
}

impl<F1, F2> Default for RotationNominal<F1, F2> {
    fn default() -> Self {
        Self { quat: nalgebra::UnitQuaternion::identity(), _marker: core::marker::PhantomData }
    }
}

impl<F1, F2> GaussianNominalType for RotationNominal<F1, F2> {
    type Value = sunpou::rotation::Rotation<F1, F2>;
    type Sigma = Vector3<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value {
        let delta_q = nalgebra::UnitQuaternion::new(*sigma);
        sunpou::rotation::Rotation::from_raw(delta_q * self.quat)
    }
}

impl<F1, F2> GaussianValueType for sunpou::rotation::Rotation<F1, F2> {
    type Nominal = RotationNominal<F1, F2>;
    type Sigma = Vector3<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma) {
        (
            RotationNominal { quat: *self.as_raw(), _marker: core::marker::PhantomData },
            Vector3::zeros(),
        )
    }
    fn error(&self, criteria: &Self) -> Self::Sigma {
        let delta_q = *self.as_raw() * criteria.as_raw().inverse();
        delta_q.scaled_axis()
    }
}

// ============================================================================
// FrameDirection<F> — frame-tagged direction (2D tangent-space error manifold)
// ============================================================================

/// Nominal type for FrameDirection<F>: stores the direction reference point.
pub struct FrameDirectionNominal<F> {
    dir: sunpou::frame_direction::FrameDirection<F>,
}

impl<F> Clone for FrameDirectionNominal<F> {
    fn clone(&self) -> Self { Self { dir: self.dir.clone() } }
}

impl<F> Default for FrameDirectionNominal<F> {
    fn default() -> Self {
        Self { dir: sunpou::frame_direction::FrameDirection::default() }
    }
}

impl<F> GaussianNominalType for FrameDirectionNominal<F> {
    type Value = sunpou::frame_direction::FrameDirection<F>;
    type Sigma = nalgebra::Vector2<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value {
        self.dir.perturb(sigma)
    }
}

impl<F> GaussianValueType for sunpou::frame_direction::FrameDirection<F> {
    type Nominal = FrameDirectionNominal<F>;
    type Sigma = nalgebra::Vector2<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma) {
        (
            FrameDirectionNominal { dir: self.clone() },
            nalgebra::Vector2::zeros(),
        )
    }
    fn error(&self, criteria: &Self) -> Self::Sigma {
        self.error_from(criteria)
    }
}

// ============================================================================
// GaussianSigmaType is already implemented for Vector1-6 in components.rs,
// so no additional impls needed. The sigma representation is always a raw
// nalgebra vector — sunpou's type safety is on the structured side.
// ============================================================================
