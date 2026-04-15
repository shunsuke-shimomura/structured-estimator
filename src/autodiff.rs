//! Automatic differentiation for EKF Jacobians via num-dual.
//!
//! Provides `autodiff_jacobian()` which computes exact Jacobians of functions
//! `f: R^N → R^M` using forward-mode dual numbers.
//!
//! # Usage
//!
//! ```rust,ignore
//! use structured_estimator::autodiff::autodiff_jacobian;
//!
//! let state = SVector::from([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]);
//! let (value, jac) = autodiff_jacobian(
//!     |x| {
//!         // x is SVector<DualSVec64<6>, 6> — use nalgebra ops as usual
//!         let pos = x.fixed_rows::<3>(0).clone_owned();
//!         let vel = x.fixed_rows::<3>(3).clone_owned();
//!         let dt = DualSVec64::<6>::from(60.0);
//!         // ... your model here
//!     },
//!     state,
//! );
//! ```

#[cfg(feature = "autodiff")]
pub use num_dual;

/// Compute the Jacobian of `f: R^N → R^M` using forward-mode automatic differentiation.
///
/// Returns `(f(x), J)` where J is the M×N Jacobian matrix.
///
/// The function `f` receives dual-number vectors and must use nalgebra operations
/// (which are generic over the scalar type).
#[cfg(feature = "autodiff")]
pub fn autodiff_jacobian<
    G,
    const N: usize,
    const M: usize,
>(
    g: G,
    x: nalgebra::SVector<f64, N>,
) -> (nalgebra::SVector<f64, M>, nalgebra::SMatrix<f64, M, N>)
where
    G: FnOnce(
        nalgebra::SVector<num_dual::DualSVec64<N>, N>,
    ) -> nalgebra::SVector<num_dual::DualSVec64<N>, M>,
{
    num_dual::jacobian(g, x)
}
