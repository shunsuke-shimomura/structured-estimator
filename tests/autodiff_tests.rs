//! Tests for num-dual automatic differentiation Jacobians.

#![cfg(feature = "autodiff")]

use nalgebra::{ComplexField, SMatrix, SVector};
use structured_estimator::autodiff::{autodiff_jacobian, num_dual::DualSVec64};

#[test]
fn test_autodiff_linear_jacobian() {
    // f(x) = A * x where A = [[1,2],[3,4],[5,6]]
    let a = SMatrix::<f64, 3, 2>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let x = SVector::<f64, 2>::new(1.0, 2.0);

    let (value, jac) = autodiff_jacobian(
        |x: SVector<DualSVec64<2>, 2>| {
            let a_dual = SMatrix::<DualSVec64<2>, 3, 2>::new(
                1.0.into(), 2.0.into(), 3.0.into(), 4.0.into(), 5.0.into(), 6.0.into(),
            );
            a_dual * x
        },
        x,
    );

    assert_eq!(value, a * x);
    // Jacobian of linear function = the matrix itself
    assert!((jac - a).norm() < 1e-14, "Jacobian should be A: {:?}", jac);
}

#[test]
fn test_autodiff_nonlinear_jacobian() {
    // f(x, y) = (x*y, x^2 + y)
    let x = SVector::<f64, 2>::new(3.0, 4.0);

    let (value, jac) = autodiff_jacobian(
        |x: SVector<DualSVec64<2>, 2>| {
            SVector::<DualSVec64<2>, 2>::new(
                x[0] * x[1],
                x[0] * x[0] + x[1],
            )
        },
        x,
    );

    assert!((value[0] - 12.0).abs() < 1e-14); // 3*4
    assert!((value[1] - 13.0).abs() < 1e-14); // 9+4

    // Jacobian: [[y, x], [2x, 1]] = [[4, 3], [6, 1]]
    assert!((jac[(0, 0)] - 4.0).abs() < 1e-14);
    assert!((jac[(0, 1)] - 3.0).abs() < 1e-14);
    assert!((jac[(1, 0)] - 6.0).abs() < 1e-14);
    assert!((jac[(1, 1)] - 1.0).abs() < 1e-14);
}

#[test]
fn test_autodiff_orbital_propagation() {
    // Simple linear orbital propagation: pos' = pos + vel*dt
    let state = SVector::<f64, 6>::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
    let dt = 60.0;

    let (value, jac) = autodiff_jacobian(
        |x: SVector<DualSVec64<6>, 6>| {
            let dt_dual: DualSVec64<6> = dt.into();
            SVector::<DualSVec64<6>, 6>::new(
                x[0] + x[3] * dt_dual,
                x[1] + x[4] * dt_dual,
                x[2] + x[5] * dt_dual,
                x[3],
                x[4],
                x[5],
            )
        },
        state,
    );

    // Value check
    assert!((value[1] - 7.5 * 60.0).abs() < 1e-10);

    // Jacobian should be:
    // | I  dt*I |
    // | 0  I    |
    for i in 0..3 {
        assert!((jac[(i, i)] - 1.0).abs() < 1e-14);
        assert!((jac[(i, i + 3)] - dt).abs() < 1e-14);
        assert!((jac[(i + 3, i + 3)] - 1.0).abs() < 1e-14);
        assert!((jac[(i + 3, i)]).abs() < 1e-14);
    }
}

#[test]
fn test_autodiff_vs_finite_diff() {
    // Compare autodiff and finite-difference Jacobians for a nonlinear function
    let x = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
    let eps = 1e-7;

    // Nonlinear function
    let f_f64 = |x: &SVector<f64, 3>| -> SVector<f64, 2> {
        SVector::<f64, 2>::new(
            x[0] * x[1].sin() + x[2] * x[2],
            x[0].exp() * x[1] - x[2],
        )
    };

    // Autodiff Jacobian
    let (_, jac_ad) = autodiff_jacobian(
        |x: SVector<DualSVec64<3>, 3>| {
            SVector::<DualSVec64<3>, 2>::new(
                x[0] * x[1].sin() + x[2] * x[2],
                x[0].exp() * x[1] - x[2],
            )
        },
        x,
    );

    // Finite-diff Jacobian
    let nom = f_f64(&x);
    let mut jac_fd = SMatrix::<f64, 2, 3>::zeros();
    for j in 0..3 {
        let mut xp = x;
        xp[j] += eps;
        let mut xm = x;
        xm[j] -= eps;
        let col = (f_f64(&xp) - f_f64(&xm)) / (2.0 * eps);
        jac_fd.set_column(j, &col);
    }

    // They should agree to high precision
    let diff = (jac_ad - jac_fd).norm();
    assert!(diff < 1e-6, "Autodiff vs finite-diff: diff = {}", diff);
}
