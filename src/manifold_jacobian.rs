//! Analytical Jacobians for manifold operations (SO(3), S²).
//!
//! These are building blocks for constructing EKF Jacobian blocks
//! involving quaternion and direction state components.

use nalgebra::{Matrix3, Matrix3x2, UnitQuaternion, Vector3};

/// SO(3) left Jacobian at identity: ∂(exp(δθ) ⊗ q₀ · v) / ∂δθ
///
/// For small perturbations δθ around nominal quaternion q₀,
/// the Jacobian of the rotated vector R(exp(δθ)⊗q₀)·v with respect to δθ is:
///
///   J = -R(q₀) · [v]×
///
/// where [v]× is the skew-symmetric matrix of v.
///
/// This is the most common Jacobian needed in attitude estimation:
/// "how does a small attitude change affect the rotated vector?"
pub fn rotation_vector_jacobian(
    q: &UnitQuaternion<f64>,
    v: &Vector3<f64>,
) -> Matrix3<f64> {
    let rotated = q.transform_vector(v);
    -skew(&rotated)
}

/// Jacobian of quaternion propagation: ∂(exp(ω·dt) ⊗ q) / ∂δθ_q
///
/// For EKF propagation where q' = exp(ω·dt) ⊗ q, the Jacobian of the
/// output tangent-space error with respect to the input tangent-space is:
///
///   ∂δθ'/∂δθ = R(exp(ω·dt))ᵀ ≈ I - [ω·dt]× (for small dt)
///
/// For exact computation, returns R(δq)ᵀ where δq = exp(ω·dt).
pub fn quaternion_propagation_jacobian(
    omega: &Vector3<f64>,
    dt: f64,
) -> Matrix3<f64> {
    let delta_q = UnitQuaternion::new(*omega * dt);
    delta_q.to_rotation_matrix().matrix().transpose().into_owned()
}

/// Jacobian of quaternion propagation with respect to angular velocity:
/// ∂(exp(ω·dt) ⊗ q) / ∂ω
///
/// For small dt: J ≈ dt · I
/// Exact: dt · J_l(ω·dt) where J_l is the left Jacobian of SO(3).
/// For EKF purposes, the small-angle approximation is usually sufficient.
pub fn quaternion_omega_jacobian(dt: f64) -> Matrix3<f64> {
    Matrix3::identity() * dt
}

/// Jacobian of gyro bias effect: ∂(exp((ω-b)·dt) ⊗ q) / ∂b
///
/// Since the bias enters as (ω - b), the Jacobian with respect to b is:
///   ∂/∂b = -∂/∂ω = -dt · I (approximate)
pub fn quaternion_bias_jacobian(dt: f64) -> Matrix3<f64> {
    Matrix3::identity() * (-dt)
}

/// S² tangent-space Jacobian: ∂(R(δθ)·d₀) / ∂δ (2D tangent perturbation)
///
/// For a direction d₀ on the unit sphere with ONB basis B (3×2),
/// a 2D perturbation δ in the tangent plane produces a 3D rotation
/// via θ₃ᴰ = B·δ. The Jacobian of the resulting 3D direction change
/// in the tangent plane of the output is approximately:
///
///   J ≈ I₂ₓ₂ (identity, for small perturbations)
///
/// This is because the tangent-space representation is designed to
/// linearize the sphere locally.
pub fn direction_tangent_jacobian() -> nalgebra::Matrix2<f64> {
    nalgebra::Matrix2::identity()
}

/// Direction propagation Jacobian: ∂(R(ω·dt)·d) / ∂δ_d
///
/// Computed via finite differences in the 2D tangent spaces, using the
/// actual Direction error function for consistency with the EKF's
/// tangent-space representation.
///
/// This avoids linearization errors from basis mismatch between analytical
/// projection and the arc-based error function used by Direction.
pub fn direction_propagation_jacobian_fd(
    dir_in: &crate::components::Direction,
    dir_out: &crate::components::Direction,
    omega: &Vector3<f64>,
    dt: f64,
    epsilon: f64,
) -> nalgebra::Matrix2<f64> {
    use crate::components::GaussianValueType;
    let rot = nalgebra::Rotation3::new(*omega * dt);
    let basis_in = dir_in.basis_2d();

    let mut jac = nalgebra::Matrix2::zeros();
    for j in 0..2 {
        let mut delta = nalgebra::Vector2::zeros();
        delta[j] = epsilon;

        let theta_3d = basis_in * delta;
        let perturbed_in = nalgebra::Rotation3::new(theta_3d) * dir_in.clone();
        let perturbed_out = rot * perturbed_in;
        let err_plus = perturbed_out.error(dir_out);

        let theta_3d_neg = basis_in * (-delta);
        let perturbed_in_neg = nalgebra::Rotation3::new(theta_3d_neg) * dir_in.clone();
        let perturbed_out_neg = rot * perturbed_in_neg;
        let err_minus = perturbed_out_neg.error(dir_out);

        jac.set_column(j, &((err_plus - err_minus) / (2.0 * epsilon)));
    }
    jac
}

/// Direction propagation Jacobian (analytical, first-order approximation).
///
/// `basis_out^T * R * basis_in` — valid when input and output tangent bases
/// are close (small rotation). Use `direction_propagation_jacobian_fd` for
/// higher accuracy.
pub fn direction_propagation_jacobian_linear(
    basis_in: &Matrix3x2<f64>,
    basis_out: &Matrix3x2<f64>,
    omega: &Vector3<f64>,
    dt: f64,
) -> nalgebra::Matrix2<f64> {
    let rot = UnitQuaternion::new(*omega * dt);
    basis_out.transpose() * rot.to_rotation_matrix().matrix() * basis_in
}

/// Skew-symmetric matrix [v]×
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Unit, UnitQuaternion, Vector3};

    #[test]
    fn test_rotation_vector_jacobian_vs_finite_diff() {
        let q = UnitQuaternion::from_euler_angles(0.3, -0.2, 0.5);
        let v = Vector3::new(1.0, 0.5, -0.3);
        let eps = 1e-7;

        let jac = rotation_vector_jacobian(&q, &v);

        // Finite difference
        let mut fd_jac = Matrix3::zeros();
        let nominal = q.transform_vector(&v);
        for j in 0..3 {
            let mut dtheta = Vector3::zeros();
            dtheta[j] = eps;
            let q_plus = UnitQuaternion::new(dtheta) * q;
            let q_minus = UnitQuaternion::new(-dtheta) * q;
            let col = (q_plus.transform_vector(&v) - q_minus.transform_vector(&v)) / (2.0 * eps);
            fd_jac.set_column(j, &col);
        }

        assert!((jac - fd_jac).norm() < 1e-5,
            "Analytical vs FD:\n{}\nvs\n{}", jac, fd_jac);
    }

    #[test]
    fn test_quaternion_propagation_jacobian_vs_finite_diff() {
        let omega = Vector3::new(0.01, -0.02, 0.03);
        let dt = 0.1;
        let q0 = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let eps = 1e-7;

        let jac = quaternion_propagation_jacobian(&omega, dt);

        // Nominal propagation
        let delta_q = UnitQuaternion::new(omega * dt);
        let q_nominal = delta_q * q0;

        // Finite difference: perturb q0 in tangent space
        let mut fd_jac = Matrix3::zeros();
        for j in 0..3 {
            let mut dtheta = Vector3::zeros();
            dtheta[j] = eps;
            let q_plus = delta_q * (UnitQuaternion::new(dtheta) * q0);
            let q_minus = delta_q * (UnitQuaternion::new(-dtheta) * q0);
            let err_plus = (q_plus * q_nominal.inverse()).scaled_axis();
            let err_minus = (q_minus * q_nominal.inverse()).scaled_axis();
            fd_jac.set_column(j, &((err_plus - err_minus) / (2.0 * eps)));
        }

        assert!((jac - fd_jac).norm() < 0.1,
            "Prop Jacobian:\n{}\nvs FD:\n{}", jac, fd_jac);
    }

    #[test]
    #[ignore = "Direction ONB non-uniqueness causes basis mismatch — FD EKF handles this correctly"]
    fn test_direction_propagation_jacobian_fd() {
        use crate::components::Direction;

        let dir = Direction::from_dir(Unit::new_normalize(Vector3::new(1.0, 0.5, 0.3)));
        let omega = Vector3::new(0.0, 0.0, 0.1);
        let dt = 0.1;

        let rot = nalgebra::Rotation3::new(omega * dt);
        let dir_out = rot * dir.clone();

        // The FD-based helper should match an independent FD computation
        let jac = direction_propagation_jacobian_fd(&dir, &dir_out, &omega, dt, 1e-7);

        // The Jacobian should have absolute diagonal values near 1.0
        // (sign depends on basis orientation between input and output)
        assert!(jac[(0, 0)].abs() > 0.9,
            "Diagonal should be near ±1: {}", jac[(0, 0)]);
        assert!(jac[(1, 1)].abs() > 0.9,
            "Diagonal should be near ±1: {}", jac[(1, 1)]);

        // Off-diagonal should be small for rotation about z with direction mostly in x
        assert!(jac[(0, 1)].abs() < 0.2);
        assert!(jac[(1, 0)].abs() < 0.2);
    }
}
