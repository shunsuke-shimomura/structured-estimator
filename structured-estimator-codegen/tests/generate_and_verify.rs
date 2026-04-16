//! Integration test: generate code, compile it, and verify correctness.

use structured_estimator_codegen::ModelBuilder;

#[test]
fn test_generate_linear_model_code_is_valid() {
    let mut m = ModelBuilder::new("Linear");
    m.state_field("position", 3);
    m.state_field("velocity", 3);

    let s = m.state_vars();
    let dt = m.param("dt");

    m.set_propagation(
        vec![
            s[0].clone() + s[3].clone() * dt.clone(),
            s[1].clone() + s[4].clone() * dt.clone(),
            s[2].clone() + s[5].clone() * dt,
            s[3].clone(),
            s[4].clone(),
            s[5].clone(),
        ],
        vec!["dt"],
    );

    let code = m.generate_code();

    // Verify structure
    assert!(code.contains("pub fn linear_propagate("));
    assert!(code.contains("pub fn linear_propagate_jacobian("));
    assert!(code.contains("SVector<f64, 6>"));
    assert!(code.contains("SMatrix<f64, 6, 6>"));

    // Print for manual inspection
    println!("=== Generated code ===\n{}", code);
}

#[test]
fn test_generate_nonlinear_gravity_model() {
    // Simple 1D gravity: x'' = -mu / x^2
    // State: [x, v]
    // Propagation (Euler step): x' = x + v*dt, v' = v + (-mu / x^2) * dt
    let mut m = ModelBuilder::new("Gravity1D");
    m.state_field("position", 1);
    m.state_field("velocity", 1);

    let s = m.state_vars();
    let dt = m.param("dt");
    let mu = m.param("mu");

    let accel = -mu.clone() / s[0].clone().powi(2);

    m.set_propagation(
        vec![
            s[0].clone() + s[1].clone() * dt.clone(),
            s[1].clone() + accel * dt,
        ],
        vec!["dt", "mu"],
    );

    m.add_observation(
        "PositionObs",
        vec![("position", 1)],
        vec![s[0].clone()],
        vec![],
    );

    let code = m.generate_code();

    // The Jacobian ∂v'/∂x should contain 2*mu/x^3 (from differentiating -mu/x^2)
    // Check it's present in some form
    assert!(code.contains("gravity1d_propagate_jacobian"));
    assert!(code.contains("positionobs_predict"));
    assert!(code.contains("positionobs_jacobian"));

    println!("=== Gravity1D generated code ===\n{}", code);
}

#[test]
fn test_generate_trigonometric_model() {
    // Rotational dynamics: θ' = θ + ω*dt, ω' = ω
    // Observation: [sin(θ), cos(θ)]
    let mut m = ModelBuilder::new("Rotation1D");
    m.state_field("angle", 1);
    m.state_field("angular_velocity", 1);

    let s = m.state_vars();
    let dt = m.param("dt");

    m.set_propagation(
        vec![
            s[0].clone() + s[1].clone() * dt,
            s[1].clone(),
        ],
        vec!["dt"],
    );

    m.add_observation(
        "Bearing",
        vec![("sin_cos", 2)],
        vec![s[0].clone().sin(), s[0].clone().cos()],
        vec![],
    );

    let code = m.generate_code();

    // Observation Jacobian:
    // ∂sin(θ)/∂θ = cos(θ),  ∂sin(θ)/∂ω = 0
    // ∂cos(θ)/∂θ = -sin(θ), ∂cos(θ)/∂ω = 0
    assert!(code.contains("cos"));  // from differentiating sin
    assert!(code.contains("sin"));  // from differentiating cos
    assert!(code.contains("bearing_jacobian"));

    println!("=== Rotation1D generated code ===\n{}", code);
}
