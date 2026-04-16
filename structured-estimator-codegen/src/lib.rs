//! structured-estimator-codegen: Build-time code generator for estimation models.
//!
//! Generates forward model functions AND their analytical Jacobians from
//! symbolic expression descriptions. Output is readable .rs source files.
//!
//! # Usage (in build.rs)
//!
//! ```rust,ignore
//! use structured_estimator_codegen::ModelBuilder;
//!
//! fn main() {
//!     let mut m = ModelBuilder::new("Orbital");
//!     m.state_field("position", 3);
//!     m.state_field("velocity", 3);
//!
//!     let s = m.state_vars();
//!     let dt = m.param("dt");
//!
//!     m.set_propagation(vec![
//!         s[0].clone() + s[3].clone() * dt.clone(),
//!         s[1].clone() + s[4].clone() * dt.clone(),
//!         s[2].clone() + s[5].clone() * dt,
//!         s[3].clone(), s[4].clone(), s[5].clone(),
//!     ], vec!["dt"]);
//!
//!     m.generate("src/generated/orbital.rs");
//! }
//! ```

pub mod expr;
pub mod model;

pub use expr::Expr;
pub use model::ModelBuilder;
