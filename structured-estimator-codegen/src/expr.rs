//! Symbolic expression AST with differentiation and code generation.

/// A symbolic mathematical expression.
#[derive(Clone, Debug)]
pub enum Expr {
    /// State variable x[i]
    Var(usize),
    /// Named parameter (e.g., "dt", "mu")
    Param(String),
    /// Numeric constant
    Const(f64),
    /// Addition
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication
    Mul(Box<Expr>, Box<Expr>),
    /// Division
    Div(Box<Expr>, Box<Expr>),
    /// Negation
    Neg(Box<Expr>),
    /// Sine
    Sin(Box<Expr>),
    /// Cosine
    Cos(Box<Expr>),
    /// Square root
    Sqrt(Box<Expr>),
    /// Integer power
    Pow(Box<Expr>, i32),
}

// ---- Convenience constructors ----

impl Expr {
    pub fn var(i: usize) -> Self {
        Expr::Var(i)
    }
    pub fn param(name: &str) -> Self {
        Expr::Param(name.to_string())
    }
    pub fn constant(v: f64) -> Self {
        Expr::Const(v)
    }
}

// ---- Operator overloading for ergonomic expression building ----

impl std::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::Div(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self))
    }
}

impl Expr {
    pub fn sin(self) -> Expr {
        Expr::Sin(Box::new(self))
    }
    pub fn cos(self) -> Expr {
        Expr::Cos(Box::new(self))
    }
    pub fn sqrt(self) -> Expr {
        Expr::Sqrt(Box::new(self))
    }
    pub fn powi(self, n: i32) -> Expr {
        Expr::Pow(Box::new(self), n)
    }
}

// ============================================================================
// Symbolic differentiation
// ============================================================================

/// Compute ∂expr/∂x[var_index] symbolically.
pub fn diff(expr: &Expr, var: usize) -> Expr {
    match expr {
        Expr::Var(i) => {
            if *i == var {
                Expr::Const(1.0)
            } else {
                Expr::Const(0.0)
            }
        }
        Expr::Param(_) | Expr::Const(_) => Expr::Const(0.0),

        // d(a + b) = da + db
        Expr::Add(a, b) => Expr::Add(Box::new(diff(a, var)), Box::new(diff(b, var))),

        // d(a - b) = da - db
        Expr::Sub(a, b) => Expr::Sub(Box::new(diff(a, var)), Box::new(diff(b, var))),

        // d(a * b) = da * b + a * db  (product rule)
        Expr::Mul(a, b) => {
            let da = diff(a, var);
            let db = diff(b, var);
            Expr::Add(
                Box::new(Expr::Mul(Box::new(da), b.clone())),
                Box::new(Expr::Mul(a.clone(), Box::new(db))),
            )
        }

        // d(a / b) = (da * b - a * db) / b²  (quotient rule)
        Expr::Div(a, b) => {
            let da = diff(a, var);
            let db = diff(b, var);
            Expr::Div(
                Box::new(Expr::Sub(
                    Box::new(Expr::Mul(Box::new(da), b.clone())),
                    Box::new(Expr::Mul(a.clone(), Box::new(db))),
                )),
                Box::new(Expr::Pow(b.clone(), 2)),
            )
        }

        // d(-a) = -da
        Expr::Neg(a) => Expr::Neg(Box::new(diff(a, var))),

        // d(sin(a)) = cos(a) * da  (chain rule)
        Expr::Sin(a) => {
            let da = diff(a, var);
            Expr::Mul(Box::new(Expr::Cos(a.clone())), Box::new(da))
        }

        // d(cos(a)) = -sin(a) * da
        Expr::Cos(a) => {
            let da = diff(a, var);
            Expr::Neg(Box::new(Expr::Mul(
                Box::new(Expr::Sin(a.clone())),
                Box::new(da),
            )))
        }

        // d(sqrt(a)) = da / (2 * sqrt(a))
        Expr::Sqrt(a) => {
            let da = diff(a, var);
            Expr::Div(
                Box::new(da),
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(2.0)),
                    Box::new(Expr::Sqrt(a.clone())),
                )),
            )
        }

        // d(a^n) = n * a^(n-1) * da  (power rule + chain rule)
        Expr::Pow(a, n) => {
            let da = diff(a, var);
            Expr::Mul(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(*n as f64)),
                    Box::new(Expr::Pow(a.clone(), n - 1)),
                )),
                Box::new(da),
            )
        }
    }
}

// ============================================================================
// Simplification
// ============================================================================

/// Algebraic simplification to reduce expression size.
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Add(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                (Expr::Const(0.0), _) => b,
                (_, Expr::Const(0.0)) => a,
                (Expr::Const(x), Expr::Const(y)) => Expr::Const(x + y),
                _ => Expr::Add(Box::new(a), Box::new(b)),
            }
        }
        Expr::Sub(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                (_, Expr::Const(0.0)) => a,
                (Expr::Const(0.0), _) => Expr::Neg(Box::new(b)),
                (Expr::Const(x), Expr::Const(y)) => Expr::Const(x - y),
                _ => Expr::Sub(Box::new(a), Box::new(b)),
            }
        }
        Expr::Mul(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0),
                (Expr::Const(1.0), _) => b,
                (_, Expr::Const(1.0)) => a,
                (Expr::Const(x), Expr::Const(y)) => Expr::Const(x * y),
                _ => Expr::Mul(Box::new(a), Box::new(b)),
            }
        }
        Expr::Div(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                (Expr::Const(0.0), _) => Expr::Const(0.0),
                (_, Expr::Const(1.0)) => a,
                (Expr::Const(x), Expr::Const(y)) if *y != 0.0 => Expr::Const(x / y),
                _ => Expr::Div(Box::new(a), Box::new(b)),
            }
        }
        Expr::Neg(a) => {
            let a = simplify(a);
            match &a {
                Expr::Const(x) => Expr::Const(-x),
                Expr::Neg(inner) => *inner.clone(),
                _ => Expr::Neg(Box::new(a)),
            }
        }
        Expr::Pow(a, n) => {
            let a = simplify(a);
            match n {
                0 => Expr::Const(1.0),
                1 => a,
                _ => Expr::Pow(Box::new(a), *n),
            }
        }
        Expr::Sin(a) => Expr::Sin(Box::new(simplify(a))),
        Expr::Cos(a) => Expr::Cos(Box::new(simplify(a))),
        Expr::Sqrt(a) => Expr::Sqrt(Box::new(simplify(a))),
        other => other.clone(),
    }
}

/// Apply simplification repeatedly until stable.
pub fn deep_simplify(expr: &Expr) -> Expr {
    let mut current = simplify(expr);
    for _ in 0..10 {
        let next = simplify(&current);
        let cs = format!("{:?}", current);
        let ns = format!("{:?}", next);
        if cs == ns {
            break;
        }
        current = next;
    }
    current
}

// ============================================================================
// Rust code generation
// ============================================================================

/// Convert an expression to a Rust code string.
///
/// - State variables become `state[i]`
/// - Parameters become their name directly (e.g., `dt`)
pub fn to_rust(expr: &Expr) -> String {
    match expr {
        Expr::Var(i) => format!("state[{}]", i),
        Expr::Param(name) => name.clone(),
        Expr::Const(v) => {
            if *v == 0.0 {
                "0.0".to_string()
            } else if *v == 1.0 {
                "1.0".to_string()
            } else if *v == -1.0 {
                "-1.0".to_string()
            } else {
                format!("{:?}_f64", v)
            }
        }
        Expr::Add(a, b) => format!("({} + {})", to_rust(a), to_rust(b)),
        Expr::Sub(a, b) => format!("({} - {})", to_rust(a), to_rust(b)),
        Expr::Mul(a, b) => format!("({} * {})", to_rust(a), to_rust(b)),
        Expr::Div(a, b) => format!("({} / {})", to_rust(a), to_rust(b)),
        Expr::Neg(a) => format!("(-{})", to_rust(a)),
        Expr::Sin(a) => format!("{}.sin()", to_rust(a)),
        Expr::Cos(a) => format!("{}.cos()", to_rust(a)),
        Expr::Sqrt(a) => format!("{}.sqrt()", to_rust(a)),
        Expr::Pow(a, n) => format!("{}.powi({})", to_rust(a), n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_var() {
        let e = Expr::var(0);
        let d = deep_simplify(&diff(&e, 0));
        assert!(matches!(d, Expr::Const(v) if (v - 1.0).abs() < 1e-15));
    }

    #[test]
    fn test_diff_other_var() {
        let e = Expr::var(0);
        let d = deep_simplify(&diff(&e, 1));
        assert!(matches!(d, Expr::Const(v) if v.abs() < 1e-15));
    }

    #[test]
    fn test_diff_product() {
        // d(x0 * x1) / dx0 = x1
        let e = Expr::var(0) * Expr::var(1);
        let d = deep_simplify(&diff(&e, 0));
        assert_eq!(to_rust(&d), "state[1]");
    }

    #[test]
    fn test_diff_linear_propagation() {
        // x' = x + v * dt → dx'/dx = 1, dx'/dv = dt
        let x = Expr::var(0);
        let v = Expr::var(1);
        let dt = Expr::param("dt");
        let expr = x + v * dt;

        let dx_dx = deep_simplify(&diff(&expr, 0));
        let dx_dv = deep_simplify(&diff(&expr, 1));

        assert_eq!(to_rust(&dx_dx), "1.0");
        assert_eq!(to_rust(&dx_dv), "dt");
    }

    #[test]
    fn test_diff_sin() {
        // d(sin(x0)) / dx0 = cos(x0)
        let e = Expr::var(0).sin();
        let d = deep_simplify(&diff(&e, 0));
        assert_eq!(to_rust(&d), "state[0].cos()");
    }

    #[test]
    fn test_diff_quadratic() {
        // d(x^2) / dx = 2x
        let e = Expr::var(0).powi(2);
        let d = deep_simplify(&diff(&e, 0));
        assert_eq!(to_rust(&d), "(2.0_f64 * state[0])");
    }

    #[test]
    fn test_to_rust_complex() {
        let mu = Expr::param("mu");
        let r = Expr::var(0).powi(2).sqrt();
        let expr = -mu / r.powi(3);
        let code = to_rust(&expr);
        assert!(code.contains("mu"));
        assert!(code.contains("powi(3)"));
    }
}
