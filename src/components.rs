use core::ops::{Add, Mul};

use nalgebra::{
    ComplexField, Matrix3, Rotation3, Unit, UnitQuaternion, UnitVector3, Vector1, Vector2, Vector3,
    Vector4, Vector5, Vector6,
};

#[derive(Debug, thiserror::Error, Clone)]
pub enum KalmanFilterError {
    #[error("Matrix is not invertible")]
    MatrixNotInvertible,
    #[error("Cholesky decomposition failed")]
    CholeskyDecompositionFailed,
    #[error("sqrt of negative number")]
    SqrtOfNegativeNumber,
}

#[derive(Clone, Debug)]
pub struct Direction {
    basis: Matrix3<f64>,
}

impl Default for Direction {
    fn default() -> Self {
        Direction {
            basis: Matrix3::identity().into_owned(),
        }
    }
}

impl Direction {
    pub fn from_dir(dir: UnitVector3<f64>) -> Self {
        let nvec = dir.into_inner();

        // dir とほぼ平行でない参照ベクトル r を選ぶ
        let r = if nvec.x.abs() < 0.5 {
            Vector3::new(1.0, 0.0, 0.0) // X 軸
        } else {
            Vector3::new(0.0, 1.0, 0.0) // Y 軸
        };

        // t1 = normalize(n × r)
        let t1 = nvec.cross(&r).normalize();

        // t2 = normalize(n × t1)
        let t2 = nvec.cross(&t1).normalize();

        let basis = Matrix3::from_columns(&[nvec, t1, t2]);
        Self { basis }
    }
    pub fn dir(&self) -> UnitVector3<f64> {
        Unit::new_normalize(self.basis.column(0).into_owned())
    }
    pub fn basis_2d(&self) -> nalgebra::Matrix3x2<f64> {
        self.basis.fixed_columns::<2>(1).into_owned()
    }
}

impl Mul<Direction> for Rotation3<f64> {
    type Output = Direction;

    fn mul(self, rhs: Direction) -> Self::Output {
        let new_basis = self * rhs.basis;
        Direction { basis: new_basis }
    }
}

pub trait GaussianNominalType {
    type Value;
    type Sigma: GaussianSigmaType;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value
    where
        Self: Sized;
}

#[derive(Clone, Default)]
pub struct Vector1EmptyNominal;

impl GaussianNominalType for Vector1EmptyNominal {
    type Value = f64;
    type Sigma = Vector1<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value
    where
        Self: Sized,
    {
        sigma[0]
    }
}

#[derive(Clone, Default)]
pub struct Vector2EmptyNominal;

impl GaussianNominalType for Vector2EmptyNominal {
    type Value = Vector2<f64>;
    type Sigma = Vector2<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value
    where
        Self: Sized,
    {
        *sigma
    }
}

#[derive(Clone, Default)]
pub struct Vector3EmptyNominal;

impl GaussianNominalType for Vector3EmptyNominal {
    type Value = Vector3<f64>;
    type Sigma = Vector3<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value
    where
        Self: Sized,
    {
        *sigma
    }
}

#[derive(Clone, Default)]
pub struct Vector4EmptyNominal;
impl GaussianNominalType for Vector4EmptyNominal {
    type Value = Vector4<f64>;
    type Sigma = Vector4<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value
    where
        Self: Sized,
    {
        *sigma
    }
}

#[derive(Clone, Default)]
pub struct Vector5EmptyNominal;
impl GaussianNominalType for Vector5EmptyNominal {
    type Value = Vector5<f64>;
    type Sigma = Vector5<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value
    where
        Self: Sized,
    {
        *sigma
    }
}

#[derive(Clone, Default)]
pub struct Vector6EmptyNominal;
impl GaussianNominalType for Vector6EmptyNominal {
    type Value = Vector6<f64>;
    type Sigma = Vector6<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value
    where
        Self: Sized,
    {
        *sigma
    }
}

impl GaussianNominalType for UnitQuaternion<f64> {
    type Value = UnitQuaternion<f64>;
    type Sigma = Vector3<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value
    where
        Self: Sized,
    {
        let delta_q = UnitQuaternion::new(*sigma);
        delta_q * (*self)
    }
}

impl GaussianNominalType for Direction {
    type Value = Direction;
    type Sigma = Vector2<f64>;
    fn merge_sigma(&self, sigma: &Self::Sigma) -> Self::Value
    where
        Self: Sized,
    {
        let theta_3d = self.basis_2d() * (*sigma);
        if theta_3d.norm() < 1e-10 {
            self.clone()
        } else {
            let axis = theta_3d / theta_3d.norm();
            let angle = theta_3d.norm();
            let rotated = angle.cos() * self.dir().into_inner() + angle.sin() * axis;
            Direction::from_dir(Unit::new_normalize(rotated))
        }
        // // Map 2D sigma to 3D tangent space (perpendicular to current direction)
        // let basis3x2 = self.basis.fixed_columns::<2>(1);
        // let theta_3d = basis3x2 * (*sigma);

        // // theta_3d is already in the tangent space (perpendicular to dir)
        // // so we can use it directly as axis-angle representation
        // let axisangle = theta_3d;

        // let rot = Rotation3::new(axisangle);
        // rot * self.clone()
    }
}

pub trait GaussianValueType {
    type Nominal;
    type Sigma;
    fn algebraize(&self) -> (Self::Nominal, Self::Sigma)
    where
        Self: Sized;
    fn error(&self, criteria: &Self) -> Self::Sigma
    where
        Self: Sized;
}

impl GaussianValueType for f64 {
    type Nominal = Vector1EmptyNominal;
    type Sigma = Vector1<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma)
    where
        Self: Sized,
    {
        (Vector1EmptyNominal, Vector1::new(*self))
    }
    fn error(&self, criteria: &Self) -> Self::Sigma
    where
        Self: Sized,
    {
        Vector1::new(*self - *criteria)
    }
}

impl GaussianValueType for Vector2<f64> {
    type Nominal = Vector2EmptyNominal;
    type Sigma = Vector2<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma)
    where
        Self: Sized,
    {
        (Vector2EmptyNominal, *self)
    }
    fn error(&self, criteria: &Self) -> Self::Sigma
    where
        Self: Sized,
    {
        *self - *criteria
    }
}

impl GaussianValueType for Vector3<f64> {
    type Nominal = Vector3EmptyNominal;
    type Sigma = Vector3<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma)
    where
        Self: Sized,
    {
        (Vector3EmptyNominal, *self)
    }
    fn error(&self, criteria: &Self) -> Self::Sigma
    where
        Self: Sized,
    {
        *self - *criteria
    }
}

impl GaussianValueType for Vector4<f64> {
    type Nominal = Vector4EmptyNominal;
    type Sigma = Vector4<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma)
    where
        Self: Sized,
    {
        (Vector4EmptyNominal, *self)
    }
    fn error(&self, criteria: &Self) -> Self::Sigma
    where
        Self: Sized,
    {
        *self - *criteria
    }
}

impl GaussianValueType for Vector5<f64> {
    type Nominal = Vector5EmptyNominal;
    type Sigma = Vector5<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma)
    where
        Self: Sized,
    {
        (Vector5EmptyNominal, *self)
    }
    fn error(&self, criteria: &Self) -> Self::Sigma
    where
        Self: Sized,
    {
        *self - *criteria
    }
}

impl GaussianValueType for Vector6<f64> {
    type Nominal = Vector6EmptyNominal;
    type Sigma = Vector6<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma)
    where
        Self: Sized,
    {
        (Vector6EmptyNominal, *self)
    }
    fn error(&self, criteria: &Self) -> Self::Sigma
    where
        Self: Sized,
    {
        *self - *criteria
    }
}

impl GaussianValueType for UnitQuaternion<f64> {
    type Nominal = Self;
    type Sigma = Vector3<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma)
    where
        Self: Sized,
    {
        (*self, Vector3::zeros())
    }
    fn error(&self, criteria: &Self) -> Self::Sigma
    where
        Self: Sized,
    {
        let delta_q = self * criteria.inverse();
        delta_q.scaled_axis()
    }
}

impl GaussianValueType for Direction {
    type Nominal = Self;
    type Sigma = Vector2<f64>;

    fn algebraize(&self) -> (Self::Nominal, Self::Sigma)
    where
        Self: Sized,
    {
        (self.clone(), Vector2::zeros())
    }
    fn error(&self, criteria: &Self) -> Self::Sigma
    where
        Self: Sized,
    {
        let dot = self.dir().dot(&criteria.dir()).clamp(-1.0, 1.0);
        let angle = dot.acos();
        let u = self.dir().into_inner() - dot * criteria.dir().into_inner();
        let u_norm = u.norm();
        if u_norm < 1e-10 {
            Vector2::zeros()
        } else {
            let axis = u / u_norm;
            let axisangle = axis * angle;
            criteria.basis_2d().transpose() * axisangle
        }
        // let criteria_basis3x2 = criteria.basis_2d();
        // // Rotation from criteria to self
        // let rotation_axis = criteria.dir().cross(&self.dir());
        // let rotation_angle = criteria.dir().angle(&self.dir());

        // // Create axis-angle representation (rotation_axis is already perpendicular to both)
        // // Normalize the axis and multiply by angle to get the axis-angle vector
        // let axis_angle_3d = if rotation_angle.abs() < 1e-10 {
        //     nalgebra::Vector3::zeros()
        // } else {
        //     let axis_normalized = rotation_axis / rotation_axis.norm();
        //     axis_normalized * rotation_angle
        // };

        // // Project to 2D tangent space of criteria
        // criteria_basis3x2.transpose() * axis_angle_3d
    }
}

pub trait GaussianSigmaType: Sized + Clone + Add {
    const DIM: usize;

    fn write_to_slice(&self, out: &mut [f64]);
    fn read_from_slice(slice: &[f64]) -> Self
    where
        Self: Sized;
}

impl GaussianSigmaType for Vector1<f64> {
    const DIM: usize = 1;

    fn write_to_slice(&self, out: &mut [f64]) {
        assert_eq!(out.len(), Self::DIM);
        out[0] = self[0];
    }

    fn read_from_slice(slice: &[f64]) -> Self {
        assert_eq!(slice.len(), Self::DIM);
        Vector1::new(slice[0])
    }
}

impl GaussianSigmaType for Vector2<f64> {
    const DIM: usize = 2;

    fn write_to_slice(&self, out: &mut [f64]) {
        assert_eq!(out.len(), Self::DIM);
        out.copy_from_slice(self.as_slice());
    }

    fn read_from_slice(slice: &[f64]) -> Self {
        assert_eq!(slice.len(), Self::DIM);
        Vector2::from_row_slice(slice)
    }
}

impl GaussianSigmaType for Vector3<f64> {
    const DIM: usize = 3;

    fn write_to_slice(&self, out: &mut [f64]) {
        assert_eq!(out.len(), Self::DIM);
        out.copy_from_slice(self.as_slice());
    }

    fn read_from_slice(slice: &[f64]) -> Self {
        assert_eq!(slice.len(), Self::DIM);
        Vector3::from_row_slice(slice)
    }
}

impl GaussianSigmaType for Vector4<f64> {
    const DIM: usize = 4;

    fn write_to_slice(&self, out: &mut [f64]) {
        assert_eq!(out.len(), Self::DIM);
        out.copy_from_slice(self.as_slice());
    }

    fn read_from_slice(slice: &[f64]) -> Self {
        assert_eq!(slice.len(), Self::DIM);
        Vector4::from_row_slice(slice)
    }
}

impl GaussianSigmaType for Vector5<f64> {
    const DIM: usize = 5;

    fn write_to_slice(&self, out: &mut [f64]) {
        assert_eq!(out.len(), Self::DIM);
        out.copy_from_slice(self.as_slice());
    }

    fn read_from_slice(slice: &[f64]) -> Self {
        assert_eq!(slice.len(), Self::DIM);
        Vector5::from_row_slice(slice)
    }
}

impl GaussianSigmaType for Vector6<f64> {
    const DIM: usize = 6;

    fn write_to_slice(&self, out: &mut [f64]) {
        assert_eq!(out.len(), Self::DIM);
        out.copy_from_slice(self.as_slice());
    }

    fn read_from_slice(slice: &[f64]) -> Self {
        assert_eq!(slice.len(), Self::DIM);
        Vector6::from_row_slice(slice)
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;
    use nalgebra::{Unit, UnitQuaternion, Vector2, Vector3};

    // Direction tests
    #[test]
    fn test_direction_from_unit_vector() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let unit_v = Unit::new_normalize(v);
        let direction = Direction::from_dir(unit_v);

        let dir = direction.dir();
        assert!((dir.x - unit_v.x).abs() < 1e-10);
        assert!((dir.y - unit_v.y).abs() < 1e-10);
        assert!((dir.z - unit_v.z).abs() < 1e-10);
    }

    #[test]
    fn test_direction_orthogonality() {
        let v = Vector3::new(1.0, 0.0, 0.0);
        let unit_v = Unit::new_normalize(v);
        let direction = Direction::from_dir(unit_v);

        let basis = direction.basis;

        let col0 = basis.column(0);
        let col1 = basis.column(1);
        let col2 = basis.column(2);

        assert!((col0.dot(&col1)).abs() < 1e-10);
        assert!((col1.dot(&col2)).abs() < 1e-10);
        assert!((col2.dot(&col0)).abs() < 1e-10);
    }

    #[test]
    fn test_direction_normalization() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let unit_v = Unit::new_normalize(v);
        let direction = Direction::from_dir(unit_v);

        // Check that dir() returns the normalized direction
        let dir = direction.dir();
        assert!((dir.norm() - 1.0).abs() < 1e-10);
        assert!((dir.x - unit_v.x).abs() < 1e-8);
        assert!((dir.y - unit_v.y).abs() < 1e-8);
        assert!((dir.z - unit_v.z).abs() < 1e-8);
    }

    #[test]
    fn test_direction_basis_2d() {
        let v = Vector3::new(0.0, 0.0, 1.0);
        let unit_v = Unit::new_normalize(v);
        let direction = Direction::from_dir(unit_v);

        let basis_2d = direction.basis_2d();

        assert_eq!(basis_2d.nrows(), 3);
        assert_eq!(basis_2d.ncols(), 2);

        let dir = direction.dir();
        assert!((dir.dot(&basis_2d.column(0))).abs() < 1e-10);
        assert!((dir.dot(&basis_2d.column(1))).abs() < 1e-10);
        assert!((basis_2d.column(0).dot(&basis_2d.column(1))).abs() < 1e-10);
    }

    #[test]
    fn test_direction_default() {
        let direction = Direction::default();
        let dir = direction.dir();

        assert!((dir.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector1_empty_nominal_merge_sigma() {
        use nalgebra::Vector1;
        let nominal = Vector1EmptyNominal;
        let sigma = Vector1::new(f64::consts::PI);
        let result = nominal.merge_sigma(&sigma);
        assert!((result - f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_empty_nominal_merge_sigma() {
        let nominal = Vector2EmptyNominal;
        let sigma = Vector2::new(1.0, 2.0);
        let result = nominal.merge_sigma(&sigma);
        assert!((result - sigma).norm() < 1e-10);
    }

    #[test]
    fn test_vector3_empty_nominal_merge_sigma() {
        let nominal = Vector3EmptyNominal;
        let sigma = Vector3::new(1.0, 2.0, 3.0);
        let result = nominal.merge_sigma(&sigma);
        assert!((result - sigma).norm() < 1e-10);
    }

    #[test]
    fn test_unit_quaternion_merge_sigma() {
        let nominal = UnitQuaternion::identity();
        let sigma = Vector3::new(0.1, 0.0, 0.0);
        let result = nominal.merge_sigma(&sigma);
        let merged_expected = UnitQuaternion::new(sigma) * nominal;
        assert!((result.angle_to(&merged_expected)).abs() < 1e-10);
    }

    #[test]
    fn test_direction_merge_sigma() {
        let unit_v = Vector3::x_axis();
        let nominal = Direction::from_dir(unit_v);
        println!("Nominal dir: {:?}", nominal);

        let sigma = Vector2::new(core::f64::consts::FRAC_PI_2, 0.0);
        let result = nominal.merge_sigma(&sigma);
        println!("Result dir: {:?}", result.dir());

        let result_dir = result.dir();
        assert!(result_dir.into_inner().x.abs() < 1e-10);
    }

    #[test]
    fn test_f64_gaussian_value_type() {
        let value1 = 5.0;
        let value2 = 3.0;

        let (_nominal, sigma) = value1.algebraize();
        assert!((sigma[0] - 5.0).abs() < 1e-10);

        let error = value1.error(&value2);
        assert!((error[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_gaussian_value_type() {
        let value1 = Vector3::new(1.0, 2.0, 3.0);
        let value2 = Vector3::new(0.5, 1.0, 1.5);

        let (_nominal, sigma) = value1.algebraize();
        assert!((sigma - value1).norm() < 1e-10);

        let error = value1.error(&value2);
        let expected = value1 - value2;
        assert!((error - expected).norm() < 1e-10);
    }

    #[test]
    fn test_unit_quaternion_gaussian_value_type() {
        let q1 = UnitQuaternion::new(Vector3::new(0.1, 0.0, 0.0));
        let q2 = UnitQuaternion::identity();

        let (_nominal, sigma) = q1.algebraize();
        assert!((sigma.norm()).abs() < 1e-10);

        let error = q1.error(&q2);
        assert!((error - Vector3::new(0.1, 0.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_direction_gaussian_value_type() {
        {
            let v1 = Vector3::new(0.0, 0.0, 1.0);
            let dir1 = Direction::from_dir(Unit::new_normalize(v1));

            let v2 = Vector3::new(1.0, 0.0, 0.0);
            let dir2 = Direction::from_dir(Unit::new_normalize(v2));

            let (_nominal, sigma) = dir1.algebraize();
            assert!(sigma.norm() < 1e-10);

            let error = dir1.error(&dir2);
            assert!(
                error.norm() - core::f64::consts::FRAC_PI_2 < 1e-6,
                "error: {:?}, ",
                error
            );
            let reconstructed_dir1 = dir2.merge_sigma(&error);
            assert!((reconstructed_dir1.dir().angle(&dir1.dir())).abs() < 1e-10);
        }
        {
            let v1 = Vector3::new(0.0, 0.0, 1.0);
            let dir1 = Direction::from_dir(Unit::new_normalize(v1));

            let v2 = Vector3::new(-1.0, 0.0, 0.0);
            let dir2 = Direction::from_dir(Unit::new_normalize(v2));

            let (_nominal, sigma) = dir1.algebraize();
            assert!(sigma.norm() < 1e-10);

            let error = dir1.error(&dir2);
            assert!((error.norm() - core::f64::consts::FRAC_PI_2).abs() < 1e-6);
            let reconstructed_dir1 = dir2.merge_sigma(&error);
            assert!((reconstructed_dir1.dir().angle(&dir1.dir())).abs() < 1e-10);
        }
    }

    #[test]
    fn test_vector1_sigma_type() {
        use nalgebra::Vector1;
        let v = Vector1::new(f64::consts::PI);
        let mut buffer = [0.0; 1];
        v.write_to_slice(&mut buffer);
        assert!((buffer[0] - f64::consts::PI).abs() < 1e-10);

        let reconstructed = Vector1::<f64>::read_from_slice(&buffer);
        assert!((reconstructed[0] - f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_sigma_type() {
        let v = Vector2::new(1.0, 2.0);
        let mut buffer = [0.0; 2];
        v.write_to_slice(&mut buffer);
        assert!((buffer[0] - 1.0).abs() < 1e-10);
        assert!((buffer[1] - 2.0).abs() < 1e-10);

        let reconstructed = Vector2::<f64>::read_from_slice(&buffer);
        assert!((reconstructed - v).norm() < 1e-10);
    }

    #[test]
    fn test_vector3_sigma_type() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let mut buffer = [0.0; 3];
        v.write_to_slice(&mut buffer);

        let reconstructed = Vector3::<f64>::read_from_slice(&buffer);
        assert!((reconstructed - v).norm() < 1e-10);
    }

    #[test]
    fn test_direction_rotation_composition() {
        println!("Testing Direction rotation composition...");
        let initial_dir = Direction::from_dir(Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)));
        println!("Initial dir: {:?}", initial_dir);

        let sigma1 = Vector2::new(core::f64::consts::FRAC_PI_2, 0.0);
        let rotated1 = initial_dir.merge_sigma(&sigma1);
        println!("Rotated dir: {:?}", rotated1);

        let sigma2 = Vector2::new(0.0, core::f64::consts::FRAC_PI_2);
        let rotated2 = rotated1.merge_sigma(&sigma2);
        println!("Rotated dir: {:?}", rotated2);
    }
}
