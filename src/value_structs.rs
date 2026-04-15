use nalgebra::{SMatrix, SVector};

use crate::components::KalmanFilterError;

pub trait ValueStructTrait {
    type NominalStruct: NominalStructTrait<SigmaStruct = Self::SigmaStruct, ValueStruct = Self>
        + Clone;
    type SigmaStruct;
    fn algebraize(&self) -> (Self::NominalStruct, Self::SigmaStruct)
    where
        Self: Sized;
}

impl ValueStructTrait for EmptyInput {
    type NominalStruct = EmptyInput;
    type SigmaStruct = EmptyInput;
    fn algebraize(&self) -> (Self::NominalStruct, Self::SigmaStruct)
    where
        Self: Sized,
    {
        (EmptyInput, EmptyInput)
    }
}

pub trait NominalStructTrait {
    type ValueStruct;
    type SigmaStruct;
    fn merge_sigma(&self, sigma: &Self::SigmaStruct) -> Self::ValueStruct
    where
        Self: Sized;
}

impl NominalStructTrait for EmptyInput {
    type ValueStruct = EmptyInput;
    type SigmaStruct = EmptyInput;
    fn merge_sigma(&self, _sigma: &Self::SigmaStruct) -> Self::ValueStruct
    where
        Self: Sized,
    {
        EmptyInput
    }
}

pub trait OutputStructTrait<const DIM: usize>: ValueStructTrait
where
    Self::SigmaStruct: From<SVector<f64, DIM>> + Into<SVector<f64, DIM>> + Clone,
{
    fn error_from(&self, criteria: &Self) -> Self::SigmaStruct
    where
        Self: Sized;
}

pub trait StateStructTrait<const DIM: usize>: OutputStructTrait<DIM>
where
    Self::SigmaStruct: From<SVector<f64, DIM>> + Into<SVector<f64, DIM>> + Clone,
{
    fn to_sigma(
        &self,
        covariance: SMatrix<f64, DIM, DIM>,
    ) -> Result<SigmaPointSetWithoutWeight<Self, DIM>, KalmanFilterError>
    where
        Self: Sized;
}

pub trait GaussianInputTrait<const DIM: usize> {
    type SigmaStruct: From<SVector<f64, DIM>> + Into<SVector<f64, DIM>> + Clone;
    type MeanStruct: ValueStructTrait<SigmaStruct = Self::SigmaStruct>;
    fn to_sigma(
        &self,
    ) -> Result<SigmaPointSetWithoutWeight<Self::MeanStruct, DIM>, KalmanFilterError>
    where
        Self: Sized;
    fn mean(&self) -> Self::MeanStruct
    where
        Self: Sized;
}

#[derive(Clone, core::marker::Copy)]
pub struct EmptyInput;

impl From<()> for EmptyInput {
    fn from(_unit: ()) -> Self {
        EmptyInput
    }
}

impl From<SVector<f64, 0>> for EmptyInput {
    fn from(_vec: SVector<f64, 0>) -> Self {
        EmptyInput
    }
}

impl From<EmptyInput> for SVector<f64, 0> {
    fn from(_empty: EmptyInput) -> Self {
        SVector::<f64, 0>::zeros()
    }
}

impl GaussianInputTrait<0> for EmptyInput {
    type SigmaStruct = EmptyInput;
    type MeanStruct = EmptyInput;
    fn to_sigma(&self) -> Result<SigmaPointSetWithoutWeight<Self, 0>, KalmanFilterError>
    where
        Self: Sized,
    {
        Ok(SigmaPointSetWithoutWeight {
            value: EmptyInput,
            positive_delta: [],
            negative_delta: [],
        })
    }

    fn mean(&self) -> Self::MeanStruct
    where
        Self: Sized,
    {
        EmptyInput
    }
}

#[derive(Clone, core::marker::Copy)]
pub struct SigmaPointSetWithoutWeight<ValueStruct: ValueStructTrait, const DIM: usize>
where
    ValueStruct::SigmaStruct: From<SVector<f64, DIM>> + Into<SVector<f64, DIM>> + Clone,
{
    pub value: ValueStruct,
    pub positive_delta: [ValueStruct::SigmaStruct; DIM],
    pub negative_delta: [ValueStruct::SigmaStruct; DIM],
}

impl<ValueStruct: ValueStructTrait, const DIM: usize> SigmaPointSetWithoutWeight<ValueStruct, DIM>
where
    ValueStruct::SigmaStruct: From<SVector<f64, DIM>> + Into<SVector<f64, DIM>> + Clone,
{
    pub fn weighed(self, gamma: f64) -> SigmaPointSetBeforeShift<ValueStruct, DIM> {
        let (nominal, center) = self.value.algebraize();
        let center_vec: SVector<f64, DIM> = center.clone().into();
        let positive = self.positive_delta.map(|sigma| {
            let sigma_vec: SVector<f64, DIM> = sigma.into();
            let weighed_vec = sigma_vec * gamma + center_vec;
            ValueStruct::SigmaStruct::from(weighed_vec)
        });
        let negative = self.negative_delta.map(|sigma| {
            let sigma_vec: SVector<f64, DIM> = sigma.into();
            let weighed_vec = sigma_vec * gamma + center_vec;
            ValueStruct::SigmaStruct::from(weighed_vec)
        });
        SigmaPointSetBeforeShift {
            nominal,
            center,
            positive,
            negative,
        }
    }
}

#[derive(Clone, core::marker::Copy)]
pub struct SigmaPointSetBeforeShift<ValueStruct: ValueStructTrait, const DIM: usize> {
    pub nominal: ValueStruct::NominalStruct,
    pub center: ValueStruct::SigmaStruct,
    pub positive: [ValueStruct::SigmaStruct; DIM],
    pub negative: [ValueStruct::SigmaStruct; DIM],
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;

    #[test]
    fn test_empty_input_value_struct_trait() {
        let empty = EmptyInput;
        let (_nominal, _sigma) = empty.algebraize();
    }

    #[test]
    fn test_empty_input_nominal_struct_trait() {
        let nominal = EmptyInput;
        let sigma = EmptyInput;
        let _value = nominal.merge_sigma(&sigma);
    }

    #[test]
    fn test_empty_input_gaussian_input_trait() {
        let empty = EmptyInput;

        let _mean = empty.mean();

        let sigma_result = empty.to_sigma();
        assert!(sigma_result.is_ok());

        let sigma_set = sigma_result.unwrap();
        assert_eq!(sigma_set.positive_delta.len(), 0);
        assert_eq!(sigma_set.negative_delta.len(), 0);
    }

    #[test]
    fn test_empty_input_from_unit() {
        let empty = EmptyInput::from(());
        let (_, _) = empty.algebraize();
    }

    #[test]
    fn test_empty_input_from_svector() {
        let vec = SVector::<f64, 0>::zeros();
        let empty = EmptyInput::from(vec);
        let (_, _) = empty.algebraize();
    }

    #[test]
    fn test_empty_input_into_svector() {
        let empty = EmptyInput;
        let vec: SVector<f64, 0> = empty.into();
        assert_eq!(vec.len(), 0);
    }

    // Note: SigmaPointSetWithoutWeight tests are performed via UKF tests in ukf.rs
    // since they require ValueStructTrait implementation which is tested there.
}
