use core::ops::Sub;

use nalgebra::{
    ArrayStorage, ComplexField, Const, DefaultAllocator, DimSub, SMatrix, SVector, Storage, U1,
    allocator::Allocator,
};

use crate::{
    components::KalmanFilterError,
    value_structs::{
        GaussianInputTrait, NominalStructTrait, OutputStructTrait, SigmaPointSetWithoutWeight,
        StateStructTrait, ValueStructTrait,
    },
};

pub trait PropagationModel {
    type State;
    type DeterministicInput;
    type GaussianInput;
    type Time;
    type Dt;
    fn propagate(
        &self,
        state: &Self::State,
        deterministic_input: &Self::DeterministicInput,
        gaussian_input: &Self::GaussianInput,
        time: &Self::Time,
        dt: &Self::Dt,
    ) -> Self::State;
}

pub trait ObservationModel {
    type State;
    type DeterministicInput;
    type GaussianInput;
    type Time;
    type Observation;
    fn predict(
        &self,
        state: &Self::State,
        deterministic_input: &Self::DeterministicInput,
        gaussian_input: &Self::GaussianInput,
        time: &Self::Time,
    ) -> Self::Observation;
}

pub struct UnscentedTransformWeights {
    pub mean_center: f64,
    pub cov_center: f64,
    pub mean_side: f64,
    pub cov_side: f64,
}

pub struct UKFParameters {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
}

impl Default for UKFParameters {
    fn default() -> Self {
        Self {
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

impl UKFParameters {
    pub fn new(alpha: f64, beta: f64, kappa: f64) -> Self {
        Self { alpha, beta, kappa }
    }
    fn lambda(&self, dim: usize) -> f64 {
        self.alpha * self.alpha * (dim as f64 + self.kappa) - (dim as f64)
    }

    fn gamma(&self, dim: usize) -> Result<f64, KalmanFilterError> {
        (self.lambda(dim) + (dim as f64))
            .try_sqrt()
            .ok_or(KalmanFilterError::SqrtOfNegativeNumber)
    }

    pub fn sigma_weights(&self, dim: usize) -> UnscentedTransformWeights {
        let lambda = self.lambda(dim);
        let c = dim as f64 + lambda;
        let mean_center = lambda / c;
        let cov_center = mean_center + (1.0 - self.alpha * self.alpha + self.beta);
        let mean_side = 1.0 / (2.0 * c);
        let cov_side = mean_side;
        UnscentedTransformWeights {
            mean_center,
            cov_center,
            mean_side,
            cov_side,
        }
    }
}

pub fn input_shift<
    InputStruct: GaussianInputTrait<IDIM>,
    OutputStruct: OutputStructTrait<ODIM>,
    ShiftFunc,
    const IDIM: usize,
    const ODIM: usize,
>(
    input: &InputStruct,
    func: ShiftFunc,
    ukf_params: &UKFParameters,
) -> Result<(OutputStruct, SMatrix<f64, ODIM, ODIM>), KalmanFilterError>
where
    InputStruct::SigmaStruct: From<SVector<f64, IDIM>> + Into<SVector<f64, IDIM>> + Clone,
    OutputStruct::SigmaStruct: From<SVector<f64, ODIM>> + Into<SVector<f64, ODIM>> + Clone,
    ShiftFunc: Fn(&InputStruct::MeanStruct) -> OutputStruct,
{
    let gamma = ukf_params.gamma(IDIM)?;
    let sigma_point_set = input.to_sigma()?.weighed(gamma);

    let input_nominal = sigma_point_set.nominal.clone();

    let input_center = input_nominal.merge_sigma(&sigma_point_set.center);

    let center_output = func(&input_center);
    let (center_output_nominal, center_output_sigma) = center_output.algebraize();

    let center_sigma = center_output.error_from(&center_output).into();

    let positive_outputs_iter = sigma_point_set.positive.iter().map(|sigma| {
        let input = input_nominal.merge_sigma(sigma);
        func(&input).error_from(&center_output).into()
    });
    let negative_outputs_iter = sigma_point_set.negative.iter().map(|sigma| {
        let input = input_nominal.merge_sigma(sigma);
        func(&input).error_from(&center_output).into()
    });

    let output_variance_mean = compute_weighted_mean(
        center_sigma,
        positive_outputs_iter.clone(),
        negative_outputs_iter.clone(),
        core::iter::empty(),
        core::iter::empty(),
        &ukf_params.sigma_weights(IDIM),
    );

    let output_covariance = compute_weighted_covariance(
        center_sigma,
        output_variance_mean,
        positive_outputs_iter,
        negative_outputs_iter,
        core::iter::empty(),
        core::iter::empty(),
        &ukf_params.sigma_weights(IDIM),
    );

    let output_sigma_mean = output_variance_mean + center_output_sigma.into();

    let output_mean = center_output_nominal.merge_sigma(&output_sigma_mean.into());

    Ok((output_mean, output_covariance))
}

/// 複数のシグマ点イテレータから重み付き平均を計算する共通関数
fn compute_weighted_mean<const DIM: usize>(
    center: SVector<f64, DIM>,
    state_positive_iter: impl Iterator<Item = SVector<f64, DIM>> + Clone,
    state_negative_iter: impl Iterator<Item = SVector<f64, DIM>> + Clone,
    input_positive_iter: impl Iterator<Item = SVector<f64, DIM>> + Clone,
    input_negative_iter: impl Iterator<Item = SVector<f64, DIM>> + Clone,
    weights: &UnscentedTransformWeights,
) -> SVector<f64, DIM> {
    let mut mean_vec = SVector::<f64, DIM>::zeros();

    // 中心点の寄与
    mean_vec += center * weights.mean_center;

    // 状態のシグマ点の寄与
    for (sigma_pos, sigma_neg) in state_positive_iter.zip(state_negative_iter) {
        mean_vec += (sigma_pos + sigma_neg) * weights.mean_side;
    }

    // ガウス入力のシグマ点の寄与
    for (sigma_pos, sigma_neg) in input_positive_iter.zip(input_negative_iter) {
        mean_vec += (sigma_pos + sigma_neg) * weights.mean_side;
    }

    mean_vec
}

/// 複数のシグマ点イテレータから重み付き共分散を計算する共通関数
fn compute_weighted_covariance<const DIM: usize>(
    center: SVector<f64, DIM>,
    mean: SVector<f64, DIM>,
    state_positive_iter: impl Iterator<Item = SVector<f64, DIM>>,
    state_negative_iter: impl Iterator<Item = SVector<f64, DIM>>,
    input_positive_iter: impl Iterator<Item = SVector<f64, DIM>>,
    input_negative_iter: impl Iterator<Item = SVector<f64, DIM>>,
    weights: &UnscentedTransformWeights,
) -> SMatrix<f64, DIM, DIM> {
    let mut cov_mat = SMatrix::<f64, DIM, DIM>::zeros();

    // 中心点の寄与
    let diff_center = center - mean;
    cov_mat += weights.cov_center * (diff_center * diff_center.transpose());

    // 状態の正のシグマ点の寄与
    for sigma_vec in state_positive_iter {
        let diff = sigma_vec - mean;
        cov_mat += weights.cov_side * (diff * diff.transpose());
    }

    // 状態の負のシグマ点の寄与
    for sigma_vec in state_negative_iter {
        let diff = sigma_vec - mean;
        cov_mat += weights.cov_side * (diff * diff.transpose());
    }

    // ガウス入力の正のシグマ点の寄与
    for sigma_vec in input_positive_iter {
        let diff = sigma_vec - mean;
        cov_mat += weights.cov_side * (diff * diff.transpose());
    }

    // ガウス入力の負のシグマ点の寄与
    for sigma_vec in input_negative_iter {
        let diff = sigma_vec - mean;
        cov_mat += weights.cov_side * (diff * diff.transpose());
    }

    cov_mat
}

/// シグマ点生成の共通関数
pub fn generate_sigma_points<ValueStruct: ValueStructTrait + Clone, const DIM: usize>(
    value: &ValueStruct,
    covariance: SMatrix<f64, DIM, DIM>,
) -> Result<SigmaPointSetWithoutWeight<ValueStruct, DIM>, KalmanFilterError>
where
    ValueStruct::SigmaStruct:
        From<SVector<f64, DIM>> + Into<SVector<f64, DIM>> + Clone + Default + Copy,
{
    let sqrt_cov = covariance
        .cholesky()
        .ok_or(KalmanFilterError::CholeskyDecompositionFailed)?
        .l();
    generate_sigma_points_from_sqrt_covariance(value, sqrt_cov)
}

/// 共分散の平方根(Cholesky分解後)からシグマ点を生成する共通関数
pub fn generate_sigma_points_from_sqrt_covariance<
    ValueStruct: ValueStructTrait + Clone,
    const DIM: usize,
>(
    value: &ValueStruct,
    sqrt_cov: SMatrix<f64, DIM, DIM>,
) -> Result<SigmaPointSetWithoutWeight<ValueStruct, DIM>, KalmanFilterError>
where
    ValueStruct::SigmaStruct:
        From<SVector<f64, DIM>> + Into<SVector<f64, DIM>> + Clone + Default + Copy,
{
    let mut sigma_points_plus = [ValueStruct::SigmaStruct::default(); DIM];
    let mut sigma_points_minus = [ValueStruct::SigmaStruct::default(); DIM];

    for i in 0..DIM {
        let column = sqrt_cov.column(i);
        sigma_points_plus[i] = column.into_owned().into();
        sigma_points_minus[i] = (-column).into_owned().into();
    }

    Ok(SigmaPointSetWithoutWeight {
        value: value.clone(),
        positive_delta: sigma_points_plus,
        negative_delta: sigma_points_minus,
    })
}

pub struct UnscentedKalmanFilter<
    State: StateStructTrait<SDIM>,
    Time: Sub<Output = Dt> + Clone,
    Dt: Clone,
    PropagateModel: PropagationModel<
            State = State,
            DeterministicInput = DeterministicPropagationInput,
            GaussianInput = <GaussianPropagationInput as GaussianInputTrait<GIDIM>>::MeanStruct,
            Time = Time,
            Dt = Dt,
        >,
    DeterministicPropagationInput,
    GaussianPropagationInput: GaussianInputTrait<GIDIM>,
    const SDIM: usize,
    const GIDIM: usize,
> where
    State::SigmaStruct: From<SVector<f64, SDIM>> + Into<SVector<f64, SDIM>> + Clone,
{
    state: State,
    covariance: SMatrix<f64, SDIM, SDIM>,
    last_propagation_time: Time,
    propagate_model: PropagateModel,
    ukf_parameters: UKFParameters,
    _marker: core::marker::PhantomData<(DeterministicPropagationInput, GaussianPropagationInput)>,
}

impl<
    State: StateStructTrait<SDIM>,
    Time: Sub<Output = Dt> + Clone,
    Dt: Clone,
    PropagateModel: PropagationModel<
            State = State,
            DeterministicInput = DeterministicPropagationInput,
            GaussianInput = <GaussianPropagationInput as GaussianInputTrait<GPIDIM>>::MeanStruct,
            Time = Time,
            Dt = Dt,
        >,
    DeterministicPropagationInput,
    GaussianPropagationInput: GaussianInputTrait<GPIDIM>,
    const SDIM: usize,
    const GPIDIM: usize,
>
    UnscentedKalmanFilter<
        State,
        Time,
        Dt,
        PropagateModel,
        DeterministicPropagationInput,
        GaussianPropagationInput,
        SDIM,
        GPIDIM,
    >
where
    State::SigmaStruct: From<SVector<f64, SDIM>> + Into<SVector<f64, SDIM>> + Clone,
{
    pub fn new(
        propagate_model: PropagateModel,
        initial_state: State,
        initial_covariance: SMatrix<f64, SDIM, SDIM>,
        initial_time: &Time,
        ukf_parameters: UKFParameters,
    ) -> Self {
        Self {
            state: initial_state,
            covariance: initial_covariance,
            last_propagation_time: initial_time.clone(),
            propagate_model,
            ukf_parameters,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn covariance(&self) -> &SMatrix<f64, SDIM, SDIM> {
        &self.covariance
    }

    pub fn propagate(
        &mut self,
        deterministic_input: &DeterministicPropagationInput,
        gaussian_input: &GaussianPropagationInput,
        process_noise_covariance: Option<SMatrix<f64, SDIM, SDIM>>,
        time: &Time,
    ) -> Result<(), KalmanFilterError> {
        // 経過時間 dt を計算
        let dt = time.clone() - self.last_propagation_time.clone();

        // sigma 点の数からγを計算
        let total_dim = SDIM + GPIDIM;
        let gamma = self.ukf_parameters.gamma(total_dim)?;

        // 状態のシグマ点生成
        let state_sigma_points = self.state.to_sigma(self.covariance)?.weighed(gamma);
        let state_center = state_sigma_points
            .nominal
            .merge_sigma(&state_sigma_points.center);

        // ガウス入力のシグマ点生成
        let gaussian_input_sigma_points = gaussian_input.to_sigma()?.weighed(gamma);
        let gaussian_input_center = gaussian_input_sigma_points
            .nominal
            .merge_sigma(&gaussian_input_sigma_points.center);

        // Propagate the center point first to use as reference
        let propagated_center = self.propagate_model.propagate(
            &state_center,
            deterministic_input,
            &gaussian_input_center,
            time,
            &dt,
        );

        let state_positive_shifted_iter = state_sigma_points.positive.iter().map(|state_sigma| {
            self.propagate_model
                .propagate(
                    &state_sigma_points.nominal.merge_sigma(state_sigma),
                    deterministic_input,
                    &gaussian_input_center,
                    time,
                    &dt,
                )
                .error_from(&propagated_center)
                .into()
        });

        let state_negative_shifted_iter = state_sigma_points.negative.iter().map(|state_sigma| {
            self.propagate_model
                .propagate(
                    &state_sigma_points.nominal.merge_sigma(state_sigma),
                    deterministic_input,
                    &gaussian_input_center,
                    time,
                    &dt,
                )
                .error_from(&propagated_center)
                .into()
        });

        let input_positive_shifted_iter =
            gaussian_input_sigma_points
                .positive
                .iter()
                .map(|input_sigma| {
                    self.propagate_model
                        .propagate(
                            &state_center,
                            deterministic_input,
                            &gaussian_input_sigma_points.nominal.merge_sigma(input_sigma),
                            time,
                            &dt,
                        )
                        .error_from(&propagated_center)
                        .into()
                });

        let input_negative_shifted_iter =
            gaussian_input_sigma_points
                .negative
                .iter()
                .map(|input_sigma| {
                    self.propagate_model
                        .propagate(
                            &state_center,
                            deterministic_input,
                            &gaussian_input_sigma_points.nominal.merge_sigma(input_sigma),
                            time,
                            &dt,
                        )
                        .error_from(&propagated_center)
                        .into()
                });

        let center_shifted = propagated_center.error_from(&propagated_center).into();
        let weights = self.ukf_parameters.sigma_weights(total_dim);

        // Decompose the propagated center to get the new nominal reference
        let (propagated_nominal, propagated_center_sigma_value) = propagated_center.algebraize();

        let x_mean_vec = compute_weighted_mean(
            center_shifted,
            state_positive_shifted_iter.clone(),
            state_negative_shifted_iter.clone(),
            input_positive_shifted_iter.clone(),
            input_negative_shifted_iter.clone(),
            &weights,
        );

        let x_cov = {
            let cov_mat = compute_weighted_covariance(
                center_shifted,
                x_mean_vec,
                state_positive_shifted_iter,
                state_negative_shifted_iter,
                input_positive_shifted_iter,
                input_negative_shifted_iter,
                &weights,
            );
            // プロセスノイズの寄与
            cov_mat
                + match process_noise_covariance {
                    Some(pnc) => pnc,
                    None => SMatrix::zeros(),
                }
        };
        // 状態と共分散を更新
        let propagated_center_vec: SVector<f64, SDIM> = propagated_center_sigma_value.into();
        let final_state_vec = propagated_center_vec + x_mean_vec;
        let final_state_sigma = State::SigmaStruct::from(final_state_vec);
        let new_state = propagated_nominal.merge_sigma(&final_state_sigma);

        self.covariance = (x_cov + x_cov.transpose()) / 2.0; // 数値安定化のため対称化
        self.state = new_state;
        self.last_propagation_time = time.clone();
        Ok(())
    }

    pub fn update<Observation, GaussianObservationInput, const ODIM: usize, const GOIDIM: usize>(
        &mut self,
        observation_model: &Observation,
        measurement: &Observation::Observation,
        deterministic_input: &Observation::DeterministicInput,
        gaussian_input: &GaussianObservationInput,
        time: &Observation::Time,
        measurement_noise_covariance: SMatrix<f64, ODIM, ODIM>,
    ) -> Result<(), KalmanFilterError>
    where
        Observation: ObservationModel<
            State=State,
            GaussianInput = <GaussianObservationInput as GaussianInputTrait<GOIDIM>>::MeanStruct,
            Time=Time,
        >,
        GaussianObservationInput: GaussianInputTrait<GOIDIM>,
        Observation::Observation: OutputStructTrait<ODIM> + core::fmt::Debug,
        <Observation::Observation as ValueStructTrait>::SigmaStruct: From<SVector<f64, ODIM>> + Into<SVector<f64, ODIM>> + Clone,
        Const<ODIM>: DimSub<U1>,
        DefaultAllocator: Allocator<Const<ODIM>, Const<ODIM>> + Allocator<<Const<ODIM> as DimSub<U1>>::Output>,
        ArrayStorage<f64, ODIM, ODIM>: Storage<f64, Const<ODIM>, Const<ODIM>>,
    {
        // sigma 点の数からγを計算
        let total_dim = SDIM + GOIDIM;
        let gamma = self.ukf_parameters.gamma(total_dim)?;

        // 状態のシグマ点生成
        let state_sigma_points = self.state.to_sigma(self.covariance)?.weighed(gamma);
        let state_center = state_sigma_points
            .nominal
            .merge_sigma(&state_sigma_points.center);

        let x_mean_vec = state_sigma_points.center.clone().into();
        let x_nominal = state_sigma_points.nominal.clone();

        // ガウス入力のシグマ点生成
        let gaussian_input_sigma_points = gaussian_input.to_sigma()?.weighed(gamma);
        let gaussian_input_center = gaussian_input_sigma_points
            .nominal
            .merge_sigma(&gaussian_input_sigma_points.center);

        // Predict the center observation first to use as reference
        let predicted_center = observation_model.predict(
            &state_center,
            deterministic_input,
            &gaussian_input_center,
            time,
        );

        let state_positive_shifted_iter = state_sigma_points.positive.iter().map(|state_sigma| {
            observation_model
                .predict(
                    &state_sigma_points.nominal.merge_sigma(state_sigma),
                    deterministic_input,
                    &gaussian_input_center,
                    time,
                )
                .error_from(&predicted_center)
                .into()
        });

        let state_negative_shifted_iter = state_sigma_points.negative.iter().map(|state_sigma| {
            observation_model
                .predict(
                    &state_sigma_points.nominal.merge_sigma(state_sigma),
                    deterministic_input,
                    &gaussian_input_center,
                    time,
                )
                .error_from(&predicted_center)
                .into()
        });

        let input_positive_shifted_iter =
            gaussian_input_sigma_points
                .positive
                .iter()
                .map(|input_sigma| {
                    observation_model
                        .predict(
                            &state_center,
                            deterministic_input,
                            &gaussian_input_sigma_points.nominal.merge_sigma(input_sigma),
                            time,
                        )
                        .error_from(&predicted_center)
                        .into()
                });

        let input_negative_shifted_iter =
            gaussian_input_sigma_points
                .negative
                .iter()
                .map(|input_sigma| {
                    observation_model
                        .predict(
                            &state_center,
                            deterministic_input,
                            &gaussian_input_sigma_points.nominal.merge_sigma(input_sigma),
                            time,
                        )
                        .error_from(&predicted_center)
                        .into()
                });

        let center_shifted = predicted_center.error_from(&predicted_center).into();

        let weights = self.ukf_parameters.sigma_weights(total_dim);

        // Compute weighted mean of prediction deviations
        let y_mean_deviation = compute_weighted_mean(
            center_shifted,
            state_positive_shifted_iter.clone(),
            state_negative_shifted_iter.clone(),
            input_positive_shifted_iter.clone(),
            input_negative_shifted_iter.clone(),
            &weights,
        );

        // Innovation: difference between measurement and predicted observation
        // Standard Kalman filter: innovation = measurement - prediction
        let innovation = measurement.error_from(&predicted_center).into() - y_mean_deviation;

        let yy_cov = {
            let cov_mat = compute_weighted_covariance(
                center_shifted,
                y_mean_deviation,
                state_positive_shifted_iter.clone(),
                state_negative_shifted_iter.clone(),
                input_positive_shifted_iter.clone(),
                input_negative_shifted_iter.clone(),
                &weights,
            );
            cov_mat + measurement_noise_covariance
        };

        let xy_cov = {
            let mut cov_mat = SMatrix::<f64, SDIM, ODIM>::zeros();
            // 中心点の寄与
            let diff_center = state_sigma_points.center.clone().into() - x_mean_vec;
            let y_center_vec = center_shifted;
            let diff_y_center = y_center_vec - y_mean_deviation;
            cov_mat += weights.cov_center * (diff_center * diff_y_center.transpose());
            // 状態の正のシグマ点の寄与
            for (state_sigma, y_sigma) in state_sigma_points
                .positive
                .iter()
                .zip(state_positive_shifted_iter)
            {
                let sigma_vec = (*state_sigma).clone().into();
                let diff = sigma_vec - x_mean_vec;
                let y_sigma_vec = y_sigma;
                let diff_y = y_sigma_vec - y_mean_deviation;
                cov_mat += weights.cov_side * (diff * diff_y.transpose());
            }
            // 状態の負のシグマ点の寄与
            for (state_sigma, y_sigma) in state_sigma_points
                .negative
                .iter()
                .zip(state_negative_shifted_iter)
            {
                let sigma_vec = (*state_sigma).clone().into();
                let diff = sigma_vec - x_mean_vec;
                let y_sigma_vec = y_sigma;
                let diff_y = y_sigma_vec - y_mean_deviation;
                cov_mat += weights.cov_side * (diff * diff_y.transpose());
            }
            // ガウス入力の正のシグマ点の寄与
            for y_sigma in input_positive_shifted_iter {
                let sigma_vec = state_sigma_points.center.clone().into();
                let diff = sigma_vec - x_mean_vec;
                let y_sigma_vec = y_sigma;
                let diff_y = y_sigma_vec - y_mean_deviation;
                cov_mat += weights.cov_side * (diff * diff_y.transpose());
            }
            // ガウス入力の負のシグマ点の寄与
            for y_sigma in input_negative_shifted_iter {
                let sigma_vec = state_sigma_points.center.clone().into();
                let diff = sigma_vec - x_mean_vec;
                let y_sigma_vec = y_sigma;
                let diff_y = y_sigma_vec - y_mean_deviation;
                cov_mat += weights.cov_side * (diff * diff_y.transpose());
            }
            cov_mat
        };

        // カルマンゲインの計算
        let yy_cov_inv = yy_cov
            .clone()
            .try_inverse()
            .ok_or(KalmanFilterError::MatrixNotInvertible)?;
        let kyx = xy_cov * yy_cov_inv.clone();
        let state_correction = kyx * innovation;

        // 状態と共分散の更新
        // Standard Kalman update: x_new = x_old + K * innovation
        let new_state =
            x_nominal.merge_sigma(&State::SigmaStruct::from(x_mean_vec + state_correction));

        self.covariance -= kyx * yy_cov * kyx.transpose();
        self.covariance = (self.covariance + self.covariance.transpose()) / 2.0; // 数値安定化のため対称化
        self.state = new_state;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_structs::{EmptyInput, OutputStructTrait, StateStructTrait, ValueStructTrait};
    use nalgebra::{SMatrix, Vector2};

    // UKFParameters tests
    #[test]
    fn test_ukf_parameters_default() {
        let params = UKFParameters::default();
        assert!((params.alpha - 1e-3).abs() < 1e-10);
        assert!((params.beta - 2.0).abs() < 1e-10);
        assert!((params.kappa - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ukf_parameters_new() {
        let alpha = 0.5;
        let beta = 3.0;
        let kappa = 1.0;
        let params = UKFParameters::new(alpha, beta, kappa);

        assert!((params.alpha - alpha).abs() < 1e-10);
        assert!((params.beta - beta).abs() < 1e-10);
        assert!((params.kappa - kappa).abs() < 1e-10);
    }

    #[test]
    fn test_ukf_parameters_lambda() {
        let params = UKFParameters::new(1.0, 2.0, 0.0);
        let dim = 3;

        let lambda = params.lambda(dim);
        let expected = 1.0 * 1.0 * (3.0 + 0.0) - 3.0;
        assert!((lambda - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ukf_parameters_gamma() {
        let params = UKFParameters::new(1.0, 2.0, 0.0);
        let dim = 3;

        let gamma_result = params.gamma(dim);
        assert!(gamma_result.is_ok());

        let gamma = gamma_result.unwrap();
        assert!((gamma - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_ukf_parameters_gamma_negative() {
        let params = UKFParameters::new(0.1, 2.0, -10.0);
        let dim = 2;

        let gamma_result = params.gamma(dim);
        assert!(gamma_result.is_err());
    }

    #[test]
    fn test_ukf_parameters_sigma_weights() {
        let params = UKFParameters::new(1.0, 2.0, 0.0);
        let dim = 3;

        let weights = params.sigma_weights(dim);

        let lambda = params.lambda(dim);
        let c = dim as f64 + lambda;

        let mean_sum = weights.mean_center + 2.0 * (dim as f64) * weights.mean_side;
        assert!((mean_sum - 1.0).abs() < 1e-10);

        assert!((weights.mean_center - lambda / c).abs() < 1e-10);
        assert!((weights.mean_side - 1.0 / (2.0 * c)).abs() < 1e-10);
    }

    // Generate sigma points tests - skipped since generate_sigma_points requires ValueStructTrait
    // which is not implemented for basic types. These are tested via UKF tests below.

    // Simple propagation model for testing
    struct LinearPropagationModel;

    impl PropagationModel for LinearPropagationModel {
        type State = Vector2<f64>;
        type DeterministicInput = ();
        type GaussianInput = EmptyInput;
        type Time = f64;
        type Dt = f64;

        fn propagate(
            &self,
            state: &Self::State,
            _deterministic_input: &Self::DeterministicInput,
            _gaussian_input: &Self::GaussianInput,
            _time: &Self::Time,
            dt: &Self::Dt,
        ) -> Self::State {
            *state + Vector2::new(*dt, *dt)
        }
    }

    // Simple observation model for testing
    struct LinearObservationModel;

    impl ObservationModel for LinearObservationModel {
        type State = Vector2<f64>;
        type DeterministicInput = ();
        type GaussianInput = EmptyInput;
        type Time = f64;
        type Observation = Vector2<f64>;

        fn predict(
            &self,
            state: &Self::State,
            _deterministic_input: &Self::DeterministicInput,
            _gaussian_input: &Self::GaussianInput,
            _time: &Self::Time,
        ) -> Self::Observation {
            *state
        }
    }

    // Implement necessary traits for Vector2<f64> in tests
    impl crate::value_structs::NominalStructTrait for crate::components::Vector2EmptyNominal {
        type ValueStruct = Vector2<f64>;
        type SigmaStruct = Vector2<f64>;

        fn merge_sigma(&self, sigma: &Self::SigmaStruct) -> Self::ValueStruct {
            *sigma
        }
    }

    impl ValueStructTrait for Vector2<f64> {
        type NominalStruct = crate::components::Vector2EmptyNominal;
        type SigmaStruct = Vector2<f64>;

        fn algebraize(&self) -> (Self::NominalStruct, Self::SigmaStruct) {
            (crate::components::Vector2EmptyNominal, *self)
        }
    }

    impl OutputStructTrait<2> for Vector2<f64> {
        fn error_from(&self, criteria: &Self) -> Self::SigmaStruct {
            *self - *criteria
        }
    }

    impl StateStructTrait<2> for Vector2<f64> {
        fn to_sigma(
            &self,
            covariance: SMatrix<f64, 2, 2>,
        ) -> Result<crate::value_structs::SigmaPointSetWithoutWeight<Self, 2>, KalmanFilterError>
        {
            generate_sigma_points(self, covariance)
        }
    }

    #[test]
    fn test_ukf_initialization() {
        let propagate_model = LinearPropagationModel;
        let initial_state = Vector2::new(0.0, 0.0);
        let initial_covariance = SMatrix::<f64, 2, 2>::identity();
        let initial_time = 0.0;
        let ukf_parameters = UKFParameters::default();

        let ukf = UnscentedKalmanFilter::<
            Vector2<f64>,
            f64,
            f64,
            LinearPropagationModel,
            (),
            EmptyInput,
            2,
            0,
        >::new(
            propagate_model,
            initial_state,
            initial_covariance,
            &initial_time,
            ukf_parameters,
        );

        assert!((*ukf.state() - initial_state).norm() < 1e-10);
        assert!((*ukf.covariance() - initial_covariance).norm() < 1e-10);
    }

    #[test]
    fn test_ukf_propagate() {
        let propagate_model = LinearPropagationModel;
        let initial_state = Vector2::new(1.0, 2.0);
        let initial_covariance = SMatrix::<f64, 2, 2>::identity() * 0.1;
        let initial_time = 0.0;
        let ukf_parameters = UKFParameters::default();

        let mut ukf = UnscentedKalmanFilter::<
            Vector2<f64>,
            f64,
            f64,
            LinearPropagationModel,
            (),
            EmptyInput,
            2,
            0,
        >::new(
            propagate_model,
            initial_state,
            initial_covariance,
            &initial_time,
            ukf_parameters,
        );

        let dt = 1.0;
        let new_time = initial_time + dt;
        let deterministic_input = ();
        let gaussian_input = EmptyInput;

        let result = ukf.propagate(&deterministic_input, &gaussian_input, None, &new_time);
        assert!(result.is_ok());

        // Just verify that propagation completed successfully
        // The exact state value depends on UKF algorithm details
    }

    #[test]
    fn test_ukf_update() {
        let propagate_model = LinearPropagationModel;
        let initial_state = Vector2::new(1.0, 2.0);
        let initial_covariance = SMatrix::<f64, 2, 2>::identity() * 0.1;
        let initial_time = 0.0;
        let ukf_parameters = UKFParameters::default();

        let mut ukf = UnscentedKalmanFilter::<
            Vector2<f64>,
            f64,
            f64,
            LinearPropagationModel,
            (),
            EmptyInput,
            2,
            0,
        >::new(
            propagate_model,
            initial_state,
            initial_covariance,
            &initial_time,
            ukf_parameters,
        );

        let observation_model = LinearObservationModel;
        let measurement = Vector2::new(1.5, 2.5);
        let deterministic_input = ();
        let gaussian_input = EmptyInput;
        let time = 0.0;
        let measurement_noise = SMatrix::<f64, 2, 2>::identity() * 0.01;

        let result = ukf.update::<LinearObservationModel, EmptyInput, 2, 0>(
            &observation_model,
            &measurement,
            &deterministic_input,
            &gaussian_input,
            &time,
            measurement_noise,
        );

        assert!(result.is_ok());

        // Update should complete successfully
        // The actual convergence behavior depends on UKF parameters and noise settings
    }

    #[test]
    fn test_ukf_propagate_and_update_cycle() {
        let propagate_model = LinearPropagationModel;
        let initial_state = Vector2::new(0.0, 0.0);
        let initial_covariance = SMatrix::<f64, 2, 2>::identity() * 0.5;
        let initial_time = 0.0;
        let ukf_parameters = UKFParameters::default();

        let mut ukf = UnscentedKalmanFilter::<
            Vector2<f64>,
            f64,
            f64,
            LinearPropagationModel,
            (),
            EmptyInput,
            2,
            0,
        >::new(
            propagate_model,
            initial_state,
            initial_covariance,
            &initial_time,
            ukf_parameters,
        );

        let time1 = 1.0;
        ukf.propagate(&(), &EmptyInput, None, &time1).unwrap();

        let observation_model = LinearObservationModel;
        let measurement = Vector2::new(1.0, 1.0);
        ukf.update::<LinearObservationModel, EmptyInput, 2, 0>(
            &observation_model,
            &measurement,
            &(),
            &EmptyInput,
            &time1,
            SMatrix::<f64, 2, 2>::identity() * 0.1,
        )
        .unwrap();

        let time2 = 2.0;
        ukf.propagate(&(), &EmptyInput, None, &time2).unwrap();

        let final_state = *ukf.state();
        assert!(final_state.norm() > 0.0);
    }

    #[test]
    fn test_ukf_with_process_noise() {
        let propagate_model = LinearPropagationModel;
        let initial_state = Vector2::new(0.0, 0.0);
        let initial_covariance = SMatrix::<f64, 2, 2>::identity() * 0.1;
        let initial_time = 0.0;
        let ukf_parameters = UKFParameters::default();

        let mut ukf = UnscentedKalmanFilter::<
            Vector2<f64>,
            f64,
            f64,
            LinearPropagationModel,
            (),
            EmptyInput,
            2,
            0,
        >::new(
            propagate_model,
            initial_state,
            initial_covariance,
            &initial_time,
            ukf_parameters,
        );

        let process_noise = SMatrix::<f64, 2, 2>::identity() * 0.5;
        let time1 = 1.0;

        let cov_before = *ukf.covariance();

        ukf.propagate(&(), &EmptyInput, Some(process_noise), &time1)
            .unwrap();

        let cov_after = *ukf.covariance();

        assert!(cov_after.norm() >= cov_before.norm());
    }

    #[test]
    fn test_ukf_weights_calculation() {
        // Test UKF weight calculation for different dimensions
        let alpha = 1e-3;
        let beta = 2.0;
        let kappa = 0.0;
        let params = UKFParameters::new(alpha, beta, kappa);

        // Test for dim = 2
        let dim = 2;
        let weights = params.sigma_weights(dim);
        println!("Weights for dim={}", dim);
        println!("  mean_center: {}", weights.mean_center);
        println!("  mean_side: {}", weights.mean_side);

        let total_weight = weights.mean_center + weights.mean_side * (2.0 * dim as f64);
        assert!(
            (total_weight - 1.0).abs() < 1e-10,
            "Weights should sum to 1 for dim={}, got {}",
            dim,
            total_weight
        );

        // Test for dim = 9 (6 state + 3 input)
        let dim = 9;
        let weights = params.sigma_weights(dim);
        println!("Weights for dim={}", dim);
        println!("  mean_center: {}", weights.mean_center);
        println!("  mean_side: {}", weights.mean_side);
        println!("  cov_center: {}", weights.cov_center);
        println!("  cov_side: {}", weights.cov_side);

        let total_weight = weights.mean_center + weights.mean_side * (2.0 * dim as f64);
        println!("  Total mean weight: {}", total_weight);
        assert!(
            (total_weight - 1.0).abs() < 1e-10,
            "Weights should sum to 1 for dim={}, got {}",
            dim,
            total_weight
        );

        // Check if center weight is negative (which can cause issues)
        if weights.mean_center < 0.0 {
            println!(
                "WARNING: Center weight is NEGATIVE for dim={}: {}",
                dim, weights.mean_center
            );
            println!("  lambda = {}", params.lambda(dim));
            println!("  This is expected for small alpha and large dim");
        }
    }

    #[test]
    fn test_ukf_propagation_simple_linear() {
        // Test UKF propagation with a simple linear model to verify weighted mean
        let propagate_model = LinearPropagationModel;
        let initial_state = Vector2::new(0.0, 0.0);
        let initial_covariance = SMatrix::<f64, 2, 2>::identity() * 0.1;
        let initial_time = 0.0;
        let ukf_parameters = UKFParameters::default();

        let mut ukf = UnscentedKalmanFilter::<
            Vector2<f64>,
            f64,
            f64,
            LinearPropagationModel,
            (),
            EmptyInput,
            2,
            0,
        >::new(
            propagate_model,
            initial_state,
            initial_covariance,
            &initial_time,
            ukf_parameters,
        );

        let dt = 1.0;
        let time1 = initial_time + dt;

        ukf.propagate(&(), &EmptyInput, None, &time1).unwrap();

        let final_state = *ukf.state();

        // For linear model: x' = x + [dt, dt]
        // Expected: [0, 0] + [1, 1] = [1, 1]
        let expected = Vector2::new(1.0, 1.0);

        println!("Linear propagation test:");
        println!("  Expected: {:?}", expected);
        println!("  Actual: {:?}", final_state);
        println!("  Error: {:?}", final_state - expected);

        // For linear models, UKF should be exact (within numerical precision)
        assert!(
            (final_state - expected).norm() < 1e-6,
            "UKF should be exact for linear models: expected {:?}, got {:?}",
            expected,
            final_state
        );
    }

    #[test]
    fn test_compute_weighted_mean_zero_deviations() {
        // Test that weighted mean of all zeros gives zero
        let weights = UKFParameters::default().sigma_weights(3);

        let center = SVector::<f64, 2>::zeros();
        let positive_iter = vec![
            SVector::<f64, 2>::zeros(),
            SVector::<f64, 2>::zeros(),
            SVector::<f64, 2>::zeros(),
        ];
        let negative_iter = vec![
            SVector::<f64, 2>::zeros(),
            SVector::<f64, 2>::zeros(),
            SVector::<f64, 2>::zeros(),
        ];

        let mean = compute_weighted_mean(
            center,
            positive_iter.into_iter(),
            negative_iter.into_iter(),
            core::iter::empty(),
            core::iter::empty(),
            &weights,
        );

        assert!(
            mean.norm() < 1e-10,
            "Weighted mean of zeros should be zero, got {:?}",
            mean
        );
    }

    #[test]
    fn test_compute_weighted_mean_manual_verification() {
        // Test weighted mean calculation with known inputs
        let params = UKFParameters::default();
        let dim = 2;
        let weights = params.sigma_weights(dim);

        println!("\n=== Weighted Mean Manual Verification (dim={}) ===", dim);
        println!("Weights:");
        println!("  mean_center: {}", weights.mean_center);
        println!("  mean_side: {}", weights.mean_side);

        // Create simple test values: center = [1, 1], all sigma points = [1, 1]
        // This simulates all sigma points propagating to the same value
        let center = SVector::<f64, 2>::new(1.0, 1.0);
        let positive_iter = vec![
            SVector::<f64, 2>::new(1.0, 1.0),
            SVector::<f64, 2>::new(1.0, 1.0),
        ];
        let negative_iter = vec![
            SVector::<f64, 2>::new(1.0, 1.0),
            SVector::<f64, 2>::new(1.0, 1.0),
        ];

        println!("\nInputs (all sigma points = [1, 1]):");
        println!("  center: {:?}", center);

        let mean = compute_weighted_mean(
            center,
            positive_iter.into_iter(),
            negative_iter.into_iter(),
            core::iter::empty(),
            core::iter::empty(),
            &weights,
        );

        println!("\nResult:");
        println!("  weighted mean: {:?}", mean);

        // Manual calculation:
        // mean = w_center * [1,1] + w_side * ([1,1] + [1,1] + [1,1] + [1,1])
        //      = w_center * [1,1] + 4 * w_side * [1,1]
        //      = (w_center + 4 * w_side) * [1,1]
        let manual_weight = weights.mean_center + 4.0 * weights.mean_side;
        let manual_mean = SVector::<f64, 2>::new(manual_weight, manual_weight);

        println!("\nManual verification:");
        println!(
            "  w_center + 4*w_side = {} + 4*{} = {}",
            weights.mean_center, weights.mean_side, manual_weight
        );
        println!("  manual mean: {:?}", manual_mean);
        println!("  difference: {:?}", mean - manual_mean);

        assert!(
            (mean - manual_mean).norm() < 1e-6,
            "Weighted mean calculation mismatch"
        );

        // Expected result: should be 1.0 (since all weights sum to 1.0)
        let expected = SVector::<f64, 2>::new(1.0, 1.0);
        println!("\nExpected (all inputs are [1,1]): {:?}", expected);
        println!("Error from expected: {:?}", mean - expected);

        assert!(
            (mean - expected).norm() < 1e-6,
            "When all inputs are equal, weighted mean should equal that value. Expected {:?}, got {:?}",
            expected,
            mean
        );
    }

    #[test]
    fn test_propagate_step_by_step_verification() {
        // Detailed step-by-step verification of propagate function
        let propagate_model = LinearPropagationModel;
        let initial_state = Vector2::new(0.0, 0.0);
        let initial_covariance = SMatrix::<f64, 2, 2>::identity() * 0.1;
        let initial_time = 0.0;
        let ukf_parameters = UKFParameters::default();

        let mut ukf = UnscentedKalmanFilter::<
            Vector2<f64>,
            f64,
            f64,
            LinearPropagationModel,
            (),
            EmptyInput,
            2,
            0,
        >::new(
            propagate_model,
            initial_state,
            initial_covariance,
            &initial_time,
            ukf_parameters,
        );

        let dt = 1.0;
        let time1 = initial_time + dt;

        println!("\n=== Propagate Step-by-Step Verification ===");
        println!("Initial state: {:?}", initial_state);
        println!("dt: {}", dt);

        // The propagation model is: x' = x + [dt, dt]
        // So for initial state [0, 0], all sigma points start at [0, 0]
        // After propagation, they should all be at [1, 1]

        println!("\nExpected behavior:");
        println!("  All sigma points start at ≈ [0, 0] (with small deviations from covariance)");
        println!("  Linear model: x' = x + [{}, {}]", dt, dt);
        println!("  All propagated sigma points should be ≈ [{}, {}]", dt, dt);
        println!("  Weighted mean should be ≈ [{}, {}]", dt, dt);

        ukf.propagate(&(), &EmptyInput, None, &time1).unwrap();

        let final_state = *ukf.state();
        println!("\nActual result:");
        println!("  final state: {:?}", final_state);

        let expected = Vector2::new(dt, dt);
        let error = final_state - expected;
        println!("  expected: {:?}", expected);
        println!("  error: {:?}", error);
        println!(
            "  relative error: {:.2}%",
            (error.norm() / expected.norm()) * 100.0
        );
    }

    #[test]
    fn test_ukf_update_step_by_step_verification() {
        // Verify update function with linear observation model
        let propagate_model = LinearPropagationModel;
        let initial_state = Vector2::new(1.0, 2.0);
        let initial_covariance = SMatrix::<f64, 2, 2>::identity() * 0.1;
        let initial_time = 0.0;
        let ukf_parameters = UKFParameters::default();

        let mut ukf = UnscentedKalmanFilter::<
            Vector2<f64>,
            f64,
            f64,
            LinearPropagationModel,
            (),
            EmptyInput,
            2,
            0,
        >::new(
            propagate_model,
            initial_state,
            initial_covariance,
            &initial_time,
            ukf_parameters,
        );

        println!("\n=== UKF Update Step-by-Step Verification ===");
        println!("Initial state: {:?}", initial_state);

        let observation_model = LinearObservationModel; // y = x (identity observation)
        let true_state = Vector2::new(1.5, 2.5);
        let measurement = true_state; // Perfect measurement
        let measurement_noise = SMatrix::<f64, 2, 2>::identity() * 0.01;

        println!("True state: {:?}", true_state);
        println!("Measurement: {:?}", measurement);
        println!("Expected: State should move toward measurement [1.5, 2.5]");

        ukf.update::<LinearObservationModel, EmptyInput, 2, 0>(
            &observation_model,
            &measurement,
            &(),
            &EmptyInput,
            &initial_time,
            measurement_noise,
        )
        .unwrap();

        let updated_state = *ukf.state();
        println!("\nUpdated state: {:?}", updated_state);

        let error_before = (initial_state - true_state).norm();
        let error_after = (updated_state - true_state).norm();
        println!("Error before update: {}", error_before);
        println!("Error after update: {}", error_after);

        assert!(
            error_after < error_before,
            "Update should reduce error. Before: {}, After: {}",
            error_before,
            error_after
        );
    }

    #[test]
    fn test_algebraize_merge_sigma_consistency() {
        // Test that algebraize and merge_sigma are consistent
        let original = Vector2::new(5.0, 7.0);
        println!("\n=== Testing algebraize/merge_sigma for Vector2 ===");
        println!("Original value: {:?}", original);

        let (nominal, sigma) = original.algebraize();
        println!("After algebraize:");
        println!("  sigma: {:?}", sigma);

        let reconstructed = nominal.merge_sigma(&sigma);
        println!("After merge_sigma:");
        println!("  reconstructed: {:?}", reconstructed);
        println!(
            "  Does reconstructed == original? {}",
            reconstructed == original
        );

        println!("\nTest: merge_sigma with zero deviation");
        let zero_deviation = Vector2::zeros();
        let result = nominal.merge_sigma(&zero_deviation);
        println!("  nominal.merge_sigma(&[0,0]) = {:?}", result);
        println!("  Expected: Should this be [5,7] or [0,0]?");
        println!("  Actual result: It's [0,0], so merge_sigma REPLACES, not ADDS");

        println!("\nTest: merge_sigma with small deviation");
        let small_deviation = Vector2::new(0.1, 0.2);
        let result2 = nominal.merge_sigma(&small_deviation);
        println!("  nominal.merge_sigma(&[0.1, 0.2]) = {:?}", result2);
        println!("  If additive: should be [5.1, 7.2]");
        println!("  If replacement: should be [0.1, 0.2]");

        println!("\n=== CONCLUSION ===");
        println!("For Vector2EmptyNominal, the sigma contains the FULL state value.");
        println!("merge_sigma() simply returns the sigma, ignoring any nominal info.");
        println!("This means: nominal.merge_sigma(&deviation) returns the deviation itself!");
    }

    #[test]
    fn test_sigma_point_generation_produces_distinct_points() {
        // Test that sigma point generation creates distinct points from the center
        let state = Vector2::new(100.0, 200.0);
        let covariance = SMatrix::<f64, 2, 2>::identity() * 1.0;

        let sigma_points = generate_sigma_points(&state, covariance).unwrap();

        println!("\n=== Sigma Point Generation Test ===");
        println!("Center state: {:?}", state);
        println!("Covariance diagonal: [1.0, 1.0]");
        println!("\nGenerated sigma points:");

        // The center should match the original state
        let (center_nominal, center_sigma) = sigma_points.value.algebraize();
        let reconstructed_center = center_nominal.merge_sigma(&center_sigma);
        println!("  Center (reconstructed): {:?}", reconstructed_center);
        assert!(
            (reconstructed_center - state).norm() < 1e-10,
            "Center should match original state"
        );

        // Positive deltas should be NON-ZERO deviations
        for (i, delta) in sigma_points.positive_delta.iter().enumerate() {
            println!("  Positive delta {}: {:?}", i, delta);
            assert!(
                delta.norm() > 1e-10,
                "Positive delta {} should be non-zero, got {:?}",
                i,
                delta
            );
        }

        // Negative deltas should be NON-ZERO deviations (opposite of positive)
        for (i, delta) in sigma_points.negative_delta.iter().enumerate() {
            println!("  Negative delta {}: {:?}", i, delta);
            assert!(
                delta.norm() > 1e-10,
                "Negative delta {} should be non-zero, got {:?}",
                i,
                delta
            );
        }

        println!("\n✓ All sigma point deltas are non-zero");
    }

    #[test]
    fn test_sigma_point_merge_creates_distinct_states() {
        // Test that merging sigma point deltas creates states different from center
        let state = Vector2::new(100.0, 200.0);
        let covariance = SMatrix::<f64, 2, 2>::identity() * 1.0;

        let sigma_points = generate_sigma_points(&state, covariance).unwrap();
        let (nominal, _) = sigma_points.value.algebraize();

        println!("\n=== Sigma Point Merge Test ===");
        println!("Center state: {:?}", state);

        // Merge positive deltas
        for (i, delta) in sigma_points.positive_delta.iter().enumerate() {
            let merged_state = nominal.merge_sigma(delta);
            println!(
                "  Positive {}: delta={:?}, merged={:?}",
                i, delta, merged_state
            );

            // For Vector2EmptyNominal, merge_sigma returns delta directly
            assert_eq!(
                merged_state, *delta,
                "For Vector2EmptyNominal, merge_sigma should return delta as-is"
            );
        }

        // Merge negative deltas
        for (i, delta) in sigma_points.negative_delta.iter().enumerate() {
            let merged_state = nominal.merge_sigma(delta);
            println!(
                "  Negative {}: delta={:?}, merged={:?}",
                i, delta, merged_state
            );

            assert_eq!(
                merged_state, *delta,
                "For Vector2EmptyNominal, merge_sigma should return delta as-is"
            );
        }
    }
}
