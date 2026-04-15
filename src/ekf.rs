//! Extended Kalman Filter (EKF) with finite-difference Jacobian computation.
//!
//! Uses the same `PropagationModel` and `ObservationModel` traits as UKF.
//! Jacobians are computed automatically via central finite differences in
//! the tangent space, using the existing `merge_sigma()` / `error_from()`
//! infrastructure.

use nalgebra::{SMatrix, SVector};

use crate::{
    components::KalmanFilterError,
    ukf::{ObservationModel, PropagationModel},
    value_structs::{
        GaussianInputTrait, NominalStructTrait, OutputStructTrait, StateStructTrait,
        ValueStructTrait,
    },
};

/// Compute the tangent-space Jacobian of a function via central finite differences.
///
/// Perturbs each tangent-space dimension by ±ε, runs the function, and computes
/// the output difference in the output's tangent space. Works with manifold
/// components (quaternion, direction) via `merge_sigma()` / `error_from()`.
pub fn finite_difference_jacobian<
    State: ValueStructTrait + Clone,
    Output: OutputStructTrait<ODIM>,
    F: Fn(&State) -> Output,
    const SDIM: usize,
    const ODIM: usize,
>(
    state: &State,
    func: &F,
    epsilon: f64,
) -> Result<SMatrix<f64, ODIM, SDIM>, KalmanFilterError>
where
    State::SigmaStruct:
        From<SVector<f64, SDIM>> + Into<SVector<f64, SDIM>> + Clone + Default + Copy,
    State::NominalStruct: Clone,
    Output::SigmaStruct: From<SVector<f64, ODIM>> + Into<SVector<f64, ODIM>> + Clone,
{
    let nominal_output = func(state);
    let (nominal_struct, nominal_sigma) = state.algebraize();
    let nominal_sigma_vec: SVector<f64, SDIM> = nominal_sigma.into();

    let mut jacobian = SMatrix::<f64, ODIM, SDIM>::zeros();

    for j in 0..SDIM {
        let mut perturb_pos = nominal_sigma_vec;
        perturb_pos[j] += epsilon;
        let state_pos = nominal_struct.merge_sigma(&State::SigmaStruct::from(perturb_pos));
        let output_pos = func(&state_pos);
        let error_pos: SVector<f64, ODIM> = output_pos.error_from(&nominal_output).into();

        let mut perturb_neg = nominal_sigma_vec;
        perturb_neg[j] -= epsilon;
        let state_neg = nominal_struct.merge_sigma(&State::SigmaStruct::from(perturb_neg));
        let output_neg = func(&state_neg);
        let error_neg: SVector<f64, ODIM> = output_neg.error_from(&nominal_output).into();

        let col = (error_pos - error_neg) / (2.0 * epsilon);
        jacobian.set_column(j, &col);
    }

    Ok(jacobian)
}

/// Extended Kalman Filter with automatic finite-difference Jacobian computation.
///
/// Uses the **same model traits** as `UnscentedKalmanFilter`. No additional
/// trait implementations needed — the Jacobian is computed automatically
/// from the existing `propagate()` and `predict()` functions.
pub struct ExtendedKalmanFilter<
    State: StateStructTrait<SDIM>,
    Time: core::ops::Sub<Output = Dt> + Clone,
    Dt: Clone,
    PropagateModel: PropagationModel<
        State = State,
        DeterministicInput = DeterministicPropagationInput,
        Time = Time,
        Dt = Dt,
    >,
    DeterministicPropagationInput,
    const SDIM: usize,
> where
    State::SigmaStruct:
        From<SVector<f64, SDIM>> + Into<SVector<f64, SDIM>> + Clone + Default + Copy,
{
    state: State,
    covariance: SMatrix<f64, SDIM, SDIM>,
    last_propagation_time: Time,
    propagate_model: PropagateModel,
    epsilon: f64,
    _marker: core::marker::PhantomData<DeterministicPropagationInput>,
}

impl<
        State: StateStructTrait<SDIM> + OutputStructTrait<SDIM> + Clone,
        Time: core::ops::Sub<Output = Dt> + Clone,
        Dt: Clone,
        PropagateModel: PropagationModel<
            State = State,
            DeterministicInput = DeterministicPropagationInput,
            Time = Time,
            Dt = Dt,
        >,
        DeterministicPropagationInput: Clone,
        const SDIM: usize,
    >
    ExtendedKalmanFilter<State, Time, Dt, PropagateModel, DeterministicPropagationInput, SDIM>
where
    State::SigmaStruct:
        From<SVector<f64, SDIM>> + Into<SVector<f64, SDIM>> + Clone + Default + Copy,
    State::NominalStruct: Clone,
{
    /// Create a new EKF.
    ///
    /// `epsilon` controls the finite-difference step size (default: 1e-7).
    pub fn new(
        propagate_model: PropagateModel,
        initial_state: State,
        initial_covariance: SMatrix<f64, SDIM, SDIM>,
        initial_time: &Time,
        epsilon: f64,
    ) -> Self {
        Self {
            state: initial_state,
            covariance: initial_covariance,
            last_propagation_time: initial_time.clone(),
            propagate_model,
            epsilon,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn covariance(&self) -> &SMatrix<f64, SDIM, SDIM> {
        &self.covariance
    }

    /// EKF predict step.
    ///
    /// Propagates the state and covariance forward. The Jacobian F = ∂f/∂x
    /// is computed automatically via finite differences.
    pub fn propagate<GaussianPropagationInput: GaussianInputTrait<GPIDIM>, const GPIDIM: usize>(
        &mut self,
        deterministic_input: &DeterministicPropagationInput,
        gaussian_input: &GaussianPropagationInput,
        process_noise_covariance: Option<SMatrix<f64, SDIM, SDIM>>,
        time: &Time,
    ) -> Result<(), KalmanFilterError>
    where
        GaussianPropagationInput::SigmaStruct:
            From<SVector<f64, GPIDIM>> + Into<SVector<f64, GPIDIM>> + Clone,
        PropagateModel: PropagationModel<
            GaussianInput = <GaussianPropagationInput as GaussianInputTrait<GPIDIM>>::MeanStruct,
        >,
    {
        let dt = time.clone() - self.last_propagation_time.clone();
        let gi_mean = gaussian_input.mean();

        // Nominal propagation
        let propagated = self.propagate_model.propagate(
            &self.state,
            deterministic_input,
            &gi_mean,
            time,
            &dt,
        );

        // Jacobian F = ∂f/∂x via finite differences
        let f_jac = finite_difference_jacobian::<State, State, _, SDIM, SDIM>(
            &self.state,
            &|s: &State| {
                self.propagate_model
                    .propagate(s, deterministic_input, &gi_mean, time, &dt)
            },
            self.epsilon,
        )?;

        // P⁻ = F P⁺ Fᵀ + Q
        self.covariance = &f_jac * &self.covariance * f_jac.transpose()
            + process_noise_covariance.unwrap_or_else(SMatrix::zeros);
        self.covariance = (&self.covariance + self.covariance.transpose()) / 2.0;

        self.state = propagated;
        self.last_propagation_time = time.clone();
        Ok(())
    }

    /// EKF update step.
    ///
    /// Updates the state and covariance with a measurement. The observation
    /// Jacobian H = ∂h/∂x is computed automatically via finite differences.
    pub fn update<
        Observation: ObservationModel<
            State = State,
            DeterministicInput = ObsDeterministicInput,
            GaussianInput = <GaussianObservationInput as GaussianInputTrait<GOIDIM>>::MeanStruct,
            Time = Time,
        >,
        ObsDeterministicInput: Clone,
        GaussianObservationInput: GaussianInputTrait<GOIDIM>,
        const ODIM: usize,
        const GOIDIM: usize,
    >(
        &mut self,
        observation_model: &Observation,
        measurement: &Observation::Observation,
        deterministic_input: &ObsDeterministicInput,
        gaussian_input: &GaussianObservationInput,
        time: &Time,
        measurement_noise_covariance: SMatrix<f64, ODIM, ODIM>,
    ) -> Result<(), KalmanFilterError>
    where
        Observation::Observation: OutputStructTrait<ODIM> + core::fmt::Debug,
        <Observation::Observation as ValueStructTrait>::SigmaStruct:
            From<SVector<f64, ODIM>> + Into<SVector<f64, ODIM>> + Clone,
    {
        let gi_mean = gaussian_input.mean();

        // Predicted observation
        let predicted =
            observation_model.predict(&self.state, deterministic_input, &gi_mean, time);

        // H = ∂h/∂x
        let h_jac =
            finite_difference_jacobian::<State, Observation::Observation, _, SDIM, ODIM>(
                &self.state,
                &|s: &State| {
                    observation_model.predict(s, deterministic_input, &gi_mean, time)
                },
                self.epsilon,
            )?;

        // Innovation
        let innovation: SVector<f64, ODIM> = measurement.error_from(&predicted).into();

        // S = H P Hᵀ + R
        let s_mat =
            &h_jac * &self.covariance * h_jac.transpose() + measurement_noise_covariance;

        // K = P Hᵀ S⁻¹
        let s_inv = s_mat
            .try_inverse()
            .ok_or(KalmanFilterError::MatrixNotInvertible)?;
        let k = &self.covariance * h_jac.transpose() * &s_inv;

        // x⁺ = x⁻ ⊕ K·innovation (tangent-space update via merge_sigma)
        let correction: SVector<f64, SDIM> = &k * innovation;
        let (nominal, sigma) = self.state.algebraize();
        let sigma_vec: SVector<f64, SDIM> = sigma.into();
        self.state = nominal.merge_sigma(&State::SigmaStruct::from(sigma_vec + correction));

        // P⁺ = (I - KH) P (I - KH)ᵀ + K R Kᵀ  (Joseph form for numerical stability)
        let i_kh = SMatrix::<f64, SDIM, SDIM>::identity() - &k * &h_jac;
        self.covariance = &i_kh * &self.covariance * i_kh.transpose()
            + &k * measurement_noise_covariance * k.transpose();
        self.covariance = (&self.covariance + self.covariance.transpose()) / 2.0;

        Ok(())
    }
}
