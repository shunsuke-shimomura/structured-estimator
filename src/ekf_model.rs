//! EKF model traits — extends PropagationModel/ObservationModel with Jacobians.
//!
//! Users can either:
//! 1. Implement `state_jacobian()` / `observation_jacobian()` manually
//! 2. Leave them unimplemented to use the finite-difference default
//! 3. Use codegen-generated implementations
//!
//! The EKF itself uses these traits for predict/update with proper
//! tangent-space handling for manifold components.

use core::ops::Sub;
use nalgebra::{SMatrix, SVector};

use crate::{
    components::KalmanFilterError,
    ekf::finite_difference_jacobian,
    ukf::{ObservationModel, PropagationModel},
    value_structs::{
        GaussianInputTrait, NominalStructTrait, OutputStructTrait, StateStructTrait,
        ValueStructTrait,
    },
};

/// EKF propagation model — extends PropagationModel with a Jacobian method.
///
/// The `state_jacobian()` method has a default implementation using
/// finite differences. Override it to provide an analytical Jacobian.
pub trait EkfPropagationModel<const SDIM: usize>: PropagationModel
where
    Self::State: StateStructTrait<SDIM> + OutputStructTrait<SDIM> + Clone,
    <Self::State as ValueStructTrait>::SigmaStruct:
        From<SVector<f64, SDIM>> + Into<SVector<f64, SDIM>> + Clone + Default + Copy,
    <Self::State as ValueStructTrait>::NominalStruct: Clone,
{
    /// Finite-difference step size for the default Jacobian.
    /// Override to customize (default: 1e-7).
    fn jacobian_epsilon(&self) -> f64 {
        1e-7
    }

    /// State-transition Jacobian F = ∂f/∂x evaluated at the given state.
    ///
    /// Default implementation uses central finite differences in the
    /// tangent space (manifold-aware via merge_sigma/error_from).
    ///
    /// Override this to provide an analytical Jacobian for better performance.
    fn state_jacobian(
        &self,
        state: &Self::State,
        deterministic_input: &Self::DeterministicInput,
        gaussian_input_mean: &Self::GaussianInput,
        time: &Self::Time,
        dt: &Self::Dt,
    ) -> Result<SMatrix<f64, SDIM, SDIM>, KalmanFilterError>
    where
        Self::DeterministicInput: Clone,
        Self::Time: Clone,
        Self::Dt: Clone,
    {
        finite_difference_jacobian::<Self::State, Self::State, _, SDIM, SDIM>(
            state,
            &|s: &Self::State| {
                self.propagate(
                    s,
                    deterministic_input,
                    gaussian_input_mean,
                    time,
                    dt,
                )
            },
            self.jacobian_epsilon(),
        )
    }
}

/// EKF observation model — extends ObservationModel with a Jacobian method.
pub trait EkfObservationModel<const SDIM: usize, const ODIM: usize>:
    ObservationModel
where
    Self::State: StateStructTrait<SDIM> + OutputStructTrait<SDIM> + Clone,
    <Self::State as ValueStructTrait>::SigmaStruct:
        From<SVector<f64, SDIM>> + Into<SVector<f64, SDIM>> + Clone + Default + Copy,
    <Self::State as ValueStructTrait>::NominalStruct: Clone,
    Self::Observation: OutputStructTrait<ODIM>,
    <Self::Observation as ValueStructTrait>::SigmaStruct:
        From<SVector<f64, ODIM>> + Into<SVector<f64, ODIM>> + Clone,
{
    fn jacobian_epsilon(&self) -> f64 {
        1e-7
    }

    /// Observation Jacobian H = ∂h/∂x.
    ///
    /// Default: finite differences. Override for analytical Jacobian.
    fn observation_jacobian(
        &self,
        state: &Self::State,
        deterministic_input: &Self::DeterministicInput,
        gaussian_input_mean: &Self::GaussianInput,
        time: &Self::Time,
    ) -> Result<SMatrix<f64, ODIM, SDIM>, KalmanFilterError>
    where
        Self::DeterministicInput: Clone,
        Self::Time: Clone,
    {
        finite_difference_jacobian::<Self::State, Self::Observation, _, SDIM, ODIM>(
            state,
            &|s: &Self::State| {
                self.predict(s, deterministic_input, gaussian_input_mean, time)
            },
            self.jacobian_epsilon(),
        )
    }
}

// ============================================================================
// Structured EKF using the trait-based Jacobians
// ============================================================================

/// Extended Kalman Filter using EkfPropagationModel/EkfObservationModel traits.
///
/// This EKF uses the `state_jacobian()` / `observation_jacobian()` methods
/// from the model traits. If the user doesn't override them, finite-difference
/// Jacobians are used automatically.
pub struct StructuredEkf<
    State: StateStructTrait<SDIM> + OutputStructTrait<SDIM> + Clone,
    Time: Sub<Output = Dt> + Clone,
    Dt: Clone,
    Model: EkfPropagationModel<SDIM, State = State, Time = Time, Dt = Dt>,
    const SDIM: usize,
> where
    State::SigmaStruct:
        From<SVector<f64, SDIM>> + Into<SVector<f64, SDIM>> + Clone + Default + Copy,
    State::NominalStruct: Clone,
{
    state: State,
    covariance: SMatrix<f64, SDIM, SDIM>,
    last_time: Time,
    model: Model,
}

impl<State, Time, Dt, Model, const SDIM: usize>
    StructuredEkf<State, Time, Dt, Model, SDIM>
where
    State: StateStructTrait<SDIM> + OutputStructTrait<SDIM> + Clone,
    Time: Sub<Output = Dt> + Clone,
    Dt: Clone,
    Model: EkfPropagationModel<SDIM, State = State, Time = Time, Dt = Dt>,
    Model::DeterministicInput: Clone,
    State::SigmaStruct:
        From<SVector<f64, SDIM>> + Into<SVector<f64, SDIM>> + Clone + Default + Copy,
    State::NominalStruct: Clone,
{
    pub fn new(
        model: Model,
        initial_state: State,
        initial_covariance: SMatrix<f64, SDIM, SDIM>,
        initial_time: &Time,
    ) -> Self {
        Self {
            state: initial_state,
            covariance: initial_covariance,
            last_time: initial_time.clone(),
            model,
        }
    }

    pub fn state(&self) -> &State { &self.state }
    pub fn covariance(&self) -> &SMatrix<f64, SDIM, SDIM> { &self.covariance }
    pub fn model(&self) -> &Model { &self.model }

    /// EKF predict step.
    pub fn propagate<GPI: GaussianInputTrait<GPIDIM>, const GPIDIM: usize>(
        &mut self,
        deterministic_input: &Model::DeterministicInput,
        gaussian_input: &GPI,
        process_noise: Option<SMatrix<f64, SDIM, SDIM>>,
        time: &Time,
    ) -> Result<(), KalmanFilterError>
    where
        GPI::SigmaStruct: From<SVector<f64, GPIDIM>> + Into<SVector<f64, GPIDIM>> + Clone,
        Model: PropagationModel<
            GaussianInput = <GPI as GaussianInputTrait<GPIDIM>>::MeanStruct,
        >,
    {
        let dt = time.clone() - self.last_time.clone();
        let gi_mean = gaussian_input.mean();

        // Jacobian F (analytical or finite-diff via trait default)
        let f_jac = self.model.state_jacobian(
            &self.state, deterministic_input, &gi_mean, time, &dt,
        )?;

        // Propagate state
        let propagated = self.model.propagate(
            &self.state, deterministic_input, &gi_mean, time, &dt,
        );

        // P⁻ = F P Fᵀ + Q
        self.covariance = &f_jac * &self.covariance * f_jac.transpose()
            + process_noise.unwrap_or_else(SMatrix::zeros);
        self.covariance = (&self.covariance + self.covariance.transpose()) / 2.0;

        self.state = propagated;
        self.last_time = time.clone();
        Ok(())
    }

    /// EKF update step.
    pub fn update<
        Obs: EkfObservationModel<SDIM, ODIM,
            State = State,
            DeterministicInput = ObsDetInput,
            GaussianInput = <GOPI as GaussianInputTrait<GOIDIM>>::MeanStruct,
            Time = Time,
        >,
        ObsDetInput: Clone,
        GOPI: GaussianInputTrait<GOIDIM>,
        const ODIM: usize,
        const GOIDIM: usize,
    >(
        &mut self,
        obs_model: &Obs,
        measurement: &Obs::Observation,
        det_input: &ObsDetInput,
        gauss_input: &GOPI,
        time: &Time,
        measurement_noise: SMatrix<f64, ODIM, ODIM>,
    ) -> Result<(), KalmanFilterError>
    where
        GOPI::SigmaStruct: From<SVector<f64, GOIDIM>> + Into<SVector<f64, GOIDIM>> + Clone,
        Obs::Observation: OutputStructTrait<ODIM> + core::fmt::Debug,
        <Obs::Observation as ValueStructTrait>::SigmaStruct:
            From<SVector<f64, ODIM>> + Into<SVector<f64, ODIM>> + Clone,
    {
        let gi_mean = gauss_input.mean();

        // Predicted observation
        let predicted = obs_model.predict(&self.state, det_input, &gi_mean, time);

        // Observation Jacobian H
        let h_jac = obs_model.observation_jacobian(
            &self.state, det_input, &gi_mean, time,
        )?;

        // Innovation
        let innovation: SVector<f64, ODIM> = measurement.error_from(&predicted).into();

        // S = H P Hᵀ + R
        let s = &h_jac * &self.covariance * h_jac.transpose() + measurement_noise;
        let s_inv = s.try_inverse().ok_or(KalmanFilterError::MatrixNotInvertible)?;

        // K = P Hᵀ S⁻¹
        let k = &self.covariance * h_jac.transpose() * &s_inv;

        // x⁺ = x⁻ ⊕ K·innovation
        let correction: SVector<f64, SDIM> = &k * innovation;
        let (nominal, sigma) = self.state.algebraize();
        let sigma_vec: SVector<f64, SDIM> = sigma.into();
        self.state = nominal.merge_sigma(&State::SigmaStruct::from(sigma_vec + correction));

        // P⁺ = (I - KH) P (I - KH)ᵀ + K R Kᵀ (Joseph form)
        let i_kh = SMatrix::<f64, SDIM, SDIM>::identity() - &k * &h_jac;
        self.covariance = &i_kh * &self.covariance * i_kh.transpose()
            + &k * measurement_noise * k.transpose();
        self.covariance = (&self.covariance + self.covariance.transpose()) / 2.0;

        Ok(())
    }
}
