use crate::grid::{Grid, GridParameters, Order};
use statrs::function::erf::erf;

pub enum ScalarConstraint {
    Omega,
    Coordinate(usize, usize),
}

pub struct ScalarFieldParameters {
    pub l: usize,
    pub m: f64,
    pub w: f64,
    pub constraint: ScalarConstraint,
}

pub struct ScalarFieldInitialGuess<'a> {
    pub parameters: &'a ScalarFieldParameters,
    pub psi_0: f64,
    pub sigma_r: f64,
    pub sigma_z: f64,
    pub rr_ext: f64,
}

impl ScalarFieldInitialGuess<'_> {
    pub fn initial_guess<'a>(&self, parameters: &'a GridParameters) -> Grid<'a> {
        let chi = self.parameters.m.powi(2) - self.parameters.w.powi(2);

        Grid::from_function(parameters, |r, z| {
            let rr = f64::sqrt(r * r + z * z);
            self.psi_0
                * (f64::exp(
                    -0.5 * (r * r / (self.sigma_r * self.sigma_r)
                        + z * z / (self.sigma_z * self.sigma_z)),
                ) + (f64::exp(-chi * rr) / rr.powi(self.parameters.l as i32 + 1))
                    * (0.5
                        + 0.5 * erf(2.0 * (rr - self.rr_ext) / std::f64::consts::FRAC_2_SQRT_PI)))
        })
    }
}
