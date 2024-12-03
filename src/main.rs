use rboson::grid::{Grid, GridParameters, Order};

fn main() {
    let parameters = GridParameters::new(0.25, 0.25, 128, 128, Order::Fourth);
    let sigma_r = 1.0;
    let sigma_z = 1.0;
    let psi_0 = 1.0;
    let scalar_field_grid = Grid::from_function(&parameters, |r, z| {
        psi_0
            * f64::exp(-0.5 * r * r / (sigma_r * sigma_r))
            * f64::exp(-0.5 * z * z / (sigma_z * sigma_z))
    });
    scalar_field_grid.save_csv("scalar_field.csv").unwrap();
    scalar_field_grid.save_npy("scalar_field.npy").unwrap();
}
