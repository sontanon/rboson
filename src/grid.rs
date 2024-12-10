use byteorder::{LittleEndian, WriteBytesExt};
use rayon::prelude::*;
use std::{fs::File, io::Write};

/// Determines the finite difference order.
///
/// This enum is used to determine the number of ghost zones and also for some boundary calculations.
///
/// # Examples
///
/// ```
/// use rboson::grid::Order;
/// let order = Order::Second;
/// assert_eq!(order as usize, 2);
/// ```
pub enum Order {
    Second = 2,
    Fourth = 4,
}

pub fn index_to_coordinate(k: usize, dx: f64, ghost: usize) -> f64 {
    ((k as f64 - ghost as f64) + 0.5) * dx
}

/// Describes the parameters for the grid.
///
/// Some fields are non-public because they are calculated from the public fields.
///
/// The `ghost` field is calculated from the `order` field, it is two times the order of the finite difference.
///
/// # Examples
/// ```
/// use rboson::grid::{GridParameters, Order};
/// let grid_parameters = GridParameters::new(0.1, 0.2, 10, 20, Order::Second);
/// // The step sizes dr and dz are public fields.
/// assert_eq!(grid_parameters.dr, 0.1);
/// assert_eq!(grid_parameters.dz, 0.2);
/// // As are the interior grid dimensions.
/// assert_eq!(grid_parameters.nr_interior, 10);
/// assert_eq!(grid_parameters.nz_interior, 20);
/// assert_eq!(grid_parameters.dim, (10 + 2 * 1) * (20 + 2 * 1)); // 264
/// // The following fields require getter methods.
/// assert_eq!(grid_parameters.ghost(), 1);
/// assert_eq!(grid_parameters.nr_total(), 10 + 2 * 1);
/// assert_eq!(grid_parameters.nz_total(), 20 + 2 * 1);
/// // The grid can be calculated/fetched via the `r` and `z` methods.
/// assert_eq!(grid_parameters.r(0), -0.05);
/// assert_eq!(grid_parameters.r(1), 0.05);
/// assert_eq!(grid_parameters.z(0), -0.1);
/// assert_eq!(grid_parameters.z(1), 0.1);
/// ```
///
/// ```
/// use rboson::grid::{GridParameters, Order};
/// let grid_parameters = GridParameters::new(0.5, 0.5, 16, 16, Order::Fourth);
/// // Fourth order finite difference requires 2 ghost zones.
/// assert_eq!(grid_parameters.ghost(), 2);
/// assert_eq!(grid_parameters.dim, 400);
/// ```
pub struct GridParameters {
    pub dr: f64,
    pub dz: f64,
    pub nr_interior: usize,
    pub nz_interior: usize,
    nr_total: usize,
    nz_total: usize,
    pub dim: usize,
    pub order: Order,
    ghost: usize,
    r: Vec<f64>,
    z: Vec<f64>,
}

impl GridParameters {
    /// Create a new GridParameters struct using the public parameters.
    ///
    /// The non-public parameters are calculated from the public parameters.
    pub fn new(
        dr: f64,
        dz: f64,
        nr_interior: usize,
        nz_interior: usize,
        order: Order,
    ) -> GridParameters {
        let ghost = match order {
            Order::Second => 1,
            Order::Fourth => 2,
        };
        let nr_total = nr_interior + 2 * ghost;
        let nz_total = nz_interior + 2 * ghost;
        let dim = nr_total * nz_total;
        let r: Vec<f64> = (0..nr_total)
            .into_par_iter()
            .map(|i| index_to_coordinate(i, dr, ghost))
            .collect();
        let z: Vec<f64> = (0..nz_total)
            .into_par_iter()
            .map(|j| index_to_coordinate(j, dz, ghost))
            .collect();

        GridParameters {
            dr,
            dz,
            nr_interior,
            nz_interior,
            nr_total,
            nz_total,
            dim,
            order,
            ghost,
            r,
            z,
        }
    }

    pub fn nr_total(&self) -> usize {
        self.nr_total
    }

    pub fn nz_total(&self) -> usize {
        self.nz_total
    }

    pub fn ghost(&self) -> usize {
        self.ghost
    }

    pub fn r(&self, i: usize) -> f64 {
        self.r[i]
    }

    pub fn z(&self, j: usize) -> f64 {
        self.z[j]
    }

    pub fn coord(&self, i: usize, j: usize) -> (f64, f64) {
        (self.r[i], self.z[j])
    }
}

/// Describes a grid variable.
///
/// * The basic parameters are a reference to a `GridParameters` struct.
/// * The private `u` field is a vector of f64 values which contains the actual grid values.
///
/// # Examples
///
/// ```
/// use rboson::grid::{Grid, GridParameters, Order};
/// let grid_parameters = GridParameters::new(0.1, 0.2, 10, 20, Order::Second);
/// let zero_grid = Grid::new(&grid_parameters);
/// assert_eq!(zero_grid.parameters.dr, 0.1);
/// assert_eq!(zero_grid.parameters.dz, 0.2);
/// assert_eq!(zero_grid._u(), &vec![0.0; 264]);
/// ```
///
/// ```
/// use rboson::grid::{Grid, GridParameters, Order};
/// let grid_parameters = GridParameters::new(0.5, 0.5, 16, 16, Order::Fourth);
/// let scalar_grid = Grid::from_scalar(&grid_parameters, 1.0);
/// assert_eq!(scalar_grid.idx(0, 0), 1.0);
/// assert_eq!(scalar_grid.idx(0, 19), 1.0);
/// assert_eq!(scalar_grid.idx(19, 0), 1.0);
/// assert_eq!(scalar_grid.idx(19, 19), 1.0);
/// ```
pub struct Grid<'a> {
    pub parameters: &'a GridParameters,
    u: Vec<f64>,
}

impl Grid<'_> {
    pub fn new(parameters: &GridParameters) -> Grid {
        Grid {
            parameters,
            u: vec![0.0; parameters.dim as usize],
        }
    }

    pub fn from_vec(parameters: &GridParameters, u: Vec<f64>) -> Grid {
        assert_eq!(u.len(), parameters.dim as usize);
        Grid { parameters, u }
    }

    pub fn from_scalar(parameters: &GridParameters, scalar: f64) -> Grid {
        Grid {
            parameters,
            u: vec![scalar; parameters.dim as usize],
        }
    }

    pub fn from_function<F>(parameters: &GridParameters, f: F) -> Grid
    where
        F: Fn(f64, f64) -> f64 + Sync + Send,
    {
        let u = (0..parameters.nr_total)
            .into_par_iter()
            .flat_map_iter(|i| {
                let f = &f;
                (0..parameters.nz_total).map(move |j| f(parameters.r(i), parameters.z(j)))
            })
            .collect();
        Grid { parameters, u }
    }

    pub fn assign(&mut self, i: usize, j: usize, value: f64) {
        self.u[i + self.parameters.nr_total * j] = value;
    }

    pub fn idx(&self, i: usize, j: usize) -> f64 {
        self.u[i + self.parameters.nr_total * j]
    }

    pub fn _u(&self) -> &Vec<f64> {
        &self.u
    }

    pub fn save_csv(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;

        for i in 0..self.parameters.nr_total {
            let row: Vec<String> = (0..self.parameters.nz_total)
                .map(|j| self.idx(i, j).to_string())
                .collect();
            writeln!(file, "{}", row.join(", "))?;
        }
        Ok(())
    }

    pub fn save_npy(&self, filename: &str) -> std::io::Result<()> {
        use std::io::Write;

        let nr = self.parameters.nr_total;
        let nz = self.parameters.nz_total;

        // Create the header dictionary
        let header_dict = format!(
            "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {})}}",
            nr, nz
        );

        // Convert the header to bytes
        let mut header_bytes = header_dict.into_bytes();

        // Calculate the total length of the header section
        let prefix_len = 10; // Magic string (6 bytes) + version (2 bytes) + header length (2 bytes)
        let total_header_len = prefix_len + header_bytes.len() + 1; // +1 for the newline character after padding.

        // Calculate padding to align to 64 bytes
        let padding_len = (64 - (total_header_len % 64)) % 64;

        // Append padding spaces
        header_bytes.extend(vec![b' '; padding_len]);

        // Append the newline character
        header_bytes.push(b'\n');

        // Update the header length
        let header_len = header_bytes.len() as u16;

        let mut file = File::create(filename)?;

        // Write the magic string and version
        file.write_all(b"\x93NUMPY")?; // Magic string
        file.write_all(&[1, 0])?; // Version number: 1.0

        // Write the header length in little-endian format
        file.write_u16::<LittleEndian>(header_len)?;

        // Write the header
        file.write_all(&header_bytes)?;

        // Write the array data
        for val in &self.u {
            file.write_f64::<LittleEndian>(*val)?;
        }

        Ok(())
    }
}
