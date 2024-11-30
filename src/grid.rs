/// This module defines the structure for the grid functions.

/// Determines the finite difference order.
///
/// This enum is used to determine the number of ghost zones and also for some boundary calculations.
///
/// # Examples
///
/// ```
/// use rboson::grid::Order;
/// let order = Order::Second;
/// assert_eq!(order as i64, 2);
/// ```
pub enum Order {
    Second = 2,
    Fourth = 4,
}

pub fn index_to_coordinate(k: i64, dr: f64, ghost: i64) -> f64 {
    ((k - ghost) as f64 + 0.5) * dr
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
/// assert_eq!(grid_parameters.dr, 0.1);
/// assert_eq!(grid_parameters.dz, 0.2);
/// assert_eq!(grid_parameters.nr_interior, 10);
/// assert_eq!(grid_parameters.nz_interior, 20);
/// assert_eq!(grid_parameters.ghost(), 1);
/// assert_eq!(grid_parameters.nr_total(), 10 + 2 * 1);
/// assert_eq!(grid_parameters.nz_total(), 20 + 2 * 1);
/// assert_eq!(grid_parameters.dim, (10 + 2 * 1) * (20 + 2 * 1)); // 264
/// assert_eq!(grid_parameters.r(0), -0.05);
/// assert_eq!(grid_parameters.r(1), 0.05);
/// assert_eq!(grid_parameters.z(0), -0.1);
/// assert_eq!(grid_parameters.z(1), 0.1);
/// ```
pub struct GridParameters {
    pub dr: f64,
    pub dz: f64,
    pub nr_interior: i64,
    pub nz_interior: i64,
    nr_total: i64,
    nz_total: i64,
    pub dim: i64,
    pub order: Order,
    ghost: i64,
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
        nr_interior: i64,
        nz_interior: i64,
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
            .map(|i| index_to_coordinate(i, dr, ghost))
            .collect();
        let z: Vec<f64> = (0..nz_total)
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

    pub fn nr_total(&self) -> i64 {
        self.nr_total
    }

    pub fn nz_total(&self) -> i64 {
        self.nz_total
    }

    pub fn ghost(&self) -> i64 {
        self.ghost
    }

    pub fn r(&self, i: i64) -> f64 {
        self.r[i as usize]
    }

    pub fn z(&self, j: i64) -> f64 {
        self.z[j as usize]
    }

    pub fn coord(&self, i: i64, j: i64) -> (f64, f64) {
        (self.r[i as usize], self.z[j as usize])
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
        Grid {
            parameters,
            u,
        }
    }

    pub fn from_scalar(parameters: &GridParameters, scalar: f64) -> Grid {
        Grid {
            parameters,
            u: vec![scalar; parameters.dim as usize],
        }
    }

    pub fn from_function<F>(parameters: &GridParameters, f: F) -> Grid
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut u = vec![0.0; parameters.dim as usize];
        for i in 0..parameters.nr_total {
            for j in 0..parameters.nz_total {
                u[(i + parameters.nr_total * j) as usize] = f(parameters.r(i), parameters.z(j));
            }
        }
        Grid {
            parameters,
            u,
        }
    }

    pub fn assign(&mut self, i: i64, j: i64, value: f64) {
        self.u[(i + self.parameters.nr_total * j) as usize] = value;
    }

    pub fn idx(&self, i: i64, j: i64) -> f64 {
        self.u[(i + self.parameters.nr_total * j) as usize]
    }

    pub fn _u(&self) -> &Vec<f64> {
        &self.u
    }
}

enum ScalarConstraint {
    Omega(f64),
    Coordinate(i64, i64),
}

struct ScalarField {
    l: i64,
    m: f64,
    constraint: ScalarConstraint,
}

const MAX_MESSAGE_WIDTH: usize = 128 - 6 - 2;

fn message_print(message: &str) {
    let message_length = message.len();
    match message_length {
        0 => println!("****{}****", "*".repeat(MAX_MESSAGE_WIDTH)),
        1..MAX_MESSAGE_WIDTH => {
            let space = MAX_MESSAGE_WIDTH - message_length;
            let left_space = space / 2;
            let right_space = space - left_space;
            println!(
                "*** {}{}{} ***",
                " ".repeat(left_space),
                message,
                " ".repeat(right_space)
            )
        }
        _ => (),
    }
}
