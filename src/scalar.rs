pub enum ScalarConstraint {
    Omega(f64),
    Coordinate(usize, usize),
}

pub struct ScalarField {
    pub l: usize,
    pub m: f64,
    pub constraint: ScalarConstraint,
}
