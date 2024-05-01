use peroxide::fuga::*;
use peroxide::fuga::anyhow::Result;
use rugfield::{grf, Kernel};
use rayon::prelude::*;
use indicatif::{ProgressBar, ParallelProgressIterator};

#[allow(non_snake_case)]
fn main() -> std::result::Result<(), Box<dyn Error>> {
    let n = 500000usize;

    println!("Generate dataset...");
    let ds = Dataset::generate(n, 0.8)?;
    ds.write_parquet()?;
    println!("Generate dataset complete");

    Ok(())
}

// ┌─────────────────────────────────────────────────────────┐
//  Dataset
// └─────────────────────────────────────────────────────────┘
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Dataset {
    pub train_u: Matrix,
    pub train_y: Matrix,
    pub train_Gu: Matrix,
    pub val_u: Matrix,
    pub val_y: Matrix,
    pub val_Gu: Matrix,
}

impl Dataset {
    #[allow(non_snake_case)]
    pub fn generate(n: usize, f_train: f64) -> Result<Self> {
        // Generate GRF
        let b = 5;  // # bases
        let m = 100; // # sensors
        let u_l = Uniform(0.1, 0.4);
        let l = u_l.sample(n);

        let grf_vec = (0 .. n).into_par_iter().zip(l.into_par_iter())
            .progress_with(ProgressBar::new(n as u64))
            .map(|(_, l)| grf(b, Kernel::SquaredExponential(l)))
            .collect::<Vec<_>>();

        // Normalize
        let grf_max_vec = grf_vec
            .iter()
            .map(|grf| grf.max())
            .collect::<Vec<_>>();

        let grf_min_vec = grf_vec
            .iter()
            .map(|grf| grf.min())
            .collect::<Vec<_>>();

        let grf_max = grf_max_vec.max();
        let grf_min = grf_min_vec.min();

        let grf_scaled_vec = grf_vec.iter()
            .map(|grf| {
                grf.fmap(|x| (x - grf_min) / (grf_max - grf_min))
            }).collect::<Vec<_>>();

        let mut stdrng = stdrng_from_seed(42);
        let mut x_dist_vec = vec![vec![0f64; n]];
        let b_step = 1f64 / (b as f64);
        for i in 0 .. b {
            let u = Uniform(b_step * (i as f64), b_step * ((i + 1) as f64));
            let x_dist = u.sample_with_rng(&mut stdrng, n);
            x_dist_vec.push(x_dist);
        }
        x_dist_vec.push(vec![1f64; n]);
        let x_dist = matrix(x_dist_vec.into_iter().flatten().collect(), n, b+2, Col);
        let x_vec = x_dist.change_shape().to_vec();

        let potential_vec = grf_scaled_vec.into_par_iter()
            .map(|grf| {
                let mut potential = grf.fmap(|x| 2f64 - 4f64 * x);
                potential.insert(0, 2f64);
                potential.push(2f64);
                potential
            })
            .collect::<Vec<_>>();

        let (y_vec, Gu_vec): (Vec<Vec<f64>>, Vec<Vec<f64>>) = potential_vec.par_iter()
            .zip(x_vec.par_iter())
            .progress_with(ProgressBar::new(n as u64))
            .map(|(potential, x)| solve_grf_ode(potential, x).unwrap())
            .unzip();

        // Filter odd data
        let mut ics = vec![];
        for (i, gu) in Gu_vec.iter().enumerate() {
           if gu.iter().any(|gu| gu.abs() > 1.1) {
               continue
           }
           ics.push(i);
        }
        let x_vec = ics.iter().map(|i| x_vec[*i].clone()).collect::<Vec<_>>();
        let potential_vec = ics.iter().map(|i| potential_vec[*i].clone()).collect::<Vec<_>>();
        let y_vec = ics.iter().map(|i| y_vec[*i].clone()).collect::<Vec<_>>();
        let Gu_vec = ics.iter().map(|i| Gu_vec[*i].clone()).collect::<Vec<_>>();
        
        let sensors = linspace(0, 1, m);
        let u_vec = potential_vec.par_iter()
            .zip(x_vec.par_iter())
            .map(|(potential, x)| {
                let cs = cubic_hermite_spline(x, potential, Quadratic)?;
                Ok(cs.eval_vec(&sensors))
            })
            .collect::<Result<Vec<_>>>()?;

        let n_train = (n as f64 * f_train).round() as usize;
        let n_val = n - n_train;

        println!("n_train: {}", n_train);
        println!("n_val: {}", n_val);

        let train_u = u_vec.iter().take(n_train).cloned().collect::<Vec<_>>();
        let train_y = y_vec.iter().take(n_train).cloned().collect::<Vec<_>>();
        let train_Gu = Gu_vec.iter().take(n_train).cloned().collect::<Vec<_>>();

        let val_u = u_vec.iter().skip(n_train).cloned().collect::<Vec<_>>();
        let val_y = y_vec.iter().skip(n_train).cloned().collect::<Vec<_>>();
        let val_Gu = Gu_vec.iter().skip(n_train).cloned().collect::<Vec<_>>();

        Ok(Self {
            train_u: py_matrix(train_u),
            train_y: py_matrix(train_y),
            train_Gu: py_matrix(train_Gu),
            val_u: py_matrix(val_u),
            val_y: py_matrix(val_y),
            val_Gu: py_matrix(val_Gu),
        })
    }

    #[allow(non_snake_case)]
    pub fn train_set(&self) -> (Matrix, Matrix, Matrix) {
        (
            self.train_u.clone(),
            self.train_y.clone(),
            self.train_Gu.clone()
        )
    }

    #[allow(non_snake_case)]
    pub fn val_set(&self) -> (Matrix, Matrix, Matrix) {
        (
            self.val_u.clone(),
            self.val_y.clone(),
            self.val_Gu.clone()
        )
    }

    #[allow(non_snake_case)]
    pub fn write_parquet(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let data_folder = "data";
        if !std::path::Path::new(data_folder).exists() {
            std::fs::create_dir(data_folder)?;
        }

        let (train_u, train_y, train_Gu) = self.train_set();
        let (val_u, val_y, val_Gu) = self.val_set();

        let mut df = DataFrame::new(vec![]);
        df.push("train_u", Series::new(train_u.data));
        df.push("train_y", Series::new(train_y.data));
        df.push("train_Gu", Series::new(train_Gu.data));

        
        let train_path = format!("{}/train.parquet", data_folder);
        df.write_parquet(&train_path, CompressionOptions::Uncompressed)?;

        let mut df = DataFrame::new(vec![]);
        df.push("val_u", Series::new(val_u.data));
        df.push("val_y", Series::new(val_y.data));
        df.push("val_Gu", Series::new(val_Gu.data));

        let val_path = format!("{}/val.parquet", data_folder);
        df.write_parquet(&val_path, CompressionOptions::Uncompressed)?;

        Ok(())
    }
}

#[allow(dead_code)]
pub struct GRFODE {
    cs: CubicHermiteSpline,
    cs_deriv: CubicHermiteSpline,
}

impl GRFODE {
    pub fn new(potential: &[f64], x: &[f64]) -> anyhow::Result<Self> {
        let cs = cubic_hermite_spline(x, potential, Quadratic)?;
        let cs_deriv = cs.derivative();
        Ok(Self { cs, cs_deriv })
    }
}

impl ODEProblem for GRFODE {
    fn initial_conditions(&self) -> Vec<f64> {
        vec![0f64, 0f64]
    }

    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        dy[0] = y[1];                           // dot(x) = p
        dy[1] = - self.cs_deriv.eval(y[0]);     // dot(p) = - partial V / partial x
        Ok(())
    }
}

pub fn solve_grf_ode(potential: &[f64], x: &[f64]) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let grf_ode = GRFODE::new(potential, x)?;
    let solver = BasicODESolver::new(RK4);
    let (t_vec, xp_vec) = solver.solve(&grf_ode, (0f64, 2f64), 1e-3)?;
    let (x_vec, _): (Vec<f64>, Vec<f64>) = xp_vec.into_iter().map(|xp| (xp[0], xp[1])).unzip();

    // Chebyshev nodes
    let n = 100;
    let cs = cubic_hermite_spline(&t_vec, &x_vec, Quadratic)?;
    let t_vec = chebyshev_nodes(n, 0f64, 2f64);
    let x_vec = cs.eval_vec(&t_vec);

    Ok((t_vec, x_vec))
}
