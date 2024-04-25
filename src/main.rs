use peroxide::fuga::*;
use peroxide::fuga::anyhow::Result;
use rugfield::{grf, Kernel};
use rayon::{prelude::*, vec};
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};

#[allow(non_snake_case)]
fn main() -> std::result::Result<(), Box<dyn Error>> {
    let n = 10000usize;

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
        let m = 100; // # sensors
        let u_l = Uniform(0.1, 0.5);
        let l = u_l.sample(n);

        let grf_vec = (0 .. n).into_par_iter().zip(l.into_par_iter())
            .progress_with(ProgressBar::new(n as u64))
            .map(|(_, l)| grf(m, Kernel::SquaredExponential(l)))
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
                grf.fmap(|x| (x - grf_min) / (grf_max- grf_min))
            }).collect::<Vec<_>>();

        let (y_vec, Gu_vec): (Vec<Vec<f64>>, Vec<Vec<f64>>) = grf_scaled_vec.par_iter()
            .progress_with(ProgressBar::new(n as u64))
            .map(|grf| solve_grf_ode(grf).unwrap())
            .unzip();

        // Filter odd data
        let mut ics = vec![];
        for (i, gu) in Gu_vec.iter().enumerate() {
            if gu.iter().any(|gu| gu.abs() > 1.1) {
                continue
            }
            ics.push(i);
        }
        let grf_scaled_vec = ics.iter().map(|i| grf_scaled_vec[*i].clone()).collect::<Vec<_>>();
        let y_vec = ics.iter().map(|i| y_vec[*i].clone()).collect::<Vec<_>>();
        let Gu_vec = ics.iter().map(|i| Gu_vec[*i].clone()).collect::<Vec<_>>();

        let x_vec = linspace(0, 1, m);
        let u_vec = grf_scaled_vec.par_iter()
            .map(|grf| grf.iter().zip(x_vec.iter()).map(|(g, t)| 2f64 - 16f64 * g * t * (1f64 - t)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

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
    pub fn new(grf: &[f64]) -> anyhow::Result<Self> {
        let x = linspace(0f64, 1f64, grf.len());
        let y = grf.iter().zip(x.iter()).map(|(g, t)| 2f64 - 16f64 * g * t * (1f64 - t)).collect::<Vec<_>>();
        let cs = cubic_hermite_spline(&x, &y, Quadratic)?;
        let cs_deriv = cs.derivative();
        Ok(Self { cs, cs_deriv })
    }
}

impl ODEProblem for GRFODE {
    fn initial_conditions(&self) -> Vec<f64> {
        vec![0f64, 0f64]
    }

    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        dy[0] = y[1];                   // dot(x) = p
        dy[1] = - self.cs_deriv.eval(y[0]);   // dot(p) = - partial V / partial x
        Ok(())
    }
}

pub fn solve_grf_ode(grf: &[f64]) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let grf_ode = GRFODE::new(grf)?;
    let solver = BasicODESolver::new(RK4);
    let (t_vec, xp_vec) = solver.solve(&grf_ode, (0f64, 2f64), 1e-3)?;
    let (x_vec, _): (Vec<f64>, Vec<f64>) = xp_vec.into_iter().map(|xp| (xp[0], xp[1])).unzip();

    // Choose 100 samples only
    let ics = linspace(0, t_vec.len() as f64 - 1f64, 100);
    let ics = ics.into_iter().map(|x| x.round() as usize).collect::<Vec<_>>();

    let t_vec = ics.iter().map(|i| t_vec[*i]).collect();
    let x_vec = ics.iter().map(|i| x_vec[*i]).collect();

    Ok((t_vec, x_vec))
}