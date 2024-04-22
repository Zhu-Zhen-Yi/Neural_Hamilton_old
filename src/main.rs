use peroxide::fuga::*;
use rugfield::{grf, Kernel};
use rayon::prelude::*;
use candle_core::{scalar::TensorScalar, DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear, VarMap, Optimizer, loss};
use candle_optimisers::adam::{Adam, ParamsAdam};
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};

fn main() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;
    println!("Device: {:?}", dev);

    let n_train = 1000usize;
    let n_val = 200usize;

    let ds = Dataset::generate(n_train, n_val, &dev)?;

    // Plot data
    let u_train = &ds.train_u.detach().to_vec2()?;
    let y_train = &ds.train_y.detach().to_vec2()?;
    let gu_train = &ds.train_Gu.detach().to_vec2()?;
    
    let x_train = linspace(0, 1, 100);

    let mut plt = Plot2D::new();
    plt
        .set_domain(x_train)
        .insert_image(u_train[0].clone())
        .set_xlabel(r"$x$")
        .set_ylabel(r"$V(x)$")
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path("potential.png")
        .savefig()?;

    let mut plt = Plot2D::new();
    plt
        .set_domain(y_train[0].clone())
        .insert_image(gu_train[0].clone())
        .set_xlabel(r"$t$")
        .set_ylabel(r"$x(t)$")
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path("x.png")
        .savefig()?;

    Ok(())
}

// ┌─────────────────────────────────────────────────────────┐
//  Neural Network
// └─────────────────────────────────────────────────────────┘
pub struct DeepONet {
    branch_net: Vec<Linear>,
    trunk_net: Vec<Linear>,
    bias: Tensor,
}

#[derive(Debug, Copy, Clone)]
pub struct HyperParams {
    pub x_sensors: usize,
    pub y_sensors: usize,
    pub p: usize,
    pub hidden_size: usize,
    pub hidden_depth: usize,
    pub learning_rate: f64,
    pub epoch: usize,
    pub batch_size: usize,
}

impl DeepONet {
    pub fn new(vs: VarBuilder, hparam: HyperParams) -> anyhow::Result<Self> {
        let m = hparam.x_sensors;
        let l = hparam.y_sensors;
        let p = hparam.p;
        let hidden_size = hparam.hidden_size;
        let hidden_depth = hparam.hidden_depth;

        let mut branch_net = vec![linear(m, hidden_size, vs.pp("branch_first"))?];
        for i in 1 .. hidden_depth {
            branch_net.push(linear(hidden_size, hidden_size, vs.pp(&format!("branch{}", i)))?);
        }
        branch_net.push(linear(hidden_size, p, vs.pp("branch_last"))?);

        let mut trunk_net = vec![linear(1, hidden_size, vs.pp("trunk_first"))?];
        for i in 1 .. hidden_depth {
            trunk_net.push(linear(hidden_size, hidden_size, vs.pp(&format!("trunk{}", i)))?);
        }
        trunk_net.push(linear(hidden_size, p, vs.pp("trunk_last"))?);

        let bias = Tensor::zeros(&[l, 1], DType::F32, vs.device())?;

        Ok(Self {
            branch_net,
            trunk_net,
            bias,
        })
    }

    pub fn forward(&self, u: &Tensor, y_vec: &Vec<Tensor>) -> anyhow::Result<Tensor> {
        let mut u = u.clone();
        let mut y_vec = y_vec.clone();
        let n = self.branch_net.len();
        for (branch, trunk) in self.branch_net.iter().take(n-1).zip(self.trunk_net.iter()) {
            u = branch.forward(&u)?;
            u = u.gelu()?;
            for y in y_vec.iter_mut() {
                *y = trunk.forward(y)?; // 1 -> p
                *y = y.gelu()?;
            }
        }
        u = self.branch_net.last().unwrap().forward(&u)?;
        for y in y_vec.iter_mut() {
            *y = self.trunk_net.last().unwrap().forward(y)?;
        }
        // u : B x p
        // y : [B x p; l]
        
        
        //let mut s = Tensor::zeros(&[y_vec[0].size()[0], 1], DType::F32, y_vec[0].device())?;

        todo!()
    }
}

// ┌─────────────────────────────────────────────────────────┐
//  Dataset
// └─────────────────────────────────────────────────────────┘
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Dataset {
    pub train_u: Tensor,
    pub train_y: Vec<Tensor>,
    pub train_Gu: Vec<Tensor>,
    pub val_u: Tensor,
    pub val_y: Vec<Tensor>,
    pub val_Gu: Vec<Tensor>,
}

impl Dataset {
    #[allow(non_snake_case)]
    pub fn generate(n_train: usize, n_val: usize, device: &Device) -> Result<Self> {
        let m = 100; // # sensors
        let n = n_train + n_val;
        let l = 0.15;

        let grf_vec = (0 .. n).into_par_iter()
            .progress_with(ProgressBar::new(n as u64))
            .map(|_| grf(m, Kernel::SquaredExponential(l)))
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
        let l = y_vec[0].len();

        let x_vec = linspace(0, 1, m);
        let u_vec = grf_scaled_vec.par_iter()
            .map(|grf| grf.iter().zip(x_vec.iter()).map(|(g, t)| 1f64 - 4f64 * g * t * (1f64 - t)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let train_u = u_vec.iter().take(n_train).flatten().cloned().collect::<Vec<_>>();
        let train_y = y_vec.iter().take(n_train).map(|y| Tensor::from_vec(y.clone(), &[n_train, 1], device).unwrap()).collect::<Vec<_>>();
        let train_Gu = Gu_vec.iter().take(n_train).map(|Gu| Tensor::from_vec(Gu.clone(), &[n_train, 1], device).unwrap()).collect::<Vec<_>>();

        let val_u = u_vec.iter().skip(n_train).flatten().cloned().collect::<Vec<_>>();
        let val_y = y_vec.iter().skip(n_train).map(|y| Tensor::from_vec(y.clone(), &[n_val, 1], device).unwrap()).collect::<Vec<_>>();
        let val_Gu = Gu_vec.iter().skip(n_train).map(|Gu| Tensor::from_vec(Gu.clone(), &[n_val, 1], device).unwrap()).collect::<Vec<_>>();

        Ok(Self {
            train_u: Tensor::from_vec(train_u, &[n_train, m], device)?,
            train_y,
            train_Gu,
            val_u: Tensor::from_vec(val_u, &[n_val, m], device)?,
            val_y,
            val_Gu,
        })
    }

    #[allow(non_snake_case)]
    pub fn train_set(&self, dev: &Device) -> anyhow::Result<(Tensor, Vec<Tensor>, Vec<Tensor>)> {
        Ok((
            self.train_u.to_device(dev)?, 
            self.train_y.iter().map(|y| y.to_device(dev).unwrap()).collect(),
            self.train_Gu.iter().map(|Gu| Gu.to_device(dev).unwrap()).collect()
        ))
    }

    #[allow(non_snake_case)]
    pub fn val_set(&self, dev: &Device) -> anyhow::Result<(Tensor, Vec<Tensor>, Vec<Tensor>)> {
        Ok((
            self.val_u.to_device(dev)?,
            self.val_y.iter().map(|y| y.to_device(dev).unwrap()).collect(),
            self.val_Gu.iter().map(|Gu| Gu.to_device(dev).unwrap()).collect()
        ))
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
        let y = grf.iter().zip(x.iter()).map(|(g, t)| 1f64 - 4f64 * g * t * (1f64 - t)).collect::<Vec<_>>();
        let cs = cubic_hermite_spline(&x, &y, Quadratic)?;
        let cs_deriv = cs.derivative();
        Ok(Self { cs, cs_deriv })
    }
}

impl ODEProblem for GRFODE {
    fn initial_conditions(&self) -> Vec<f64> {
        vec![1f64, 0f64]
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
    let (t_vec, xp_vec) = solver.solve(&grf_ode, (0f64, 10f64), 1e-3)?;
    let (x_vec, _): (Vec<f64>, Vec<f64>) = xp_vec.into_iter().map(|xp| (xp[0], xp[1])).unzip();
    Ok((t_vec, x_vec))
}
