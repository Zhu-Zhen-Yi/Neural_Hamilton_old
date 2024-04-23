use peroxide::fuga::*;
use peroxide::fuga::anyhow::Result;
use rugfield::{grf, Kernel};
use rayon::prelude::*;
use candle_core::{Device, Module, Tensor, DType};
use candle_nn::{Linear, VarBuilder, linear, VarMap, Optimizer, loss};
use candle_optimisers::adam::{Adam, ParamsAdam};
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    println!("Device: {:?}", dev);

    let n_train = 1000usize;
    let n_val = 200usize;

    let ds = Dataset::generate(n_train, n_val, &dev)?;

    // Plot data
    let u_train = &ds.train_u.detach().to_vec2()?;
    let y_train = &ds.train_y[0].to_vec1()?;
    let gu_train = &ds.train_Gu[0].to_vec1()?;
    
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
        .set_domain(y_train.clone())
        .insert_image(gu_train.clone())
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
//  Train
// └─────────────────────────────────────────────────────────┘
#[allow(non_snake_case)]
pub fn train(ds: Dataset, dev: &Device, rng: &mut SmallRng) -> Result<DeepONet> {
    let (train_u, train_y, train_Gu) = ds.train_set(dev)?;
    let (val_u, val_y, val_Gu) = ds.val_set(dev)?;

    let hparam = HyperParams {
        x_sensors: 100,
        y_sensors: 100,
        p: 10,
        hidden_size: 40,
        hidden_depth: 3,
        learning_rate: 1e-3,
        batch_size: 256,
        epoch: 200
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

    let model = DeepONet::new(vb, hparam)?;

    let adam_param = ParamsAdam {
        lr: hparam.learning_rate,
        ..Default::default()
    };
    let mut adam = Adam::new(varmap.all_vars(), adam_param)?;

    let mut train_loss = vec![0f32; hparam.epoch];
    let mut val_loss = vec![0f32; hparam.epoch];

    let train_batch = train_u.dims()[0] / hparam.batch_size;
    
    let pb = ProgressBar::new(hparam.epoch as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("##-"));

    for epoch in 0 .. hparam.epoch {
        pb.set_position(epoch as u64);
        //TODO
        //let msg = format!("epoch: {}, train_loss: {:.4e}, val_loss: {:.4e}", epoch, train_loss[], val_loss);
        pb.set_message(msg);

        let mut ics = (0u32 .. train_u.dims()[0] as u32).collect::<Vec<_>>();
        ics.shuffle(rng);

        let mut epoch_loss = 0f32;
        for i in 0 .. 
    }

    todo!()
}

// ┌─────────────────────────────────────────────────────────┐
//  Neural Network
// └─────────────────────────────────────────────────────────┘
pub struct DeepONet {
    branch_net: Vec<Linear>,
    trunk_net: Vec<Linear>,
    bias: Vec<Tensor>,
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
    pub fn new(vb: VarBuilder, hparam: HyperParams) -> Result<Self> {
        let m = hparam.x_sensors;
        let l = hparam.y_sensors;
        let p = hparam.p;
        let hidden_size = hparam.hidden_size;
        let hidden_depth = hparam.hidden_depth;

        let mut branch_net = vec![linear(m, hidden_size, vb.pp("branch_first"))?];
        for i in 1 .. hidden_depth {
            branch_net.push(linear(hidden_size, hidden_size, vb.pp(&format!("branch{}", i)))?);
        }
        branch_net.push(linear(hidden_size, p, vb.pp("branch_last"))?);

        let mut trunk_net = vec![linear(1, hidden_size, vb.pp("trunk_first"))?];
        for i in 1 .. hidden_depth {
            trunk_net.push(linear(hidden_size, hidden_size, vb.pp(&format!("trunk{}", i)))?);
        }
        trunk_net.push(linear(hidden_size, p, vb.pp("trunk_last"))?);

        let mut bias = vec![];
        for i in 0 .. l {
            bias.push(vb.get(1, &format!("bias{}", i))?);
        }

        Ok(Self {
            branch_net,
            trunk_net,
            bias,
        })
    }

    pub fn forward(&self, u: &Tensor, y_vec: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut u = u.clone();
        let mut y_vec = y_vec.to_vec();
        let n = self.branch_net.len();
        for (branch, trunk) in self.branch_net.iter().take(n-1).zip(self.trunk_net.iter()) {
            u = branch.forward(&u)?;
            u = u.gelu()?;
            for y in y_vec.iter_mut() {
                *y = trunk.forward(y)?; // 1 -> p
                *y = y.gelu()?;
            }
        }
        u = self.branch_net.last().unwrap().forward(&u)?;       // u: B x p
        for (y, bias) in y_vec.iter_mut().zip(self.bias.iter()) {
            *y = self.trunk_net.last().unwrap().forward(y)?;    // y: B x p
            *y = y.mul(&u)?.sum(1)?.broadcast_add(bias)?;       // y: B
        }

        Ok(y_vec)
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

        let x_vec = linspace(0, 1, m);
        let u_vec = grf_scaled_vec.par_iter()
            .map(|grf| grf.iter().zip(x_vec.iter()).map(|(g, t)| 1f64 - 4f64 * g * t * (1f64 - t)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let train_u = u_vec.iter().take(n_train).flatten().cloned().collect::<Vec<_>>();
        let train_y = y_vec.iter().take(n_train).cloned().collect::<Vec<_>>();
        let train_y = vec_to_tensor(train_y, device)?;
        let train_Gu = Gu_vec.iter().take(n_train).cloned().collect::<Vec<_>>();
        let train_Gu = vec_to_tensor(train_Gu, device)?;

        let val_u = u_vec.iter().skip(n_train).flatten().cloned().collect::<Vec<_>>();
        let val_y = y_vec.iter().skip(n_train).cloned().collect::<Vec<_>>();
        let val_y = vec_to_tensor(val_y, device)?;
        let val_Gu = Gu_vec.iter().skip(n_train).cloned().collect::<Vec<_>>();
        let val_Gu = vec_to_tensor(val_Gu, device)?;

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

// ┌─────────────────────────────────────────────────────────┐
//  Utils
// └─────────────────────────────────────────────────────────┘
pub fn vec_to_tensor(vec: Vec<Vec<f64>>, dev: &Device) -> Result<Vec<Tensor>> {
    let mat = py_matrix(vec);
    let mut tensors = vec![];
    for i in 0 .. mat.col {
        tensors.push(Tensor::from_vec(mat.col(i), &[mat.row, 1], dev)?);
    }
    Ok(tensors)
}

pub fn batch_slicing(vec: &[Tensor], ics: &Tensor) -> Result<Vec<Tensor>> {
    Ok(vec.iter().map(|v| v.index_select(ics, 0)).collect::<std::result::Result<Vec<_>, _>>()?)
}
