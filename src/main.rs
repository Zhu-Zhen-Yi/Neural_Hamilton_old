use peroxide::fuga::*;
use peroxide::fuga::anyhow::Result;
use rugfield::{grf, Kernel};
use rayon::prelude::*;
use candle_core::{scalar::TensorScalar, DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear, VarMap, Optimizer, loss};
use candle_optimisers::adam::{Adam, ParamsAdam};
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};

#[allow(non_snake_case)]
fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    println!("Device: {:?}", dev);

    let mut rng = smallrng_from_seed(42);

    let n_train = 10000usize;
    let n_val = 2000usize;

    let ds = Dataset::generate(n_train, n_val, &dev)?;

    // Plot data
    //let u_train = &ds.train_u.detach().to_vec2()?;
    //let y_train = ds.train_y
    //    .iter()
    //    .flat_map(|y| {
    //        let y: Vec<f32> = y.detach().to_vec1().unwrap();
    //        y.into_iter().map(|x| x as f64).collect::<Vec<f64>>()
    //    })
    //    .collect();
    //let y_train = matrix(y_train, n_train, 100, Col);
    //let gu_train = ds.train_Gu
    //    .iter()
    //    .flat_map(|Gu| {
    //        let Gu: Vec<f32> = Gu.detach().to_vec1().unwrap();
    //        Gu.into_iter().map(|x| x as f64).collect::<Vec<f64>>()
    //    })
    //    .collect();
    //let gu_train = matrix(gu_train, n_train, 100, Col);
    //
    //let x_train = linspace(0, 1, 100);

    //let mut plt = Plot2D::new();
    //plt
    //    .set_domain(x_train)
    //    .insert_image(u_train[0].clone())
    //    .set_xlabel(r"$x$")
    //    .set_ylabel(r"$V(x)$")
    //    .set_style(PlotStyle::Nature)
    //    .tight_layout()
    //    .set_dpi(600)
    //    .set_path("potential.png")
    //    .savefig()?;

    //let mut plt = Plot2D::new();
    //plt
    //    .set_domain(y_train.row(0))
    //    .insert_image(gu_train.row(0))
    //    .set_xlabel(r"$t$")
    //    .set_ylabel(r"$x(t)$")
    //    .set_style(PlotStyle::Nature)
    //    .tight_layout()
    //    .set_dpi(600)
    //    .set_path("x.png")
    //    .savefig()?;

    let (model, train_history, val_history) = train(ds, &dev, &mut rng)?;

    let epochs = linspace(1, train_history.len() as f64, train_history.len());

    let mut plt = Plot2D::new();
    plt
        .set_domain(epochs.clone())
        .insert_image(train_history.clone())
        .set_xlabel(r"Epoch")
        .set_ylabel(r"Train loss")
        .set_yscale(PlotScale::Log)
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path("train_loss.png")
        .savefig()?;

    let mut plt = Plot2D::new();
    plt
        .set_domain(epochs.clone())
        .insert_image(val_history.clone())
        .set_xlabel(r"Epoch")
        .set_ylabel(r"Val loss")
        .set_yscale(PlotScale::Log)
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path("val_loss.png")
        .savefig()?;

    let ds_test = Dataset::generate(1, 1, &dev)?;
    let (test_u, test_y, test_Gu) = ds_test.train_set(&dev)?;
    let Gu_hat = model.forward(&test_u, &test_y)?;

    let test_u: Vec<f32> = test_u.detach().reshape(100).unwrap().to_vec1()?;
    let test_u = test_u.into_iter().map(|x| x as f64).collect::<Vec<f64>>();


    let test_Gu: Vec<f32> = test_Gu
        .iter()
        .map(|Gu| {
            let Gu = Gu.detach().reshape(1).unwrap().to_vec1().unwrap();
            Gu[0]
        })
        .collect();
    let test_Gu = test_Gu.into_iter().map(|x| x as f64).collect::<Vec<f64>>();

    let test_y: Vec<f32> = test_y
        .iter()
        .map(|y| {
            let y = y.detach().reshape(1).unwrap().to_vec1().unwrap();
            y[0]
        })
        .collect();
    let test_y = test_y.into_iter().map(|x| x as f64).collect::<Vec<f64>>();

    let Gu_hat: Vec<f32> = Gu_hat
        .iter()
        .map(|Gu_hat| {
            let Gu_hat = Gu_hat.detach().reshape(1).unwrap().to_vec1().unwrap();
            Gu_hat[0]
        })
        .collect();
    let Gu_hat = Gu_hat.into_iter().map(|x| x as f64).collect::<Vec<f64>>();

    let x_train = linspace(0, 1, 100);

    let mut plt = Plot2D::new();
    plt
        .set_domain(x_train)
        .insert_image(test_u)
        .set_xlabel(r"$x$")
        .set_ylabel(r"$V(x)$")
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path("potential_test.png")
        .savefig()?;

    let mut plt = Plot2D::new();
    plt
        .set_domain(test_y)
        .insert_image(test_Gu)
        .insert_image(Gu_hat)
        .set_xlabel(r"$t$")
        .set_ylabel(r"$x(t)$")
        .set_line_style(vec![(0, LineStyle::Solid), (1, LineStyle::Dashed)])
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path("x_test.png")
        .savefig()?;

    Ok(())
}

// ┌─────────────────────────────────────────────────────────┐
//  Train
// └─────────────────────────────────────────────────────────┘
#[allow(non_snake_case)]
pub fn train(ds: Dataset, dev: &Device, rng: &mut SmallRng) -> Result<(DeepONet, Vec<f64>, Vec<f64>)> {
    let (train_u, train_y, train_Gu) = ds.train_set(dev)?;
    let (val_u, val_y, val_Gu) = ds.val_set(dev)?;

    println!("train_u Dtype: {:?}", train_u.dtype());
    println!("train_y Dtype: {:?}", train_y[0].dtype());
    println!("train_Gu Dtype: {:?}", train_Gu[0].dtype());

    let hparam = HyperParams {
        x_sensors: 100,
        y_sensors: 100,
        p: 10,
        hidden_size: 40,
        hidden_depth: 5,
        learning_rate: 1e-2,
        batch_size: 1000,
        epoch: 500
    };

    let mut lr = hparam.learning_rate;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

    let model = DeepONet::new(vb, hparam)?;

    let adam_param = ParamsAdam {
        lr: hparam.learning_rate,
        ..Default::default()
    };
    let mut adam = Adam::new(varmap.all_vars(), adam_param)?;

    let mut train_history = vec![0f64; hparam.epoch];
    let mut val_history = vec![0f64; hparam.epoch];

    let mut train_loss = 0f32;
    let mut val_loss = 0f32;

    let train_batch = train_u.dims()[0] / hparam.batch_size;
    
    let pb = ProgressBar::new(hparam.epoch as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("##-"));

    for epoch in 0 .. hparam.epoch {
        pb.set_position(epoch as u64);
        let msg = format!("epoch: {}, train_loss: {:.4e}, val_loss: {:.4e}", epoch, train_loss, val_loss);
        pb.set_message(msg);

        let mut ics = (0u32 .. train_u.dims()[0] as u32).collect::<Vec<_>>();
        ics.shuffle(rng);

        let mut epoch_loss = 0f32;
        for i in 0 .. train_batch {
            let batch_ics = Tensor::from_slice(
                &ics[i * hparam.batch_size .. (i+1) * hparam.batch_size],
                hparam.batch_size,
                dev
            )?;
            let batch_u = train_u.index_select(&batch_ics, 0)?;
            let batch_y = batch_slicing(&train_y, &batch_ics)?;
            let batch_Gu = batch_slicing(&train_Gu, &batch_ics)?;

            let Gu_hat = model.forward(&batch_u, &batch_y)?;
            let loss: Tensor = batch_Gu.iter().zip(Gu_hat.iter())
                .map(|(Gu, Gu_hat)| loss::mse(Gu, Gu_hat).unwrap())
                .reduce(|a, b| (a + b).unwrap()).unwrap();
            let loss = (loss / Tensor::new(100f32, dev)?)?;
            adam.backward_step(&loss)?;

            let loss: f32 = loss.to_scalar()?;
            epoch_loss += loss;
        }
        epoch_loss /= train_batch as f32;
        train_loss = epoch_loss;

        let Gu_hat = model.forward(&val_u, &val_y)?;
        let loss: Tensor = val_Gu.iter().zip(Gu_hat.iter())
            .map(|(Gu, Gu_hat)| loss::mse(Gu, Gu_hat).unwrap())
            .reduce(|a, b| (a + b).unwrap()).unwrap();
        val_loss = loss.to_scalar()?;
        val_loss /= 100f32;

        train_history[epoch] = train_loss as f64;
        val_history[epoch] = val_loss as f64;
        adam.set_learning_rate(lr);
    }

    println!("train_loss: {:.4e}, val_loss: {:.4e}", train_loss, val_loss);

    Ok((model, train_history, val_history))
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
        let mut u = u.clone();          // u: B x m
        let mut y_vec = y_vec.to_vec(); // y_vec: B x l
        let n = self.branch_net.len();
        for (branch, trunk) in self.branch_net.iter().take(n-1).zip(self.trunk_net.iter()) {
            u = branch.forward(&u)?;
            u = u.gelu()?;
            for y in y_vec.iter_mut() {
                *y = trunk.forward(y)?; // 1 -> p
                *y = y.gelu()?;
            }
        }
        u = self.branch_net.last().unwrap().forward(&u)?;           // u: B x p
        for (y, bias) in y_vec.iter_mut().zip(self.bias.iter()) {
            *y = self.trunk_net.last().unwrap().forward(y)?;        // y: B x p
            *y = y.mul(&u)?.sum(1)?.broadcast_add(bias)?;           // y: B
            *y = y.reshape((y.dims()[0],1))?;                       // y: B x 1
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

        let u_vec = u_vec.into_iter().map(|u| u.into_iter().map(|u| u as f32).collect::<Vec<_>>()).collect::<Vec<_>>();
        let y_vec = y_vec.into_iter().map(|y| y.into_iter().map(|y| y as f32).collect::<Vec<_>>()).collect::<Vec<_>>();
        let Gu_vec = Gu_vec.into_iter().map(|Gu| Gu.into_iter().map(|Gu| Gu as f32).collect::<Vec<_>>()).collect::<Vec<_>>();

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

    // Choose 100 samples only
    let ics = linspace(0, t_vec.len() as f64 - 1f64, 100);
    let ics = ics.into_iter().map(|x| x.round() as usize).collect::<Vec<_>>();

    let t_vec = ics.iter().map(|i| t_vec[*i]).collect();
    let x_vec = ics.iter().map(|i| x_vec[*i]).collect();

    Ok((t_vec, x_vec))
}

// ┌─────────────────────────────────────────────────────────┐
//  Utils
// └─────────────────────────────────────────────────────────┘
pub fn vec_to_tensor(vec: Vec<Vec<f32>>, dev: &Device) -> Result<Vec<Tensor>> {
    let mat = py_matrix(vec);
    let mut tensors = vec![];
    for i in 0 .. mat.col {
        let col = mat.col(i).into_iter().map(|x| x as f32).collect::<Vec<_>>();
        tensors.push(Tensor::from_vec(col, &[mat.row, 1], dev)?);
    }
    Ok(tensors)
}

pub fn batch_slicing(vec: &[Tensor], ics: &Tensor) -> Result<Vec<Tensor>> {
    Ok(vec.iter().map(|v| v.index_select(ics, 0)).collect::<std::result::Result<Vec<_>, _>>()?)
}
