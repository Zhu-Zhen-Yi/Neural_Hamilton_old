use peroxide::fuga::*;
use peroxide::fuga::anyhow::Result;
use rugfield::{grf, Kernel};
use rayon::prelude::*;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear, VarMap, Optimizer, loss};
use candle_optimisers::adam::{Adam, ParamsAdam};
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};

#[allow(non_snake_case)]
fn main() -> std::result::Result<(), Box<dyn Error>> {
    let dev = Device::cuda_if_available(0)?;
    println!("Device: {:?}", dev);

    let mut rng = stdrng_from_seed(42);

    let n_train = 10000usize;
    let n_val = 2000usize;

    println!("Generate dataset...");
    let ds = Dataset::generate(n_train, n_val, &dev)?;
    ds.write_parquet()?;
    println!("Generate dataset complete");

    // println!("Total train data: {}", ds.train_u.dims()[0]);
    // println!("Total val data: {}", ds.val_u.dims()[0]);

    // // Plot data
    // let u_train: Vec<Vec<f32>> = ds.train_u.detach().to_vec2()?;
    // let u_train = py_matrix(u_train.clone());
    // let y_train = vec_tensor_to_matrix(&ds.train_y);
    // let gu_train = vec_tensor_to_matrix(&ds.train_Gu);
    
    // let x_train = linspace(0, 1, 100);

    // let mut plt = Plot2D::new();
    // plt
    //    .set_domain(x_train)
    //    .insert_image(u_train.row(0))
    //    .set_xlabel(r"$x$")
    //    .set_ylabel(r"$V(x)$")
    //    .set_style(PlotStyle::Nature)
    //    .tight_layout()
    //    .set_dpi(600)
    //    .set_path("potential.png")
    //    .savefig()?;

    // let mut plt = Plot2D::new();
    // plt
    //    .set_domain(y_train.row(0))
    //    .insert_image(gu_train.row(0))
    //    .set_xlabel(r"$t$")
    //    .set_ylabel(r"$x(t)$")
    //    .set_style(PlotStyle::Nature)
    //    .tight_layout()
    //    .set_dpi(600)
    //    .set_path("x.png")
    //    .savefig()?;

    // // Train
    // let hparams = HyperParams {
    //     x_sensors: 100,
    //     y_sensors: 100,
    //     p: 20,
    //     hidden_size: 40,
    //     hidden_depth: 3,
    //     learning_rate: 1e-1,
    //     epoch: 100,
    //     batch_size: 500,
    // };
    // let lr_scheduler = PolyLRScheduler {
    //     lr: hparams.learning_rate,
    //     max_lr: 1e-1,
    //     min_lr: 1e-4,
    //     power: 2.0,
    //     total_epoch: hparams.epoch,
    // };

    // let mut trainer = Trainer {
    //     dataset: ds,
    //     hyperparams: hparams,
    //     dev: &dev,
    //     rng: &mut rng,
    //     lr_scheduler,
    // };
    // let (model, train_history, val_history, lr_history) = trainer.train()?;

    // let epochs = linspace(1, train_history.len() as f64, train_history.len());
    // let mut plt = Plot2D::new();
    // plt
    //     .set_domain(epochs.clone())
    //     .insert_image(train_history.clone())
    //     .set_xlabel(r"Epoch")
    //     .set_ylabel(r"Train loss")
    //     .set_yscale(PlotScale::Log)
    //     .set_style(PlotStyle::Nature)
    //     .tight_layout()
    //     .set_dpi(600)
    //     .set_path("train_loss.png")
    //     .savefig()?;

    // let mut plt = Plot2D::new();
    // plt
    //     .set_domain(epochs.clone())
    //     .insert_image(val_history.clone())
    //     .set_xlabel(r"Epoch")
    //     .set_ylabel(r"Val loss")
    //     .set_yscale(PlotScale::Log)
    //     .set_style(PlotStyle::Nature)
    //     .tight_layout()
    //     .set_dpi(600)
    //     .set_path("val_loss.png")
    //     .savefig()?;

    // let mut plt = Plot2D::new();
    // plt
    //     .set_domain(epochs.clone())
    //     .insert_image(lr_history.clone())
    //     .set_xlabel(r"Epoch")
    //     .set_ylabel(r"Learning rate")
    //     .set_yscale(PlotScale::Log)
    //     .set_style(PlotStyle::Nature)
    //     .tight_layout()
    //     .set_dpi(600)
    //     .set_path("lr.png")
    //     .savefig()?;

    // // Test
    // let ds_test = Dataset::generate(1, 1, &dev)?;
    // let (test_u, test_y, test_Gu) = ds_test.train_set(&dev)?;
    // let Gu_hat = model.forward(&test_u, &test_y)?;

    // let test_u: Vec<f32> = test_u.detach().reshape(100).unwrap().to_vec1()?;
    // let test_u = test_u.into_iter().map(|x| x as f64).collect::<Vec<f64>>();
    // let test_y = vec_tensor_to_matrix(&test_y);
    // let test_Gu = vec_tensor_to_matrix(&test_Gu);
    // let Gu_hat = vec_tensor_to_matrix(&Gu_hat);

    // let x_train = linspace(0, 1, 100);

    // let mut plt = Plot2D::new();
    // plt
    //     .set_domain(x_train)
    //     .insert_image(test_u)
    //     .set_xlabel(r"$x$")
    //     .set_ylabel(r"$V(x)$")
    //     .set_style(PlotStyle::Nature)
    //     .tight_layout()
    //     .set_dpi(600)
    //     .set_path("potential_test.png")
    //     .savefig()?;

    // let mut plt = Plot2D::new();
    // plt
    //     .set_domain(test_y.row(0))
    //     .insert_image(test_Gu.row(0))
    //     .insert_image(Gu_hat.row(0))
    //     .set_xlabel(r"$t$")
    //     .set_ylabel(r"$x(t)$")
    //     .set_line_style(vec![(0, LineStyle::Solid), (1, LineStyle::Dashed)])
    //     .set_style(PlotStyle::Nature)
    //     .tight_layout()
    //     .set_dpi(600)
    //     .set_path("x_test.png")
    //     .savefig()?;

    Ok(())
}

// ┌─────────────────────────────────────────────────────────┐
//  Train
// └─────────────────────────────────────────────────────────┘
pub trait LRScheduler {
    fn step(&mut self, epoch: usize, loss: f64);
    fn get_lr(&self) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub struct PolyLRScheduler {
    lr: f64,
    max_lr: f64,
    min_lr: f64,
    power: f64,
    total_epoch: usize,
}

impl LRScheduler for PolyLRScheduler {
    fn step(&mut self, epoch: usize, _loss: f64) {
        self.lr = (self.max_lr - self.min_lr) * (1f64 - ((epoch + 1) as f64) / self.total_epoch as f64).powf(self.power) + self.min_lr;
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}

pub struct Trainer<'a, R: Rng, S: LRScheduler> {
    pub dataset: Dataset,
    pub hyperparams: HyperParams,
    pub dev: &'a Device,
    pub rng: &'a mut R,
    pub lr_scheduler: S,
}

impl<'a, R: Rng, S: LRScheduler> Trainer<'a, R, S> {
    #[allow(non_snake_case)]
    pub fn train(&mut self) -> Result<(DeepONet, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let dev = self.dev;
        let hparam = self.hyperparams;
        let mut lr = self.lr_scheduler.get_lr();

        let (train_u, train_y, train_Gu) = self.dataset.train_set(dev)?;
        let (val_u, val_y, val_Gu) = self.dataset.val_set(dev)?;

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
        let mut lr_history = vec![0f64; hparam.epoch];

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
            let msg = format!("epoch: {}, train_loss: {:.4e}, val_loss: {:.4e}, lr: {:.4e}", epoch, train_loss, val_loss, lr);
            pb.set_message(msg);

            let mut ics = (0u32 .. train_u.dims()[0] as u32).collect::<Vec<_>>();
            ics.shuffle(self.rng);

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
            lr_history[epoch] = lr;

            self.lr_scheduler.step(epoch, val_loss as f64);
            lr = self.lr_scheduler.get_lr();
            adam.set_learning_rate(lr);
        }

        println!("train_loss: {:.4e}, val_loss: {:.4e}", train_loss, val_loss);

        Ok((model, train_history, val_history, lr_history))
    }
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
    pub fn new(vb: VarBuilder, hparam: HyperParams) -> Result<Self> {
        let m = hparam.x_sensors;
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

        let bias = vb.get(1, "bias")?;

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
        for y in y_vec.iter_mut() {
            *y = self.trunk_net.last().unwrap().forward(y)?;        // y: B x p
            //*y = y.mul(&u)?.sum(1)?.broadcast_add(&self.bias)?;     // y: B
            *y = y.mul(&u)?.sum(1)?;     // y: B
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

        // Filter odd data
        let mut ics = vec![];
        for (i, gu) in Gu_vec.iter().enumerate() {
            if gu.iter().any(|gu| gu.abs() > 1.2) {
                continue
            }
            ics.push(i);
        }
        let grf_scaled_vec = ics.iter().map(|i| grf_scaled_vec[*i].clone()).collect::<Vec<_>>();
        let y_vec = ics.iter().map(|i| y_vec[*i].clone()).collect::<Vec<_>>();
        let Gu_vec = ics.iter().map(|i| Gu_vec[*i].clone()).collect::<Vec<_>>();

        let x_vec = linspace(0, 1, m);
        let u_vec = grf_scaled_vec.par_iter()
            .map(|grf| grf.iter().zip(x_vec.iter()).map(|(g, t)| 2f64 - 8f64 * g * t * (1f64 - t)).collect::<Vec<_>>())
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

    #[allow(non_snake_case)]
    pub fn write_parquet(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (train_u, train_y, train_Gu) = self.train_set(&Device::Cpu)?;
        let (val_u, val_y, val_Gu) = self.val_set(&Device::Cpu)?;

        let train_u: Vec<Vec<f32>> = train_u.detach().to_vec2()?;
        let train_u = py_matrix(train_u);
        let train_y = train_y.iter().flat_map(|y| {
            let y: Vec<f32> = y.detach().reshape(train_u.row).unwrap().to_vec1().unwrap();
            y.into_iter().map(|x| x as f64).collect::<Vec<f64>>()
        }).collect::<Vec<_>>();
        let train_y = matrix(train_y, train_u.row, 100, Col);
        let train_Gu = train_Gu.iter()
            .flat_map(|Gu| {
                let Gu: Vec<f32> = Gu.detach().reshape(train_u.row).unwrap().to_vec1().unwrap();
                Gu.into_iter().map(|x| x as f64).collect::<Vec<f64>>()
            }).collect::<Vec<_>>();
        let train_Gu = matrix(train_Gu, train_u.row, 100, Col);

        let train_y = train_y.change_shape();
        let train_Gu = train_Gu.change_shape();

        let train_u = train_u.data;
        let train_y = train_y.data;
        let train_Gu = train_Gu.data;

        let val_u: Vec<Vec<f32>> = val_u.detach().to_vec2()?;
        let val_u = py_matrix(val_u);
        let val_y = val_y.iter().flat_map(|y| {
            let y: Vec<f32> = y.detach().reshape(val_u.row).unwrap().to_vec1().unwrap();
            y.into_iter().map(|x| x as f64).collect::<Vec<f64>>()
        }).collect::<Vec<_>>();
        let val_y = matrix(val_y, val_u.row, 100, Col);
        let val_Gu = val_Gu.iter()
            .flat_map(|Gu| {
                let Gu: Vec<f32> = Gu.detach().reshape(val_u.row).unwrap().to_vec1().unwrap();
                Gu.into_iter().map(|x| x as f64).collect::<Vec<f64>>()
            }).collect::<Vec<_>>();
        let val_Gu = matrix(val_Gu, val_u.row, 100, Col);

        let val_y = val_y.change_shape();
        let val_Gu = val_Gu.change_shape();

        let val_u = val_u.data;
        let val_y = val_y.data;
        let val_Gu = val_Gu.data;

        let mut df = DataFrame::new(vec![]);
        df.push("train_u", Series::new(train_u));
        df.push("train_y", Series::new(train_y));
        df.push("train_Gu", Series::new(train_Gu));

        let data_folder = "data";
        if !std::path::Path::new(data_folder).exists() {
            std::fs::create_dir(data_folder)?;
        }

        let train_path = format!("{}/train.parquet", data_folder);
        df.write_parquet(&train_path, CompressionOptions::Uncompressed)?;

        let mut df = DataFrame::new(vec![]);
        df.push("val_u", Series::new(val_u));
        df.push("val_y", Series::new(val_y));
        df.push("val_Gu", Series::new(val_Gu));

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
        let y = grf.iter().zip(x.iter()).map(|(g, t)| 2f64 - 8f64 * g * t * (1f64 - t)).collect::<Vec<_>>();
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

pub fn vec_tensor_to_matrix(vec: &[Tensor]) -> Matrix {
    let row = vec[0].dims()[0];
    let col = vec.len();
    let vec: Vec<f64> = vec.iter()
        .flat_map(|t| {
            let t: Vec<f32> = t.detach().reshape(t.dims()[0]).unwrap().to_vec1().unwrap();
            t.into_iter().map(|x| x as f64).collect::<Vec<f64>>()
        })
        .collect();
    matrix(vec, row, col, Col)
}
