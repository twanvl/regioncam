use std::time::Instant;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use regioncam::nn::{Linear, NNModule};
use regioncam::Partition;

/// A (relatively) large MLP with a single hidden layer
fn large_random_mlp() {
    let mut rng = SmallRng::seed_from_u64(42);
    let dim_in = 2;
    let dim_hidden = 1000;
    let dim_out = 700;
    let layer1 = Linear::new_uniform(dim_in, dim_hidden, &mut rng);
    let layer2 = Linear::new_uniform(dim_hidden, dim_out, &mut rng);
    // partition
    let mut p = Partition::square(1.0);
    layer1.apply(&mut p);
    layer2.apply(&mut p);
    p.relu();
    println!("{} faces", p.num_faces());
}

fn main() {
    let now = Instant::now();
    large_random_mlp();
    let elapsed = now.elapsed();
    println!("Time: {elapsed:.3?}s");
}