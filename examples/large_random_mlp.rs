use std::time::Instant;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use regioncam::nn::Linear;
use regioncam::Regioncam;

/// A (relatively) large MLP with a single hidden layer
fn large_random_mlp() {
    let mut rng = SmallRng::seed_from_u64(42);
    let dim_in = 2;
    let dim_hidden = 1000;
    let dim_out = 700;
    let layer1 = Linear::new_uniform(dim_in, dim_hidden, &mut rng);
    let layer2 = Linear::new_uniform(dim_hidden, dim_out, &mut rng);
    // partition
    let mut rc = Regioncam::square(1.0);
    rc.add(&layer1);
    rc.add(&layer2);
    rc.relu();
    println!("{} faces", rc.num_faces());
}

fn main() {
    let now = Instant::now();
    large_random_mlp();
    let elapsed = now.elapsed();
    println!("Time: {elapsed:.3?}");
}