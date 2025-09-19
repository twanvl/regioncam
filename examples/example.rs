use std::fs::File;
use rand::prelude::*;
use regioncam::{NNBuilder, Regioncam, RenderOptions, nn::Linear};

fn main() -> std::io::Result<()> {
    // Create a regioncam object, with the region [-1..1]^2
    let mut rc = Regioncam::square(1.0);
    // Apply a linear layer
    let mut rng = SmallRng::seed_from_u64(42);
    let layer1 = Linear::new_uniform(2, 30, &mut rng);
    rc.add(&layer1);
    // Apply a relu activation function
    rc.relu();
    // Write to an svg file
    let render_options = RenderOptions::default();
    let mut file = File::create("example.svg")?;
    rc.write_svg(&render_options, &mut file)?;
    // Inspect regions
    println!("Created {} regions", rc.num_faces());
    println!("Face with the most edges has {} edges",
        rc.faces().map(|face| rc.vertices_on_face(face).count()).max().unwrap()
    );
    Ok(())
}