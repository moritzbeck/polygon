extern crate polygon;

use polygon::*;
use polygon::generate;

fn main() {
    let pts = vec![Point::new(0.0, 0.0); 10];
    let mut p = Polygon::from_points(&pts[..]);
    p.randomize(generate::Mode::QuickStarLike);
    println!("Polygon: {:?}", p);
}
