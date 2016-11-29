extern crate polygon;

use polygon::*;

fn main() {
    let pts = vec![Point::new(0.0, 0.0); 10];
    let mut p = Polygon::from_points(&pts[..]);
    generate::randomize_polygon(&mut p, generate::Mode::QuickStarLike);
    println!("Polygon: {:?}", p);
}
