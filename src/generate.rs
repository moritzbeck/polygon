//! Provides methods to generate random polygons.

extern crate rand;

use super::*;
use self::rand::{thread_rng, sample};
use self::rand::distributions::{IndependentSample, Range};

/// Describes the mode used to generate a polygon.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Mode {
    /// Generates random points and then tries to disentangle the
    /// resulting polygon by swapping endpoints of intersecting lines.
    TwoOptLike,
    /// Generates random points `p_1`, … ,`p_n`
    /// and chooses a point `q` in their convex hull.
    /// Then sorts the vertices by the angle between the line
    /// from `q` to `p_i` and the x-axis.
    /// The resulting polygon will be star-shaped.
    QuickStarLike
}

impl Polygon {
    /// Randomizes `self`, i.e generates a random polygon.
    ///
    /// `self` will be overwritten with a newly generated polygon.
    /// The `mode` specifies how the polygon will be generated.
    pub fn randomize(&mut self, mode: Mode) {
        let mut rng = thread_rng();
        let range = Range::new(1.0, 500.0);
        let len = self.size();
        {
            let points = self.points_mut();
            for i in 0..len {
                let rand_x = range.ind_sample(&mut rng);
                let rand_y = range.ind_sample(&mut rng);
                points[i] = Point::new(rand_x, rand_y);
            }
        }
        match mode {
            Mode::TwoOptLike => {
                let d = _two_opt_like(self);
                if d.is_none() {
                    self.randomize(mode); // try again
                    // TODO?: remove recursion
                }
            },
            Mode::QuickStarLike => {
                _quick_star_like(self);
            }
        }
        assert!(self.is_simple());
    }
}
/// Returns the number of swaps if it succeeds.
fn _two_opt_like(p: &mut Polygon) -> Option<u32> {
        // Ω(n^2)
        let len = p.size();
        assert!(len >= 3);
        let points = p.points_mut();
        let mut tangled = true;
        let mut swaps = 0;
        while tangled {
            tangled = false;
            let l1 = Line::new(points[len-1], points[0]);
            for j in 1..len-2 {
                let l2 = Line::new(points[j], points[j+1]);
                if l1.intersects(&l2) {
                    points.swap(0, j);
                    swaps += 1;
                    tangled = true;
                    break;
                }
            }
            for i in 0..len-3 {
                for j in i+2..len-1 {
                    let l1 = Line::new(points[i], points[i+1]);
                    let l2 = Line::new(points[j], points[j+1]);
                    if l1.intersects(&l2) {
                        points.swap(i+1, j);
                        swaps += 1;
                        tangled = true;
                        break;
                    }
                }
            }
            if swaps >= 20_000 {
                // TODO: choose a good limit;
                // this limit should be dependent on the size of the polygon
                return None;
            }
        }
        Some(swaps)
}
/// Returns a star shaped polygon with the given vertices.
///
/// First it randomly selects 3 vertices and
/// computes a random point `p` in the induced triangle.
/// Then it orders the vertices by their circular order around `p`.
fn _quick_star_like(pol: &mut Polygon) {
    use std::cmp::Ordering;

    let mut rng = thread_rng();
    let sample = sample(&mut rng, 0..pol.size(), 3).into_iter()
        .map(|s| { pol.points()[s] })
        .collect::<Vec<_>>();

    // t0 + t1 + t2 == 1.0
    let t0 = Range::new(0.0, 1.0).ind_sample(&mut rng);
    let t1 = Range::new(0.0, 1.0 - t0).ind_sample(&mut rng);
    let t2 = 1.0 - t0 - t1;
    let p = sample[0] * t0 + sample[1] * t1 + sample[2] * t2;

    // Sorts the points around p in CCW order.
    // The minimal angle is the top direction.
    // Ties are broken sometimes, by a strange method that is easy to implement.
    pol.points_mut().sort_by(|p1, p2| {
        let p1 = *p1;
        let p2 = *p2;
        if p1 == p2 { return Ordering::Equal; }
        if p1 == p  { return Ordering::Less; }
        if p2 == p  { return Ordering::Greater; }
        if p1.x < p.x && p2.x > p.x { return Ordering::Less; }
        if p2.x < p.x && p1.x > p.x { return Ordering::Greater; }
        match (Line::new(p, p1).slope(), Line::new(p, p2).slope()) {
            (Slope::Vertical, Slope::Vertical) => {
                p2.y.partial_cmp(&p1.y).expect("y coordinate should never be NaN!")
            },
            (Slope::Vertical, _) => { //p1.x == p.x
                if p1.y > p.y { Ordering::Less }
                else { Ordering::Greater }
            },
            (_, Slope::Vertical) => {
                if p2.y > p.y { Ordering::Greater }
                else { Ordering::Less }
            },
            (Slope::Slope(m1), Slope::Slope(m2)) => {
                m1.partial_cmp(&m2).expect("Slope should never be NaN!")
            }
        }
    });
}

