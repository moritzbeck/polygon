//! Provides methods to generate random polygons.

extern crate rand;

use super::*;
use self::rand::{thread_rng, sample};
use self::rand::distributions::{IndependentSample, Range};
use std::cmp::Ordering;

/// Describes the mode used to generate a polygon.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Mode {
    /// Generates random points and then tries to disentangle the
    /// resulting polygon by swapping endpoints of intersecting lines.
    TwoOptLike,
    /// Generates random points `p_1`, … ,`p_n`
    /// and chooses a point `q` in their convex hull.
    /// Then it orders the vertices by their circular order around `q`.
    /// The resulting polygon will be star-shaped.
    QuickStarLike,
    /// Generates random points, sort them by x-coordinate
    /// and disentangles the resulting polygon.
    Xmonotone,
    /// Generates random points with a gap between the lower and upper line.
    /// The leftmost and rightmost points vertically lie in this gap.
    XmonotoneGap
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
        assert!(len >= 3);
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
                let d = self._two_opt_like();
                if d.is_none() {
                    self.randomize(mode); // try again
                    // TODO?: remove recursion
                }
            },
            Mode::QuickStarLike => {
                self._quick_star_like();
            },
            Mode::Xmonotone => {
                self._x_monotone();
            },
            Mode::XmonotoneGap => {
                self._x_monotone_gap(0.5, 100);
            },
        }
        assert!(self.is_simple());
    }
    /// Rounds all coordinates to `dec_places` decimal places.
    pub fn trim_coordinates(&mut self, dec_places: i8) {
        for mut p in &mut self.points {
            p.trim_coordinates(dec_places);
        }
    }

    /// Returns the number of swaps if it succeeds.
    fn _two_opt_like(&mut self) -> Option<u32> {
        // Ω(n^2)
        let mut swaps = 0;
        {
            let len = self.size();
            let points = self.points_mut();
            let mut tangled = true;
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
        }
        Some(swaps)
    }
    /// Forms a star shaped polygon by reordering the given vertices.
    ///
    /// First it randomly selects 3 vertices and
    /// computes a random point `p` in the induced triangle.
    /// Then it orders the vertices by their circular order around `p`.
    // TODO: This function could be useful even without first generating
    //       random points (to get a random star with the same vertex set).
    fn _quick_star_like(&mut self) {
        let mut rng = thread_rng();
        let sample = sample(&mut rng, 0..self.size(), 3).into_iter()
            .map(|s| { self.points()[s] })
            .collect::<Vec<_>>();

        // t0 + t1 + t2 == 1.0
        let t0 = Range::new(0.0, 1.0).ind_sample(&mut rng);
        let t1 = Range::new(0.0, 1.0 - t0).ind_sample(&mut rng);
        let t2 = 1.0 - t0 - t1;
        let p = sample[0] * t0 + sample[1] * t1 + sample[2] * t2;

        // Sorts the points around p in CCW order.
        // The minimal angle is the top direction.
        // Ties are broken sometimes, by a strange method that is easy to implement.
        self.points_mut().sort_by(|p1, p2| {
            let p1 = *p1;
            let p2 = *p2;
            if p1 == p2 { return Ordering::Equal; }
            if p1 == p  { return Ordering::Less; }
            if p2 == p  { return Ordering::Greater; }
            if p1.x < p.x && p2.x > p.x { return Ordering::Less; }
            if p2.x < p.x && p1.x > p.x { return Ordering::Greater; }
            match (Line::new(p, p1).slope(), Line::new(p, p2).slope()) {
                (Slope::Vertical, Slope::Vertical) => {
                    if p1.y > p2.y { Ordering::Less }
                    else { Ordering::Greater }
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
                    if m1 > m2 { Ordering::Greater }
                    else { Ordering::Less }
                }
            }
        });
    }
    /// Returns an x-monotone simple polygon with the given vertices.
    ///
    /// First it sorts the vertices by x-coordinate.
    /// Then it tests for intersections of the 'closing edge' `p_{n-1}p_0`
    /// with another edge from left to right.
    /// If an intersection is found the vertex with the higher index
    /// is pushed to the beginning of the backing `Vec`.
    /// After one sweep the polygon is simple.
    fn _x_monotone(&mut self) {
        #![allow(unused_variables)]
        let n = self.size();
        self.points_mut().sort_by(|p1, p2| {
            f64_cmp(p1.x, p2.x).then(f64_cmp(p1.y, p2.y))
        });
        //println!("{:#?}", self.points);
        { // new scope for the borrow in `let mut points = self.points_mut();`.
            let mut points = self.points_mut();

            let mut idx_leftmost = 0;
            // The lower chain starts at index n-1 and ends at index `idx_leftmost`.
            // The upper chain starts at index `idx_leftmost` and ends at index n-1.
            let mut idx_upper = 1;
            let mut idx_lower = n-1;

            while idx_upper < n-2 { // The line from index n-2 to index n-1 necessarily intersects the one from n-1 to 0.
               // println!("idx_upper: {}", idx_upper);
                while idx_lower != idx_leftmost {
                    let upper_line = Line::new(points[idx_upper], points[idx_upper + 1]);
                    let lower_line = Line::new(points[idx_lower], points[(idx_lower + 1) % n]);
                    //println!("{} {}, {} {}", idx_upper, idx_upper+1, idx_lower, (idx_lower + 1) % n);

                    idx_lower = (idx_lower + 1) % n;
                    if upper_line.intersects(&lower_line) {
                       // println!("moving {} to front", idx_upper + 1);
                        move_to_front(points, idx_upper + 1);
                        idx_leftmost += 1;
                        idx_lower = n-1; // reset this
                        idx_upper += if idx_upper < n-3 { 1 } else { 0 }; // it has been shifted
                    }
                }
                idx_lower = n-1;
                idx_upper += 1;
            }



            //for i in 1..n-2 {
            //let mut i = 1;
            //while i < n-2 {
                //let closing_edge = Line::new(points[0], points[n-1]);
                //let edge = Line::new(point[i], points[i+1]);
                //let mut j = 0;
                //if Line::new(points[n-1], points[0]).intersects(&Line::new(points[i], points[i+1])) {
                    //println!("moving {} to front", i+1);
                    //move_to_front(points, i+1);
                    //i += 1; // now this point has been shifted in the array
                //}
                //while points[j].x > points[i].x {
                    //if Line::new(points[j], points[j+1]).intersects(&Line::new(points[i], points[i+1])) {
                        //println!("moving {} to front", i+1);
                        //move_to_front(points, i+1);
                        //i += 1; // now this point has been shifted in the array
                        //j -= 1;
                    //}
                    //j += 1;
                //}
                //i += 1; // get to the next point
            //}
        }
        //println!("{:#?}", self.points);
        assert!(self.is_x_monotone());
    }
    fn _x_monotone_gap(&mut self, ratio: f32, min_gap: u16) {
        let n = self.points().len();

        let mut rng = rand::thread_rng();
        {
        let mut points = self.points_mut();

        // points[0] is the leftmost point
        // points[..rightmost_idx] is the lower chain
        // points[rightmost_idx] is the rightmost point
        // points[rightmost_idx..] is the upper chain
        let rightmost_idx = (n as f32 * ratio) as usize;

        let x_range = Range::new(10, 490);
        let y_range = Range::new(min_gap/2, 250);

        let endpoint_y_range = Range::new(0, min_gap);

        points[0].x             = 5.0;
        points[0].y             = (250 - min_gap/2 + endpoint_y_range.ind_sample(&mut rng)) as f64;
        points[rightmost_idx].x = 500.0;
        points[rightmost_idx].y = (250 - min_gap/2 + endpoint_y_range.ind_sample(&mut rng)) as f64;

        // generate the lower chain
        for p in points[1..rightmost_idx].iter_mut() {
            p.x = x_range.ind_sample(&mut rng) as f64;
            p.y = (250 - y_range.ind_sample(&mut rng)) as f64;
        }
        // generate the upper chain
        for p in points[(rightmost_idx+1)..].iter_mut() {
            p.x = x_range.ind_sample(&mut rng) as f64;
            p.y = (250 + y_range.ind_sample(&mut rng)) as f64;
        }
        // sort the upper points by x-coordinate
        points[1..rightmost_idx].sort_by(|p1, p2| f64_cmp(p1.x, p2.x).then(f64_cmp(p1.y, p2.y)));
        // sort the lower points by x-coordinate (reversed)
        points[(rightmost_idx+1)..].sort_by(|p1, p2| f64_cmp(p2.x, p1.x).then(f64_cmp(p2.y, p1.y)));
        }

        assert!(self.is_x_monotone());
    }
}
fn f64_cmp(f1: f64, f2: f64) -> Ordering {
    if f1 == f2 {
        Ordering::Equal
    } else if f1 < f2 {
        Ordering::Less
    } else {
        // also for cases where f1 == NaN or f2 == NaN
        Ordering::Greater
    }
}
fn move_to_front<T: Copy>(s: &mut [T], index: usize) {
    assert!(s.len() > index);
    let temp = s[index];
    for i in (0..index).rev() {
        s[i+1] = s[i];
    }
    s[0] = temp;
}

#[test]
fn move_to_front_works() {
    let mut slice = [0, 1, 2, 3, 4, 5];
    move_to_front(&mut slice, 3);
    assert_eq!(slice, [3, 0, 1, 2, 4, 5]);
    move_to_front(&mut slice, 1);
    assert_eq!(slice, [0, 3, 1, 2, 4, 5]);
    move_to_front(&mut slice, 4);
    assert_eq!(slice, [4, 0, 3, 1, 2, 5]);

    let mut slice = [Ok(0.0), Err(-1), Ok(2.0), Ok(3.0), Err(-4)];
    move_to_front(&mut slice, 2);
    assert_eq!(slice, [Ok(2.0), Ok(0.0), Err(-1), Ok(3.0), Err(-4)]);
}
