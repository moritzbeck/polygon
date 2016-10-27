fn min_max(a: f64, b: f64) -> (f64, f64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[derive(Debug, PartialEq)]
pub enum Slope {
    Vertical,
    Slope(f64)
}
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64
}
impl Point {
    pub fn new(x: f64, y: f64) -> Point {
        Point {
            x: x,
            y: y
        }
    }
    pub fn new_u(x: u32, y: u32) -> Point {
        Point {
            x: x as f64,
            y: y as f64
        }
    }
}
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Line {
    pub from:   Point,
    pub to:     Point
}
impl Line {
    pub fn new(from: Point, to: Point) -> Line {
        Line {
            from: from,
            to: to
        }
    }
    pub fn reverse(&self) -> Line {
        Line {
            from: self.to,
            to: self.from
       }
    }
    pub fn slope(&self) -> Slope {
        if self.from.x == self.to.x {
            Slope::Vertical
        } else {
            Slope::Slope((self.to.y - self.from.y) / (self.to.x - self.from.x))
        }
    }
    pub fn intersects(&self, line: &Line) -> bool {
        // first check if one point is an endpoint of both lines
        // this check is needed because otherwise precision problems surface
        if self.from == line.from || self.from == line.to ||
           self.to   == line.from || self.to   == line.to {
            return true;
        }
        let (s_min_x, s_max_x) = min_max(self.from.x, self.to.x);
        let (s_min_y, s_max_y) = min_max(self.from.y, self.to.y);
        let (l_min_x, l_max_x) = min_max(line.from.x, line.to.x);
        let (l_min_y, l_max_y) = min_max(line.from.y, line.to.y);
        if s_min_x > l_max_x || s_max_x < l_min_x ||
           s_min_y > l_max_y || s_max_y < l_min_y {
            return false; //disjunct bounding box
        }
        match (self.slope(), line.slope()) {
            (Slope::Vertical, Slope::Vertical) => true, //bounding boxes intersect
            (Slope::Vertical, Slope::Slope(m)) => {
                // s_min_x == s_max_x
                let t = line.from.y - m * line.from.x;
                let y = m * s_min_x + t;
                s_min_y <= y && y <= s_max_y
            },
            (Slope::Slope(m), Slope::Vertical) => {
                // l_min_x == l_max_x
                let t = self.from.y - m * self.from.x;
                let y = m * l_min_x + t;
                l_min_y <= y && y <= l_max_y
            },
            (Slope::Slope(m1), Slope::Slope(m2)) => { // m*x +t = y
                let t1 = self.from.y - m1 * self.from.x;
                let t2 = line.from.y - m2 * line.from.x;
                if m1 == m2 { t1 == t2 }
                else {
                    let x = (t2 - t1) / (m1 - m2);
                    s_min_x <= x && x <= s_max_x && // lies on &self
                    l_min_x <= x && x <= l_max_x    // lies on &line
                }
            }
        }
    }
    fn contains(&self, p: Point) -> bool {
        self.intersects(&Line::new(p, p))
    }
}
#[derive(Debug, PartialEq)]
pub struct Polygon {
    points: Vec<Point>
}
impl Polygon {
    pub fn new() -> Polygon {
        Polygon {
            points: Vec::new()
        }
    }
    pub fn with_capacity(c: usize) -> Polygon {
        Polygon {
            points: Vec::with_capacity(c)
        }
    }
    pub fn from_points(pts: &[Point]) -> Polygon {
        Polygon {
            points: pts.to_vec() //TODO: do it without copying?!
        }
    }
    pub fn size(&self) -> usize {
        self.points.len()
    }
    pub fn add(&mut self, p: Point) {
        self.points.push(p);
    }
    pub fn points(&self) -> &[Point] {
        //self.points.clone()
        &self.points[..]
    }
    pub fn points_mut(&mut self) -> &mut [Point] {
        &mut self.points[..]
    }
    pub fn edges(&self) -> Vec<Line> {
        if self.points.len() <= 1 { return Vec::new(); }
        let mut lines = Vec::with_capacity(self.points.len());
        lines.push(Line::new(self.points[self.points.len()-1], self.points[0]));
        for l in self.points.windows(2) {
            let l = Line::new(l[0], l[1]);
            lines.push(l);
        }
        lines
    }
    pub fn is_simple(&self) -> bool {
        for l1 in self.edges() {
            for l2 in self.edges() {
                if l1.from == l2.from || l1.from == l2.to ||
                   l1.to   == l2.from || l1.to   == l2.to {
                    break; // TODO: fails for the edge case when a vertex has more
                           //   than two incident edges, i.e. the polygon touches or
                           //   crosses itself at a vertex.
                }
                if l1.intersects(&l2) {
                    return false;
                }
            }
        }
        true
    }
    pub fn contains(&self, p: Point) -> bool {
        /*
        Casts a horizontal ray to the Point p
        and counts how many edges to the right of p were hit.
        If the count is odd, p lies within &self.

        An edge is not considered hit, if
        - it is an horizontal edge OR
        - it is only hit at the bottom end point.

        Additionally, to avoid treating the upper and lower boundary differently
        (e.g. top edge considered part of the polygon while bottom one is not)
        it is checked if p lies on some edge of &self (and true is returned).

        running time: O(n)
        */
        if self.points.len() == 0 {
            return false;
        }
        if self.points.len() == 1 {
            return self.points[0] == p;
        }
        let mut odd_hit_count = false;
        for l in self.edges() {
            if l.contains(p) { return true; } // p lies on the boundary of &self
            odd_hit_count ^= Polygon::_horizontal_ray_hits_line(p, l);
        }
        odd_hit_count
    }
    fn _horizontal_ray_hits_line(p: Point, l: Line) -> bool {
        let (min_x, max_x) = min_max(l.from.x, l.to.x);
        let (min_y, max_y) = min_max(l.from.y, l.to.y);
        if p.y <= min_y || max_y < p.y { return false; } // too low or high
        if max_x < p.x { return false; } // l too far on the left
        if min_y == max_y { return false; } // ignore horizontal edges
        if p.x <= min_x { return true; } // l is to the right of p

        // now we know that p lies within the bounding box of l
        let l_lower_point = if min_y == l.from.y { l.from } else { l.to };
        if l_lower_point == p { return false; }
        let lslope = match l.slope() {
            Slope::Vertical => return true,
            Slope::Slope(m) => m
        };
        match Line::new(l_lower_point, p).slope() {
            Slope::Vertical => min_x == l_lower_point.x,
            Slope::Slope(m) => lslope <= m
        }
    }
}
/*
impl Iterator for Polygon {
    type Item = Point;

    fn next(&mut self) -> Option<Point> {
        self.points.iter().cloned().next()
    }
}
*/

#[cfg(test)]
mod tests {
    extern crate rand;

    use self::rand::distributions::{IndependentSample, Range};
    use super::*;

    const P1: Point = Point{ x: 0., y: 0.};
    const P2: Point = Point{ x: 1., y: 0.};
    const P3: Point = Point{ x: 0., y: 1.};
    const P4: Point = Point{ x: 1., y: 1.};
    const P5: Point = Point{ x: 2., y: 2.};
    const P6: Point = Point{ x: 3., y: 3.};
    const P7: Point = Point{ x: 0., y: 2.};
    const P8: Point = Point{ x: 1., y: 2.};
    const SLO1: Line = Line { from: P1, to: P4 };
    const SLO2: Line = Line { from: P4, to: P5 };
    const SLO3: Line = Line { from: P5, to: P6 };
    const SLO4: Line = Line { from: P2, to: P3 };
    const VER5: Line = Line { from: P1, to: P3 };
    const VER6: Line = Line { from: P2, to: P4 };
    const VER7: Line = Line { from: P3, to: P7 };
    const VER8: Line = Line { from: P4, to: P8 };
    const HOR9: Line = Line { from: P5, to: P8 };

    #[test]
    fn line_slope_works() {
        assert_eq!(SLO1.slope(),           Slope::Slope( 1f64));
        assert_eq!(SLO1.reverse().slope(), Slope::Slope( 1f64));
        assert_eq!(SLO2.slope(),           Slope::Slope( 1f64));
        assert_eq!(SLO2.reverse().slope(), Slope::Slope( 1f64));
        assert_eq!(SLO3.slope(),           Slope::Slope( 1f64));
        assert_eq!(SLO4.slope(),           Slope::Slope(-1f64));
        assert_eq!(HOR9.slope(),           Slope::Slope( 0f64));
        assert_eq!(VER5.slope(),           Slope::Vertical);
        assert_eq!(VER5.reverse().slope(), Slope::Vertical);
        assert_eq!(VER6.slope(),           Slope::Vertical);
        assert_eq!(VER7.slope(),           Slope::Vertical);
        assert_eq!(VER8.slope(),           Slope::Vertical);
    }
    #[test]
    fn line_intersects_works_both_w_slope() {
        assert!( SLO1.intersects(&SLO1), "Line doesn't intersect itself");
        assert!( SLO1.intersects(&SLO1.reverse()), "Line doesn't intersect itself reversed");
        assert!( SLO1.intersects(&SLO2), "Line doesn't intersect one it should");
        assert!( SLO2.intersects(&SLO1), "Line doesn't intersect one it should");
        assert!(!SLO1.intersects(&SLO3), "Line does intersect one it shouldn't");
        assert!(!SLO3.intersects(&SLO1), "Line does intersect one it shouldn't");
        assert!( SLO1.intersects(&SLO4), "Line doesn't intersect one it should");
        assert!( SLO4.intersects(&SLO1), "Line doesn't intersect one it should");
        assert!(!SLO2.intersects(&SLO4), "Line does intersect one it shouldn't");
        assert!(!SLO4.intersects(&SLO2), "Line does intersect one it shouldn't");
    }
    #[test]
    fn line_intersects_works_one_vertical() {
        assert!( SLO1.intersects(&VER5), "Line doesn't intersect one it should");
        assert!( VER5.intersects(&SLO1), "Line doesn't intersect one it should");
        assert!( SLO1.intersects(&VER6), "Line doesn't intersect one it should");
        assert!( VER6.intersects(&SLO1), "Line doesn't intersect one it should");
        assert!(!SLO1.intersects(&VER7), "Line does intersect one it shouldn't");
        assert!(!VER7.intersects(&SLO1), "Line does intersect one it shouldn't");
        assert!( SLO4.intersects(&VER7), "Line doesn't intersect one it should");
        assert!( VER7.intersects(&SLO4), "Line doesn't intersect one it should");
        assert!(!SLO4.intersects(&VER8), "Line does intersect one it shouldn't");
        assert!(!VER8.intersects(&SLO4), "Line does intersect one it shouldn't");
    }
    #[test]
    fn line_intersects_works_both_vertical() {
        assert!( VER5.intersects(&VER5), "Line doesn't intersect itself");
        assert!( VER6.intersects(&VER6.reverse()), "Line doesn't intersect itself reversed");
        assert!(!VER5.intersects(&VER6), "Line does intersect one it shouldn't");
        assert!(!VER6.intersects(&VER5), "Line does intersect one it shouldn't");
        assert!( VER5.intersects(&VER7), "Line doesn't intersect one it should");
        assert!( VER7.intersects(&VER5), "Line doesn't intersect one it should");
        assert!(!VER6.intersects(&VER7), "Line does intersect one it shouldn't");
        assert!(!VER7.intersects(&VER6), "Line does intersect one it shouldn't");
    }
    #[test]
    fn line_intersects_works_rand_triangle() {
        let mut rng = rand::thread_rng();
        let range = Range::new(1.0, 500.0);

        for _ in 0..10000 {
            let x1 = range.ind_sample(&mut rng);
            let y1 = range.ind_sample(&mut rng);
            let x2 = range.ind_sample(&mut rng);
            let y2 = range.ind_sample(&mut rng);
            let x3 = range.ind_sample(&mut rng);
            let y3 = range.ind_sample(&mut rng);
            let p1 = Point::new(x1, y1);
            let p2 = Point::new(x2, y2);
            let p3 = Point::new(x3, y3);
            let l1 = Line::new(p1, p2);
            let l2 = Line::new(p2, p3);
            assert!(l1.intersects(&l2));
            assert!(l2.intersects(&l1));
        }
    }// TODO: add tests where endpoints of the lines lie very close together (to check for precision problems)
    #[test]
    fn polygon_contains_works_1() {
        let points = [Point::new_u(0,0), Point::new_u(3,0), Point::new_u(4, 2), Point::new_u(1,2)];
        let parallelogram = Polygon::from_points(&points);
        assert!(!parallelogram.contains(Point::new(3.5, 0.9)));
        assert!( parallelogram.contains(Point::new(3.5, 1.0)));
        assert!( parallelogram.contains(Point::new(3.5, 1.2)));
        assert!( parallelogram.contains(Point::new(0.5, 0.9)));
        assert!( parallelogram.contains(Point::new(0.5, 1.0)));
        assert!(!parallelogram.contains(Point::new(0.5, 1.2)));
        assert!( parallelogram.contains(Point::new(0.0, 0.0)));
        assert!( parallelogram.contains(Point::new(3.0, 0.0)));
        assert!( parallelogram.contains(Point::new(4.0, 2.0)));
        assert!( parallelogram.contains(Point::new(1.0, 2.0)));
        assert!( parallelogram.contains(Point::new(3.0, 2.0)));
        assert!( parallelogram.contains(Point::new(2.0, 1.0)));
        assert!(!parallelogram.contains(Point::new(4.0, 1.0)));
        assert!(!parallelogram.contains(Point::new(5.0, 1.0)));
        assert!(!parallelogram.contains(Point::new(5.0, 5.0)));
        assert!(!parallelogram.contains(Point::new(1.0, 5.0)));
    }
    #[test]
    fn polygon_contains_works_2() {
//                8     6
//   polygon =    |\   /|
//                | \ / |
//          a-----9  7  5
//          |           |
//          |           |
//          |           |
//          |           |
//          |           |
//          b--0     3--4
//             |     |
//             |     |
//             1-----2
        let points = [Point::new_u(1,1), Point::new_u(1,0), Point::new_u(3, 0),
            Point::new_u(3,1), Point::new_u(4,1), Point::new_u(4,3),
            Point::new_u(4,4), Point::new_u(3,3), Point::new_u(2,4),
            Point::new_u(2,3), Point::new_u(0,3), Point::new_u(0,1)];
        let polygon = Polygon::from_points(&points);
        assert!(!polygon.contains(Point::new(0.5, 0.5)));
        assert!( polygon.contains(Point::new(0.5, 1.5)));
        assert!(!polygon.contains(Point::new(0.5, 3.5)));
        assert!( polygon.contains(Point::new(1.0, 0.5)));
        assert!( polygon.contains(Point::new(1.0, 1.0)));
        assert!( polygon.contains(Point::new(1.0, 3.0)));
        assert!(!polygon.contains(Point::new(1.0, 3.5)));
        assert!(!polygon.contains(Point::new(1.0, 4.0)));
        assert!( polygon.contains(Point::new(2.0, 0.5)));
        assert!( polygon.contains(Point::new(2.0, 2.5)));
        assert!( polygon.contains(Point::new(2.0, 4.0)));
        assert!(!polygon.contains(Point::new(2.0, 5.0)));
        assert!( polygon.contains(Point::new(2.5, 3.0)));
        assert!( polygon.contains(Point::new(3.5, 3.0)));
        assert!(!polygon.contains(Point::new(4.0, 0.5)));
        assert!( polygon.contains(Point::new(4.0, 3.5)));
        assert!(!polygon.contains(Point::new(4.0, 4.5)));
    }
}
