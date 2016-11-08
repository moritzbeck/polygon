#![warn(missing_docs)]

fn min_max(a: f64, b: f64) -> (f64, f64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Describes the slope of a Line.
#[derive(Debug, PartialEq)]
pub enum Slope {
    /// The line is exactly vertical, i.e l.from.x == l.to.x
    Vertical,
    /// The line is not vertical and the slope is given.
    Slope(f64)
}
/// A point in the 2D-plane.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Point {
    /// x-coordinate
    pub x: f64,
    /// y-coordinate
    pub y: f64
}
impl Point {
    /// Creates a new Point with the given coordinates.
    pub fn new(x: f64, y: f64) -> Point {
        Point {
            x: x,
            y: y
        }
    }
    /// Creates a new Point with the given integer coordinates.
    pub fn new_u(x: u32, y: u32) -> Point {
        Point {
            x: x as f64,
            y: y as f64
        }
    }
}
/// A line in the 2D-plane.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Line {
    /// starting point
    pub from:   Point,
    /// endpoint
    pub to:     Point
}
impl Line {
    /// Creates a new line between two points.
    pub fn new(from: Point, to: Point) -> Line {
        Line {
            from: from,
            to: to
        }
    }
    /// Reverses the direction of the current line.
    ///
    /// ```
    /// use polygon::*;
    ///
    /// let p1 = Point::new(0.0, 0.0);
    /// let p2 = Point::new(0.7, 1.5);
    /// let line = Line::new(p1, p2);
    ///
    /// assert_eq!(line.reverse().reverse(), line);
    /// ```
    pub fn reverse(&self) -> Line {
        Line {
            from: self.to,
            to: self.from
       }
    }
    /// Calculates the slope of the given line.
    ///
    /// ```
    /// use polygon::*;
    ///
    /// let p1 = Point::new(0.0, 0.0);
    /// let p2 = Point::new(1.5, 4.5);
    /// let p3 = Point::new(0.0, 4.5);
    /// let line = Line::new(p1, p2);
    /// let vertical_line = Line::new(p1, p3);
    ///
    /// assert_eq!(line.slope(), Slope::Slope(3.0));
    /// assert_eq!(vertical_line.slope(), Slope::Vertical);
    /// ```
    pub fn slope(&self) -> Slope {
        if self.from.x == self.to.x {
            Slope::Vertical
        } else {
            Slope::Slope((self.to.y - self.from.y) / (self.to.x - self.from.x))
        }
    }
    /// Returns if two lines intersect.
    ///
    /// ```
    /// use polygon::*;
    ///
    /// let p1 = Point::new(0.0, 0.0);
    /// let p2 = Point::new(1.5, 4.5);
    /// let p3 = Point::new(0.0, 1.5);
    /// let p4 = Point::new(2.0, 0.5);
    /// let line1 = Line::new(p1, p2);
    /// let line2 = Line::new(p3, p4);
    ///
    /// assert!(line1.intersects(&line2));
    /// ```
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
                if m1 == m2 { t1 == t2 } //bounding boxes intersect
                else {
                    let x = (t2 - t1) / (m1 - m2); // x-coord of intersection point
                    s_min_x <= x && x <= s_max_x && // lies on &self
                    l_min_x <= x && x <= l_max_x    // lies on &line
                    // TODO?: use an epsilon for the comparisons above
                }
            }
        }
    }
    /// Returns if a points lies on this line.
    ///
    /// ``
    /// use polygon::*;
    ///
    /// let p1 = Point::new(0.0, 0.0);
    /// let p2 = Point::new(1.5, 4.5);
    /// let p3 = Point::new(1.0, 3.0);
    /// let line = Line::new(p1, p2);
    ///
    /// assert!(line.contains(p3));
    /// ``
    fn contains(&self, p: Point) -> bool {
        self.intersects(&Line::new(p, p))
    }
}
/// A polygon (without holes) in the 2D-plane.
/// Represented by a its vertex points in order.
#[derive(Debug, PartialEq)]
pub struct Polygon {
    points: Vec<Point>
}
impl Polygon {
    /// Creates a new empty polygon.
    pub fn new() -> Polygon {
        Polygon {
            points: Vec::new()
        }
    }
    /// Creates a new empty polygon with capacity for `c` vertex points.
    pub fn with_capacity(c: usize) -> Polygon {
        Polygon {
            points: Vec::with_capacity(c)
        }
    }
    /// Creates a new polygon from the given points.
    pub fn from_points(pts: &[Point]) -> Polygon {
        Polygon {
            points: pts.to_vec() //TODO: do it without copying?!
        }
    }
    /// Returns the number of vertices of this polygon.
    pub fn size(&self) -> usize {
        self.points.len()
    }
    /// Adds a new vertex between the first and the last vertex of this polygon.
    pub fn add(&mut self, p: Point) {
        self.points.push(p);
    }
    /// Return a slice of all the vertices of the polygon.
    pub fn points(&self) -> &[Point] {
        //self.points.clone()
        &self.points[..]
    }
    /// Return a mutable slice of all the vertices of the polygon.
    pub fn points_mut(&mut self) -> &mut [Point] {
        &mut self.points[..]
    }
    /// Returns all edges of ths polygon as `Vec` of `Line`s.
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
    /// Returns whether this polygon is simple, i.e has no self intersections.
    pub fn is_simple(&self) -> bool {
        for l1 in self.edges() {
            for l2 in self.edges() {
                if l1.from == l2.from || l1.from == l2.to ||
                   l1.to   == l2.from || l1.to   == l2.to {
                    continue; // TODO: fails for the edge case when a vertex has more
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
    /// Returns whether the given point is contained in this polygon
    /// (i.e. it lies in the interior or on the boundary).
    /// This function uses the even-odd rule to determine insideness.
    // TODO: example in doc
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
    #[test]
    fn is_simple_works() {
        let points = [Point::new(274.96843119671473, 383.51923995101777),
                      Point::new(206.10110844061037, 158.0229249347057),
                      Point::new(389.4722193425373, 220.6053166681628),
                      Point::new(468.8266446074043, 10.428792883777806),
                      Point::new(221.9383956061111, 291.53969226732926)];
        let polygon = Polygon::from_points(&points);
        assert!(!polygon.is_simple());
        let points = [Point::new(392.8896421386376, 325.62334672370304),
                      Point::new(169.53141009383438, 447.7137894321314),
                      Point::new(403.17601678365895, 397.65944288976203),
                      Point::new(264.2450152314188, 459.00996294545286),
                      Point::new(393.4736279391549, 263.26340095968646)];
        let polygon = Polygon::from_points(&points);
        assert!(!polygon.is_simple());
    }
}
