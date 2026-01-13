$fn = 180;

bead_dia = 2.8;
row_length = 75;
wall_height = 20;
row_halfcount = 4;
row_spacing = 0.4;
floor_thickness = 1;
wall_thickness = 2;

depth_factor = 0.45;


Main();

module Main() {
    difference() {
        walls();
        translate([0, row_length, 0])
        translate([-150, 0, 0])
        rotate([60, 0, 0])
        translate([0, 0, -15])
        cube([300, 300, 80]);
    }
    flooring();
    
    module walls() {
        linear_extrude(floor_thickness + wall_height)
        difference() {
            offset(wall_thickness)
            hull()
            projection()
            rows();

            hull() {
                projection()
                rows();
                
                translate([0, 20, 0])
                projection()
                rows();
            }
        }
    }
    
    module flooring() {
        difference() {
            core();
            rows();
        }
    }
    
    module core() {
        linear_extrude(floor_thickness + bead_dia * depth_factor)
        hull()
        projection()
        rows();
    }
    
    module rows() {
        for(n = [-row_halfcount : row_halfcount]) {
            translate([n * (bead_dia + row_spacing), 0, 0])
            translate([0, 0, floor_thickness])
            translate([0, 0, bead_dia / 2])
            rotate([-90, 0, 0])
            difference() {
                cylinder(d = bead_dia, h = row_length);
            }
        }
    
    }
}
