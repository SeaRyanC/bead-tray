//! Bead Tray Physics Simulator
//!
//! This simulator evaluates how well a bead sorting tray performs at aligning beads
//! axially when shaken. It uses a simplified physics model to simulate bead dynamics.

use nalgebra::{Vector3, UnitQuaternion};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, Write};
use std::process::Command;

// Physical constants from problem statement
#[allow(dead_code)]
const BEAD_MASS_G: f64 = 0.01;          // grams
#[allow(dead_code)]
const BEAD_LENGTH_MM: f64 = 2.8;         // mm
const BEAD_OUTER_DIA_MM: f64 = 2.5;      // mm
#[allow(dead_code)]
const BEAD_INNER_DIA_MM: f64 = 1.5;      // mm
#[allow(dead_code)]
const TRAY_MASS_G: f64 = 10.45;          // grams
const FRICTION_ANGLE_DEG: f64 = 55.0;    // degrees (angle at which bead starts sliding)
const GRAVITY_MM_S2: f64 = 9810.0;       // mm/s^2

// Simulation parameters
const NUM_BEADS: usize = 100;
const SIMULATION_TIME_S: f64 = 5.0;      // Total simulation time
const DT: f64 = 0.0001;                  // Time step (100 microseconds)
const TRAY_TILT_DEG: f64 = 10.0;         // Tilt toward closed end

// Shaking parameters (reasonable hand shaking)
const SHAKE_AMPLITUDE_MM: f64 = 15.0;    // Amplitude of shake
const SHAKE_FREQUENCY_HZ: f64 = 3.0;     // Frequency (3 Hz is typical hand shake)

// Number of simulation runs per configuration (median is taken)
const SIMULATION_RUNS: usize = 5;

/// Represents the tray geometry extracted from STL
#[derive(Debug, Clone)]
struct TrayGeometry {
    /// Bounding box min
    min_bound: Vector3<f64>,
    /// Bounding box max
    max_bound: Vector3<f64>,
    /// Triangle vertices for collision detection
    triangles: Vec<Triangle>,
    /// Row groove centers (X positions)
    groove_centers: Vec<f64>,
    /// Groove depth
    groove_depth: f64,
    /// Groove radius (half of bead diameter)
    groove_radius: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct Triangle {
    v0: Vector3<f64>,
    v1: Vector3<f64>,
    v2: Vector3<f64>,
    normal: Vector3<f64>,
}

/// Represents a single bead
#[derive(Debug, Clone)]
struct Bead {
    position: Vector3<f64>,      // Center of mass position
    velocity: Vector3<f64>,      // Linear velocity
    orientation: UnitQuaternion<f64>, // Orientation quaternion
    angular_velocity: Vector3<f64>,   // Angular velocity
}

/// Configuration for the tray geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrayConfig {
    bead_dia: f64,
    row_length: f64,
    wall_height: f64,
    row_halfcount: i32,
    row_spacing: f64,
    floor_thickness: f64,
    wall_thickness: f64,
    depth_factor: f64,
}

impl Default for TrayConfig {
    fn default() -> Self {
        Self {
            bead_dia: 2.8,
            row_length: 75.0,
            wall_height: 20.0,
            row_halfcount: 4,
            row_spacing: 0.4,
            floor_thickness: 1.0,
            wall_thickness: 2.0,
            depth_factor: 0.45,
        }
    }
}

/// Simulation results
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulationResult {
    config: TrayConfig,
    alignment_score: f64,
    beads_in_grooves: usize,
    average_axial_alignment: f64,
    simulation_time_ms: u128,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 {
        match args[1].as_str() {
            "optimize" => run_optimization(false),
            "quick-test" => run_optimization(true),
            "help" | "--help" | "-h" => print_help(),
            _ => {
                eprintln!("Unknown command: {}", args[1]);
                print_help();
                std::process::exit(1);
            }
        }
    } else {
        run_single_simulation();
    }
}

fn print_help() {
    println!("Bead Tray Physics Simulator");
    println!();
    println!("USAGE:");
    println!("  bead_simulator              Run single simulation with default config");
    println!("  bead_simulator optimize     Run full optimization (245 configurations)");
    println!("  bead_simulator quick-test   Run quick optimization test (8 configurations)");
    println!("  bead_simulator help         Show this help message");
    println!();
    println!("DESCRIPTION:");
    println!("  This simulator evaluates how well a bead sorting tray performs at");
    println!("  aligning beads axially when shaken. It uses physics simulation to");
    println!("  model 100 beads being shaken in the tray.");
    println!();
    println!("  The optimization mode searches over bead_dia (groove size), depth_factor,");
    println!("  and row_spacing parameters to find the best tray configuration.");
    println!("  Note: bead_dia affects groove size in the SCAD file, but simulated beads");
    println!("  always use the physical bead diameter (2.5mm).");
    println!();
    println!("OUTPUT:");
    println!("  Results are saved to optimization_results.json when running optimize mode.");
}

fn run_single_simulation() {
    println!("=== Bead Tray Physics Simulator ===\n");
    
    let config = TrayConfig::default();
    println!("Configuration: {:?}\n", config);
    
    // Generate STL from config
    let stl_path = generate_stl(&config).expect("Failed to generate STL");
    
    // Load tray geometry
    let geometry = load_tray_geometry(&stl_path, &config);
    println!("Loaded tray geometry:");
    println!("  Bounds: {:?} to {:?}", geometry.min_bound, geometry.max_bound);
    println!("  Groove centers: {:?}", geometry.groove_centers);
    println!("  Groove depth: {} mm", geometry.groove_depth);
    println!("  Triangles: {}\n", geometry.triangles.len());
    
    // Run multiple simulations and take median
    println!("Running {} simulations to get median result...\n", SIMULATION_RUNS);
    let (result, beads) = run_multiple_simulations(&geometry, &config);
    
    println!("\n=== Simulation Results (Median of {} runs) ===", SIMULATION_RUNS);
    println!("Alignment Score: {:.2}%", result.alignment_score * 100.0);
    println!("Beads in grooves: {}/{}", result.beads_in_grooves, NUM_BEADS);
    println!("Average axial alignment: {:.2}%", result.average_axial_alignment * 100.0);
    println!("Simulation time: {} ms", result.simulation_time_ms);
    
    // Render visualization image
    match render_simulation_image(&beads, &config) {
        Ok(image_path) => println!("Visualization saved to: {}", image_path),
        Err(e) => eprintln!("Failed to render visualization: {}", e),
    }
}

fn run_optimization(quick_test: bool) {
    if quick_test {
        println!("=== Bead Tray Quick Test ===\n");
        println!("Running quick optimization with reduced search space\n");
    } else {
        println!("=== Bead Tray Optimization ===\n");
    }
    println!("Optimizing parameters: bead_dia (groove size), depth_factor, row_spacing\n");
    
    // Define parameter ranges to search
    // bead_dia affects groove diameter in SCAD but simulated beads stay at BEAD_OUTER_DIA_MM
    let (bead_dias, depth_factors, row_spacings): (Vec<f64>, Vec<f64>, Vec<f64>) = if quick_test {
        // Quick test: 2x2x2 grid = 8 configs
        (vec![2.6, 2.8], vec![0.40, 0.50], vec![0.3, 0.5])
    } else {
        // Full search: 5x7x7 grid = 245 configs
        (
            vec![2.4, 2.5, 2.6, 2.7, 2.8],  // Groove diameters from tight to loose fit
            (30..=60).step_by(5).map(|x| x as f64 / 100.0).collect(),
            (2..=8).step_by(1).map(|x| x as f64 / 10.0).collect(),
        )
    };
    
    println!("Search space:");
    println!("  bead_dia (groove size): {:?}", bead_dias);
    println!("  depth_factor: {:?}", depth_factors);
    println!("  row_spacing: {:?}", row_spacings);
    
    let mut best_result: Option<SimulationResult> = None;
    let mut results: Vec<SimulationResult> = Vec::new();
    
    // Grid search over all parameters
    let mut configs: Vec<TrayConfig> = Vec::new();
    for &bd in &bead_dias {
        for &df in &depth_factors {
            for &rs in &row_spacings {
                let mut config = TrayConfig::default();
                config.bead_dia = bd;
                config.depth_factor = df;
                config.row_spacing = rs;
                configs.push(config);
            }
        }
    }
    
    println!("\nTotal configurations to test: {}\n", configs.len());
    println!("Each configuration will run {} simulations (median taken)\n", SIMULATION_RUNS);
    
    // Run simulations (sequentially since STL generation needs file system)
    for (i, config) in configs.iter().enumerate() {
        println!(
            "[{}/{}] Testing bead_dia={:.2}, depth_factor={:.2}, row_spacing={:.2}...",
            i + 1,
            configs.len(),
            config.bead_dia,
            config.depth_factor,
            config.row_spacing
        );
        
        match generate_stl(config) {
            Ok(stl_path) => {
                let geometry = load_tray_geometry(&stl_path, config);
                let (result, beads) = run_multiple_simulations(&geometry, config);
                
                println!(
                    "  -> Median Score: {:.2}%, Beads in grooves: {}",
                    result.alignment_score * 100.0,
                    result.beads_in_grooves
                );
                
                // Render visualization image
                match render_simulation_image(&beads, config) {
                    Ok(image_path) => println!("  -> Image saved: {}", image_path),
                    Err(e) => eprintln!("  -> Failed to render image: {}", e),
                }
                
                if best_result.is_none()
                    || result.alignment_score > best_result.as_ref().unwrap().alignment_score
                {
                    best_result = Some(result.clone());
                }
                results.push(result);
            }
            Err(e) => {
                eprintln!("  -> Failed to generate STL: {}", e);
            }
        }
    }
    
    // Output results
    println!("\n=== Optimization Complete ===\n");
    
    if let Some(best) = &best_result {
        println!("Best configuration found:");
        println!("  bead_dia (groove size): {:.2}", best.config.bead_dia);
        println!("  depth_factor: {:.2}", best.config.depth_factor);
        println!("  row_spacing: {:.2}", best.config.row_spacing);
        println!("  Alignment Score: {:.2}%", best.alignment_score * 100.0);
        println!("  Beads in grooves: {}", best.beads_in_grooves);
        println!("  Axial alignment: {:.2}%", best.average_axial_alignment * 100.0);
    }
    
    // Save all results to JSON
    let results_json = serde_json::to_string_pretty(&results).unwrap();
    std::fs::write("optimization_results.json", results_json).unwrap();
    println!("\nAll results saved to optimization_results.json");
}

fn generate_stl(config: &TrayConfig) -> Result<String, String> {
    // Generate modified SCAD file
    let scad_content = format!(
        r#"$fn = 180;

bead_dia = {bead_dia};
row_length = {row_length};
wall_height = {wall_height};
row_halfcount = {row_halfcount};
row_spacing = {row_spacing};
floor_thickness = {floor_thickness};
wall_thickness = {wall_thickness};

depth_factor = {depth_factor};


Main();

module Main() {{
    difference() {{
        walls();
        translate([0, row_length, 0])
        translate([-150, 0, 0])
        rotate([60, 0, 0])
        translate([0, 0, -15])
        cube([300, 300, 80]);
    }}
    flooring();
    
    module walls() {{
        linear_extrude(floor_thickness + wall_height)
        difference() {{
            offset(wall_thickness)
            hull()
            projection()
            rows();

            hull() {{
                projection()
                rows();
                
                translate([0, 20, 0])
                projection()
                rows();
            }}
        }}
    }}
    
    module flooring() {{
        difference() {{
            core();
            rows();
        }}
    }}
    
    module core() {{
        linear_extrude(floor_thickness + bead_dia * depth_factor)
        hull()
        projection()
        rows();
    }}
    
    module rows() {{
        for(n = [-row_halfcount : row_halfcount]) {{
            translate([n * (bead_dia + row_spacing), 0, 0])
            translate([0, 0, floor_thickness])
            translate([0, 0, bead_dia / 2])
            rotate([-90, 0, 0])
            difference() {{
                cylinder(d = bead_dia, h = row_length);
            }}
        }}
    
    }}
}}
"#,
        bead_dia = config.bead_dia,
        row_length = config.row_length,
        wall_height = config.wall_height,
        row_halfcount = config.row_halfcount,
        row_spacing = config.row_spacing,
        floor_thickness = config.floor_thickness,
        wall_thickness = config.wall_thickness,
        depth_factor = config.depth_factor,
    );
    
    let temp_dir = std::env::temp_dir();
    let scad_path = temp_dir.join("tray_temp.scad");
    let stl_path = temp_dir.join("tray_temp.stl");
    
    std::fs::write(&scad_path, scad_content)
        .map_err(|e| format!("Failed to write SCAD file: {}", e))?;
    
    // Run OpenSCAD to generate STL
    let output = Command::new("/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD")
        .args(["-o",  &stl_path.to_string_lossy(), &scad_path.to_string_lossy()])
        .output()
        .map_err(|e| format!("Failed to run OpenSCAD: {}", e))?;
    
    if !output.status.success() {
        return Err(format!(
            "OpenSCAD failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    
    Ok(stl_path.to_string_lossy().to_string())
}

fn load_tray_geometry(stl_path: &str, config: &TrayConfig) -> TrayGeometry {
    let file = File::open(stl_path).expect("Failed to open STL file");
    let mut reader = BufReader::new(file);
    
    let stl = stl_io::read_stl(&mut reader).expect("Failed to parse STL file");
    
    let mut min_bound = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max_bound = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
    let mut triangles = Vec::new();
    
    for face in stl.faces.iter() {
        let v0 = Vector3::new(
            stl.vertices[face.vertices[0]][0] as f64,
            stl.vertices[face.vertices[0]][1] as f64,
            stl.vertices[face.vertices[0]][2] as f64,
        );
        let v1 = Vector3::new(
            stl.vertices[face.vertices[1]][0] as f64,
            stl.vertices[face.vertices[1]][1] as f64,
            stl.vertices[face.vertices[1]][2] as f64,
        );
        let v2 = Vector3::new(
            stl.vertices[face.vertices[2]][0] as f64,
            stl.vertices[face.vertices[2]][1] as f64,
            stl.vertices[face.vertices[2]][2] as f64,
        );
        
        let normal = Vector3::new(
            face.normal[0] as f64,
            face.normal[1] as f64,
            face.normal[2] as f64,
        );
        
        for v in [&v0, &v1, &v2] {
            min_bound.x = min_bound.x.min(v.x);
            min_bound.y = min_bound.y.min(v.y);
            min_bound.z = min_bound.z.min(v.z);
            max_bound.x = max_bound.x.max(v.x);
            max_bound.y = max_bound.y.max(v.y);
            max_bound.z = max_bound.z.max(v.z);
        }
        
        triangles.push(Triangle { v0, v1, v2, normal });
    }
    
    // Calculate groove centers based on config
    let groove_centers: Vec<f64> = (-config.row_halfcount..=config.row_halfcount)
        .map(|n| n as f64 * (config.bead_dia + config.row_spacing))
        .collect();
    
    let groove_depth = config.bead_dia * config.depth_factor;
    let groove_radius = config.bead_dia / 2.0;
    
    TrayGeometry {
        min_bound,
        max_bound,
        triangles,
        groove_centers,
        groove_depth,
        groove_radius,
    }
}

fn run_simulation(geometry: &TrayGeometry, config: &TrayConfig) -> (SimulationResult, Vec<Bead>) {
    let start_time = std::time::Instant::now();
    
    // Initialize beads randomly on the tray surface
    let mut beads = initialize_beads(geometry, config);
    
    // Friction coefficient from problem statement
    let friction_coeff = (FRICTION_ANGLE_DEG * PI / 180.0).tan();
    
    // Tray tilt
    let tilt_rad = TRAY_TILT_DEG * PI / 180.0;
    
    // Simulation loop
    let num_steps = (SIMULATION_TIME_S / DT) as usize;
    let report_interval = num_steps / 10;
    
    for step in 0..num_steps {
        // Current time
        let t = step as f64 * DT;
        
        // Tray acceleration from shaking (second derivative of sinusoidal motion)
        // We use acceleration to model pseudo-force on beads in tray's reference frame
        let shake_accel_x = -SHAKE_AMPLITUDE_MM * (2.0 * PI * SHAKE_FREQUENCY_HZ).powi(2)
            * (2.0 * PI * SHAKE_FREQUENCY_HZ * t).sin();
        let shake_accel_y = -SHAKE_AMPLITUDE_MM * 0.5 * (2.0 * PI * SHAKE_FREQUENCY_HZ).powi(2)
            * (2.0 * PI * SHAKE_FREQUENCY_HZ * t + PI / 3.0).sin();
        
        // Update each bead
        for bead in beads.iter_mut() {
            update_bead(
                bead,
                geometry,
                config,
                friction_coeff,
                tilt_rad,
                shake_accel_x,
                shake_accel_y,
                DT,
            );
        }
        
        // Bead-bead collisions
        resolve_bead_collisions(&mut beads);
        
        if step % report_interval == 0 && step > 0 {
            let progress = (step as f64 / num_steps as f64) * 100.0;
            print!("\rSimulation progress: {:.0}%", progress);
            std::io::stdout().flush().ok();
        }
    }
    println!();
    
    // Calculate final metrics
    let (alignment_score, beads_in_grooves, average_axial_alignment) =
        calculate_alignment_metrics(&beads, geometry, config);
    
    let result = SimulationResult {
        config: config.clone(),
        alignment_score,
        beads_in_grooves,
        average_axial_alignment,
        simulation_time_ms: start_time.elapsed().as_millis(),
    };
    
    (result, beads)
}

/// Run multiple simulations and return the result with the median alignment score along with
/// the beads from that median run
fn run_multiple_simulations(geometry: &TrayGeometry, config: &TrayConfig) -> (SimulationResult, Vec<Bead>) {
    let mut run_results: Vec<(SimulationResult, Vec<Bead>)> = Vec::with_capacity(SIMULATION_RUNS);
    
    for run_idx in 0..SIMULATION_RUNS {
        print!("  Run {}/{}: ", run_idx + 1, SIMULATION_RUNS);
        let (result, beads) = run_simulation(geometry, config);
        println!("Score: {:.2}%", result.alignment_score * 100.0);
        run_results.push((result, beads));
    }
    
    // Sort by alignment score to find median
    run_results.sort_by(|a, b| {
        a.0.alignment_score
            .partial_cmp(&b.0.alignment_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    
    // Take the median (middle element for 5 runs)
    let median_idx = SIMULATION_RUNS / 2;
    run_results.remove(median_idx)
}

/// Generate a SCAD file visualizing the final bead positions
fn generate_visualization_scad(beads: &[Bead], config: &TrayConfig) -> String {
    let mut scad = String::new();
    
    // Include the tray geometry
    scad.push_str(&format!(
        r#"$fn = 60;

bead_dia = {bead_dia};
row_length = {row_length};
wall_height = {wall_height};
row_halfcount = {row_halfcount};
row_spacing = {row_spacing};
floor_thickness = {floor_thickness};
wall_thickness = {wall_thickness};
depth_factor = {depth_factor};

// Tray geometry
Main();

module Main() {{
    difference() {{
        walls();
        translate([0, row_length, 0])
        translate([-150, 0, 0])
        rotate([60, 0, 0])
        translate([0, 0, -15])
        cube([300, 300, 80]);
    }}
    flooring();
    
    module walls() {{
        linear_extrude(floor_thickness + wall_height)
        difference() {{
            offset(wall_thickness)
            hull()
            projection()
            rows();

            hull() {{
                projection()
                rows();
                
                translate([0, 20, 0])
                projection()
                rows();
            }}
        }}
    }}
    
    module flooring() {{
        difference() {{
            core();
            rows();
        }}
    }}
    
    module core() {{
        linear_extrude(floor_thickness + bead_dia * depth_factor)
        hull()
        projection()
        rows();
    }}
    
    module rows() {{
        for(n = [-row_halfcount : row_halfcount]) {{
            translate([n * (bead_dia + row_spacing), 0, 0])
            translate([0, 0, floor_thickness])
            translate([0, 0, bead_dia / 2])
            rotate([-90, 0, 0])
            difference() {{
                cylinder(d = bead_dia, h = row_length);
            }}
        }}
    
    }}
}}

// Beads
"#,
        bead_dia = config.bead_dia,
        row_length = config.row_length,
        wall_height = config.wall_height,
        row_halfcount = config.row_halfcount,
        row_spacing = config.row_spacing,
        floor_thickness = config.floor_thickness,
        wall_thickness = config.wall_thickness,
        depth_factor = config.depth_factor,
    ));
    
    // Add each bead as a cylinder with its position and orientation
    for (i, bead) in beads.iter().enumerate() {
        // Convert quaternion to axis-angle for OpenSCAD rotation
        let axis_angle = bead.orientation.axis_angle();
        let (axis, angle_rad) = match axis_angle {
            Some((axis, angle)) => (axis.into_inner(), angle),
            None => (Vector3::new(0.0, 0.0, 1.0), 0.0),
        };
        let angle_deg = angle_rad * 180.0 / PI;
        
        scad.push_str(&format!(
            r#"// Bead {i}
translate([{x}, {y}, {z}])
rotate(a={angle}, v=[{ax}, {ay}, {az}])
rotate([90, 0, 0])
color("red")
difference() {{
    cylinder(d={outer_d}, h={length}, center=true);
    cylinder(d={inner_d}, h={length}+1, center=true);
}}
"#,
            i = i,
            x = bead.position.x,
            y = bead.position.y,
            z = bead.position.z,
            angle = angle_deg,
            ax = axis.x,
            ay = axis.y,
            az = axis.z,
            outer_d = BEAD_OUTER_DIA_MM,
            inner_d = BEAD_INNER_DIA_MM,
            length = BEAD_LENGTH_MM,
        ));
    }
    
    scad
}

/// Render the simulation result as an image and save it to the images directory
fn render_simulation_image(beads: &[Bead], config: &TrayConfig) -> Result<String, String> {
    // Create images directory if it doesn't exist
    std::fs::create_dir_all("images")
        .map_err(|e| format!("Failed to create images directory: {}", e))?;
    
    // Generate SCAD visualization
    let scad_content = generate_visualization_scad(beads, config);
    
    // Create filename based on parameters
    let filename = format!(
        "bd{:.2}_df{:.2}_rs{:.2}",
        config.bead_dia,
        config.depth_factor,
        config.row_spacing
    );
    
    let temp_dir = std::env::temp_dir();
    let scad_path = temp_dir.join(format!("viz_{}.scad", filename));
    let scad_path_str = scad_path.to_string_lossy();
    let image_path = format!("images/{}.png", filename);
    
    std::fs::write(&scad_path, scad_content)
        .map_err(|e| format!("Failed to write visualization SCAD file: {}", e))?;
    
    // Render with OpenSCAD
    // Use a camera angle that shows the tray from above at an angle
    let output = Command::new("xvfb-run")
        .args([
            "-a",
            "openscad",
            "--camera=0,35,50,60,0,30,120",  // Camera: eye position, center, distance
            "--imgsize=800,600",
            "-o", &image_path,
            &scad_path_str,
        ])
        .output()
        .map_err(|e| format!("Failed to run OpenSCAD for rendering: {}", e))?;
    
    if !output.status.success() {
        return Err(format!(
            "OpenSCAD rendering failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    
    Ok(image_path)
}

fn initialize_beads(geometry: &TrayGeometry, config: &TrayConfig) -> Vec<Bead> {
    let mut rng = rand::thread_rng();
    let mut beads = Vec::with_capacity(NUM_BEADS);
    
    // Tray usable area (inside walls)
    let margin = config.wall_thickness + BEAD_OUTER_DIA_MM / 2.0;
    let x_min = geometry.min_bound.x + margin;
    let x_max = geometry.max_bound.x - margin;
    let y_min = geometry.min_bound.y + margin;
    let y_max = geometry.max_bound.y - margin;
    
    // Place beads with initial random positions and orientations
    for _ in 0..NUM_BEADS {
        let x = rng.gen_range(x_min..x_max);
        let y = rng.gen_range(y_min..y_max);
        let z = config.floor_thickness + config.bead_dia * config.depth_factor + BEAD_OUTER_DIA_MM;
        
        // Random orientation
        let axis_angle = Vector3::new(
            rng.gen_range(-PI..PI),
            rng.gen_range(-PI..PI),
            rng.gen_range(-PI..PI),
        );
        let angle = axis_angle.magnitude();
        let orientation = if angle > 1e-6 {
            UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(axis_angle / angle),
                angle,
            )
        } else {
            UnitQuaternion::identity()
        };
        
        beads.push(Bead {
            position: Vector3::new(x, y, z),
            velocity: Vector3::zeros(),
            orientation,
            angular_velocity: Vector3::zeros(),
        });
    }
    
    beads
}

fn update_bead(
    bead: &mut Bead,
    geometry: &TrayGeometry,
    config: &TrayConfig,
    friction_coeff: f64,
    tilt_rad: f64,
    shake_accel_x: f64,
    shake_accel_y: f64,
    dt: f64,
) {
    let bead_radius = BEAD_OUTER_DIA_MM / 2.0;
    
    // Gravity with tray tilt (tilted toward -Y direction)
    let gravity = Vector3::new(
        0.0,
        GRAVITY_MM_S2 * tilt_rad.sin(),
        -GRAVITY_MM_S2 * tilt_rad.cos(),
    );
    
    // Pseudo-force from tray acceleration (opposite direction)
    let shake_force = Vector3::new(-shake_accel_x, -shake_accel_y, 0.0);
    
    // Total acceleration
    let mut accel = gravity + shake_force;
    
    // Find nearest groove
    let (nearest_groove_x, groove_dist) = find_nearest_groove(bead.position.x, &geometry.groove_centers);
    
    // Groove constraint force (pushes bead toward groove center)
    let groove_depth_at_pos = get_groove_depth_at(bead.position.y, geometry, config);
    let in_groove = bead.position.z < config.floor_thickness + groove_depth_at_pos + bead_radius * 1.5;
    
    if in_groove && groove_dist < geometry.groove_radius {
        // Apply force toward groove center
        let groove_force_x = (nearest_groove_x - bead.position.x) * 100.0;
        accel.x += groove_force_x;
    }
    
    // Floor collision
    let floor_z = config.floor_thickness;
    let groove_bottom_z = floor_z + groove_depth_at_pos - geometry.groove_radius;
    
    // Check if in a groove
    let min_z = if groove_dist < geometry.groove_radius {
        // In a groove - calculate height based on groove geometry
        let lateral_offset = groove_dist.abs();
        let groove_bottom = groove_bottom_z;
        if lateral_offset < geometry.groove_radius - bead_radius {
            // Deep in groove
            groove_bottom + bead_radius + 
                (geometry.groove_radius.powi(2) - lateral_offset.powi(2)).sqrt() - geometry.groove_radius
        } else {
            // On edge of groove
            floor_z + groove_depth_at_pos + bead_radius * 0.5
        }
    } else {
        // On flat floor
        floor_z + bead_radius
    };
    
    if bead.position.z < min_z {
        bead.position.z = min_z;
        if bead.velocity.z < 0.0 {
            bead.velocity.z = -bead.velocity.z * 0.3; // Bounce with energy loss
        }
        
        // Friction
        let vel_horizontal = Vector3::new(bead.velocity.x, bead.velocity.y, 0.0);
        let vel_mag = vel_horizontal.magnitude();
        if vel_mag > 1e-6 {
            let normal_force = GRAVITY_MM_S2;
            let friction_force = friction_coeff * normal_force;
            let friction_accel = friction_force;
            
            let friction_decel = vel_horizontal.normalize() * friction_accel.min(vel_mag / dt);
            accel -= friction_decel;
        }
        
        // Rolling resistance (beads can roll along Y axis when aligned in groove)
        let bead_axis = bead.orientation * Vector3::y();
        let alignment_with_y = bead_axis.dot(&Vector3::y()).abs();
        
        // If well aligned with groove, allow rolling
        if alignment_with_y > 0.9 {
            // Reduce friction in Y direction (rolling)
            let rolling_friction = friction_coeff * 0.1;
            if bead.velocity.y.abs() > 1e-6 {
                let rolling_decel = bead.velocity.y.signum() * rolling_friction * GRAVITY_MM_S2;
                accel.y -= rolling_decel;
            }
        }
    }
    
    // Wall collisions
    let wall_margin = config.wall_thickness;
    if bead.position.x < geometry.min_bound.x + wall_margin + bead_radius {
        bead.position.x = geometry.min_bound.x + wall_margin + bead_radius;
        bead.velocity.x = -bead.velocity.x * 0.3;
    }
    if bead.position.x > geometry.max_bound.x - wall_margin - bead_radius {
        bead.position.x = geometry.max_bound.x - wall_margin - bead_radius;
        bead.velocity.x = -bead.velocity.x * 0.3;
    }
    if bead.position.y < geometry.min_bound.y + bead_radius {
        bead.position.y = geometry.min_bound.y + bead_radius;
        bead.velocity.y = -bead.velocity.y * 0.3;
    }
    // Open end (Y max) - tilt keeps beads from falling out
    if bead.position.y > config.row_length - bead_radius {
        bead.position.y = config.row_length - bead_radius;
        bead.velocity.y = -bead.velocity.y * 0.3;
    }
    
    // Cap Z to prevent escaping
    let max_z = config.floor_thickness + config.wall_height;
    if bead.position.z > max_z {
        bead.position.z = max_z;
        bead.velocity.z = -bead.velocity.z * 0.3;
    }
    
    // Update velocity and position
    bead.velocity += accel * dt;
    
    // Apply damping
    bead.velocity *= 0.999;
    
    bead.position += bead.velocity * dt;
    
    // Update orientation based on angular velocity
    if bead.angular_velocity.magnitude() > 1e-6 {
        let angle = bead.angular_velocity.magnitude() * dt;
        let axis = nalgebra::Unit::new_normalize(bead.angular_velocity);
        let rotation = UnitQuaternion::from_axis_angle(&axis, angle);
        bead.orientation = rotation * bead.orientation;
    }
    
    // Derive angular velocity from linear motion when in contact with ground
    if bead.position.z < min_z + 0.1 {
        // Rolling motion - angular velocity from linear velocity
        // When rolling along Y, axis is along X
        let rolling_omega_x = bead.velocity.y / bead_radius;
        let rolling_omega_y = -bead.velocity.x / bead_radius;
        
        // Blend toward rolling angular velocity
        bead.angular_velocity.x = bead.angular_velocity.x * 0.9 + rolling_omega_x * 0.1;
        bead.angular_velocity.y = bead.angular_velocity.y * 0.9 + rolling_omega_y * 0.1;
    }
    
    // Damping on angular velocity
    bead.angular_velocity *= 0.995;
}

fn find_nearest_groove(x: f64, groove_centers: &[f64]) -> (f64, f64) {
    let mut min_dist = f64::MAX;
    let mut nearest = 0.0;
    
    for &center in groove_centers {
        let dist = (x - center).abs();
        if dist < min_dist {
            min_dist = dist;
            nearest = center;
        }
    }
    
    (nearest, min_dist)
}

fn get_groove_depth_at(y: f64, geometry: &TrayGeometry, config: &TrayConfig) -> f64 {
    // Grooves are full depth until near the open end
    if y < config.row_length * 0.9 {
        geometry.groove_depth
    } else {
        // Taper at open end
        let taper = (config.row_length - y) / (config.row_length * 0.1);
        geometry.groove_depth * taper.max(0.0)
    }
}

fn resolve_bead_collisions(beads: &mut [Bead]) {
    let min_dist = BEAD_OUTER_DIA_MM * 0.95; // Allow slight overlap for stability
    
    for i in 0..beads.len() {
        for j in (i + 1)..beads.len() {
            let delta = beads[j].position - beads[i].position;
            let dist = delta.magnitude();
            
            if dist < min_dist && dist > 1e-6 {
                let overlap = min_dist - dist;
                let normal = delta / dist;
                
                // Separate beads
                beads[i].position -= normal * (overlap * 0.5);
                beads[j].position += normal * (overlap * 0.5);
                
                // Elastic collision response
                let rel_vel = beads[j].velocity - beads[i].velocity;
                let vel_along_normal = rel_vel.dot(&normal);
                
                if vel_along_normal < 0.0 {
                    // Beads approaching
                    let impulse = normal * vel_along_normal * 0.8; // 0.8 = restitution
                    beads[i].velocity += impulse;
                    beads[j].velocity -= impulse;
                }
            }
        }
    }
}

// Threshold for considering a bead "fully axially aligned"
// 0.95 corresponds to approximately 18 degrees from perfect alignment
const AXIAL_ALIGNMENT_THRESHOLD: f64 = 0.95;

fn calculate_alignment_metrics(
    beads: &[Bead],
    geometry: &TrayGeometry,
    config: &TrayConfig,
) -> (f64, usize, f64) {
    let groove_tolerance = geometry.groove_radius * 0.5;
    
    let mut fully_aligned_beads = 0;
    let mut beads_in_grooves = 0;
    let mut total_axial_alignment = 0.0;
    
    for bead in beads {
        // Check if bead is in a groove
        let (_, groove_dist) = find_nearest_groove(bead.position.x, &geometry.groove_centers);
        let in_groove = groove_dist < groove_tolerance;
        
        // Check Z position (should be settled in groove)
        let expected_z = config.floor_thickness + geometry.groove_depth / 2.0;
        let z_ok = (bead.position.z - expected_z).abs() < geometry.groove_radius;
        
        // Axial alignment: how well the bead's long axis aligns with Y
        let bead_axis = bead.orientation * Vector3::y();
        let alignment = bead_axis.dot(&Vector3::y()).abs();
        total_axial_alignment += alignment;
        
        // A bead only counts toward the positive score if it's fully axially aligned
        // in a groove. Partial axial alignment doesn't score anything.
        let fully_aligned = alignment >= AXIAL_ALIGNMENT_THRESHOLD;
        
        if in_groove && z_ok {
            beads_in_grooves += 1;
            if fully_aligned {
                fully_aligned_beads += 1;
            }
        }
    }
    
    let average_axial_alignment = total_axial_alignment / beads.len() as f64;
    
    // Score is based only on beads that are both in groove AND fully axially aligned
    let alignment_score = fully_aligned_beads as f64 / beads.len() as f64;
    
    (alignment_score, beads_in_grooves, average_axial_alignment)
}
