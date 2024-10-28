use anyhow::Result;
use image::{ImageBuffer, Rgb, codecs::jpeg::JpegEncoder};
use std::fs::File;
use std::io::BufWriter;
use rayon::prelude::*;
use std::time::Instant;
use num_cpus;

fn init_rayon() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();
}

fn main() -> Result<()> {
    init_rayon();
    
    let total_start = Instant::now();
    
    // Load and convert image once
    let input_path = "images/LightRoom-7.jpg";
    let img = image::open(input_path)?;
    let rgb_img = img.to_rgb8();
    
    let sizes = [1024, 2048, 4096];
    for &size in &sizes {
        println!("\nProcessing size: {}", size);
        convert_jpg_to_cubemap(&rgb_img, size, 95)?;
    }
    
    println!("\nTotal processing time for all sizes: {:?}", total_start.elapsed());
    Ok(())
}

fn cube_to_spherical(x: u32, y: u32, size: u32, face: &str) -> (f32, f32) {
    let x = (2.0 * x as f32 / size as f32) - 1.0;
    let y = (2.0 * y as f32 / size as f32) - 1.0;

    match face {
        "right" => {
            let r = (x * x + y * y + 1.0).sqrt();
            let phi = (y).atan2(1.0);
            let theta = (x / r).acos();
            ((phi / (2.0 * std::f32::consts::PI) + 0.5),
             (theta / std::f32::consts::PI))
        }
        "left" => {
            let r = (x * x + y * y + 1.0).sqrt();
            let phi = (y).atan2(-1.0);
            let theta = (-x / r).acos();
            ((phi / (2.0 * std::f32::consts::PI) + 0.5),
             (theta / std::f32::consts::PI))
        }
        "up" => {
            let r = (x * x + y * y + 1.0).sqrt();
            let phi = (-x).atan2(y);
            let theta = (1.0 / r).acos();
            ((phi / (2.0 * std::f32::consts::PI) + 0.5),
             (theta / std::f32::consts::PI))
        }
        "down" => {
            let r = (x * x + y * y + 1.0).sqrt();
            let phi = (x).atan2(-y);
            let theta = (-1.0 / r).acos();
            ((phi / (2.0 * std::f32::consts::PI) + 0.5),
             (theta / std::f32::consts::PI))
        }
        "front" => {
            let r = (x * x + y * y + 1.0).sqrt();
            let phi = (x).atan2(1.0);
            let theta = (y / r).acos();
            ((phi / (2.0 * std::f32::consts::PI) + 0.5),
             (theta / std::f32::consts::PI))
        }
        "back" => {
            let r = (x * x + y * y + 1.0).sqrt();
            let phi = (-x).atan2(-1.0);
            let theta = (-y / r).acos();
            ((phi / (2.0 * std::f32::consts::PI) + 0.5),
             (theta / std::f32::consts::PI))
        }
        _ => (0.0, 0.0)
    }
}

fn convert_jpg_to_cubemap(rgb_img: &image::RgbImage, size: u32, quality: u8) -> Result<()> {
    let start = Instant::now();
    println!("Starting conversion at {}x{}", size, size);

    let width = rgb_img.width();
    let height = rgb_img.height();
    
    println!("Starting processing at {:?}", start.elapsed());
    
    // Create output directory
    std::fs::create_dir_all(format!("output/cubemap_{}", size))?;
    
    // Process faces in parallel with larger chunks
    let faces = ["right", "left", "up", "down", "front", "back"];
    faces.par_iter().try_for_each(|face| -> Result<()> {
        let face_start = Instant::now();
        let mut face_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(size, size);
        
        // Use larger chunks for better cache utilization
        let chunk_size = (size * 16) as usize; // Adjust chunk size based on face size
        face_buffer.enumerate_pixels_mut()
            .collect::<Vec<_>>()
            .par_chunks_mut(chunk_size.min(size as usize * size as usize))
            .for_each(|chunk| {
                for (x, y, pixel) in chunk {
                    let (u, v) = cube_to_spherical(*x, *y, size, face);
                    
                    let x = (u * width as f32).rem_euclid(width as f32);
                    let y = (v * height as f32).rem_euclid(height as f32);
                    
                    let x0 = x.floor() as u32;
                    let y0 = y.floor() as u32;
                    let x1 = (x0 + 1) % width;
                    let y1 = (y0 + 1) % height;
                    
                    let fx = x.fract();
                    let fy = y.fract();
                    
                    let p00 = rgb_img.get_pixel(x0, y0);
                    let p10 = rgb_img.get_pixel(x1, y0);
                    let p01 = rgb_img.get_pixel(x0, y1);
                    let p11 = rgb_img.get_pixel(x1, y1);
                    
                    **pixel = Rgb([
                        bilerp(p00[0], p10[0], p01[0], p11[0], fx, fy),
                        bilerp(p00[1], p10[1], p01[1], p11[1], fx, fy),
                        bilerp(p00[2], p10[2], p01[2], p11[2], fx, fy),
                    ]);
                }
            });
        
        // Save with optimized buffer size
        let output_path = format!("output/cubemap_{}/{}.jpg", size, face);
        let file = File::create(&output_path)?;
        let buf_writer = BufWriter::with_capacity(65536, file); // 64KB buffer
        let mut encoder = JpegEncoder::new_with_quality(buf_writer, quality);
        encoder.encode(
            face_buffer.as_raw(),
            size,
            size,
            image::ColorType::Rgb8
        )?;
        
        println!("Face {} completed in {:?}", face, face_start.elapsed());
        Ok(())
    })?;
    
    println!("Total conversion time: {:?}", start.elapsed());
    Ok(())
}

#[inline(always)]
fn bilerp(c00: u8, c10: u8, c01: u8, c11: u8, fx: f32, fy: f32) -> u8 {
    let c0 = c00 as f32 * (1.0 - fx) + c10 as f32 * fx;
    let c1 = c01 as f32 * (1.0 - fx) + c11 as f32 * fx;
    ((c0 * (1.0 - fy) + c1 * fy) + 0.5) as u8
}
