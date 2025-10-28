use std::{env::var, io::Write, time::Instant, usize};

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyModifiers},
    execute, queue, terminal,
};
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tokio::sync::watch;

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = execute!(std::io::stdout(), cursor::Show);
        let _ = terminal::disable_raw_mode();
    }
}

const ASCII_CHARS: [char; 70] = [
    '$', '@', 'B', '%', '8', '&', 'W', 'M', '#', '*', 'o', 'a', 'h', 'k', 'b', 'd', 'p', 'q', 'w',
    'm', 'Z', 'O', '0', 'Q', 'L', 'C', 'J', 'U', 'Y', 'X', 'z', 'c', 'v', 'u', 'n', 'x', 'r', 'j',
    'f', 't', '/', '\\', '|', '(', ')', '1', '{', '}', '[', ']', '?', '-', '_', '+', '~', '<', '>',
    'i', '!', 'l', 'I', ';', ':', ',', '"', '^', '`', '\'', '.', ' ',
];
const TARGET_FRAME_TIME_MS: u128 = 16;

enum SobelEdge {
    None,
    Horizontal,
    Vertical,
    DiagonalUp,
    DiagonalDown,
}

struct DecodedFrame {
    buffer: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

fn sobel_detect_edge(
    decoded: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    threshold: f32,
) -> SobelEdge {
    if x == 0 || y == 0 || x >= width - 1 || y >= height - 1 {
        return SobelEdge::None;
    }

    let get_brightness = |px: u32, py: u32| -> i32 {
        let pixel = decoded.get_pixel(px, py);
        ((pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3) as i32
    };

    // 3x3 neighborhood
    let nw = get_brightness((x - 1) as u32, (y - 1) as u32);
    let n = get_brightness((x) as u32, (y - 1) as u32);
    let ne = get_brightness((x + 1) as u32, (y - 1) as u32);
    let w = get_brightness((x - 1) as u32, (y) as u32);
    let e = get_brightness((x + 1) as u32, (y) as u32);
    let sw = get_brightness((x - 1) as u32, (y + 1) as u32);
    let s = get_brightness((x) as u32, (y + 1) as u32);
    let se = get_brightness((x + 1) as u32, (y + 1) as u32);

    // Sobel operator kernels
    // Gx (horizontal gradient):     Gy (vertical gradient):
    //   -1  0  +1                      -1  -2  -1
    //   -2  0  +2                       0   0   0
    //   -1  0  +1                      +1  +2  +1

    let gx = -nw + ne - 2 * w + 2 * e - sw + se;
    let gy = -nw - 2 * n - ne + sw + 2 * s + se;

    let magnitude = ((gx * gx + gy * gy) as f32).sqrt();

    if magnitude <= threshold {
        return SobelEdge::None;
    }

    let angle = (gy as f32).atan2(gx as f32);

    let degrees = angle.to_degrees();
    let normalized = if degrees < 0.0 {
        degrees + 360.0
    } else {
        degrees
    };

    match normalized {
        a if a >= 337.5 || a < 22.5 => SobelEdge::Vertical,
        a if a >= 22.5 && a < 67.5 => SobelEdge::DiagonalDown,
        a if a >= 67.5 && a < 112.5 => SobelEdge::Horizontal,
        a if a >= 112.5 && a < 157.5 => SobelEdge::DiagonalUp,
        a if a >= 157.5 && a < 202.5 => SobelEdge::Vertical,
        a if a >= 202.5 && a < 247.5 => SobelEdge::DiagonalDown,
        a if a >= 247.5 && a < 292.5 => SobelEdge::Horizontal,
        _ => SobelEdge::DiagonalUp,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    terminal::enable_raw_mode()?;
    let _guard = TerminalGuard;

    let mut stdout = std::io::stdout();
    execute!(
        stdout,
        terminal::Clear(terminal::ClearType::All),
        cursor::Hide
    )?;

    let (frame_tx, mut frame_rx) = watch::channel(None);
    let (quit_tx, mut quit_rx) = watch::channel(false);

    tokio::spawn(async move {
        loop {
            if let Ok(Event::Key(key)) = event::read() {
                match key.code {
                    KeyCode::Char('q') => {
                        let _ = quit_tx.send(true);
                        break;
                    }
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        let _ = quit_tx.send(true);
                        break;
                    }
                    _ => {}
                }
            }
        }
    });

    // KNOWN ISSUE: First run may hang on camera initialization
    // This is a hardware/driver warm-up issue, not a Rust problem
    // Workaround: Run twice, or wait ~30s on first run
    println!("Stream opened. Warming up...");
    println!("NOTE: First run may take 30s while camera initializes...");

    tokio::task::spawn_blocking(move || {
        let index = CameraIndex::Index(0);
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        let mut camera = match Camera::new(index, requested) {
            Ok(cam) => cam,
            Err(e) => {
                eprintln!("Error creating camera: {}", e);
                return;
            }
        };

        if let Err(e) = camera.open_stream() {
            eprintln!("Error opening stream: {}", e);
            return;
        }

        loop {
            let frame_data = camera.frame();

            if let Ok(frame) = frame_data {
                match frame.decode_image::<RgbFormat>() {
                    Ok(decoded) => {
                        let width = frame.resolution().width() as usize;
                        let height = frame.resolution().height() as usize;
                        let pixels = decoded.as_raw().to_vec();
                        if frame_tx
                            .send(Some(DecodedFrame {
                                buffer: decoded,
                                width,
                                height,
                                pixels,
                            }))
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("Decode error: {}", e);
                        continue;
                    }
                }
            }
        }
    });

    let mut prev_frame: Option<Vec<u8>> = None;
    let mut frame_buffer = String::with_capacity(2_000_000);
    let mut should_skip_next_frame = false;
    let mut prev_rows: Option<Vec<String>> = None;

    let color_lookup: Vec<String> = (0..4096)
        .map(|i| {
            let r = ((i >> 8) & 0xF) * 17;
            let g = ((i >> 4) & 0xF) * 17;
            let b = (i & 0xF) * 17;
            format!("\x1b[38;2;{};{};{}m", r, g, b)
        })
        .collect();

    loop {
        tokio::select! {
            Ok(_) = frame_rx.changed() => {
                if *quit_rx.borrow() {
                    break;
                }

                let (term_cols, term_rows) = terminal::size()?;
                let term_width = term_cols as usize;
                let term_height = term_rows as usize;
                let total_pixels = term_width * term_height;
                let estimated_size = term_width * term_height * 25;
                let sobel_sample_rate = if total_pixels > 200_000 {
                    20
                } else if total_pixels > 100_000 {
                    10
                } else {
                    1
                };

                if frame_buffer.capacity() < estimated_size {
                    frame_buffer.reserve(estimated_size - frame_buffer.capacity());
                }

                if let Some(frame) = frame_rx.borrow().as_ref() {
                    let frame_start = Instant::now();

                    if should_skip_next_frame {
                        prev_frame = Some(frame.pixels.clone());
                        should_skip_next_frame = false;
                        continue;
                    }

                    frame_buffer.clear();

                    let width = frame.width;
                    let height = frame.height;
                    let decoded = &frame.buffer;
                    let current_pixels = &frame.pixels;

                    let rows: Vec<String> = (0..term_height)
                        .into_par_iter()
                        .map(|ty| {
                            let mut row_buffer = String::with_capacity(term_width * 20);

                            let mut last_color_idx = usize::MAX;

                            for tx in 0..term_width {
                                let x = tx * width / term_width;
                                let y = ty * height / term_height;
                                let pixel = decoded.get_pixel(x as u32, y as u32);
                                let mut r = pixel[0];
                                let mut g = pixel[1];
                                let mut b = pixel[2];

                                if let Some(prev) = &prev_frame {
                                    let idx = (y * width + x) * 3;
                                    if idx + 2 < prev.len() {
                                        r = ((r as u16 * 7 + prev[idx] as u16 * 3) / 10) as u8;
                                        g = ((g as u16 * 7 + prev[idx + 1] as u16 * 3) / 10) as u8;
                                        b = ((b as u16 * 7 + prev[idx + 2] as u16 * 3) / 10) as u8;
                                    }
                                }


                                let should_sample_sobel = (tx % sobel_sample_rate == 0) && (ty % sobel_sample_rate == 0);
                                let sobel_edge = if should_sample_sobel {
                                    sobel_detect_edge(&decoded, x, y, width, height, 30.0)
                                } else {
                                    SobelEdge::None
                                };

                                let ascii_char = match sobel_edge {
                                    SobelEdge::Horizontal => '═',
                                    SobelEdge::Vertical => '║',
                                    SobelEdge::DiagonalUp => '/',
                                    SobelEdge::DiagonalDown => '\\',
                                    SobelEdge::None => pixel_to_ascii(r, g, b),
                                };

                                let r_idx = (r / 16) as usize;
                                let g_idx = (g / 16) as usize;
                                let b_idx = (b / 16) as usize;
                                let color_idx = (r_idx << 8) | (g_idx << 4) | b_idx;
                                if color_idx != last_color_idx {
                                    row_buffer.push_str(&color_lookup[color_idx]);
                                    last_color_idx = color_idx;
                                }
                                row_buffer.push(ascii_char);
                            }

                            row_buffer
                        })
                        .collect();

                    // frame_buffer.clear();
                    // for (i, row) in rows.iter().enumerate() {
                    //     frame_buffer.push_str(row);
                    //     if i < term_height - 1 {
                    //         frame_buffer.push_str("\r\n");
                    //     }
                    // }

                    if let Some(prev) = &prev_rows {
                        for (row_idx, current_row) in rows.iter().enumerate() {
                            if row_idx >= prev.len() || current_row != &prev[row_idx] {
                                queue!(stdout, cursor::MoveTo(0, row_idx as u16))?;
                                write!(stdout, "{}", current_row)?;
                            }
                        }
                    } else {
                        queue!(stdout, cursor::MoveTo(0, 0))?;

                        for (i, row) in rows.iter().enumerate() {
                            write!(stdout, "{}", row)?;

                            if i < term_height - 1 {
                                write!(stdout, "\r\n")?;
                            }
                        }

                    }


                    stdout.flush()?;

                    prev_frame = Some(current_pixels.to_vec());

                    let frame_duration = frame_start.elapsed();
                    should_skip_next_frame = frame_duration.as_millis() > TARGET_FRAME_TIME_MS;

                }
            },
                Ok(_) = quit_rx.changed() => {
                if *quit_rx.borrow() {
                    break;
                }
            }
        }
    }

    stdout.flush()?;

    Ok(())
}

fn pixel_to_ascii(r: u8, g: u8, b: u8) -> char {
    let brightness = ((r as u32 + g as u32 + b as u32) / 3) as u8;
    let index = (brightness as usize * ASCII_CHARS.len()) / 256;

    ASCII_CHARS[index]
}
