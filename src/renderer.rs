use std::{
    io::{self, Write},
    time::Instant,
};

use crossterm::{cursor, queue, terminal};
use dashmap::DashMap;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::capture::Frame;

const ASCII_CHARS: [char; 70] = [
    '$', '@', 'B', '%', '8', '&', 'W', 'M', '#', '*', 'o', 'a', 'h', 'k', 'b', 'd', 'p', 'q', 'w',
    'm', 'Z', 'O', '0', 'Q', 'L', 'C', 'J', 'U', 'Y', 'X', 'z', 'c', 'v', 'u', 'n', 'x', 'r', 'j',
    'f', 't', '/', '\\', '|', '(', ')', '1', '{', '}', '[', ']', '?', '-', '_', '+', '~', '<', '>',
    'i', '!', 'l', 'I', ';', ':', ',', '"', '^', '`', '\'', '.', ' ',
];
const HIGH_RESOLUTION_THRESHOLD: usize = 200_000;
const MEDIUM_RESOLUTION_THRESHOLD: usize = 100_000;

const SOBEL_SAMPLE_RATE_HIGH_RES: usize = 20; // sample every 20th pixel
const SOBEL_SAMPLE_RATE_MEDIUM_RES: usize = 10; // sample every 10th pixel
const SOBEL_SAMPLE_RATE_LOW_RES: usize = 1; // sample every pixel
const SOBEL_THRESHOLD: f32 = 30.0; // gradient magnitude threshold

const TARGET_FRAME_TIME_MS: u128 = 16; // ~60 FPS
const INITIAL_BUFFER_CAPACITY: usize = 2_000_000; // 2MB for ANSI codes
const ESTIMATED_BYTES_PER_CELL: usize = 25; // ANSI color codes + char

enum SobelEdge {
    None,
    Horizontal,
    Vertical,
    DiagonalUp,
    DiagonalDown,
}

struct SobelConfig {
    enabled: bool,
    sample_rate: usize,
}

pub struct Renderer {
    bits_per_channel: u8,
    quantization_divisor: u8,
    red_shift: u32,
    green_shift: u32,

    color_cache: DashMap<usize, String>,
    term_width: usize,
    term_height: usize,

    prev_frame: Option<Frame>,
    blended_pixels: Vec<u8>,
    should_skip_next_frame: bool,

    frame_buffer: String,
    final_buffer: String,

    stdout: std::io::Stdout,
}

impl Renderer {
    pub fn new(bits_per_channel: u8) -> io::Result<Self> {
        if bits_per_channel == 0 || bits_per_channel > 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("bits_per_channel must be 1-8, got {}", bits_per_channel),
            ));
        }

        let quantization_divisor = 1 << (8 - bits_per_channel);
        let red_shift = (bits_per_channel * 2) as u32;
        let green_shift = bits_per_channel as u32;

        let (cols, rows) = terminal::size()?;
        let term_width = cols as usize;
        let term_height = rows as usize;

        Ok(Renderer {
            bits_per_channel,
            quantization_divisor,
            red_shift,
            green_shift,

            color_cache: DashMap::with_capacity(150_000),
            term_width,
            term_height,

            prev_frame: None,
            blended_pixels: Vec::with_capacity(term_width * term_height),
            should_skip_next_frame: false,

            frame_buffer: String::with_capacity(INITIAL_BUFFER_CAPACITY),
            final_buffer: String::with_capacity(INITIAL_BUFFER_CAPACITY),

            stdout: io::stdout(),
        })
    }

    pub fn render(&mut self, frame: &Frame) -> io::Result<()> {
        self.update_terminal_size()?;
        let sobel_config = self.calculate_sobel_config();
        let estimated_size = self.term_width * self.term_height * ESTIMATED_BYTES_PER_CELL;

        if self.frame_buffer.capacity() < estimated_size {
            self.frame_buffer
                .reserve(estimated_size - self.frame_buffer.capacity());
        }
        let frame_start = Instant::now();

        if self.should_skip_next_frame {
            self.prev_frame = Some(frame.clone());
            self.should_skip_next_frame = false;

            return Ok(());
        }

        self.prepare_frame_data(frame);

        let rows: Vec<String> = (0..self.term_height)
            .into_par_iter()
            .map(|term_y| self.render_row(term_y, frame, &sobel_config))
            .collect();

        self.write_to_terminal(&rows)?;
        self.update_frame_state(frame_start, frame)?;

        Ok(())
    }

    fn render_row(&self, term_y: usize, frame: &Frame, sobel_config: &SobelConfig) -> String {
        let mut row_buffer = String::with_capacity(self.term_width * 20);
        let mut last_color_idx = usize::MAX;

        for term_x in 0..self.term_width {
            let (ascii_char, color_idx, (r, g, b)) =
                self.render_pixel(term_x, term_y, frame, &sobel_config);

            if color_idx != last_color_idx {
                row_buffer.push_str(&self.get_color_code(color_idx, r, g, b));
                last_color_idx = color_idx;
            }
            row_buffer.push(ascii_char);
        }

        row_buffer
    }

    fn render_pixel(
        &self,
        term_x: usize,
        term_y: usize,
        frame: &Frame,
        sobel_config: &SobelConfig,
    ) -> (char, usize, (u8, u8, u8)) {
        let (x, y) = self.calculate_sample_position(term_x, term_y, frame.width, frame.height);

        let (r, g, b) = Self::get_pixel_rgb_from_slice(&self.blended_pixels, x, y, frame.width);

        let should_sample_sobel = sobel_config.enabled
            && (term_x % sobel_config.sample_rate == 0)
            && (term_y % sobel_config.sample_rate == 0);

        let sobel_edge = if should_sample_sobel {
            self.sobel_detect_edge(
                &frame.pixels,
                x,
                y,
                frame.width,
                frame.height,
                SOBEL_THRESHOLD,
            )
        } else {
            SobelEdge::None
        };

        let ascii_char = match sobel_edge {
            SobelEdge::Horizontal => '═',
            SobelEdge::Vertical => '║',
            SobelEdge::DiagonalUp => '/',
            SobelEdge::DiagonalDown => '\\',
            SobelEdge::None => Self::pixel_to_ascii(r, g, b),
        };

        let color_idx = self.calculate_color_index(r, g, b);

        (ascii_char, color_idx, (r, g, b))
    }

    fn write_to_terminal(&mut self, rows: &[String]) -> io::Result<()> {
        queue!(self.stdout, cursor::MoveTo(0, 0))?;

        self.final_buffer.clear();
        for (index, row) in rows.iter().enumerate() {
            self.final_buffer.push_str(row);
            if index < self.term_height - 1 {
                self.final_buffer.push_str("\r\n");
            }
        }
        write!(self.stdout, "{}", self.final_buffer)?;

        self.stdout.flush()?;
        Ok(())
    }

    fn update_frame_state(&mut self, frame_start: Instant, frame: &Frame) -> io::Result<()> {
        let frame_duration = frame_start.elapsed();

        self.prev_frame = Some(frame.clone());
        self.should_skip_next_frame = frame_duration.as_millis() > TARGET_FRAME_TIME_MS;
        Ok(())
    }

    fn update_terminal_size(&mut self) -> io::Result<()> {
        let (cols, rows) = terminal::size()?;
        let new_width = cols as usize;
        let new_height = rows as usize;

        if new_width != self.term_width || new_height != self.term_height {
            self.term_height = new_height;
            self.term_width = new_width;

            let new_frame_buffer_capacity = new_width * new_height;
            if self.frame_buffer.capacity() < new_frame_buffer_capacity {
                self.frame_buffer.reserve(new_frame_buffer_capacity);
            }
        }

        Ok(())
    }

    fn calculate_sobel_config(&self) -> SobelConfig {
        let total_pixels = self.term_width * self.term_height;
        let sobel_sample_rate = if total_pixels > HIGH_RESOLUTION_THRESHOLD {
            SOBEL_SAMPLE_RATE_HIGH_RES
        } else if total_pixels > MEDIUM_RESOLUTION_THRESHOLD {
            SOBEL_SAMPLE_RATE_MEDIUM_RES
        } else {
            SOBEL_SAMPLE_RATE_LOW_RES
        };

        SobelConfig {
            enabled: total_pixels < MEDIUM_RESOLUTION_THRESHOLD,
            sample_rate: sobel_sample_rate,
        }
    }

    fn calculate_sample_position(
        &self,
        term_x: usize,
        term_y: usize,
        frame_width: usize,
        frame_height: usize,
    ) -> (usize, usize) {
        let x =
            ((term_x * frame_width * 2 + frame_width) / (self.term_width * 2)).min(frame_width - 1);
        let y = ((term_y * frame_height * 2 + frame_height) / (self.term_height * 2))
            .min(frame_height - 1);

        (x, y)
    }

    fn get_color_code(&self, color_idx: usize, r: u8, g: u8, b: u8) -> String {
        self.color_cache
            .entry(color_idx)
            .or_insert_with(|| {
                let r_q = (r / self.quantization_divisor) * self.quantization_divisor;
                let g_q = (g / self.quantization_divisor) * self.quantization_divisor;
                let b_q = (b / self.quantization_divisor) * self.quantization_divisor;
                format!("\x1b[38;2;{};{};{}m", r_q, g_q, b_q)
            })
            .clone()
    }

    fn get_pixel_rgb_from_slice(pixels: &[u8], x: usize, y: usize, width: usize) -> (u8, u8, u8) {
        let idx = (y * width + x) * 3;
        (pixels[idx], pixels[idx + 1], pixels[idx + 2])
    }

    fn sobel_detect_edge(
        &self,
        pixels: &[u8],
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        threshold: f32,
    ) -> SobelEdge {
        if x == 0 || y == 0 || x >= width - 1 || y >= height - 1 {
            return SobelEdge::None;
        }

        let get_brightness = |px: usize, py: usize| -> i32 {
            let (r, g, b) = Self::get_pixel_rgb_from_slice(pixels, px, py, width);

            Self::calculate_brightness(r, g, b) as i32
        };

        // 3x3 neighborhood
        let nw = get_brightness(x - 1, y - 1);
        let n = get_brightness(x, y - 1);
        let ne = get_brightness(x + 1, y - 1);
        let w = get_brightness(x - 1, y);
        let e = get_brightness(x + 1, y);
        let sw = get_brightness(x - 1, y + 1);
        let s = get_brightness(x, y + 1);
        let se = get_brightness(x + 1, y + 1);

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

    fn calculate_color_index(&self, r: u8, g: u8, b: u8) -> usize {
        let r_idx = (r / self.quantization_divisor) as usize;
        let g_idx = (g / self.quantization_divisor) as usize;
        let b_idx = (b / self.quantization_divisor) as usize;

        (r_idx << self.red_shift) | (g_idx << self.green_shift) | b_idx
    }

    fn generate_color_lookup(bits_per_channel: u8) -> Vec<String> {
        let colors_per_channel = (1 << bits_per_channel) as u32;
        let total_colors = colors_per_channel.pow(3); // TODO: why pow 3? what is 3;
        let channel_mask = colors_per_channel - 1;

        let multiplier = if colors_per_channel == 256 {
            1
        } else {
            255 / channel_mask
        };

        (0..total_colors)
            .map(|i| {
                let r_shift = bits_per_channel * 2;

                let r = ((i >> r_shift) & channel_mask) * multiplier;
                let g = ((i >> bits_per_channel) & channel_mask) * multiplier;
                let b = (i & channel_mask) * multiplier;
                format!("\x1b[38;2;{};{};{}m", r, g, b)
            })
            .collect()
    }

    fn pixel_to_ascii(r: u8, g: u8, b: u8) -> char {
        let brightness = Self::calculate_brightness(r, g, b);
        let index = (brightness as usize * ASCII_CHARS.len()) / 256;

        ASCII_CHARS[index]
    }

    fn prepare_frame_data(&mut self, frame: &Frame) {
        self.frame_buffer.clear();

        if self.blended_pixels.len() < frame.pixels.len() {
            self.blended_pixels.resize(frame.pixels.len(), 0);
        }

        self.blended_pixels.copy_from_slice(&frame.pixels);
    }

    fn calculate_brightness(r: u8, g: u8, b: u8) -> u8 {
        ((r as u32 + g as u32 + b as u32) / 3) as u8
    }
}
