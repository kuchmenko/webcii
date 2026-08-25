# webcii

Real-time color ASCII webcam renderer for the terminal.

## Demo

https://github.com/user-attachments/assets/0b7924b0-cf55-426f-8c18-b6a13d6a36a7

## Features

- Webcam frames rendered as brightness-mapped ASCII with 4-bit RGB colors
- Sobel edge characters on smaller terminal grids
- Parallel row rendering with Rayon
- Frame skipping when rendering exceeds the 16 ms target
- Cached ANSI color codes
- Output that adapts to terminal resizing

## Usage

```bash
cargo run --release
```

Press `q` or `Ctrl+C` to exit.
