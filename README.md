# webcii

Real-time ASCII webcam renderer with edge detection

## Demo
https://github.com/user-attachments/assets/0b7924b0-cf55-426f-8c18-b6a13d6a36a7

## Features

- Camera → ASCII conversion (configurable 4-8 bit color depth)
- Sobel edge detection with adaptive sampling
- Parallel row rendering (rayon)
- Adaptive frame skipping (60 FPS target)
- Color quantization cache (150k entries)
- Dynamic terminal resize support

## Usage

```bash
cargo run --release
```
