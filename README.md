# webcii

Real-time ASCII webcam renderer with edge detection

<video autoplay loop muted playsinline>
  <source src="https://github.com/user-attachments/assets/f8641be7-035a-441f-8267-c4e78c205de5" type="video/mp4">
</video>

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
