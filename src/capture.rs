use std::sync::Arc;

use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
};
use tokio::{
    sync::{
        broadcast,
        watch::{self, Sender},
    },
    task,
};

pub struct FrameData {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

pub type Frame = Arc<FrameData>;

pub struct Capturer {
    frame_tx: broadcast::Sender<Frame>,
    shutdown_tx: watch::Sender<bool>,
    task_handle: task::JoinHandle<()>,
}

impl Capturer {
    pub fn start(
        camera_index: u32,
        buffer_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (frame_tx, _) = broadcast::channel::<Frame>(buffer_size);
        let (shutdown_tx, shutdown_rx) = watch::channel::<bool>(false);
        let handler_frame_tx = frame_tx.clone();

        let task_handle = task::spawn_blocking(move || {
            Self::capture_loop(camera_index, handler_frame_tx, shutdown_rx);
        });

        Ok(Capturer {
            frame_tx,
            shutdown_tx,
            task_handle,
        })
    }

    pub fn subscribe(&mut self) -> broadcast::Receiver<Frame> {
        self.frame_tx.subscribe()
    }

    fn capture_loop(
        camera_index: u32,
        frame_tx: broadcast::Sender<Frame>,
        shutdown_rx: watch::Receiver<bool>,
    ) {
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        let mut camera = match Camera::new(CameraIndex::Index(camera_index), requested) {
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
            if *shutdown_rx.borrow() {
                break;
            }

            let frame_data = camera.frame();

            if let Ok(frame) = frame_data {
                match frame.decode_image::<RgbFormat>() {
                    Ok(decoded) => {
                        let width = frame.resolution().width() as usize;
                        let height = frame.resolution().height() as usize;
                        let pixels = decoded.as_raw().to_vec();
                        let frame_data = FrameData {
                            width,
                            height,
                            pixels,
                        };
                        if frame_tx.send(Arc::new(frame_data)).is_err() {
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
    }
}
