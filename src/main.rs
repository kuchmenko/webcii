mod capture;
mod renderer;

use capture::Capturer;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyModifiers},
    execute, terminal,
};
use renderer::Renderer;
use tokio::sync::watch;

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = execute!(std::io::stdout(), cursor::Show);
        let _ = terminal::disable_raw_mode();
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

    let mut render = Renderer::new(4)?;
    let mut capturer = Capturer::start(0, 2)?;
    let mut frame_rx = capturer.subscribe();

    loop {
        tokio::select! {
            Ok(frame) = frame_rx.recv() => {
                render.render(&frame)?;
            }
            Ok(_) = quit_rx.changed() => break,
        }
    }

    Ok(())
}
