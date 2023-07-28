use crossbeam_channel::{unbounded, Receiver, SendError, Sender};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone)]
pub struct Text {
    // transcription
    text: String,
    // number of frames (number of 1x80 mel spectrograms)
    frames: u32,
    // frame index in the overall stream where this segment begins
    cutsec: u64,
}

impl Text {
    pub fn new(cutsec: u64, frames: u32, text: String) -> Self {
        Self {
            cutsec,
            frames,
            text,
        }
    }

    // segment duration in ms
    pub fn duration(&self, fft_size: u32, hop_size: u32, sampling_rate: u32) -> u32 {
        let frame_duration = (fft_size as f32 / sampling_rate as f32 * 1000.0) as u32;
        let total_duration = frame_duration * self.frames;
        total_duration
    }
}

/// thread safe sorted buffer for receiving transcriptions
pub struct TextBuffer {
    text_tx: Option<Sender<Text>>,
    text_rx: Receiver<Text>,
    texts: Arc<Mutex<HashMap<u64, Text>>>,
}

impl TextBuffer {
    pub fn new() -> Self {
        let (text_tx, text_rx): (Sender<Text>, Receiver<Text>) = unbounded();
        let texts = Arc::new(Mutex::new(HashMap::new()));

        Self {
            text_tx: Some(text_tx),
            text_rx,
            texts,
        }
    }

    pub fn start(&mut self) -> thread::JoinHandle<()> {
        let text_rx = self.text_rx.clone();
        let texts_clone = Arc::clone(&self.texts);
        let handle = thread::spawn(move || {
            while let Ok(txt) = text_rx.recv() {
                texts_clone.lock().unwrap().insert(txt.cutsec, txt);
            }
        });

        return handle;
    }

    pub fn add(&self, txt: Text) -> Result<(), SendError<Text>> {
        self.text_tx.as_ref().expect("not closed").send(txt)
    }

    // to signal we are done streaming into the pipeline
    pub fn close_ingress(&mut self) {
        self.text_tx = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_buffer() {
        let mut text_buffer = TextBuffer::new();
        let handle = text_buffer.start();

        let text1 = Text {
            text: "Message 1".to_string(),
            frames: 10,
            cutsec: 1,
        };
        let text2 = Text {
            text: "Message 2".to_string(),
            frames: 20,
            cutsec: 2,
        };
        let text3 = Text {
            text: "Message 3".to_string(),
            frames: 30,
            cutsec: 3,
        };

        text_buffer.add(text1.clone());
        text_buffer.add(text3.clone());
        text_buffer.add(text2.clone());

        text_buffer.close_ingress();

        handle.join().unwrap();

        let texts = text_buffer.texts.lock().unwrap();
        assert_eq!(texts.get(&1).unwrap().text, text1.text);
        assert_eq!(texts.get(&2).unwrap().text, text2.text);
        assert_eq!(texts.get(&3).unwrap().text, text3.text);
    }
}
