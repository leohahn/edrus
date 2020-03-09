use crate::text_buffer::{SimplePieceTable, TextBuffer};
use std::path::Path;

pub struct Cursor {
    pos: usize,
    col: usize,
}

impl Cursor {
    pub fn position(&self) -> usize {
        self.pos
    }
}

pub struct Buffer {
    piece_table: SimplePieceTable,
    contents: String,
    pub cursor: Cursor,
}

impl Buffer {
    pub fn new(path: impl AsRef<Path>) -> Result<Buffer, std::io::Error> {
        let raw_contents = std::fs::read_to_string(path)?;
        let piece_table = SimplePieceTable::new(raw_contents);
        let contents = piece_table.contents();
        Ok(Buffer {
            piece_table: piece_table,
            contents: contents,
            cursor: Cursor { pos: 0, col: 0 },
        })
    }

    pub fn contents(&self) -> &str {
        self.contents.as_ref()
    }

    pub fn move_left(&mut self) {
        self.cursor.pos -= 1;
    }

    pub fn move_right(&mut self) {
        self.cursor.pos += 1;
    }

    pub fn move_down(&mut self) {
        // FIXME: this is slow.
        if let Some(pos) = self
            .contents
            .chars()
            .enumerate()
            .skip(self.cursor.pos)
            .skip_while(|(_, c)| *c != '\n')
            .skip(self.cursor.col)
            .nth(0)
        {
            self.cursor.pos = pos.0;
        }
    }

    pub fn move_up(&mut self) {
        // FIXME: this is slow.
        self.cursor.pos += 1;
    }
}
