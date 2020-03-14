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

    pub fn current_char(&self) -> char {
        self.contents.chars().nth(self.cursor.pos).unwrap()
    }

    pub fn move_left(&mut self) {
        println!("===== move_left =====");
        println!("prev char: {}", self.current_char());
        self.contents.chars().nth(self.cursor.pos - 1).map(|_| {
            self.cursor.pos -= 1;
            self.cursor.col -= 1;
        });
        println!("next char: {}", self.current_char());
        println!("=====================");
    }

    pub fn move_right(&mut self) {
        println!("===== move_right =====");
        println!("prev char: {}", self.current_char());
        self.contents.chars().nth(self.cursor.pos + 1).map(|_| {
            self.cursor.pos += 1;
            self.cursor.col += 1;
        });
        println!("next char: {}", self.current_char());
        println!("=====================");
    }

    pub fn move_down(&mut self) {
        // FIXME: this is slow.
        println!("===== move_down =====");
        println!("prev char: {}", self.current_char());
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
        println!("next char: {}", self.current_char());
        println!("=====================");
    }

    pub fn move_up(&mut self) {
        // FIXME: this is slow.
        println!("===== move_up =====");
        println!("prev char: {}", self.current_char());

        let (before_cursor, _) = self.contents.split_at(self.cursor.pos);
        if let Some(init_previous_line) = before_cursor
            .rfind('\n')
            .map(|index| self.contents.split_at(index))
            .and_then(|(s, _)| s.rfind('\n'))
            .map(|i| i + 1)
        {
            self.cursor.pos = init_previous_line + self.cursor.col;
        }

        println!("next char: {}", self.current_char());
        println!("=====================");
    }
}
