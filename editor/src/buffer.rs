use crate::text_buffer::{HorizonalOffset, SimplePieceTable, TextBuffer};
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
        self.contents[self.cursor.pos..]
            .chars()
            .next()
            .expect("should not fail")
    }

    pub fn move_left(&mut self) -> Option<usize> {
        self.piece_table.prev(self.cursor.pos).map(|offset| {
            println!("===== move_left =====");
            println!("prev char: {}", self.current_char());
            self.cursor.pos = offset;
            self.cursor.col = self
                .piece_table
                .column_for_offset(self.cursor.pos)
                .expect("should not fail")
                .0;
            println!("next char: {}", self.current_char());
            println!("=====================");
            offset
        })
    }

    pub fn move_right(&mut self) -> Option<usize> {
        self.piece_table.next(self.cursor.pos).map(|offset| {
            println!("===== move_right =====");
            println!("prev char: {}", self.current_char());
            self.cursor.pos = offset;
            self.cursor.col = self
                .piece_table
                .column_for_offset(self.cursor.pos)
                .expect("should not fail")
                .0;
            println!("next char: {}", self.current_char());
            println!("=====================");
            offset
        })
    }

    pub fn move_down(&mut self) -> Option<usize> {
        self.piece_table.next_line(self.cursor.pos).map(|offset| {
            println!("===== move_down =====");
            println!(
                "prev char: offset={} char={}",
                self.cursor.pos,
                self.current_char()
            );
            self.cursor.pos = offset;
            self.cursor.col = self
                .piece_table
                .column_for_offset(self.cursor.pos)
                .expect("should not fail")
                .0;
            println!(
                "next char: offset={} char={}",
                self.cursor.pos,
                self.current_char()
            );
            println!("=====================");
            offset
        })
    }

    pub fn move_up(&mut self) -> Option<usize> {
        self.piece_table.prev_line(self.cursor.pos).map(|offset| {
            println!("===== move_up =====");
            println!("prev char: {}", self.current_char());
            self.cursor.pos = offset;
            self.cursor.col = self
                .piece_table
                .column_for_offset(self.cursor.pos)
                .expect("should not fail")
                .0;
            println!(
                "next char: {}, col={}",
                self.current_char(),
                self.cursor.col
            );
            println!("=====================");
            offset
        })
    }

    pub fn column(&self, offset: usize) -> Option<HorizonalOffset> {
        self.piece_table.column_for_offset(offset)
    }
}
