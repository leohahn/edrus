use crate::text_buffer::{HorizontalOffset, SimplePieceTable, TextBuffer};
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

    pub fn find_before(&self, character: char) -> Option<usize> {
        self.piece_table.find_before(self.cursor.pos, character)
    }

    pub fn move_to(&mut self, offset: usize) -> Option<()> {
        self.piece_table
            .column_for_offset(offset)
            .map(|HorizontalOffset(col)| {
                self.cursor.pos = offset;
                self.cursor.col = col;
            })
    }

    pub fn move_left(&mut self) -> Option<usize> {
        if self.current_char() == '\n' {
            None
        } else {
            self.piece_table.prev(self.cursor.pos).and_then(|offset| {
                if self.piece_table.char_at(offset).unwrap() == '\n' {
                    None
                } else {
                    println!("===== move_left =====");
                    println!("prev char: {}", self.current_char());
                    self.cursor.pos = offset;
                    self.cursor.col = self
                        .piece_table
                        .column_for_offset(self.cursor.pos)
                        .expect("should not fail")
                        .0;
                    println!("next char: {}, offset={}", self.current_char(), self.cursor.pos);
                    println!("=====================");
                    Some(offset)
                }
            })
        }
    }

    pub fn move_right(&mut self) -> Option<usize> {
        if self.current_char() == '\n' {
            None
        } else {
            self.piece_table.next(self.cursor.pos).and_then(|offset| {
                if self.piece_table.char_at(offset).unwrap() == '\n' {
                    None
                } else {
                    println!("===== move_right =====");
                    println!("prev char: {}", self.current_char());
                    self.cursor.pos = offset;
                    self.cursor.col = self
                        .piece_table
                        .column_for_offset(self.cursor.pos)
                        .expect("should not fail")
                        .0;
                    println!("next char: {}, offset={}", self.current_char(), self.cursor.pos);
                    println!("=====================");
                    Some(offset)
                }
            })
        }
    }

    pub fn move_down(&mut self) -> Option<usize> {
        self.piece_table.next_line(self.cursor.pos).map(|offset| {
            println!("===== move_down =====");
            println!(
                "prev char: offset={} char={:?}",
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
                "next char: offset={} char={:?}",
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
            println!("prev char: {:?}", self.current_char());
            self.cursor.pos = offset;
            self.cursor.col = self
                .piece_table
                .column_for_offset(self.cursor.pos)
                .expect("should not fail")
                .0;
            println!(
                "next char: {:?}, col={} offset={}",
                self.current_char(),
                self.cursor.col,
                self.cursor.pos
            );
            println!("=====================");
            offset
        })
    }

    pub fn insert_before(&mut self, text: &str) {
        self.piece_table
            .insert(self.cursor.pos, text)
            .expect("insertion failed");
        self.contents = self.piece_table.contents();
    }

    pub fn remove_current_char(&mut self) {
        if self.current_char() == '\n' {
            println!("cannot remove newline");
            return;
        }

        self.piece_table
            .remove(self.cursor.pos..self.cursor.pos + 1)
            .expect("remove failed");
        self.contents = self.piece_table.contents();
        // println!(
        //     "new contents after removing in pos={}: {:?}",
        //     self.cursor.pos, self.contents
        // );
    }

    pub fn column(&self, offset: usize) -> Option<HorizontalOffset> {
        self.piece_table.column_for_offset(offset)
    }
}
