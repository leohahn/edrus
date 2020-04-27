use crate::text_buffer::{CharMetric, HorizontalOffset, Line, LineMetric, SimplePieceTable, TextBuffer};
use std::path::Path;

#[derive(Debug)]
pub struct Cursor {
    pub pos: usize,
    pub col: HorizontalOffset,
    pub line: Line,
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
            cursor: Cursor {
                pos: 0,
                col: HorizontalOffset(1),
                line: Line(1),
            },
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

    pub fn find_after(&self, character: char) -> Option<usize> {
        self.piece_table.find_after(self.cursor.pos, character)
    }

    pub fn move_to(&mut self, offset: usize) -> Option<()> {
        self.column(offset).and_then(|col| {
            self.piece_table.line_for_offset(offset).map(|line| {
                self.cursor.pos = offset;
                self.cursor.col = col;
                self.cursor.line = line;
            })
        })
    }

    pub fn move_left_unbounded(&mut self) -> Option<usize> {
        self.piece_table.prev(self.cursor.pos).and_then(|offset| {
            println!("===== move_left =====");
            println!("prev char: {:?}", self.current_char());
            self.cursor.pos = offset;
            self.cursor.col = self.column(self.cursor.pos).expect("should not fail");
            self.cursor.line = self
                .piece_table
                .line_for_offset(self.cursor.pos)
                .expect("should not fail");
            println!(
                "next char: {:?}, offset={} line={}",
                self.current_char(),
                self.cursor.pos,
                self.cursor.line.0
            );
            println!("=====================");
            Some(offset)
        })
    }

    pub fn move_left(&mut self) -> Option<usize> {
        self.piece_table.prev(self.cursor.pos).and_then(|offset| {
            if self.piece_table.char_at(offset).unwrap() == '\n' {
                None
            } else {
                println!("===== move_left =====");
                println!("prev char: {:?}", self.current_char());
                self.cursor.pos = offset;
                self.cursor.col = self.column(self.cursor.pos).expect("should not fail");
                self.cursor.line = self
                    .piece_table
                    .line_for_offset(self.cursor.pos)
                    .expect("should not fail");
                println!(
                    "next char: {:?}, offset={} line={}",
                    self.current_char(),
                    self.cursor.pos,
                    self.cursor.line.0
                );
                println!("=====================");
                Some(offset)
            }
        })
    }

    pub fn move_right_unbounded(&mut self) -> Option<usize> {
        self.piece_table
            .next::<CharMetric>(self.cursor.pos)
            .and_then(|offset| {
                println!("===== move_left =====");
                println!("prev char: {:?}", self.current_char());
                self.cursor.pos = offset;
                self.cursor.col = self.column(self.cursor.pos).expect("should not fail");
                self.cursor.line = self
                    .piece_table
                    .line_for_offset(self.cursor.pos)
                    .expect("should not fail");
                println!("next char: {:?}, offset={}", self.current_char(), self.cursor.pos);
                println!("=====================");
                Some(offset)
            })
    }

    pub fn move_right(&mut self) -> Option<usize> {
        if self.current_char() == '\n' {
            None
        } else {
            self.piece_table
                .next::<CharMetric>(self.cursor.pos)
                .and_then(|offset| {
                    if self.piece_table.char_at(offset).unwrap() == '\n' {
                        None
                    } else {
                        println!("===== move_right =====");
                        println!("prev char: {:?}", self.current_char());
                        self.cursor.pos = offset;
                        self.cursor.col = self.column(self.cursor.pos).expect("should not fail");
                        self.cursor.line = self
                            .piece_table
                            .line_for_offset(self.cursor.pos)
                            .expect("should not fail");
                        println!("next char: {:?}, offset={}", self.current_char(), self.cursor.pos);
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
            self.cursor.col = self.column(self.cursor.pos).expect("should not fail");
            self.cursor.line = self
                .piece_table
                .line_for_offset(self.cursor.pos)
                .expect("should not fail");
            println!(
                "next char: offset={} col={} line={} char={:?}",
                self.cursor.pos,
                self.cursor.col.0,
                self.cursor.line.0,
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
            self.cursor.col = self.column(self.cursor.pos).expect("should not fail");
            self.cursor.line = self
                .piece_table
                .line_for_offset(self.cursor.pos)
                .expect("should not fail");
            println!(
                "next char: {:?}, col={} line={} offset={}",
                self.current_char(),
                self.cursor.col.0,
                self.cursor.line.0,
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
        self.piece_table
            .remove(self.cursor.pos..self.cursor.pos + 1)
            .expect("remove failed");
        self.contents = self.piece_table.contents();
    }

    pub fn column(&self, offset: usize) -> Option<HorizontalOffset> {
        self.piece_table.column_for_offset(offset)
    }
}
