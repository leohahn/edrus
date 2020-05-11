use crate::text_buffer::{CharMetric, HorizontalOffset, Line, SimplePieceTable, TextBuffer};
use std::path::Path;

#[derive(Debug)]
pub struct Normal {
    piece_table: SimplePieceTable,
    contents: String,
    is_on_eof: bool,
    pos: usize,
    pub col: HorizontalOffset,
    pub line: Line,
}

#[derive(Debug)]
pub struct Insert {
    piece_table: SimplePieceTable,
    contents: String,
    pub pos: usize,
    pub col: HorizontalOffset,
    pub line: Line,
}

impl Insert {
    fn from_normal(normal: Normal) -> Self {
        Self {
            piece_table: normal.piece_table,
            contents: normal.contents,
            pos: normal.pos,
            col: normal.col,
            line: normal.line,
        }
    }

    pub fn current_char(&self) -> char {
        self.contents[self.pos..].chars().next().expect("should not fail")
    }

    pub fn remove_previous_char(&mut self) -> Option<()> {
        self.move_left(false).map(|_| {
            self.piece_table
                .remove(self.pos..self.pos + 1)
                .expect("remove failed");
            self.contents = self.piece_table.contents();
        })
    }

    pub fn insert(&mut self, text: &str) {
        self.piece_table.insert(self.pos, text).expect("insertion failed");
        self.contents = self.piece_table.contents();
    }

    pub fn exit(mut self) -> Normal {
        let _ = self.move_left(true);
        Normal::from_insert(self)
    }

    pub fn move_right(&mut self) -> Option<()> {
        self.piece_table.next::<CharMetric>(self.pos).and_then(|offset| {
            if self.piece_table.char_at(offset) == Some('\n') {
                return None;
            }

            self.pos = offset;
            self.col = self
                .piece_table
                .column_for_offset(offset)
                .expect("should not fail");
            self.line = self.piece_table.line_for_offset(offset).expect("should not fail");
            Some(())
        })
    }

    pub fn move_left(&mut self, stop_at_newline: bool) -> Option<()> {
        if self.pos == self.contents.len() {
            // FIXME(lhahn): this will break for non ascii.
            self.pos -= 1;
            self.col = self
                .piece_table
                .column_for_offset(self.pos)
                .expect("should not fail");
            self.line = self
                .piece_table
                .line_for_offset(self.pos)
                .expect("should not fail");
            return Some(());
        }

        self.piece_table.prev(self.pos).and_then(|offset| {
            if stop_at_newline && self.piece_table.char_at(offset) == Some('\n') {
                return None;
            }

            self.pos = offset;
            self.col = self
                .piece_table
                .column_for_offset(offset)
                .expect("should not fail");
            self.line = self.piece_table.line_for_offset(offset).expect("should not fail");
            Some(())
        })
    }

    pub fn contents(&self) -> &str {
        self.contents.as_ref()
    }
}

impl Normal {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let raw_contents = std::fs::read_to_string(path)?;
        let piece_table = SimplePieceTable::new(raw_contents);
        let contents = piece_table.contents();
        let is_on_eof = contents.is_empty();
        Ok(Self {
            piece_table: piece_table,
            contents: contents,
            is_on_eof: is_on_eof,
            pos: 0,
            col: HorizontalOffset(1),
            line: Line(1),
        })
    }

    pub fn from_insert(insert: Insert) -> Self {
        let is_on_eof = insert.contents.is_empty();
        Self {
            piece_table: insert.piece_table,
            contents: insert.contents,
            is_on_eof: is_on_eof,
            pos: insert.pos,
            col: insert.col,
            line: insert.line,
        }
    }

    pub fn contents(&self) -> &str {
        self.contents.as_ref()
    }

    pub fn current_char(&self) -> char {
        self.contents[self.pos..].chars().next().expect("should not fail")
    }

    pub fn find_before(&self, character: char) -> Option<usize> {
        self.piece_table.find_before(self.pos, character)
    }

    pub fn find_after(&self, character: char) -> Option<usize> {
        self.piece_table.find_after(self.pos, character)
    }

    fn set_pos(&mut self, offset: usize) {
        self.pos = offset;
        self.is_on_eof = offset == self.contents.len();
        self.piece_table.line_for_offset(offset).map(|line| {
            self.line = line;
        });
        self.piece_table.column_for_offset(offset).map(|col| {
            self.col = col;
        });
    }

    pub fn move_to(&mut self, offset: usize) -> Option<()> {
        self.column(offset).and_then(|col| {
            self.piece_table.line_for_offset(offset).map(|line| {
                self.set_pos(offset);
                self.col = col;
                self.line = line;
            })
        })
    }

    pub fn move_left_unbounded(&mut self) -> Option<usize> {
        self.piece_table.prev(self.pos).and_then(|offset| {
            println!("===== move_left =====");
            println!("prev char: {:?}", self.current_char());
            self.set_pos(offset);
            println!(
                "next char: {:?}, offset={} line={}",
                self.current_char(),
                self.pos,
                self.line.0
            );
            println!("=====================");
            Some(offset)
        })
    }

    pub fn move_left(&mut self) -> Option<usize> {
        self.piece_table.prev(self.pos).and_then(|offset| {
            if self.piece_table.char_at(offset).unwrap() == '\n' {
                None
            } else {
                println!("===== move_left =====");
                println!("prev char: {:?}", self.current_char());
                self.set_pos(offset);
                println!(
                    "next char: {:?}, offset={} line={}",
                    self.current_char(),
                    self.pos,
                    self.line.0
                );
                println!("=====================");
                Some(offset)
            }
        })
    }

    pub fn enter_insert(self) -> Insert {
        Insert::from_normal(self)
    }

    pub fn enter_append(mut self) -> Insert {
        if let Some(offset) = self.piece_table.next::<CharMetric>(self.pos) {
            self.set_pos(offset);
            Insert::from_normal(self)
        } else {
            // we are at the end of the file.
            assert_eq!(self.pos, self.contents.len() - 1);
            self.pos += 1;
            // FIXME: this is not going to work for characters that are not ascii.
            self.col = HorizontalOffset(self.col.0 + 1);
            Insert::from_normal(self)
        }
    }

    pub fn move_right_unbounded(&mut self) -> Option<usize> {
        self.piece_table.next::<CharMetric>(self.pos).and_then(|offset| {
            println!("===== move_left =====");
            println!("prev char: {:?}", self.current_char());
            self.set_pos(offset);
            self.col = self.column(self.pos).expect("should not fail");
            self.line = self
                .piece_table
                .line_for_offset(self.pos)
                .expect("should not fail");
            println!("next char: {:?}, offset={}", self.current_char(), self.pos);
            println!("=====================");
            Some(offset)
        })
    }

    pub fn move_right(&mut self) -> Option<usize> {
        if self.current_char() == '\n' {
            None
        } else {
            self.piece_table.next::<CharMetric>(self.pos).and_then(|offset| {
                if self.piece_table.char_at(offset).unwrap() == '\n' {
                    None
                } else {
                    println!("===== move_right =====");
                    println!("prev char: {:?}", self.current_char());
                    self.set_pos(offset);
                    self.col = self.column(self.pos).expect("should not fail");
                    self.line = self
                        .piece_table
                        .line_for_offset(self.pos)
                        .expect("should not fail");
                    println!("next char: {:?}, offset={}", self.current_char(), self.pos);
                    println!("=====================");
                    Some(offset)
                }
            })
        }
    }

    pub fn move_down(&mut self) -> Option<usize> {
        self.piece_table.next_line(self.pos).map(|offset| {
            println!("done");
            println!("===== move_down =====");
            println!("prev char: offset={} char={:?}", self.pos, self.current_char());
            self.set_pos(offset);
            self.col = self.column(self.pos).expect("should not fail");
            self.line = self
                .piece_table
                .line_for_offset(self.pos)
                .expect("should not fail");
            println!(
                "next char: offset={} col={} line={} char={:?}",
                self.pos,
                self.col.0,
                self.line.0,
                self.current_char()
            );
            println!("=====================");
            offset
        })
    }

    pub fn move_up(&mut self) -> Option<usize> {
        self.piece_table.prev_line(self.pos).map(|offset| {
            println!("===== move_up =====");
            println!("prev char: {:?}", self.current_char());
            self.set_pos(offset);
            self.col = self.column(self.pos).expect("should not fail");
            self.line = self
                .piece_table
                .line_for_offset(self.pos)
                .expect("should not fail");
            println!(
                "next char: {:?}, col={} line={} offset={}",
                self.current_char(),
                self.col.0,
                self.line.0,
                self.pos
            );
            println!("=====================");
            offset
        })
    }

    pub fn remove_current_char(&mut self) {
        self.piece_table
            .remove(self.pos..self.pos + 1)
            .expect("remove failed");
        self.contents = self.piece_table.contents();
    }

    pub fn column(&self, offset: usize) -> Option<HorizontalOffset> {
        self.piece_table.column_for_offset(offset)
    }
}
