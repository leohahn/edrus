use crate::error::Error;

pub trait TextBuffer {
    fn insert(&mut self, pos: usize, text: &str) -> Result<(), Error>;
    fn remove(&mut self, range: std::ops::Range<usize>) -> Result<(), Error>;
    fn contents(&self) -> String;
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Buffer {
    Original,
    Added,
}

#[derive(Debug, Clone)]
struct Piece {
    buffer: Buffer,
    start: usize,
    len: usize,
}

// NOTE: This data structure is inefficient, eventually this should be replaced
// with a RedBlack tree implementation or something similar.
pub struct SimplePieceTable {
    original: String,
    added: String,
    pieces: Vec<Piece>,
}

impl SimplePieceTable {
    pub fn new(text: String) -> Self {
        let text_len = text.len();
        SimplePieceTable {
            original: text,
            added: String::new(),
            pieces: vec![Piece {
                buffer: Buffer::Original,
                start: 0,
                len: text_len,
            }],
        }
    }

    fn display_piece(&self, piece: &Piece) -> String {
        let b: &str = match piece.buffer {
            Buffer::Original => self.original.as_ref(),
            Buffer::Added => self.added.as_ref(),
        };
        format!(
            "start={} len={} contents={}",
            piece.start,
            piece.len,
            &b[piece.start..piece.start + piece.len]
        )
    }
}

impl TextBuffer for SimplePieceTable {
    fn insert(&mut self, pos: usize, text: &str) -> Result<(), Error> {
        let added_start = self.added.len();
        self.added.push_str(text);

        let mut current_pos = 0;
        let mut insert_pieces = Vec::new();
        let mut remove_indexes = Vec::new();

        let new_piece = Piece {
            buffer: Buffer::Added,
            start: added_start,
            len: text.len(),
        };

        for (index, piece) in self.pieces.iter().enumerate() {
            let start_piece_pos = current_pos;
            let end_piece_pos = current_pos + piece.len;
            let last_index = index == self.pieces.len() - 1;

            // Insert at the beggining of the piece.
            if pos == start_piece_pos {
                println!("BEGIN");
                // We insert at the start of the piece, therefore
                // we create a new piece before it.
                insert_pieces.push((index, new_piece));
                break;
            }

            // Insert at the end of the piece.
            if end_piece_pos == pos {
                println!("END");
                insert_pieces.push((index + 1, new_piece));
                break;
            }

            if pos > end_piece_pos {
                if last_index {
                    return Err(Error::InvalidBufferPosition(pos));
                } else {
                    current_pos = end_piece_pos;
                    continue;
                }
            }

            // Insert at the middle of an existing piece.
            assert!(pos > start_piece_pos && pos < end_piece_pos);

            let diff_end = end_piece_pos - pos;

            println!(
                "MIDDLE: end={} pos={}, diff={}",
                end_piece_pos, pos, diff_end
            );

            let piece_before = Piece {
                buffer: piece.buffer,
                start: piece.start,
                len: piece.len - diff_end,
            };

            println!("CURRENT PIECE: {}", self.display_piece(&piece));

            println!("PIECE BEFORE: {}", self.display_piece(&piece_before));

            let piece_after = Piece {
                buffer: piece.buffer,
                start: piece_before.start + piece_before.len,
                len: diff_end,
            };

            println!("PIECE NEW: {}", self.display_piece(&new_piece));

            println!("PIECE AFTER: {}", self.display_piece(&piece_after));

            remove_indexes.push(index);
            insert_pieces.push((index, piece_after));
            insert_pieces.push((index, new_piece));
            insert_pieces.push((index, piece_before));
            break;
        }

        for i in remove_indexes {
            self.pieces.remove(i);
        }

        for (i, piece) in insert_pieces {
            self.pieces.insert(i, piece);
        }

        Ok(())
    }

    fn remove(&mut self, range: std::ops::Range<usize>) -> Result<(), Error> {
        let remove_start_pos = range.start;
        let remove_end_pos = range.end;

        #[derive(Debug)]
        struct Removal {
            start_index: usize,
            start_pos: usize,
            end_index: usize,
            end_pos: usize,
        }

        let mut maybe_removal = None;
        let mut current_pos = 0;
        for (index, piece) in self.pieces.iter().enumerate() {
            let start_piece_pos = current_pos;
            let end_piece_pos = current_pos + piece.len;
            let last_index = index == self.pieces.len() - 1;

            // Find the start of the range inside the pieces.
            if maybe_removal.is_none() {
                if end_piece_pos <= remove_start_pos {
                    current_pos = end_piece_pos;
                    continue;
                } else {
                    maybe_removal = Some(Removal {
                        start_index: index,
                        start_pos: remove_start_pos - start_piece_pos,
                        end_index: index,
                        end_pos: remove_start_pos - start_piece_pos,
                    });
                }
            }

            if end_piece_pos < remove_end_pos && last_index {
                return Err(Error::InvalidRange(range));
            } else if end_piece_pos < remove_end_pos {
                current_pos = end_piece_pos;
                continue;
            }

            maybe_removal.as_mut().map(|removal| {
                removal.end_index = index;
                removal.end_pos = remove_end_pos - start_piece_pos;
            });

            break;
        }

        if let Some(removal) = maybe_removal.as_mut() {
            let first_piece = self.pieces[removal.start_index].clone();
            let last_piece = self.pieces[removal.end_index].clone();

            // Check if we need to insert a piece at the beginning.
            if removal.start_pos != first_piece.start {
                // Partial removal.
                let piece_before = Piece {
                    buffer: first_piece.buffer,
                    start: first_piece.start,
                    len: removal.start_pos,
                };
                self.pieces.insert(removal.start_index, piece_before);
                removal.start_index += 1;
                removal.end_index += 1;
            }

            if removal.end_pos != last_piece.start + last_piece.len {
                let piece_after = Piece {
                    buffer: last_piece.buffer,
                    start: removal.end_pos,
                    len: last_piece.len - removal.end_pos,
                };
                self.pieces.insert(removal.end_index + 1, piece_after);
            }

            self.pieces.drain(removal.start_index..=removal.end_index);
        } else {
            return Err(Error::InvalidRange(range));
        }

        Ok(())
    }

    fn contents(&self) -> String {
        let mut contents = String::new();

        for piece in self.pieces.iter() {
            let buffer: &str = match piece.buffer {
                Buffer::Original => self.original.as_ref(),
                Buffer::Added => self.added.as_ref(),
            };
            let slice = &buffer[piece.start..(piece.start + piece.len)];
            contents.push_str(slice);
        }

        contents
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple_piece_table_can_be_created() {
        let text = "the dog";
        let mut table = SimplePieceTable::new(text.to_owned());
        assert_eq!(table.contents(), text);

        table.insert(0, "Carl, ").unwrap();
        assert_eq!(table.contents(), "Carl, the dog");

        table.insert(13, ", is an awesome dog").unwrap();
        assert_eq!(table.contents(), "Carl, the dog, is an awesome dog");

        table.insert(15, "and not the cat, ").unwrap();
        assert_eq!(
            table.contents(),
            "Carl, the dog, and not the cat, is an awesome dog"
        );

        assert_eq!(
            table.insert(50, "my text").unwrap_err(),
            Error::InvalidBufferPosition(50)
        );

        table.insert(49, ". He is 13 years old.").unwrap();
        assert_eq!(
            table.contents(),
            "Carl, the dog, and not the cat, is an awesome dog. He is 13 years old."
        );

        table.remove(0..6).unwrap();
        assert_eq!(
            table.contents(),
            "the dog, and not the cat, is an awesome dog. He is 13 years old."
        );

        table.remove(40..43).unwrap();
        assert_eq!(
            table.contents(),
            "the dog, and not the cat, is an awesome . He is 13 years old."
        );

        println!("================================");
        table.insert(40, "alpaca").unwrap();
        assert_eq!(
            table.contents(),
            "the dog, and not the cat, is an awesome alpaca. He is 13 years old."
        );
    }
}
