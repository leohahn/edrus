use crate::error::Error;

pub trait TextBuffer {
    fn insert(&mut self, pos: usize, text: &str) -> Result<(), Error>;
    fn remove(&mut self, range: std::ops::Range<usize>) -> Result<(), Error>;
    fn contents(&self) -> String;

    fn next(&self, offset: usize) -> Option<(char, usize)>;
    fn prev(&self, offset: usize) -> Option<(char, usize)>;
    fn below(&self, offset: usize) -> Option<(char, usize)>;
    fn above(&self, offset: usize) -> Option<(char, usize)>;
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

struct CurrentPiece {
    len_until: usize,
    index: usize,
    piece: Piece,
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

    fn find_prev_offset(offset: usize, buf: &str) -> usize {
        let mut prev_offset = offset - 1;
        while !buf.is_char_boundary(prev_offset) {
            assert!(prev_offset > 0);
            prev_offset -= 1;
        }
        prev_offset
    }

    fn get_buffer(&self, piece: &Piece) -> &str {
        match piece.buffer {
            Buffer::Original => &self.original,
            Buffer::Added => &self.added,
        }
    }

    fn display_piece(&self, piece: &Piece) -> String {
        let b = self.get_buffer(piece);
        format!(
            "start={} len={} contents={}",
            piece.start,
            piece.len,
            &b[piece.start..piece.start + piece.len]
        )
    }

    fn get_current_piece(&self, offset: usize) -> Option<CurrentPiece> {
        let mut total_offset = 0;
        for (idx, piece) in self.pieces.iter().enumerate() {
            total_offset += piece.len;
            if total_offset > offset {
                return Some(CurrentPiece {
                    len_until: total_offset - piece.len,
                    index: idx,
                    piece: piece.clone(),
                });
            }
        }
        None
    }
}

/// Given the inital byte of a UTF-8 codepoint, returns the number of
/// bytes required to represent the codepoint.
/// RFC reference : https://tools.ietf.org/html/rfc3629#section-4
fn len_utf8_from_first_byte(b: u8) -> usize {
    match b {
        b if b < 0x80 => 1,
        b if b < 0xe0 => 2,
        b if b < 0xf0 => 3,
        _ => 4,
    }
}

impl TextBuffer for SimplePieceTable {
    fn next(&self, offset: usize) -> Option<(char, usize)> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = offset - current_piece.len_until;
        let buffer = self.get_buffer(&current_piece.piece);

        assert!(buffer.is_char_boundary(current_piece.piece.start + piece_offset));

        let current_char_len =
            len_utf8_from_first_byte(buffer.as_bytes()[current_piece.piece.start + piece_offset]);
        let next_piece_offset = piece_offset + current_char_len;

        assert!(next_piece_offset <= current_piece.piece.len);
        if next_piece_offset == current_piece.piece.len {
            // The next char is at the beggining of the next piece.
            return self
                .pieces
                .get(current_piece.index + 1)
                .and_then(|next_piece| {
                    let buffer = self.get_buffer(next_piece);
                    let c = &buffer[next_piece.start..].chars().next()?;
                    Some((*c, current_piece.len_until + next_piece_offset))
                });
        }

        let next_char = &buffer[current_piece.piece.start + next_piece_offset..]
            .chars()
            .next()?;

        Some((*next_char, current_piece.len_until + next_piece_offset))
    }

    fn prev(&self, offset: usize) -> Option<(char, usize)> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = offset - current_piece.len_until;
        let buffer = self.get_buffer(&current_piece.piece);

        assert!(buffer.is_char_boundary(current_piece.piece.start + piece_offset));

        if piece_offset == 0 {
            // The previous character is on the prvious piece.
            if current_piece.index == 0 {
                return None;
            }

            return self
                .pieces
                .get(current_piece.index - 1)
                .and_then(|prev_piece| {
                    let buffer = self.get_buffer(prev_piece);
                    let prev_offset = SimplePieceTable::find_prev_offset(
                        prev_piece.len,
                        &buffer[prev_piece.start..],
                    );
                    let c = &buffer[prev_piece.start + prev_offset..].chars().next()?;
                    Some((*c, current_piece.len_until - (prev_piece.len - prev_offset)))
                });
        }

        let prev_piece_offset =
            SimplePieceTable::find_prev_offset(piece_offset, &buffer[current_piece.piece.start..]);

        let prev_char = &buffer[current_piece.piece.start + prev_piece_offset..]
            .chars()
            .next()?;

        Some((*prev_char, current_piece.len_until + prev_piece_offset))
    }

    fn below(&self, offset: usize) -> Option<(char, usize)> {
        let current_piece = self.get_current_piece(offset)?;

        None
    }

    fn above(&self, index: usize) -> Option<(char, usize)> {
        None
    }

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
                // We insert at the start of the piece, therefore
                // we create a new piece before it.
                insert_pieces.push((index, new_piece));
                break;
            }

            // Insert at the end of the piece.
            if end_piece_pos == pos {
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

            let piece_before = Piece {
                buffer: piece.buffer,
                start: piece.start,
                len: piece.len - diff_end,
            };

            let piece_after = Piece {
                buffer: piece.buffer,
                start: piece_before.start + piece_before.len,
                len: diff_end,
            };

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

    #[test]
    fn next() {
        let text = "the dog";
        let mut table = SimplePieceTable::new(text.to_owned());
        assert_eq!(table.contents(), text);

        table.insert(0, "Carl, ").unwrap();
        assert_eq!(table.contents(), "Carl, the dog");

        {
            let (c, idx) = table.next(3).expect("should not fail");
            assert_eq!((c, idx), (',', 4));

            let (c, idx) = table.next(4).expect("should not fail");
            assert_eq!((c, idx), (' ', 5));

            let (c, idx) = table.next(5).expect("should not fail");
            assert_eq!((c, idx), ('t', 6));

            let (c, idx) = table.next(6).expect("should not fail");
            assert_eq!((c, idx), ('h', 7));

            let (c, idx) = table.next(7).expect("should not fail");
            assert_eq!((c, idx), ('e', 8));

            let (c, idx) = table.next(8).expect("should not fail");
            assert_eq!((c, idx), (' ', 9));

            let (c, idx) = table.next(9).expect("should not fail");
            assert_eq!((c, idx), ('d', 10));

            let (c, idx) = table.next(10).expect("should not fail");
            assert_eq!((c, idx), ('o', 11));

            let (c, idx) = table.next(11).expect("should not fail");
            assert_eq!((c, idx), ('g', 12));
        }

        table.insert(6, "cartão ótimo, ").unwrap();
        assert_eq!(table.contents(), "Carl, cartão ótimo, the dog");

        {
            let (c, idx) = table.next(4).expect("should not fail");
            assert_eq!((c, idx), (' ', 5));

            let (c, idx) = table.next(5).expect("should not fail");
            assert_eq!((c, idx), ('c', 6));

            let (c, idx) = table.next(6).expect("should not fail");
            assert_eq!((c, idx), ('a', 7));

            let (c, idx) = table.next(7).expect("should not fail");
            assert_eq!((c, idx), ('r', 8));

            let (c, idx) = table.next(8).expect("should not fail");
            assert_eq!((c, idx), ('t', 9));

            let (c, idx) = table.next(9).expect("should not fail");
            assert_eq!((c, idx), ('ã', 10));

            let (c, idx) = table.next(10).expect("should not fail");
            assert_eq!((c, idx), ('o', 12));

            let (c, idx) = table.next(12).expect("should not fail");
            assert_eq!((c, idx), (' ', 13));

            let (c, idx) = table.next(13).expect("should not fail");
            assert_eq!((c, idx), ('ó', 14));

            let (c, idx) = table.next(14).expect("should not fail");
            assert_eq!((c, idx), ('t', 16));

            let (c, idx) = table.next(16).expect("should not fail");
            assert_eq!((c, idx), ('i', 17));

            let (c, idx) = table.next(17).expect("should not fail");
            assert_eq!((c, idx), ('m', 18));

            let (c, idx) = table.next(18).expect("should not fail");
            assert_eq!((c, idx), ('o', 19));
        }
    }

    #[test]
    fn prev() {
        let text = "the dog";
        let mut table = SimplePieceTable::new(text.to_owned());
        assert_eq!(table.contents(), text);

        table.insert(0, "Carl, ").unwrap();
        assert_eq!(table.contents(), "Carl, the dog");

        {
            let (c, idx) = table.prev(12).expect("should not fail");
            assert_eq!((c, idx), ('o', 11));

            let (c, idx) = table.prev(11).expect("should not fail");
            assert_eq!((c, idx), ('d', 10));

            let (c, idx) = table.prev(10).expect("should not fail");
            assert_eq!((c, idx), (' ', 9));

            let (c, idx) = table.prev(9).expect("should not fail");
            assert_eq!((c, idx), ('e', 8));

            let (c, idx) = table.prev(8).expect("should not fail");
            assert_eq!((c, idx), ('h', 7));

            let (c, idx) = table.prev(7).expect("should not fail");
            assert_eq!((c, idx), ('t', 6));

            let (c, idx) = table.prev(6).expect("should not fail");
            assert_eq!((c, idx), (' ', 5));

            let (c, idx) = table.prev(5).expect("should not fail");
            assert_eq!((c, idx), (',', 4));

            let (c, idx) = table.prev(4).expect("should not fail");
            assert_eq!((c, idx), ('l', 3));
        }

        table.insert(6, "cartão ótimo, ").unwrap();
        assert_eq!(table.contents(), "Carl, cartão ótimo, the dog");

        {
            let (c, idx) = table.prev(19).expect("should not fail");
            assert_eq!((c, idx), ('m', 18));

            let (c, idx) = table.prev(18).expect("should not fail");
            assert_eq!((c, idx), ('i', 17));

            let (c, idx) = table.prev(17).expect("should not fail");
            assert_eq!((c, idx), ('t', 16));

            let (c, idx) = table.prev(16).expect("should not fail");
            assert_eq!((c, idx), ('ó', 14));

            let (c, idx) = table.prev(14).expect("should not fail");
            assert_eq!((c, idx), (' ', 13));

            let (c, idx) = table.prev(13).expect("should not fail");
            assert_eq!((c, idx), ('o', 12));

            let (c, idx) = table.prev(12).expect("should not fail");
            assert_eq!((c, idx), ('ã', 10));

            let (c, idx) = table.prev(10).expect("should not fail");
            assert_eq!((c, idx), ('t', 9));

            let (c, idx) = table.prev(9).expect("should not fail");
            assert_eq!((c, idx), ('r', 8));

            let (c, idx) = table.prev(8).expect("should not fail");
            assert_eq!((c, idx), ('a', 7));

            let (c, idx) = table.prev(7).expect("should not fail");
            assert_eq!((c, idx), ('c', 6));

            let (c, idx) = table.prev(6).expect("should not fail");
            assert_eq!((c, idx), (' ', 5));

            let (c, idx) = table.prev(5).expect("should not fail");
            assert_eq!((c, idx), (',', 4));
        }
    }
}
