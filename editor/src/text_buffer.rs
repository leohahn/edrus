use crate::error::Error;
use memchr::{memchr, memrchr};

pub trait TextBuffer {
    fn insert(&mut self, pos: usize, text: &str) -> Result<(), Error>;
    fn remove(&mut self, range: std::ops::Range<usize>) -> Result<(), Error>;
    fn contents(&self) -> String;

    fn next(&self, offset: usize) -> Option<usize>;
    fn prev(&self, offset: usize) -> Option<usize>;
    fn prev_line(&self, offset: usize) -> Option<usize>;
    fn next_line(&self, offset: usize) -> Option<usize>;

    fn char_at(&self, offset: usize) -> Option<char>;
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

impl Piece {
    fn get_slice<'a>(&self, buffer: &'a str) -> &'a str {
        &buffer[self.start..self.start + self.len]
    }
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

    fn get_absolute_offset(&self, piece_index: usize, piece_offset: usize) -> usize {
        assert!(piece_index < self.pieces.len());
        let mut offset = 0;
        for (index, piece) in self.pieces.iter().enumerate().take(piece_index + 1) {
            if index == piece_index {
                assert!(piece_offset < piece.len);
                offset += piece_offset;
            } else {
                offset += piece.len;
            }
        }
        offset
    }

    fn scan_lines(
        &self,
        num_lines: isize,
        piece_index: usize,
        piece_offset: usize,
        // TODO(lhahn): better return type here.
    ) -> Vec<(usize, usize)> {
        assert!(num_lines != 0);
        assert!(piece_index < self.pieces.len());

        let mut offsets = Vec::new();
        let mut curr_num_lines = num_lines.abs();
        let mut first_call = true;

        if num_lines < 0 {
            for (index, piece) in self
                .pieces
                .iter()
                .enumerate()
                .rev()
                .skip(self.pieces.len() - (piece_index + 1))
            {
                let end_offset = if first_call {
                    first_call = false;
                    piece_offset
                } else {
                    piece.len
                };
                let buffer = self.get_buffer(piece);
                let slice = &buffer.as_bytes()[piece.start..piece.start + end_offset];
                if let Some(line_offset) = memrchr('\n' as u8, slice) {
                    offsets.push((index, line_offset));
                    curr_num_lines -= 1;
                }
                if curr_num_lines == 0 {
                    break;
                }
            }
        } else {
            for (index, piece) in self.pieces.iter().skip(piece_index).enumerate() {
                let start_offset = if first_call {
                    first_call = false;
                    piece_offset
                } else {
                    piece.start
                };
                let buffer = self.get_buffer(piece);
                let slice =
                    &buffer.as_bytes()[start_offset..start_offset + (piece.len - start_offset)];
                if let Some(line_offset) = memchr('\n' as u8, slice) {
                    offsets.push((index, line_offset));
                    curr_num_lines -= 1;
                }
                if curr_num_lines == 0 {
                    break;
                }
            }
        }

        offsets
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
    fn char_at(&self, offset: usize) -> Option<char> {
        // FIXME(lhahn): this method is slow O(n).
        let mut total_len = 0;
        for piece in self.pieces.iter() {
            total_len += piece.len;
            if total_len > offset {
                let buffer = self.get_buffer(piece);
                let piece_offset = offset - (total_len - piece.len);
                return buffer[piece.start + piece_offset..].chars().next();
            }
        }
        None
    }

    fn next(&self, offset: usize) -> Option<usize> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = offset - current_piece.len_until;
        let buffer = self.get_buffer(&current_piece.piece);

        assert!(buffer.is_char_boundary(current_piece.piece.start + piece_offset));

        let current_char_len =
            len_utf8_from_first_byte(buffer.as_bytes()[current_piece.piece.start + piece_offset]);
        let next_piece_offset = piece_offset + current_char_len;

        assert!(next_piece_offset <= current_piece.piece.len);
        Some(current_piece.len_until + next_piece_offset)
    }

    fn prev(&self, offset: usize) -> Option<usize> {
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
                    Some(current_piece.len_until - (prev_piece.len - prev_offset))
                });
        }

        let prev_piece_offset =
            SimplePieceTable::find_prev_offset(piece_offset, &buffer[current_piece.piece.start..]);

        Some(current_piece.len_until + prev_piece_offset)
    }

    fn prev_line(&self, offset: usize) -> Option<usize> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = offset - current_piece.len_until;
        let lines = self.scan_lines(-2, current_piece.index, piece_offset);

        if lines.is_empty() {
            return None;
        }

        let (first_piece_index, first_newline_offset) = lines[0];

        dbg!(first_piece_index);
        dbg!(first_newline_offset);

        let current_col = if first_piece_index == current_piece.index {
            println!("THEN");
            piece_offset - first_newline_offset
        } else {
            println!("ELSE");
            let mut col =
                piece_offset + (self.pieces[first_piece_index].len - (first_newline_offset + 1));
            if first_piece_index + 1 < current_piece.index {
                for piece in &self.pieces[first_piece_index..current_piece.index] {
                    col += piece.len;
                }
            }
            col
        };

        dbg!(current_col);

        if lines.len() == 1 {
            // It means that the above line is the first line.
            Some(current_col.min(first_newline_offset - 1))
        } else {
            let (second_piece_index, second_newline_offset) = lines[1];
            let mut current_piece_index = dbg!(second_piece_index);
            let mut current_piece = &self.pieces[second_piece_index];

            if second_piece_index == first_newline_offset {
                return Some(current_col.min(first_newline_offset - second_newline_offset - 1));
            } else if second_piece_index < first_newline_offset {
                let mut correct_offset = dbg!(second_newline_offset + current_col);
                for (i, piece) in self
                    .pieces
                    .iter()
                    .enumerate()
                    .take(first_piece_index + 1)
                    .skip(second_piece_index)
                {
                    if i == first_piece_index {
                        let abs = self
                            .get_absolute_offset(i, correct_offset.min(first_newline_offset - 1));
                        return Some(abs);
                    }
                    if correct_offset < piece.len {
                        let abs = self.get_absolute_offset(i, correct_offset);
                        return Some(abs);
                    }
                    correct_offset -= current_piece.len;
                }
            } else {
                unreachable!();
            }

            None
        }
    }

    fn next_line(&self, index: usize) -> Option<usize> {
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
            let idx = table.next(3).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (',', 4));

            let idx = table.next(4).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 5));

            let idx = table.next(5).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 6));

            let idx = table.next(6).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('h', 7));

            let idx = table.next(7).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('e', 8));

            let idx = table.next(8).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 9));

            let idx = table.next(9).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('d', 10));

            let idx = table.next(10).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('o', 11));

            let idx = table.next(11).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('g', 12));
        }

        table.insert(6, "cartão ótimo, ").unwrap();
        assert_eq!(table.contents(), "Carl, cartão ótimo, the dog");

        {
            let idx = table.next(4).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 5));

            let idx = table.next(5).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('c', 6));

            let idx = table.next(6).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('a', 7));

            let idx = table.next(7).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('r', 8));

            let idx = table.next(8).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 9));

            let idx = table.next(9).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('ã', 10));

            let idx = table.next(10).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('o', 12));

            let idx = table.next(12).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 13));

            let idx = table.next(13).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('ó', 14));

            let idx = table.next(14).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 16));

            let idx = table.next(16).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('i', 17));

            let idx = table.next(17).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('m', 18));

            let idx = table.next(18).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('o', 19));
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
            let idx = table.prev(12).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('o', 11));

            let idx = table.prev(11).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('d', 10));

            let idx = table.prev(10).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 9));

            let idx = table.prev(9).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('e', 8));

            let idx = table.prev(8).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('h', 7));

            let idx = table.prev(7).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 6));

            let idx = table.prev(6).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 5));

            let idx = table.prev(5).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (',', 4));

            let idx = table.prev(4).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('l', 3));
        }

        table.insert(6, "cartão ótimo, ").unwrap();
        assert_eq!(table.contents(), "Carl, cartão ótimo, the dog");

        {
            let idx = table.prev(19).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('m', 18));

            let idx = table.prev(18).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('i', 17));

            let idx = table.prev(17).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 16));

            let idx = table.prev(16).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('ó', 14));

            let idx = table.prev(14).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 13));

            let idx = table.prev(13).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('o', 12));

            let idx = table.prev(12).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('ã', 10));

            let idx = table.prev(10).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 9));

            let idx = table.prev(9).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('r', 8));

            let idx = table.prev(8).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('a', 7));

            let idx = table.prev(7).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('c', 6));

            let idx = table.prev(6).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 5));

            let idx = table.prev(5).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (',', 4));
        }
    }

    #[test]
    fn scan_lines() {
        let text = "the dog\n";
        let mut table = SimplePieceTable::new(text.to_owned());
        assert_eq!(table.contents(), text);

        table.insert(8, "cool dog\n").unwrap();
        assert_eq!(table.contents(), "the dog\ncool dog\n");

        table.insert(17, "golden retrivier\n").unwrap();
        assert_eq!(table.contents(), "the dog\ncool dog\ngolden retrivier\n");

        {
            let lines = table.scan_lines(-1, 0, 0);
            assert_eq!(lines, vec![]);
        }
        {
            let last_piece = table.pieces.last().unwrap();
            let lines = table.scan_lines(1, table.pieces.len() - 1, last_piece.len);
            assert_eq!(lines, vec![]);
        }
        {
            let lines = table.scan_lines(1, 0, 0);
            assert_eq!(lines, vec![(0, 7)]);
        }
        {
            let lines = table.scan_lines(2, 0, 0);
            assert_eq!(lines, vec![(0, 7), (1, 8)]);
        }
        {
            let lines = table.scan_lines(2, 0, 0);
            assert_eq!(lines, vec![(0, 7), (1, 8)]);
        }
        {
            let lines = table.scan_lines(-5, 2, 7);
            assert_eq!(lines, vec![(1, 8), (0, 7)]);
        }
        {
            let lines = table.scan_lines(-1, 1, 0);
            assert_eq!(lines, vec![(0, 7)]);
        }
    }

    #[test]
    fn prev_line() {
        let text = "the dog\n";
        let mut table = SimplePieceTable::new(text.to_owned());
        assert_eq!(table.contents(), text);

        table.insert(8, "cool dog\n").unwrap();
        assert_eq!(table.contents(), "the dog\ncool dog\n");

        table.insert(17, "golden retrivier\n").unwrap();
        assert_eq!(table.contents(), "the dog\ncool dog\ngolden retrivier\n");
        // r#"
        // the dog
        // cool dog
        // golden retrivier
        // "#;

        {
            let idx = table.prev_line(8).expect("should not fail");
            assert_eq!(table.char_at(8), Some('c'));
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 0));
        }
        {
            let idx = table.prev_line(12).expect("should not fail");
            assert_eq!(table.char_at(12), Some(' '));
            assert_eq!((table.char_at(idx).unwrap(), idx), ('d', 4));
        }
        {
            let idx = table.prev_line(32).expect("should not fail");
            assert_eq!(table.char_at(32), Some('r'));
            assert_eq!((table.char_at(idx).unwrap(), idx), ('g', 15));
        }
    }
}
