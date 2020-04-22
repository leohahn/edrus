use crate::error::Error;
use memchr::{memchr, memrchr};

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct HorizontalOffset(pub usize);

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct Line(pub usize);

pub trait TextBuffer {
    fn insert(&mut self, pos: usize, text: &str) -> Result<(), Error>;
    fn remove(&mut self, range: std::ops::Range<usize>) -> Result<(), Error>;
    fn contents(&self) -> String;

    fn next<M: Metric>(&self, offset: usize) -> Option<usize>;
    fn prev(&self, offset: usize) -> Option<usize>;
    fn prev_line(&self, offset: usize) -> Option<usize>;
    fn next_line(&self, offset: usize) -> Option<usize>;

    fn char_at(&self, offset: usize) -> Option<char>;
    fn column_for_offset(&self, offset: usize) -> Option<HorizontalOffset>;
    fn line_for_offset(&self, offset: usize) -> Option<Line>;

    fn find_before(&self, offset: usize, character: char) -> Option<usize>;
}

// This is based on the Metric trait from xi-editor.
// TODO: find a better name.
pub trait Metric {
    fn next(piece_slice: &PieceSlice, offset: PieceOffset) -> Option<(char, PieceOffset)>;
    fn prev(piece_slice: &PieceSlice, offset: PieceOffset) -> Option<(char, PieceOffset)>;
    fn at_boundary(piece_slice: &PieceSlice, offset: PieceOffset) -> bool;
}

#[derive(Debug)]
pub struct LineMetric(());

impl Metric for LineMetric {
    fn next(piece_slice: &PieceSlice, offset: PieceOffset) -> Option<(char, PieceOffset)> {
        if offset.0 == piece_slice.len() - 1 {
            None
        } else {
            piece_slice
                .memchr_from(offset + 1, '\n' as u8)
                .map(|newline| ('\n', newline))
        }
    }

    fn prev(piece_slice: &PieceSlice, offset: PieceOffset) -> Option<(char, PieceOffset)> {
        if offset.0 == 0 {
            None
        } else {
            piece_slice
                .memrchr_from(offset - 1, '\n' as u8)
                .map(|newline| ('\n', newline))
        }
    }

    fn at_boundary(piece_slice: &PieceSlice, offset: PieceOffset) -> bool {
        piece_slice.char_at(offset).map(|c| c == '\n').unwrap_or(false)
    }
}

#[derive(Debug)]
pub struct CharMetric(());

impl Metric for CharMetric {
    fn next(piece_slice: &PieceSlice, offset: PieceOffset) -> Option<(char, PieceOffset)> {
        piece_slice.byte_at(offset).and_then(|byte| {
            let next_offset = offset + len_utf8_from_first_byte(byte);
            piece_slice.char_at(next_offset).map(|c| (c, next_offset))
        })
    }

    fn prev(piece_slice: &PieceSlice, offset: PieceOffset) -> Option<(char, PieceOffset)> {
        if offset.0 == 0 {
            None
        } else {
            let mut prev_offset = offset - 1;
            while !piece_slice.is_char_boundary(prev_offset) {
                assert!(prev_offset.0 > 0);
                prev_offset -= 1;
            }
            let c = piece_slice
                .char_at(prev_offset)
                .expect("prev_offset is not valid");
            Some((c, prev_offset))
        }
    }

    fn at_boundary(piece_slice: &PieceSlice, offset: PieceOffset) -> bool {
        piece_slice.is_char_boundary(offset)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Buffer {
    Original,
    Added,
}

pub struct PieceSlice<'a>(&'a str);

impl<'a> PieceSlice<'a> {
    fn memchr_from(&self, offset: PieceOffset, byte: u8) -> Option<PieceOffset> {
        let bytes = &self.0.as_bytes()[offset.0..];
        memchr(byte, bytes).map(|newline| PieceOffset(newline + offset.0))
    }

    fn memrchr_from(&self, offset: PieceOffset, byte: u8) -> Option<PieceOffset> {
        let bytes = &self.0.as_bytes()[0..=offset.0];
        memrchr(byte, bytes).map(|newline| PieceOffset(newline))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_char_boundary(&self, offset: PieceOffset) -> bool {
        self.0.is_char_boundary(offset.0)
    }

    fn char_at(&self, offset: PieceOffset) -> Option<char> {
        self.0.get(offset.0..).and_then(|slice| slice.chars().next())
    }

    fn byte_at(&self, offset: PieceOffset) -> Option<u8> {
        self.0.as_bytes().get(offset.0).map(|b| b.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PieceOffset(usize);

impl std::ops::Add<usize> for PieceOffset {
    type Output = Self;
    fn add(self, other: usize) -> Self {
        Self(self.0 + other)
    }
}

impl std::ops::Sub<usize> for PieceOffset {
    type Output = Self;
    fn sub(self, other: usize) -> Self {
        Self(self.0 - other)
    }
}

impl std::ops::SubAssign<usize> for PieceOffset {
    fn sub_assign(&mut self, offset: usize) {
        self.0 -= offset;
    }
}

#[derive(Debug, Clone)]
struct Piece {
    buffer: Buffer,
    start: usize,
    len: usize,
}

impl Piece {
    fn get_slice<'a>(&'a self, buf: &'a str) -> PieceSlice {
        let piece_buf = &buf[self.start..self.start + self.len];
        PieceSlice(piece_buf)
    }
}

// NOTE: This data structure is inefficient, eventually this should be replaced
// with a RedBlack tree implementation or something similar.
pub struct SimplePieceTable {
    original: String,
    added: String,
    pieces: Vec<Piece>,
}

#[derive(Debug)]
struct CurrentPiece {
    len_until: usize,
    index: usize,
    piece: Piece,
}

impl SimplePieceTable {
    #[flamer::flame]
    pub fn new(text: String) -> Self {
        // TODO: consider supporting \r\n here instead of only \n.
        let formatted_text = text.replace("\r\n", "\n");
        let text_len = formatted_text.len();
        assert!(formatted_text.is_ascii(), "only works with ascii currently");
        SimplePieceTable {
            original: formatted_text,
            added: String::new(),
            pieces: vec![Piece {
                buffer: Buffer::Original,
                start: 0,
                len: text_len,
            }],
        }
    }

    #[flamer::flame]
    fn find_prev_offset(offset: usize, buf: &str) -> usize {
        let mut prev_offset = offset - 1;
        while !buf.is_char_boundary(prev_offset) {
            assert!(prev_offset > 0);
            prev_offset -= 1;
        }
        prev_offset
    }

    #[flamer::flame]
    fn get_buffer(&self, piece: &Piece) -> &str {
        match piece.buffer {
            Buffer::Original => &self.original,
            Buffer::Added => &self.added,
        }
    }

    #[flamer::flame]
    fn display_piece(&self, piece: &Piece) -> String {
        let b = self.get_buffer(piece);
        format!(
            "start={} len={} contents={}",
            piece.start,
            piece.len,
            &b[piece.start..piece.start + piece.len]
        )
    }

    #[flamer::flame]
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

    #[flamer::flame]
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

    #[flamer::flame]
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
                    if piece_offset == 0 {
                        continue;
                    } else {
                        piece_offset
                    }
                } else {
                    piece.len
                };

                let buffer = self.get_buffer(piece);
                let mut slice = &buffer.as_bytes()[piece.start..piece.start + end_offset];

                while let Some(line_offset) = memrchr('\n' as u8, slice) {
                    offsets.push((index, line_offset));
                    curr_num_lines -= 1;
                    if curr_num_lines == 0 {
                        break;
                    }
                    slice = &buffer.as_bytes()[piece.start..piece.start + line_offset];
                }
            }
        } else {
            for (index, piece) in self.pieces.iter().enumerate().skip(piece_index) {
                let start_offset = if first_call {
                    first_call = false;
                    if piece_offset + 1 > piece.len {
                        continue;
                    } else {
                        piece_offset + 1
                    }
                } else {
                    0
                };

                let piece_start = piece.start + start_offset;

                let buffer = self.get_buffer(piece);
                let mut slice = &buffer.as_bytes()[piece_start..piece_start + (piece.len - start_offset)];

                let mut offset_accum = start_offset;
                while let Some(line_offset) = memchr('\n' as u8, slice) {
                    offsets.push((index, offset_accum + line_offset));
                    curr_num_lines -= 1;

                    if curr_num_lines == 0 {
                        break;
                    }

                    offset_accum += line_offset + 1;
                    if line_offset == slice.len() - 1 {
                        break;
                    }

                    let next_offset = line_offset + 1;
                    slice = &slice[next_offset..next_offset + (slice.len() - next_offset)];
                }
                if curr_num_lines == 0 {
                    break;
                }
            }
        }

        offsets
    }

    #[flamer::flame]
    fn is_on_the_right_of(
        &self,
        index: usize,
        offset: usize,
        target_index: usize,
        target_offset: usize,
    ) -> bool {
        let abs = self.get_absolute_offset(index, offset);
        let target_abs = self.get_absolute_offset(target_index, target_offset);
        abs == target_abs + 1
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
    fn find_before(&self, offset: usize, character: char) -> Option<usize> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = offset - current_piece.len_until;

        for (index, piece) in self
            .pieces
            .iter()
            .enumerate()
            .rev()
            .skip(self.pieces.len() - (current_piece.index + 1))
        {
            let curr_buffer = self.get_buffer(piece);
            let start = piece.start;
            let end = if index == current_piece.index {
                piece_offset
            } else {
                piece.len
            };

            let str_slice = &curr_buffer[start..start + end];
            let maybe_newline_offset = str_slice.rfind('\n');
            let maybe_char_offset = str_slice.rfind(character);

            match (maybe_char_offset, maybe_newline_offset) {
                (None, None) => continue,
                (None, Some(_)) => {
                    return None;
                }
                (Some(_), None) => {
                    return maybe_char_offset.map(|char_offset| self.get_absolute_offset(index, char_offset));
                }
                (Some(char_offset), Some(newline_offset)) => {
                    if char_offset >= newline_offset {
                        return maybe_char_offset
                            .map(|char_offset| self.get_absolute_offset(index, char_offset));
                    } else {
                        return None;
                    }
                }
            };
        }

        None
    }

    fn line_for_offset(&self, offset: usize) -> Option<Line> {
        let current_piece = self.get_current_piece(offset)?;
        let current_piece_offset = PieceOffset(offset - current_piece.len_until);
        // TODO: consider make lines start at 0.
        let mut line = 1;

        for (i, piece) in self.pieces.iter().enumerate().take(current_piece.index + 1) {
            let last_index = i == current_piece.index;
            let piece_slice = piece.get_slice(self.get_buffer(piece));
            let mut piece_offset = PieceOffset(0);

            if LineMetric::at_boundary(&piece_slice, piece_offset) {
                line += 1;
            }

            while let Some((_, offset)) = LineMetric::next(&piece_slice, piece_offset) {
                piece_offset = offset;
                if last_index {
                    if offset < current_piece_offset {
                        line += 1;
                    }
                } else {
                    line += 1;
                }
            }
        }

        Some(Line(line))
    }

    #[flamer::flame]
    fn column_for_offset(&self, offset: usize) -> Option<HorizontalOffset> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = offset - current_piece.len_until;
        let buffer = self.get_buffer(&current_piece.piece);

        if buffer.as_bytes()[current_piece.piece.start + piece_offset] == '\n' as u8 {
            return Some(HorizontalOffset(1));
        }

        let prev_lines = self.scan_lines(-1, current_piece.index, piece_offset);

        if prev_lines.is_empty() {
            return Some(HorizontalOffset(offset + 1));
        }

        let (prev_line_index, prev_line_offset) = prev_lines[0];
        if prev_line_index == current_piece.index {
            return Some(HorizontalOffset(piece_offset - prev_line_offset));
        }

        let mut col_offset = 0;

        for (i, piece) in self
            .pieces
            .iter()
            .enumerate()
            .take(current_piece.index + 1)
            .skip(prev_line_index)
        {
            let start_piece_offset = if i == prev_line_index { prev_line_offset } else { 0 };
            let end_offset = if i == current_piece.index {
                piece_offset
            } else {
                piece.len
            };

            col_offset += end_offset - start_piece_offset;
        }

        Some(HorizontalOffset(col_offset))
    }

    #[flamer::flame]
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

    #[flamer::flame]
    fn next<M: Metric>(&self, offset: usize) -> Option<usize> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = PieceOffset(offset - current_piece.len_until);

        for (i, piece) in self.pieces.iter().enumerate().skip(current_piece.index) {
            let piece_slice = piece.get_slice(self.get_buffer(&piece));
            if i == current_piece.index {
                match M::next(&piece_slice, piece_offset) {
                    None => (),
                    Some((_ch, offset)) => {
                        let abs = self.get_absolute_offset(i, offset.0);
                        return Some(abs);
                    }
                }
            } else {
                let at_boundary = M::at_boundary(&piece_slice, PieceOffset(0));
                if at_boundary {
                    let abs = self.get_absolute_offset(i, 0);
                    return Some(abs);
                }

                match M::next(&piece_slice, PieceOffset(0)) {
                    None => (),
                    Some((_ch, offset)) => {
                        let abs = self.get_absolute_offset(i, offset.0);
                        return Some(abs);
                    }
                }
            }
        }

        None
    }

    #[flamer::flame]
    fn prev(&self, offset: usize) -> Option<usize> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = PieceOffset(offset - current_piece.len_until);

        for (i, piece) in self
            .pieces
            .iter()
            .enumerate()
            .rev()
            .skip(self.pieces.len() - (current_piece.index + 1))
        {
            let start_offset = if i == current_piece.index {
                piece_offset
            } else {
                PieceOffset(piece.len)
            };
            let piece_slice = piece.get_slice(self.get_buffer(piece));
            match CharMetric::prev(&piece_slice, start_offset) {
                Some((_, offset)) => {
                    let abs = self.get_absolute_offset(i, offset.0);
                    return Some(abs);
                }
                None => (),
            };
        }

        None
    }

    #[flamer::flame]
    fn prev_line(&self, offset: usize) -> Option<usize> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = offset - current_piece.len_until;
        let lines = self.scan_lines(-2, current_piece.index, piece_offset);

        if lines.is_empty() {
            return None;
        }

        let (first_piece_index, first_newline_offset) = lines[0];

        let current_piece_buffer = self.get_buffer(&current_piece.piece);
        let is_current_char_newline =
            current_piece_buffer.as_bytes()[current_piece.piece.start + piece_offset] == '\n' as u8;

        let current_col = if is_current_char_newline {
            1
        } else {
            let mut col = 0;

            for (i, piece) in self
                .pieces
                .iter()
                .enumerate()
                .take(current_piece.index + 1)
                .skip(first_piece_index)
            {
                let start_offset = if i == first_piece_index {
                    first_newline_offset
                } else {
                    0
                };
                let end_offset = if i == current_piece.index {
                    piece_offset
                } else {
                    piece.len
                };
                col += end_offset - start_offset;
            }
            col
        };

        if lines.len() == 1 {
            // It means that the above line is the first line.
            // NOTE: We take current_col - 1 here since the col starts at 1 instead of 0.
            // TODO: maybe change columns internally to start at 0?
            let abs_first_newline_offset = self.get_absolute_offset(first_piece_index, first_newline_offset);

            Some((current_col - 1).min(abs_first_newline_offset - 1))
        } else if is_current_char_newline {
            let is_last_char =
                current_piece.index == self.pieces.len() - 1 && piece_offset == current_piece.piece.len - 1;
            let (index, offset) = if is_last_char {
                let first_piece = &self.pieces[first_piece_index];
                if first_newline_offset == first_piece.len - 1 {
                    // If the newline is the last char of the piece, we return the next,
                    // piece 0 offset.
                    (first_piece_index + 1, 0)
                } else {
                    (first_piece_index, first_newline_offset + 1)
                }
            } else {
                let (second_piece_index, second_newline_offset) = lines[1];
                let second_piece = &self.pieces[second_piece_index];
                if second_newline_offset == second_piece.len - 1 {
                    // If the newline is the last char of the piece, we return the next,
                    // piece 0 offset.
                    (second_piece_index + 1, 0)
                } else {
                    (second_piece_index, second_newline_offset + 1)
                }
            };
            let abs = self.get_absolute_offset(index, offset);
            Some(abs)
        } else {
            let (second_piece_index, second_newline_offset) = lines[1];
            let mut correct_offset = second_newline_offset + current_col;
            let mut correct_index = second_piece_index;

            if self.is_on_the_right_of(
                first_piece_index,
                first_newline_offset,
                second_piece_index,
                second_newline_offset,
            ) {
                // We return the first previous newline if both newlines are consecutive.
                let abs = self.get_absolute_offset(first_piece_index, first_newline_offset);
                return Some(abs);
            }

            for (i, piece) in self
                .pieces
                .iter()
                .enumerate()
                .take(first_piece_index + 1)
                .skip(second_piece_index)
            {
                if correct_offset >= piece.len {
                    correct_offset -= piece.len;
                    correct_index += 1;
                    if i != first_piece_index {
                        continue;
                    }
                }

                if correct_index == first_piece_index {
                    let abs =
                        self.get_absolute_offset(correct_index, correct_offset.min(first_newline_offset - 1));
                    return Some(abs);
                } else if correct_index > first_piece_index {
                    let abs = self.get_absolute_offset(first_piece_index, first_newline_offset - 1);
                    return Some(abs);
                } else {
                    let abs = self.get_absolute_offset(correct_index, correct_offset);
                    return Some(abs);
                }
            }

            unreachable!()
        }
    }

    #[flamer::flame]
    fn next_line(&self, offset: usize) -> Option<usize> {
        let current_piece = self.get_current_piece(offset)?;
        let piece_offset = offset - current_piece.len_until;
        let prev_lines = self.scan_lines(-1, current_piece.index, piece_offset);
        let next_lines = self.scan_lines(1, current_piece.index, piece_offset);

        if next_lines.is_empty() {
            return None;
        }

        let current_piece_buffer = self.get_buffer(&current_piece.piece);
        let current_char_is_newline =
            current_piece_buffer.as_bytes()[current_piece.piece.start + piece_offset] == '\n' as u8;

        if current_char_is_newline {
            return Some(offset + 1);
        }

        let current_col = if prev_lines.is_empty() {
            // The current line is the first line.
            offset + 1
        } else if current_char_is_newline {
            1
        } else {
            let (prev_line_index, prev_line_offset) = prev_lines[0];
            let abs_offset = self.get_absolute_offset(prev_line_index, prev_line_offset);
            offset - abs_offset
        };

        assert!(next_lines.len() == 1);
        let (next_line_index, next_line_offset) = next_lines[0];

        let mut correct_offset = next_line_offset + current_col;
        let mut correct_index = next_line_index;

        for (i, piece) in self.pieces.iter().enumerate().skip(next_line_index) {
            let start_offset = if i == next_line_index {
                next_line_offset + 1
            } else {
                0
            };

            if start_offset >= piece.len {
                // the start offset is on the next piece.
                correct_offset -= piece.len;
                correct_index += 1;

                if correct_index >= self.pieces.len() {
                    // The correct index is past the end of the text,
                    // therfore we return the next newline offset.
                    let abs = self.get_absolute_offset(next_line_index, next_line_offset);
                    return Some(abs);
                }
                continue;
            }

            if correct_offset >= piece.len {
                correct_offset -= piece.len;
                correct_index += 1;
            }

            let buffer = self.get_buffer(piece);
            let piece_slice = &buffer.as_bytes()[piece.start + start_offset..piece.start + piece.len];

            if correct_index >= self.pieces.len() {
                // If the correct index is larger than the number of pieces, we return
                // the last offset.
                let abs =
                    self.get_absolute_offset(self.pieces.len() - 1, self.pieces.last().unwrap().len - 1);
                return Some(abs);
            }

            // Try to find a newline or the current column.
            if let Some(newline) = memchr('\n' as u8, piece_slice) {
                if i == next_line_index && newline == 0 {
                    // If right after the first newline we have another newline,
                    // we just return it as the current position.
                    // he[y]\n\nman --> hey\n[\n]man.
                    let abs = self.get_absolute_offset(i, start_offset + newline);
                    return Some(abs);
                }

                let max_col_offset = start_offset + newline - 1;

                if i != correct_index || max_col_offset < correct_offset {
                    let abs = self.get_absolute_offset(i, max_col_offset);
                    return Some(abs);
                }

                let abs = self.get_absolute_offset(i, correct_offset);
                return Some(abs);
            } else if correct_index == i {
                let abs = self.get_absolute_offset(correct_index, correct_offset);
                return Some(abs);
            }
        }

        None
    }

    #[flamer::flame]
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
            assert!(piece_before.len > 0);

            let piece_after = Piece {
                buffer: piece.buffer,
                start: piece_before.start + piece_before.len,
                len: diff_end,
            };
            assert!(piece_after.len > 0);

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

    #[flamer::flame]
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
            if removal.start_pos != 0 {
                // Partial removal.
                let piece_before = Piece {
                    buffer: first_piece.buffer,
                    start: first_piece.start,
                    len: removal.start_pos,
                };
                assert!(piece_before.len > 0);
                self.pieces.insert(removal.start_index, piece_before);
                removal.start_index += 1;
                removal.end_index += 1;
            }

            // Check if we need to insert a piece at the end.
            if removal.end_pos != last_piece.len {
                let piece_after = Piece {
                    buffer: last_piece.buffer,
                    start: last_piece.start + removal.end_pos,
                    len: last_piece.len - removal.end_pos,
                };
                assert!(piece_after.len > 0);
                self.pieces.insert(removal.end_index + 1, piece_after);
            }

            self.pieces.drain(removal.start_index..=removal.end_index);
        } else {
            return Err(Error::InvalidRange(range));
        }

        Ok(())
    }

    #[flamer::flame]
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
            let idx = table.next::<CharMetric>(3).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (',', 4));

            let idx = table.next::<CharMetric>(4).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 5));

            let idx = table.next::<CharMetric>(5).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 6));

            let idx = table.next::<CharMetric>(6).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('h', 7));

            let idx = table.next::<CharMetric>(7).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('e', 8));

            let idx = table.next::<CharMetric>(8).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 9));

            let idx = table.next::<CharMetric>(9).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('d', 10));

            let idx = table.next::<CharMetric>(10).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('o', 11));

            let idx = table.next::<CharMetric>(11).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('g', 12));
        }

        table.insert(6, "cartão ótimo, ").unwrap();
        assert_eq!(table.contents(), "Carl, cartão ótimo, the dog");

        {
            let idx = table.next::<CharMetric>(4).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 5));

            let idx = table.next::<CharMetric>(5).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('c', 6));

            let idx = table.next::<CharMetric>(6).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('a', 7));

            let idx = table.next::<CharMetric>(7).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('r', 8));

            let idx = table.next::<CharMetric>(8).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 9));

            let idx = table.next::<CharMetric>(9).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('ã', 10));

            let idx = table.next::<CharMetric>(10).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('o', 12));

            let idx = table.next::<CharMetric>(12).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 13));

            let idx = table.next::<CharMetric>(13).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('ó', 14));

            let idx = table.next::<CharMetric>(14).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 16));

            let idx = table.next::<CharMetric>(16).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('i', 17));

            let idx = table.next::<CharMetric>(17).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('m', 18));

            let idx = table.next::<CharMetric>(18).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('o', 19));
        }

        table.insert(29, "\n:)").unwrap();
        assert_eq!(table.contents(), "Carl, cartão ótimo, the dog\n:)");

        {
            assert_eq!(table.char_at(28).unwrap(), 'g');
            assert_eq!(table.next::<CharMetric>(28), Some(29));
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

        table.insert(29, "\n:)").unwrap();
        assert_eq!(table.contents(), "Carl, cartão ótimo, the dog\n:)");

        {
            assert_eq!(table.char_at(30).unwrap(), ':');
            assert_eq!(table.prev(30), Some(29));
        }
    }

    #[test]
    fn scan_lines_multiple_lines_on_same_piece() {
        let text = "the dog\nis awesome.\nI really like the dog.\nHello\n";
        let table = SimplePieceTable::new(text.to_owned());

        {
            let lines = table.scan_lines(-2, 0, 29);
            assert_eq!(lines, vec![(0, 19), (0, 7)]);
        }
        {
            let lines = table.scan_lines(2, 0, 0);
            assert_eq!(lines, vec![(0, 7), (0, 19)]);
        }
        {
            let lines = table.scan_lines(3, 0, 0);
            assert_eq!(lines, vec![(0, 7), (0, 19), (0, 42)]);
        }
        {
            let lines = table.scan_lines(-2, 0, 48);
            assert_eq!(lines, vec![(0, 42), (0, 19)]);
        }
        {
            let lines = table.scan_lines(-1, 0, 8);
            assert_eq!(lines, vec![(0, 7)]);
        }

        let text = r#"extern crate memchr;
pub mod buffer;
pub mod error;
pub mod text_buffer;
mod tree;
"#;
        let table = SimplePieceTable::new(text.to_owned());
        {
            assert_eq!(table.char_at(82), Some('\n'));
            let lines = table.scan_lines(-2, 0, 82);
            assert_eq!(lines, vec![(0, 72), (0, 51)]);

            assert_eq!(table.prev_line(82), Some(73));
            assert_eq!(table.char_at(73), Some('m'));
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
        {
            let lines = table.scan_lines(-1, 2, 16);
            assert_eq!(lines, vec![(1, 8), (0, 7)]);
        }
    }

    #[test]
    fn prev_line() {
        let text = "the dog\n";
        let mut table = SimplePieceTable::new(text.to_owned());

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
            assert_eq!(table.char_at(8), Some('c'));
            let idx = table.prev_line(8).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('t', 0));
        }
        {
            assert_eq!(table.column_for_offset(12), Some(HorizontalOffset(5)));
            assert_eq!(table.char_at(12), Some(' '));
            let idx = table.prev_line(12).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('d', 4));
        }
        {
            assert_eq!(table.char_at(32), Some('r'));
            let idx = table.prev_line(32).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('g', 15));
        }
        {
            assert_eq!(table.column_for_offset(21), Some(HorizontalOffset(5)));
            assert_eq!(table.char_at(21), Some('e'));
            let idx = table.prev_line(21).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), (' ', 12));
        }
        {
            assert_eq!(table.char_at(3), Some(' '));
            assert_eq!(table.prev_line(3), None);
        }
        {
            assert_eq!(table.char_at(15), Some('g'));
            let idx = table.prev_line(15).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('g', 6));
        }
    }

    #[test]
    fn prev_line_2() {
        let table_str = r#"# This file is automatically @generated by Cargo.
# It is not intended for manual editing.
[[package]]
name = "aho-corasick"
version = "0.7.9""#;

        let table = SimplePieceTable::new(table_str.to_owned());
        assert_eq!(table.char_at(91), Some('['));
        let offset = table.prev_line(91).expect("should not fail");
        assert_eq!(offset, 50);
        assert_eq!(table.char_at(50), Some('#'));
    }

    #[test]
    fn next_line_1() {
        let text = "the dog is amazing\n";
        let mut table = SimplePieceTable::new(text.to_owned());
        assert_eq!(table.contents(), text);

        table.insert(19, "cool dog\n").unwrap();
        assert_eq!(table.contents(), "the dog is amazing\ncool dog\n");

        table.insert(28, "golden retrivier\n").unwrap();
        assert_eq!(
            table.contents(),
            "the dog is amazing\ncool dog\ngolden retrivier\n"
        );
        // r#"
        // the dog is amazing
        // cool dog
        // golden retrivier
        // "#;
        {
            assert_eq!(table.char_at(19), Some('c'));
            let idx = table.next_line(19).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('g', 28));
        }
        {
            assert_eq!(table.char_at(23), Some(' '));
            let idx = table.next_line(23).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('e', 32));
        }
        {
            assert_eq!(table.char_at(43), Some('r'));
            let idx = table.next_line(43).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('\n', 44));
        }
        {
            assert_eq!(table.char_at(32), Some('e'));
            let idx = table.next_line(32).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('\n', 44));
        }
        {
            assert_eq!(table.char_at(14), Some('z'));
            let idx = table.next_line(14).expect("should not fail");
            assert_eq!((table.char_at(idx).unwrap(), idx), ('g', 26));
        }
    }

    #[test]
    fn next_line_2() {
        let table_str = r#"# This file is automatically @generated by Cargo.
# It is not intended for manual editing.
[[package]]
name = "aho-corasick"
version = "0.7.9""#;

        let table = SimplePieceTable::new(table_str.to_owned());

        let offset = table.next_line(0).expect("should not fail");
        assert_eq!(offset, 50);
        assert_eq!(table.char_at(50).expect("should not fail"), '#');
    }

    #[test]
    fn column_for_offset_1() {
        let table_str = r#"# This file is automatically @generated by Cargo.
# It is not intended for manual editing.
[[package]]
name = "aho-corasick"
version = "0.7.9""#;

        let table = SimplePieceTable::new(table_str.to_owned());
        assert_eq!(table.column_for_offset(7), Some(HorizontalOffset(8)));
        assert_eq!(table.column_for_offset(50), Some(HorizontalOffset(1)));
        assert_eq!(table.column_for_offset(69), Some(HorizontalOffset(20)));
        assert_eq!(table.column_for_offset(141), Some(HorizontalOffset(17)));
    }

    #[test]
    fn multiple_lines_edge_cases() {
        let table_str = r#"source = "registry"
checksum = "d5e6"
dependencies = []

[[package]]
name = "alga"
version = "0.9.3"
"#;

        let table = SimplePieceTable::new(table_str.to_owned());
        {
            assert_eq!(table.char_at(54), Some(']'));
            assert_eq!(table.column_for_offset(54), Some(HorizontalOffset(17)));
            assert_eq!(table.next_line(54), Some(56));
        }
        {
            assert_eq!(table.char_at(56), Some('\n'));
            assert_eq!(table.column_for_offset(56), Some(HorizontalOffset(1)));
            assert_eq!(table.next::<CharMetric>(56), Some(57));
            assert_eq!(table.prev(56), Some(55));
            assert_eq!(table.next_line(56), Some(57));
            assert_eq!(table.char_at(57), Some('['));
        }
        {
            assert_eq!(table.char_at(38), Some('d'));
            assert_eq!(table.prev_line(56), Some(38));
        }
        {
            assert_eq!(table.char_at(67), Some(']'));
            assert_eq!(table.prev_line(67), Some(56));
        }
        {
            assert_eq!(table.char_at(94), Some('0'));
            assert_eq!(table.next_line(94), Some(100));
            assert_eq!(table.column_for_offset(100), Some(HorizontalOffset(1)));
        }
        {
            assert_eq!(table.char_at(100), Some('\n'));
            assert_eq!(table.next_line(100), None);
        }
    }

    const WORKSPACE_TEXT: &str = r#"[workspace]
members = [
    "editor",
    "gui",
]

[profile.release]
debug = true
"#;
    #[test]
    fn workspace_text_deletion() -> Result<(), Error> {
        let mut table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());

        {
            table.remove(0..1)?;
            assert_eq!(
                table.contents(),
                "workspace]\nmembers = [\n    \"editor\",\n    \"gui\",\n]\n\n[profile.release]\ndebug = true\n"
            );
            table.remove(0..1)?;
            assert_eq!(
                table.contents(),
                "orkspace]\nmembers = [\n    \"editor\",\n    \"gui\",\n]\n\n[profile.release]\ndebug = true\n"
            );
            Ok(())
        }
    }

    #[test]
    fn insert_and_jump_to_empty_line() -> Result<(), Error> {
        let mut table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());

        assert_eq!(table.char_at(49), Some(']'));
        table.insert(49, "i")?;

        assert_eq!(table.next_line(49), Some(52));

        Ok(())
    }

    #[test]
    fn multiple_inserts_and_next_line() -> Result<(), Error> {
        let mut table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());

        assert_eq!(table.char_at(38), Some(' '));
        assert_eq!(table.column_for_offset(38), Some(HorizontalOffset(1)));

        table.insert(38, "i")?;
        table.insert(39, "i")?;
        table.insert(40, "i")?;
        table.insert(41, "i")?;
        table.insert(42, "i")?;

        assert_eq!(table.next_line(42), Some(54));
        assert_eq!(table.char_at(54), Some(']'));
        assert_eq!(table.next_line(54), Some(56));
        assert_eq!(table.prev_line(56), Some(54));

        Ok(())
    }

    #[test]
    fn workspace_text_insertion_and_deletion_idempotent() -> Result<(), Error> {
        let mut table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());

        for _ in 0..100 {
            table.insert(0, "i")?;
            table.remove(0..1)?;
        }

        Ok(())
    }

    #[test]
    fn workspace_text_insertion_and_movement_1() -> Result<(), Error> {
        let mut table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());

        {
            assert_eq!(table.char_at(46), Some('"'));
            assert_eq!(table.next_line(46), Some(49));
            assert_eq!(table.char_at(49), Some(']'));
        }

        {
            assert_eq!(table.column_for_offset(0), Some(HorizontalOffset(1)));
            assert_eq!(table.next_line(0), Some(12));
            assert_eq!(table.char_at(12), Some('m'));
            assert_eq!(table.column_for_offset(12), Some(HorizontalOffset(1)));
        }

        table.insert(0, "i")?;

        {
            assert_eq!(table.column_for_offset(0), Some(HorizontalOffset(1)));
            assert_eq!(table.next_line(0), Some(13));
            assert_eq!(table.char_at(13), Some('m'));
            assert_eq!(table.column_for_offset(13), Some(HorizontalOffset(1)));
            assert_eq!(table.prev_line(13), Some(0));
            assert_eq!(table.column_for_offset(0), Some(HorizontalOffset(1)));
        }

        table.insert(23, "i")?;
        assert_eq!(table.char_at(24), Some('['));

        {
            assert_eq!(table.next_line(23), Some(36));
            assert_eq!(table.char_at(36), Some('r'));
            assert_eq!(table.column_for_offset(36), Some(HorizontalOffset(11)));
        }

        {
            assert_eq!(table.char_at(10), Some('e'));
            assert_eq!(table.next_line(10), Some(23));
            assert_eq!(table.char_at(23), Some('i'));
        }

        Ok(())
    }

    #[test]
    fn workspace_text_insertion_and_movement_2() -> Result<(), Error> {
        let mut table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());

        assert_eq!(table.char_at(12), Some('m'));
        table.insert(12, "i")?;
        assert_eq!(table.char_at(12), Some('i'));

        {
            assert_eq!(table.char_at(25), Some(' '));
            assert_eq!(table.prev_line(25), Some(12));
        }

        Ok(())
    }

    #[test]
    fn workspace_text_find_before() -> Result<(), Error> {
        let table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());

        {
            assert_eq!(table.char_at(6), Some('p'));
            assert_eq!(table.find_before(6, '\n'), None);
            assert_eq!(table.find_before(6, '['), Some(0));
        }

        {
            assert_eq!(table.char_at(18), Some('s'));
            assert_eq!(table.find_before(18, '\n'), Some(11));
        }

        {
            assert_eq!(table.char_at(13), Some('e'));
            assert_eq!(table.find_before(18, ']'), None);
        }

        Ok(())
    }

    //     const WORKSPACE_TEXT: &str = r#"[workspace]
    // members = [
    //     "editor",
    //     "gui",
    // ]

    // [profile.release]
    // debug = true
    // "#;
    #[test]

    fn line_for_offset() {
        let table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());
        assert_eq!(table.line_for_offset(0), Some(Line(1)));
        assert_eq!(table.line_for_offset(5), Some(Line(1)));
        assert_eq!(table.line_for_offset(11), Some(Line(1)));
        assert_eq!(table.line_for_offset(12), Some(Line(2)));
        assert_eq!(table.line_for_offset(15), Some(Line(2)));
        assert_eq!(table.line_for_offset(23), Some(Line(2)));
        assert_eq!(table.line_for_offset(24), Some(Line(3)));
        assert_eq!(table.line_for_offset(36), Some(Line(3)));
        assert_eq!(table.line_for_offset(37), Some(Line(3)));
        assert_eq!(table.line_for_offset(49), Some(Line(5)));
        assert_eq!(table.line_for_offset(50), Some(Line(5)));
        assert_eq!(table.line_for_offset(51), Some(Line(6)));
        assert_eq!(table.line_for_offset(52), Some(Line(7)));
        assert_eq!(table.line_for_offset(69), Some(Line(7)));
        assert_eq!(table.line_for_offset(70), Some(Line(8)));
        assert_eq!(table.line_for_offset(82), Some(Line(8)));
        assert_eq!(table.line_for_offset(83), None);
    }

    #[test]
    fn char_movement() {
        let table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());
        let first_piece = &table.pieces[0];
        let piece_slice = first_piece.get_slice(table.get_buffer(first_piece));

        struct TestCase {
            input_offset: PieceOffset,
            expected: Option<(char, PieceOffset)>,
            next: bool,
        }

        let table = vec![
            TestCase {
                input_offset: PieceOffset(10),
                expected: Some(('\n', PieceOffset(11))),
                next: true,
            },
            TestCase {
                input_offset: PieceOffset(11),
                expected: Some(('m', PieceOffset(12))),
                next: true,
            },
            TestCase {
                input_offset: PieceOffset(11),
                expected: Some(('m', PieceOffset(12))),
                next: true,
            },
            TestCase {
                input_offset: PieceOffset(20),
                expected: Some((' ', PieceOffset(21))),
                next: true,
            },
            TestCase {
                input_offset: PieceOffset(WORKSPACE_TEXT.len() - 1),
                expected: None,
                next: true,
            },
            TestCase {
                input_offset: PieceOffset(WORKSPACE_TEXT.len() - 2),
                expected: Some(('\n', PieceOffset(WORKSPACE_TEXT.len() - 1))),
                next: true,
            },
            TestCase {
                input_offset: PieceOffset(20),
                expected: Some((' ', PieceOffset(19))),
                next: false,
            },
            TestCase {
                input_offset: PieceOffset(1),
                expected: Some(('[', PieceOffset(0))),
                next: false,
            },
            TestCase {
                input_offset: PieceOffset(0),
                expected: None,
                next: false,
            },
        ];

        for test_case in table {
            let res = if test_case.next {
                CharMetric::next(&piece_slice, test_case.input_offset)
            } else {
                CharMetric::prev(&piece_slice, test_case.input_offset)
            };
            assert_eq!(res, test_case.expected);
        }
    }

    #[test]
    fn line_movement() {
        let table = SimplePieceTable::new(WORKSPACE_TEXT.to_owned());
        let first_piece = &table.pieces[0];
        let piece_slice = first_piece.get_slice(table.get_buffer(first_piece));

        struct TestCase {
            input_offset: PieceOffset,
            expected: Option<(char, PieceOffset)>,
            next: bool,
        }

        let table = vec![
            TestCase {
                input_offset: PieceOffset(0),
                expected: Some(('\n', PieceOffset(11))),
                next: true,
            },
            TestCase {
                input_offset: PieceOffset(11),
                expected: Some(('\n', PieceOffset(23))),
                next: true,
            },
            TestCase {
                input_offset: PieceOffset(WORKSPACE_TEXT.len() - 1),
                expected: None,
                next: true,
            },
            TestCase {
                input_offset: PieceOffset(5),
                expected: None,
                next: false,
            },
            TestCase {
                input_offset: PieceOffset(11),
                expected: None,
                next: false,
            },
            TestCase {
                input_offset: PieceOffset(12),
                expected: Some(('\n', PieceOffset(11))),
                next: false,
            },
        ];

        for test_case in table {
            let res = if test_case.next {
                LineMetric::next(&piece_slice, test_case.input_offset)
            } else {
                LineMetric::prev(&piece_slice, test_case.input_offset)
            };
            assert_eq!(res, test_case.expected);
        }
    }
}
