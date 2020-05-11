extern crate bytemuck;
extern crate dirs;
extern crate edrus;
extern crate env_logger;
extern crate flame;
extern crate flamer;
extern crate futures;
extern crate gluon;
extern crate gluon_codegen;
extern crate log;
extern crate nalgebra as na;
extern crate rusttype;
extern crate serde_derive;
extern crate winit;

mod footer;
mod keyboard;
mod rendering;
mod scripting;

use bytemuck::{Pod, Zeroable};
use edrus::text_buffer::{HorizontalOffset, Line};
use na::{Matrix4, Point2, Point3, Vector2, Vector3, Vector4};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};
use wgpu_glyph::{GlyphBrushBuilder, Scale, Section};
use winit::{
    event,
    event::VirtualKeyCode,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct ColorBufferObject {
    color: Vector4<f32>,
}

unsafe impl Pod for ColorBufferObject {}
unsafe impl Zeroable for ColorBufferObject {}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct CameraBufferObject {
    view_projection: Matrix4<f32>,
    model: Matrix4<f32>,
}

unsafe impl Pod for CameraBufferObject {}
unsafe impl Zeroable for CameraBufferObject {}

struct FontCache<'a> {
    font: rusttype::Font<'static>,
    hash: HashMap<char, rusttype::ScaledGlyph<'a>>,
    scale: rusttype::Scale,
}

impl<'a> FontCache<'a> {
    fn new(scale: rusttype::Scale, font_bytes: &'static [u8]) -> Self {
        let collection = rusttype::FontCollection::from_bytes(font_bytes).unwrap();
        let font = collection.into_font().unwrap();
        Self {
            hash: HashMap::new(),
            font: font,
            scale: scale,
        }
    }

    fn get_glyph(&mut self, c: char) -> &rusttype::ScaledGlyph<'a> {
        let scale = self.scale.clone();
        let entry = self.hash.entry(c).or_insert(self.font.glyph(c).scaled(scale));
        entry
    }

    fn v_metrics(&self) -> rusttype::VMetrics {
        self.font.v_metrics(self.scale)
    }
}

fn get_view_matrix(eye: &Point3<f32>) -> Matrix4<f32> {
    let target = eye + Vector3::new(0.0, 0.0, 1.0);
    na::Matrix4::look_at_rh(eye, &target, &-Vector3::y())
}

struct EditorView {
    height: u32,
    width: u32,
    normal_cursor: Option<edrus::cursor::Normal>,
    insert_cursor: Option<edrus::cursor::Insert>,
    projection_matrix: Matrix4<f32>,
    view_matrix: Matrix4<f32>,
    eye: Point3<f32>,
}

impl EditorView {
    fn new(filepath: impl AsRef<Path>, width: u32, height: u32) -> Self {
        let projection_matrix = na::Matrix4::new_orthographic(
            0.0,              // left
            width as f32,     // right
            -(height as f32), // bottom
            0.0,              // top
            0.1,              // znear
            100.0,            // zfar
        );

        let eye = Point3::new(0.0, 0.0, -5.0);

        Self {
            height: height,
            width: width,
            normal_cursor: Some(edrus::cursor::Normal::new(filepath).expect("buffer creation failed")),
            insert_cursor: None,
            projection_matrix: projection_matrix,
            view_matrix: get_view_matrix(&eye),
            eye: eye,
        }
    }

    fn top_y(&self) -> f32 {
        self.eye.y
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn update_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        self.projection_matrix = Matrix4::new_orthographic(
            0.0,              // left
            width as f32,     // right
            -(height as f32), // bottom
            0.0,              // top
            0.1,              // znear
            100.0,            // zfar
        );
    }

    fn scroll_down(&mut self, lines: u32, font_cache: &FontCache) {
        assert!(lines > 0);
        let vmetrics = font_cache.v_metrics();
        let vertical_offset = ((vmetrics.ascent - vmetrics.descent) + vmetrics.line_gap) * lines as f32;
        self.eye.y += vertical_offset;
        self.view_matrix = get_view_matrix(&self.eye);
    }

    fn scroll_up(&mut self, lines: u32, font_cache: &FontCache) {
        assert!(lines > 0);
        let vmetrics = font_cache.v_metrics();
        let vertical_offset = ((vmetrics.ascent - vmetrics.descent) + vmetrics.line_gap) * lines as f32;
        self.eye.y -= vertical_offset;
        self.view_matrix = get_view_matrix(&self.eye);
    }

    fn follow_cursor(&mut self, font_cache: &FontCache) {
        let line = if let Some(normal_cursor) = self.normal_cursor.as_ref() {
            normal_cursor.line
        } else if let Some(insert_cursor) = self.insert_cursor.as_ref() {
            insert_cursor.line
        } else {
            unreachable!()
        };

        let vmetrics = font_cache.v_metrics();
        let vertical_offset = vmetrics.ascent - vmetrics.descent;

        let closeness_top = VisualCursor::y(line, font_cache) - self.eye.y;
        if closeness_top < 0.0 {
            self.scroll_up(1, font_cache);
            return;
        }

        let closeness_bottom =
            (self.eye.y + self.height as f32) - (VisualCursor::y(line, font_cache) + vertical_offset);
        if closeness_bottom < 0.0 {
            self.scroll_down(1, font_cache);
        }
    }

    fn move_end_of_line(&mut self, font_cache: &mut FontCache) {
        if let Some(normal_cursor) = self.normal_cursor.as_mut() {
            if normal_cursor.current_char() == '\n' {
                return;
            }
            let new_offset = if let Some(offset) = normal_cursor.find_after('\n') {
                offset - 1
            } else {
                normal_cursor.contents().len() - 1
            };
            normal_cursor.move_to(new_offset);
        }
    }

    fn move_left(&mut self, font_cache: &mut FontCache) -> Option<()> {
        let (line, col) = if let Some(normal_cursor) = self.normal_cursor.as_mut() {
            normal_cursor
                .move_left()
                .map(|_| (normal_cursor.line, normal_cursor.col))
        } else {
            unreachable!()
        }?;

        Some(())
    }

    fn move_left_unbounded(&mut self, font_cache: &mut FontCache) -> Option<()> {
        let (line, col) = if let Some(normal_cursor) = self.normal_cursor.as_mut() {
            normal_cursor
                .move_left_unbounded()
                .map(|_| (normal_cursor.line, normal_cursor.col))
        } else {
            unreachable!()
        }?;

        Some(())
    }

    fn move_right(&mut self, font_cache: &mut FontCache) -> Option<()> {
        let (line, col) = if let Some(normal_cursor) = self.normal_cursor.as_mut() {
            normal_cursor
                .move_right()
                .map(|_| (normal_cursor.line, normal_cursor.col))
        } else {
            unreachable!()
        }?;

        Some(())
    }

    // fn move_right_unbounded(&mut self, font_cache: &mut FontCache) -> Option<()> {
    //     if let Some(normal_cursor) = self.normal_cursor.as_mut() {
    //         return normal_cursor.move_right_unbounded().map(|_| {
    //             self.visual_cursor
    //                 .set_position(normal_cursor.line, normal_cursor.col, font_cache);
    //         });
    //     }
    //     None
    // }

    fn contents(&self) -> &str {
        if let Some(normal_cursor) = self.normal_cursor.as_ref() {
            normal_cursor.contents()
        } else if let Some(insert_cursor) = self.insert_cursor.as_ref() {
            insert_cursor.contents()
        } else {
            ""
        }
    }

    fn draw_cursor(&self, font_cache: &mut FontCache) -> rusttype::Rect<f32> {
        VisualCursor::draw_cursor_for(
            font_cache,
            self.normal_cursor.as_ref(),
            self.insert_cursor.as_ref(),
        )
    }

    fn view_projection(&self) -> Matrix4<f32> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let mx_correction: Matrix4<f32> = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );
        mx_correction * self.projection_matrix * self.view_matrix
    }

    fn insert_text(&mut self, text: &str, font_cache: &mut FontCache) {
        let insert_cursor = self
            .insert_cursor
            .as_mut()
            .expect("should only be called on insert mode");
        insert_cursor.insert(text);
        // FIXME: this seems inneficient.
        for _ in 0..text.len() {
            if let None = insert_cursor.move_right() {
                break;
            }
        }
    }

    fn handle_key(&mut self, key: VirtualKeyCode, kb: &keyboard::Keyboard, font_cache: &mut FontCache) {
        match key {
            VirtualKeyCode::A => {
                if let Some(normal_cursor) = self.normal_cursor.take() {
                    self.insert_cursor = Some(normal_cursor.enter_append());
                } else {
                    self.insert_text("a", font_cache);
                }
            }
            VirtualKeyCode::B => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): Implement back by work.
                } else {
                    self.insert_text(if kb.shift_pressed() { "B" } else { "b" }, font_cache);
                }
            }
            VirtualKeyCode::C => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement delete line after cursor and enter insert mode.
                } else {
                    self.insert_text(if kb.shift_pressed() { "C" } else { "c" }, font_cache);
                }
            }
            VirtualKeyCode::D => {
                if kb.ctrl_pressed() {
                    self.normal_cursor
                        .as_mut()
                        .map(|cursor| {
                            let mut lines_scrolled = 0;
                            let lines_to_scroll = 5;
                            for _ in 0..lines_to_scroll {
                                cursor.move_down().map(|_| lines_scrolled += 1);
                            }
                            lines_scrolled
                        })
                        .map(|lines_scrolled| {
                            self.scroll_down(lines_scrolled, font_cache);
                        });
                }

                if self.insert_cursor.is_some() {
                    self.insert_text(if kb.shift_pressed() { "D" } else { "d" }, font_cache);
                }
            }
            VirtualKeyCode::E => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    if kb.ctrl_pressed() {
                        self.scroll_down(1, font_cache);
                    }
                } else {
                    self.insert_text(if kb.shift_pressed() { "E" } else { "e" }, font_cache);
                }
            }
            VirtualKeyCode::F => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): nothing to implement here?
                } else {
                    self.insert_text(if kb.shift_pressed() { "F" } else { "f" }, font_cache);
                }
            }
            VirtualKeyCode::G => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement jump to the end of the file here.
                } else {
                    self.insert_text(if kb.shift_pressed() { "G" } else { "g" }, font_cache);
                }
            }
            VirtualKeyCode::H => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    if kb.shift_pressed() {
                        // TODO(lhahn): implement something here?
                    } else {
                        let _ = normal_cursor.move_left();
                    }
                } else {
                    self.insert_text(if kb.shift_pressed() { "H" } else { "h" }, font_cache);
                }
            }
            VirtualKeyCode::I => {
                if let Some(normal_cursor) = self.normal_cursor.take() {
                    if kb.shift_pressed() {
                        // TODO(lhahn): implement go to start line and insert mode.
                    } else {
                        self.insert_cursor = Some(normal_cursor.enter_insert());
                    }
                } else {
                    self.insert_text(if kb.shift_pressed() { "I" } else { "i" }, font_cache);
                }
            }
            VirtualKeyCode::J => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    if kb.shift_pressed() {
                        // TODO(lhahn): implement join lines.
                    } else {
                        let _ = normal_cursor.move_down();
                    }
                } else {
                    self.insert_text(if kb.shift_pressed() { "J" } else { "j" }, font_cache);
                }
            }
            VirtualKeyCode::K => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    if kb.shift_pressed() {
                        // TODO(lhahn): implement something here?
                    } else {
                        let _ = normal_cursor.move_up();
                    }
                } else {
                    self.insert_text(if kb.shift_pressed() { "K" } else { "k" }, font_cache);
                }
            }
            VirtualKeyCode::L => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    if kb.shift_pressed() {
                        // TODO(lhahn): implement something here?
                    } else {
                        let _ = normal_cursor.move_right();
                    }
                } else {
                    self.insert_text(if kb.shift_pressed() { "L" } else { "l" }, font_cache);
                }
            }
            VirtualKeyCode::M => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement something here?
                } else {
                    self.insert_text(if kb.shift_pressed() { "M" } else { "m" }, font_cache);
                }
            }
            VirtualKeyCode::N => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement something here?
                } else {
                    self.insert_text(if kb.shift_pressed() { "N" } else { "n" }, font_cache);
                }
            }
            VirtualKeyCode::O => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement enter insert mode previous line.
                } else {
                    self.insert_text(if kb.shift_pressed() { "O" } else { "o" }, font_cache);
                }
            }
            VirtualKeyCode::P => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement something here?
                } else {
                    self.insert_text(if kb.shift_pressed() { "p" } else { "P" }, font_cache);
                }
            }
            VirtualKeyCode::Q => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement something here?
                } else {
                    self.insert_text(if kb.shift_pressed() { "q" } else { "Q" }, font_cache);
                }
            }
            VirtualKeyCode::R => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement after cursor replace.
                } else {
                    self.insert_text(if kb.shift_pressed() { "r" } else { "S" }, font_cache);
                }
            }
            VirtualKeyCode::S => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement whole line delete and insert mode here.
                } else {
                    self.insert_text(if kb.shift_pressed() { "S" } else { "s" }, font_cache);
                }
            }
            VirtualKeyCode::T => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement something here?
                } else {
                    self.insert_text(if kb.shift_pressed() { "T" } else { "t" }, font_cache);
                }
            }
            VirtualKeyCode::U => {
                if kb.ctrl_pressed() {
                    self.normal_cursor
                        .as_mut()
                        .map(|cursor| {
                            let mut lines_scrolled = 0;
                            let lines_to_scroll = 5;
                            for _ in 0..lines_to_scroll {
                                cursor.move_up().map(|_| lines_scrolled += 1);
                            }
                            lines_scrolled
                        })
                        .map(|lines_scrolled| {
                            self.scroll_up(lines_scrolled, font_cache);
                        });
                }

                if self.insert_cursor.is_some() {
                    self.insert_text(if kb.shift_pressed() { "U" } else { "u" }, font_cache);
                }
            }
            VirtualKeyCode::V => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement whole line select here.
                } else {
                    self.insert_text(if kb.shift_pressed() { "V" } else { "v" }, font_cache);
                }
            }
            VirtualKeyCode::X => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    normal_cursor.remove_current_char();
                } else {
                    self.insert_text(if kb.shift_pressed() { "X" } else { "x" }, font_cache);
                }
            }
            VirtualKeyCode::W => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement jump by whitespaces.
                } else {
                    self.insert_text(if kb.shift_pressed() { "W" } else { "w" }, font_cache);
                }
            }
            VirtualKeyCode::Y => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    if kb.ctrl_pressed() {
                        self.scroll_up(1, font_cache);
                    }
                } else {
                    self.insert_text(if kb.shift_pressed() { "Y" } else { "y" }, font_cache);
                }
            }
            VirtualKeyCode::Z => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement jump middle top bottom here.
                } else {
                    self.insert_text(if kb.shift_pressed() { "Z" } else { "z" }, font_cache);
                }
            }
            VirtualKeyCode::Comma => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement indent here.
                } else {
                    self.insert_text(if kb.shift_pressed() { "<" } else { "," }, font_cache);
                }
            }
            VirtualKeyCode::Period => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    // TODO(lhahn): implement indent here.
                    // TODO(lhahn): implement command repeat here.
                } else {
                    self.insert_text(if kb.shift_pressed() { ">" } else { "." }, font_cache);
                }
            }
            VirtualKeyCode::Minus => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                } else {
                    self.insert_text(if kb.shift_pressed() { "_" } else { "-" }, font_cache);
                }
            }
            VirtualKeyCode::Equals => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                } else {
                    self.insert_text(if kb.shift_pressed() { "+" } else { "=" }, font_cache);
                }
            }
            VirtualKeyCode::Divide => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                } else {
                    self.insert_text(if kb.shift_pressed() { "?" } else { "/" }, font_cache);
                }
            }
            VirtualKeyCode::Backslash => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                } else {
                    self.insert_text(if kb.shift_pressed() { "|" } else { "\\" }, font_cache);
                }
            }
            VirtualKeyCode::Tab => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                } else {
                    // FIXME: tabs are not correctly rendered.
                    self.insert_text("    ", font_cache)
                }
            }
            VirtualKeyCode::Subtract => {
                if self.insert_cursor.is_some() {
                    self.insert_text("-", font_cache);
                }
            }
            VirtualKeyCode::Add => {
                if self.insert_cursor.is_some() {
                    self.insert_text("+", font_cache);
                }
            }
            VirtualKeyCode::At => {
                if self.insert_cursor.is_some() {
                    self.insert_text("@", font_cache);
                }
            }
            VirtualKeyCode::Apostrophe => {
                if self.insert_cursor.is_some() {
                    self.insert_text("'", font_cache);
                }
            }
            VirtualKeyCode::Semicolon => {
                if self.insert_cursor.is_some() {
                    self.insert_text(";", font_cache);
                }
            }
            VirtualKeyCode::Slash => {
                if self.insert_cursor.is_some() {
                    self.insert_text("/", font_cache);
                }
            }
            VirtualKeyCode::Return => {
                if self.insert_cursor.is_some() {
                    self.insert_text("\n", font_cache);
                }
            }
            VirtualKeyCode::Key0 => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    let new_offset = if let Some(offset) = normal_cursor.find_before('\n') {
                        offset + 1
                    } else {
                        0
                    };
                    normal_cursor.move_to(new_offset);
                } else {
                    self.insert_text("0", font_cache);
                }
            }
            VirtualKeyCode::Key1 => {
                if self.insert_cursor.is_some() {
                    self.insert_text("1", font_cache);
                }
            }
            VirtualKeyCode::Key2 => {
                if self.insert_cursor.is_some() {
                    self.insert_text("2", font_cache);
                }
            }
            VirtualKeyCode::Key3 => {
                if self.insert_cursor.is_some() {
                    self.insert_text("3", font_cache);
                }
            }
            VirtualKeyCode::Key4 => {
                if kb.shift_pressed() {
                    if self.normal_cursor.is_some() {
                        self.move_end_of_line(font_cache);
                    }
                }
                if self.insert_cursor.is_some() {
                    self.insert_text("4", font_cache);
                }
            }
            VirtualKeyCode::Key5 => {
                if self.insert_cursor.is_some() {
                    self.insert_text("5", font_cache);
                }
            }
            VirtualKeyCode::Key6 => {
                if self.insert_cursor.is_some() {
                    self.insert_text("6", font_cache);
                }
            }
            VirtualKeyCode::Key7 => {
                if self.insert_cursor.is_some() {
                    self.insert_text("7", font_cache);
                }
            }
            VirtualKeyCode::Key8 => {
                if self.insert_cursor.is_some() {
                    self.insert_text("8", font_cache);
                }
            }
            VirtualKeyCode::Key9 => {
                if self.insert_cursor.is_some() {
                    self.insert_text("9", font_cache);
                }
            }
            VirtualKeyCode::Space => {
                if self.insert_cursor.is_some() {
                    self.insert_text(" ", font_cache);
                }
            }
            VirtualKeyCode::Back => {
                if let Some(normal_cursor) = self.normal_cursor.as_mut() {
                    normal_cursor.move_left();
                }
                if let Some(insert_cursor) = self.insert_cursor.as_mut() {
                    insert_cursor.remove_previous_char();
                }
            }
            VirtualKeyCode::Escape => {
                if let Some(insert_cursor) = self.insert_cursor.take() {
                    self.normal_cursor = Some(insert_cursor.exit());
                }
            }
            _ => (),
        }
        self.follow_cursor(font_cache);
    }
}

struct VisualCursor {}

impl VisualCursor {
    fn x_from_horizontal_offset(horizontal_offset: HorizontalOffset, font_cache: &mut FontCache) -> f32 {
        assert!(horizontal_offset.0 > 0);
        // FIXME(lhahn): this is a hack, only works because I am using a
        // monospaced font.
        let glyph = font_cache.get_glyph('a');
        let hmetrics = glyph.h_metrics();
        hmetrics.advance_width * (horizontal_offset.0 - 1) as f32
    }

    fn y(line: Line, font_cache: &FontCache) -> f32 {
        let vmetrics = font_cache.v_metrics();
        let vertical_offset = (vmetrics.ascent - vmetrics.descent) + vmetrics.line_gap;
        vertical_offset * (line.0 - 1) as f32
    }

    fn get_position(line: Line, col: HorizontalOffset, font_cache: &mut FontCache) -> Point2<f32> {
        let vmetrics = font_cache.v_metrics();
        let vertical_offset = (vmetrics.ascent - vmetrics.descent) + vmetrics.line_gap;
        Point2::new(
            Self::x_from_horizontal_offset(col, font_cache),
            vertical_offset * (line.0 - 1) as f32,
        )
    }

    fn draw_cursor_for(
        font_cache: &mut FontCache,
        normal_cursor: Option<&edrus::cursor::Normal>,
        insert_cursor: Option<&edrus::cursor::Insert>,
    ) -> rusttype::Rect<f32> {
        let (advance_width, position) = if let Some(normal_cursor) = normal_cursor {
            (
                font_cache
                    .get_glyph(normal_cursor.current_char())
                    .h_metrics()
                    .advance_width,
                Self::get_position(normal_cursor.line, normal_cursor.col, font_cache),
            )
        } else if let Some(insert_cursor) = insert_cursor {
            (
                2.0, // TODO: remove hardcoded advance width
                Self::get_position(insert_cursor.line, insert_cursor.col, font_cache),
            )
        } else {
            unreachable!()
        };
        let mut rect = rusttype::Rect {
            min: rusttype::point(position.x, position.y),
            max: rusttype::point(position.x, position.y),
        };

        let vmetrics = font_cache.v_metrics();
        rect.max.x += advance_width;
        rect.max.y += vmetrics.ascent - vmetrics.descent;
        rect
    }
}

struct EditorWindow<'a> {
    width: u32,
    height: u32,
    focused: bool,
    editor_view: &'a EditorView,
}

struct FooterWindow {
    width: u32,
    height: u32,
    line: Line,
    col: HorizontalOffset,
}

// TODO: is this safe to assume for vector2?
unsafe impl Pod for PositionVertex {}
unsafe impl Zeroable for PositionVertex {}

#[derive(Copy, Clone, Debug)]
struct PositionVertex {
    position: Vector2<f32>,
}

impl PositionVertex {
    fn new(x: f32, y: f32) -> Self {
        Self {
            position: Vector2::new(x, y),
        }
    }
}

fn create_cursor() -> Vec<PositionVertex> {
    let xo = 0.0;
    let yo = 0.0;
    let width = 1.0;
    let height = 1.0;
    vec![
        PositionVertex::new(xo, yo),
        PositionVertex::new(xo + width, yo),
        PositionVertex::new(xo + width, yo + height),
        PositionVertex::new(xo + width, yo + height),
        PositionVertex::new(xo, yo + height),
        PositionVertex::new(xo, yo),
    ]
}

async fn run_async(
    event_loop: EventLoop<()>,
    window: Window,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    filepath: String,
) {
    let vm = scripting::startup_engine().expect("failed to startup scripting engine");
    let editor_config = scripting::get_editor_config(&vm);

    let adapter = wgpu::Adapter::request(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: None, // TODO: check the correct value.
        },
        wgpu::BackendBit::PRIMARY,
    )
    .await
    .unwrap();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        })
        .await;

    let font_data: &[u8] = include_bytes!("../fonts/iosevka-fixed-regular.ttf");
    let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(font_data)
        .expect("Load fonts")
        .build(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

    let mut sc_descriptor = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &sc_descriptor);

    let cursor_vertex_data = create_cursor();

    let cursor_vertex_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&cursor_vertex_data),
        wgpu::BufferUsage::VERTEX,
    );

    let vertex_shader = include_bytes!("../shaders/cursor/vertex.spirv");
    let fragment_shader = include_bytes!("../shaders/cursor/fragment.spirv");

    let color_uniform_buffer = rendering::UniformBuffer::new::<ColorBufferObject>(&device);
    let camera_uniform_buffer = rendering::UniformBuffer::new::<CameraBufferObject>(&device);

    let shader = rendering::Shader::new(
        &device,
        vertex_shader,
        fragment_shader,
        &[
            rendering::ShaderBinding::Uniform {
                index: 0,
                uniform_buffer: &color_uniform_buffer,
                visibility: wgpu::ShaderStage::FRAGMENT,
            },
            rendering::ShaderBinding::Uniform {
                index: 1,
                uniform_buffer: &camera_uniform_buffer,
                visibility: wgpu::ShaderStage::VERTEX,
            },
        ],
    );

    let widget_pipeline = rendering::WidgetPipeline::new(&device, &shader);

    let mut keyboard = keyboard::Keyboard::new(Duration::from_millis(200));
    let font_scale = Scale {
        x: editor_config.font_scale,
        y: editor_config.font_scale,
    };
    let mut font_cache = FontCache::new(font_scale, font_data);

    let mut cursor_inside_window = true;

    let mut editor_view = EditorView::new(filepath, sc_descriptor.width, sc_descriptor.height);

    let main_window = EditorWindow {
        width: 0,
        height: 0,
        focused: true,
        editor_view: &editor_view,
    };

    let footer_window = FooterWindow {
        width: 0,
        height: 0,
        line: Line(1),
        col: HorizontalOffset(1),
    };

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            event::Event::WindowEvent {
                event: event::WindowEvent::Resized(size),
                ..
            } => {
                sc_descriptor.width = size.width;
                sc_descriptor.height = size.height;
                swap_chain = device.create_swap_chain(&surface, &sc_descriptor);
                editor_view.update_size(sc_descriptor.width, sc_descriptor.height);
                window.request_redraw();
            }
            event::Event::RedrawRequested(_) => {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let frame = swap_chain
                    .get_next_texture()
                    .expect("timed out getting texture from the GPU");

                // Clear frame
                {
                    let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: &frame.view,
                            resolve_target: None,
                            load_op: wgpu::LoadOp::Clear,
                            store_op: wgpu::StoreOp::Store,
                            clear_color: wgpu::Color {
                                r: 0.01,
                                g: 0.01,
                                b: 0.01,
                                a: 1.0,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });
                }
                // Draw the cursor before everything else, since it should be behind
                // the text.
                {
                    color_uniform_buffer.fill_buffer(
                        &device,
                        &mut encoder,
                        bytemuck::bytes_of(&ColorBufferObject {
                            // The color of the cursor.
                            color: Vector4::new(1.0, 0.0, 0.0, 1.0),
                        }),
                    );

                    {
                        let cursor_rect = editor_view.draw_cursor(&mut font_cache);
                        let mut model = Matrix4::new_translation(&Vector3::new(
                            cursor_rect.min.x,
                            cursor_rect.min.y,
                            0.0,
                        ));
                        model.prepend_nonuniform_scaling_mut(&Vector3::new(
                            cursor_rect.width(),
                            cursor_rect.height(),
                            1.0,
                        ));

                        camera_uniform_buffer.fill_buffer(
                            &device,
                            &mut encoder,
                            bytemuck::bytes_of(&CameraBufferObject {
                                view_projection: editor_view.view_projection(),
                                model: model,
                            }),
                        );
                    }

                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: &frame.view,
                            resolve_target: None,
                            load_op: wgpu::LoadOp::Load,
                            store_op: wgpu::StoreOp::Store,
                            clear_color: wgpu::Color {
                                r: 0.01,
                                g: 0.01,
                                b: 0.01,
                                a: 1.0,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });
                    render_pass.set_pipeline(widget_pipeline.render_pipeline());
                    render_pass.set_bind_group(0, shader.bind_group(), &[]);
                    // TODO: are these parameters correct?
                    render_pass.set_vertex_buffer(0, &cursor_vertex_buffer, 0, 0);
                    render_pass.draw(0..cursor_vertex_data.len() as u32, 0..1);
                }

                // Render the text
                {
                    {
                        let _guard = flame::start_guard("process text");
                        let now = std::time::Instant::now();

                        let contents = editor_view.contents();
                        let lines = contents.split('\n');

                        let vmetrics = font_cache.v_metrics();
                        let line_height = vmetrics.ascent - vmetrics.descent;

                        let mut line_y = 0.0;

                        for line in lines {
                            if (line_y + line_height) < editor_view.top_y() {
                                // The bottom of the line is not visible, therefore
                                // the line will not be rendered.
                                line_y += line_height + vmetrics.line_gap;
                                continue;
                            }

                            // Avoid iterating over all lines if they are not going to be drawn.
                            if line_y > editor_view.top_y() + editor_view.height() as f32 {
                                break;
                            }

                            glyph_brush.queue(Section {
                                text: line,
                                screen_position: (0.0, line_y),
                                color: [1.0, 1.0, 1.0, 1.0],
                                scale: font_scale,
                                bounds: (editor_view.width() as f32, editor_view.height() as f32),
                                ..Section::default()
                            });

                            line_y += line_height + vmetrics.line_gap;
                        }

                        println!("process text used {} ms", now.elapsed().as_millis());
                    }

                    let view_proj: [f32; 16] = {
                        let mut arr: [f32; 16] = Default::default();
                        arr.copy_from_slice(editor_view.view_projection().as_slice());
                        arr
                    };

                    // Draw the text!

                    {
                        let _guard = flame::start_guard("draw text");
                        glyph_brush
                            .draw_queued_with_transform(&device, &mut encoder, &frame.view, view_proj)
                            .expect("Draw queued");
                    }
                }

                queue.submit(&[encoder.finish()]);
            }
            event::Event::DeviceEvent {
                event:
                    event::DeviceEvent::MouseWheel {
                        delta: mouse_scroll_delta,
                    },
                ..
            } => match mouse_scroll_delta {
                event::MouseScrollDelta::LineDelta(_, y) => {
                    if !cursor_inside_window {
                        return;
                    }

                    if y < 0.0 {
                        editor_view.scroll_down(1, &font_cache);
                    } else {
                        editor_view.scroll_up(1, &font_cache);
                    }
                    window.request_redraw();
                }
                _ => println!("this scroll format is not yet supported"),
            },
            event::Event::WindowEvent {
                event: event::WindowEvent::ModifiersChanged(modifiers_state),
                ..
            } => {
                keyboard.set_ctrl(modifiers_state.ctrl());
                keyboard.set_shift(modifiers_state.shift());
            }
            event::Event::WindowEvent {
                event: event::WindowEvent::CursorEntered { .. },
                ..
            } => {
                cursor_inside_window = true;
            }
            event::Event::WindowEvent {
                event: event::WindowEvent::CursorLeft { .. },
                ..
            } => {
                cursor_inside_window = false;
            }
            event::Event::WindowEvent {
                event: event::WindowEvent::KeyboardInput { input: key, .. },
                ..
            } => {
                if let None = key.virtual_keycode {
                    return;
                }

                let virtual_keycode = key.virtual_keycode.unwrap();

                keyboard.update(Instant::now(), virtual_keycode, key.state);

                let should_process_key = {
                    let key_state = keyboard.key_state(&virtual_keycode);
                    key_state.is_repeat() || key_state.was_just_pressed()
                };

                if should_process_key {
                    editor_view.handle_key(virtual_keycode, &keyboard, &mut font_cache);
                    window.request_redraw();
                }
            }
            event::Event::WindowEvent {
                event: event::WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
                // println!("saving flamegraph");
                // flame::dump_html(std::fs::File::create("flamegraph.html").unwrap()).unwrap();
            }
            _ => (),
        }
    });
}

fn main() {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "edrus=debug");
    }
    env_logger::init();

    let filepath = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            log::error!("failed to get filepath from first argument");
            std::process::exit(1);
        }
    };

    let event_loop = EventLoop::new();

    let (window, size, surface) = {
        let window = WindowBuilder::new()
            .with_title("edrus")
            .build(&event_loop)
            .unwrap();
        let size = window.inner_size();
        println!("width={}, height={}", size.width, size.height);
        let surface = wgpu::Surface::create(&window);
        (window, size, surface)
    };

    futures::executor::block_on(run_async(event_loop, window, size, surface, filepath));
}
