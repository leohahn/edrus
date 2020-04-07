extern crate edrus;
extern crate env_logger;
extern crate log;
extern crate nalgebra as na;
extern crate rusttype;
extern crate winit;

use edrus::text_buffer::HorizontalOffset;
use na::{Matrix4, Point2, Point3, Vector2, Vector3, Vector4};
use std::collections::HashMap;
use std::mem;
use std::path::Path;
use std::rc::Rc;
use wgpu_glyph::{GlyphBrushBuilder, Scale, Section};
use winit::{
    event,
    event::VirtualKeyCode,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct UBO {
    color: Vector4<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CoordinateUniformBuffer {
    view_projection: Matrix4<f32>,
    model: Matrix4<f32>,
}

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
        let entry = self
            .hash
            .entry(c)
            .or_insert(self.font.glyph(c).scaled(scale));
        entry
    }

    fn v_metrics(&self) -> rusttype::VMetrics {
        self.font.v_metrics(self.scale)
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum VisualCursorMode {
    Normal,
    Edit,
}

struct VisualCursor {
    position: Point2<f32>,
    mode: VisualCursorMode,
}

impl VisualCursor {
    fn mode(&self) -> VisualCursorMode {
        self.mode
    }
}

fn get_view_matrix(eye: &Point3<f32>) -> Matrix4<f32> {
    let target = eye + Vector3::new(0.0, 0.0, 1.0);
    na::Matrix4::look_at_rh(eye, &target, &-Vector3::y())
}

struct EditorView {
    top_y: f32,
    height: u32,
    visual_cursor: VisualCursor,
    buffer: edrus::buffer::Buffer,
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
            top_y: 0 as f32,
            height: height,
            visual_cursor: VisualCursor::new(0.0, 0.0),
            buffer: edrus::buffer::Buffer::new(filepath).expect("buffer creation failed"),
            projection_matrix: projection_matrix,
            view_matrix: get_view_matrix(&eye),
            eye: eye,
        }
    }

    fn update_size(&mut self, width: u32, height: u32) {
        self.projection_matrix = Matrix4::new_orthographic(
            0.0,              // left
            width as f32,     // right
            -(height as f32), // bottom
            0.0,              // top
            0.1,              // znear
            100.0,            // zfar
        );
    }

    fn scroll_down(&mut self, font_cache: &FontCache) {
        let vmetrics = font_cache.v_metrics();
        let vertical_offset = (vmetrics.ascent - vmetrics.descent) + vmetrics.line_gap;
        self.top_y += vertical_offset;
        self.eye.y += vertical_offset;
        self.view_matrix = get_view_matrix(&self.eye);
    }

    fn scroll_up(&mut self, font_cache: &FontCache) {
        let vmetrics = font_cache.v_metrics();
        let vertical_offset = (vmetrics.ascent - vmetrics.descent) + vmetrics.line_gap;
        self.top_y -= vertical_offset;
        self.eye.y -= vertical_offset;
        self.view_matrix = get_view_matrix(&self.eye);
    }

    fn move_up(&mut self, font_cache: &mut FontCache) -> Option<()> {
        self.buffer.move_up().map(|new_offset| {
            let hoffset = self.buffer.column(new_offset).expect("should not fail");
            self.visual_cursor.move_up(font_cache, hoffset);

            let closeness_top = self.visual_cursor.position.y - self.top_y;
            if closeness_top < 0.0 {
                self.scroll_up(font_cache);
            }
        })
    }

    fn move_down(&mut self, font_cache: &mut FontCache) -> Option<()> {
        self.buffer.move_down().map(|new_offset| {
            let hoffset = self.buffer.column(new_offset).expect("should not fail");
            self.visual_cursor.move_down(font_cache, hoffset);

            let vmetrics = font_cache.v_metrics();
            let vertical_offset = vmetrics.ascent - vmetrics.descent;

            let closeness_bottom = (self.top_y + self.height as f32)
                - (self.visual_cursor.position.y + vertical_offset);
            if closeness_bottom < 0.0 {
                self.scroll_down(font_cache);
            }
        })
    }

    fn move_left(&mut self, font_cache: &mut FontCache) -> Option<()> {
        self.buffer.move_left().map(|_| {
            self.visual_cursor
                .move_left(font_cache, self.buffer.current_char());
        })
    }

    fn move_right(&mut self, font_cache: &mut FontCache) -> Option<()> {
        self.buffer.move_right().map(|_| {
            self.visual_cursor
                .move_right(font_cache, self.buffer.current_char());
        })
    }

    fn contents(&self) -> &str {
        self.buffer.contents()
    }

    fn draw_cursor(&self, font_cache: &mut FontCache) -> rusttype::Rect<f32> {
        self.visual_cursor
            .draw_cursor_for(font_cache, self.buffer.current_char())
    }

    fn view_projection(&self) -> Matrix4<f32> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let mx_correction: Matrix4<f32> = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );
        mx_correction * self.projection_matrix * self.view_matrix
    }

    fn insert_text(&mut self, text: &str) {
        self.buffer.insert_before(text);
    }
}

impl VisualCursor {
    fn new(x: f32, y: f32) -> Self {
        Self {
            position: Point2::new(x, y),
            mode: VisualCursorMode::Normal,
        }
    }

    fn enter_edit_mode(&mut self) {
        assert!(self.mode != VisualCursorMode::Edit);
        self.mode = VisualCursorMode::Edit;
    }

    fn enter_normal_mode(&mut self) {
        self.mode = VisualCursorMode::Normal;
    }

    fn x_from_horizontal_offset(
        horizontal_offset: HorizontalOffset,
        font_cache: &mut FontCache,
    ) -> f32 {
        assert!(horizontal_offset.0 > 0);
        // FIXME(lhahn): this is a hack, only works because I am using a
        // monospaced font.
        let glyph = font_cache.get_glyph('a');
        let hmetrics = glyph.h_metrics();
        hmetrics.advance_width * (horizontal_offset.0 - 1) as f32
    }

    fn move_left(&mut self, font_cache: &mut FontCache, left_char: char) {
        let left_glyph = font_cache.get_glyph(left_char);
        let left_hmetrics = left_glyph.h_metrics();
        self.position.x -= left_hmetrics.advance_width;
    }

    fn move_right(&mut self, font_cache: &mut FontCache, curr_char: char) {
        let curr_glyph = font_cache.get_glyph(curr_char);
        let curr_hmetrics = curr_glyph.h_metrics();
        self.position.x += curr_hmetrics.advance_width;
    }

    fn move_down(&mut self, font_cache: &mut FontCache, hoffset: HorizontalOffset) {
        let vmetrics = font_cache.v_metrics();
        let vertical_offset = (vmetrics.ascent - vmetrics.descent) + vmetrics.line_gap;
        self.position.y += vertical_offset;
        self.position.x = Self::x_from_horizontal_offset(hoffset, font_cache);
    }

    fn move_up(&mut self, font_cache: &mut FontCache, hoffset: HorizontalOffset) {
        let vmetrics = font_cache.v_metrics();
        let vertical_offset = (vmetrics.ascent - vmetrics.descent) + vmetrics.line_gap;
        self.position.y -= vertical_offset;
        self.position.x = Self::x_from_horizontal_offset(hoffset, font_cache);
    }

    fn draw_cursor_for(&self, font_cache: &mut FontCache, c: char) -> rusttype::Rect<f32> {
        let mut rect = rusttype::Rect {
            min: rusttype::point(self.position.x, self.position.y),
            max: rusttype::point(self.position.x, self.position.y),
        };

        let vmetrics = font_cache.v_metrics();
        let hmetrics = font_cache.get_glyph(c).h_metrics();
        rect.max.x += match self.mode {
            VisualCursorMode::Normal => hmetrics.advance_width,
            _ => 2.0, // TODO: remove hardcoded value here.
        };
        rect.max.y += vmetrics.ascent - vmetrics.descent;

        rect
    }
}

fn create_cursor() -> Vec<Vector2<f32>> {
    let xo = 0.0;
    let yo = 0.0;
    let width = 1.0;
    let height = 1.0;
    vec![
        Vector2::new(xo, yo),
        Vector2::new(xo + width, yo),
        Vector2::new(xo + width, yo + height),
        Vector2::new(xo + width, yo + height),
        Vector2::new(xo, yo + height),
        Vector2::new(xo, yo),
    ]
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

    let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::Default,
        backends: wgpu::BackendBit::PRIMARY,
    })
    .unwrap();

    let (mut device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });

    let font_data: &[u8] = include_bytes!("../fonts/iosevka-fixed-regular.ttf");

    let collection = rusttype::FontCollection::from_bytes(font_data).unwrap();
    let font = collection.into_font().unwrap();

    let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(font_data)
        .expect("Load fonts")
        .build(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

    let vertex_shader = include_bytes!("../shaders/cursor/vertex.spirv");
    let vs_module = device.create_shader_module(
        &wgpu::read_spirv(std::io::Cursor::new(&vertex_shader[..]))
            .expect("failed to read vertex shader"),
    );

    let fragment_shader = include_bytes!("../shaders/cursor/fragment.spirv");
    let fs_module = device.create_shader_module(
        &wgpu::read_spirv(std::io::Cursor::new(&fragment_shader[..]))
            .expect("failed to read fragment shader"),
    );

    let mut sc_descriptor = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Vsync,
    };

    let cursor_vertex_data = create_cursor();
    let mut editor_view = EditorView::new(filepath, sc_descriptor.width, sc_descriptor.height);

    let cursor_vertex_buffer = device
        .create_buffer_mapped(cursor_vertex_data.len(), wgpu::BufferUsage::VERTEX)
        .fill_from_slice(&cursor_vertex_data);

    let uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
            ],
        });

    let uniform_buffer_size = mem::size_of::<UBO>() as wgpu::BufferAddress;
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: uniform_buffer_size,
        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    });

    let coordinate_uniform_buffer_size =
        mem::size_of::<CoordinateUniformBuffer>() as wgpu::BufferAddress;
    let coordinate_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: coordinate_uniform_buffer_size,
        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &uniform_bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniform_buffer,
                    range: 0..uniform_buffer_size,
                },
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &coordinate_uniform_buffer,
                    range: 0..coordinate_uniform_buffer_size,
                },
            },
        ],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&uniform_bind_group_layout],
        }),
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: None,
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[wgpu::VertexBufferDescriptor {
            stride: mem::size_of::<Vector2<f32>>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[wgpu::VertexAttributeDescriptor {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float2,
            }],
        }],
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    });

    let mut swap_chain = device.create_swap_chain(&surface, &sc_descriptor);

    // let glyph = font.glyph('s');
    let font_scale = Scale { x: 16.0, y: 16.0 };
    let mut font_cache = FontCache::new(font_scale, font_data);
    let mut ctrl_pressed = false;

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
            event::Event::WindowEvent {
                event: event::WindowEvent::ModifiersChanged(modifiers_state),
                ..
            } => {
                ctrl_pressed = modifiers_state.ctrl();
            }
            event::Event::RedrawRequested(_) => {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

                let frame = swap_chain.get_next_texture();

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
                    {
                        let ubo = UBO {
                            color: Vector4::new(1.0, 0.0, 0.0, 1.0),
                        };
                        let temp_buf = device
                            .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
                            .fill_from_slice(&[ubo]);

                        encoder.copy_buffer_to_buffer(
                            &temp_buf,
                            0,
                            &uniform_buffer,
                            0,
                            mem::size_of::<UBO>() as wgpu::BufferAddress,
                        );
                    }
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

                        let coordinate_ubo = CoordinateUniformBuffer {
                            view_projection: editor_view.view_projection(),
                            model: model,
                        };
                        let temp_buf = device
                            .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
                            .fill_from_slice(&[coordinate_ubo]);
                        encoder.copy_buffer_to_buffer(
                            &temp_buf,
                            0,
                            &coordinate_uniform_buffer,
                            0,
                            mem::size_of::<CoordinateUniformBuffer>() as wgpu::BufferAddress,
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
                    render_pass.set_pipeline(&render_pipeline);
                    render_pass.set_bind_group(0, &uniform_bind_group, &[]);
                    render_pass.set_vertex_buffers(0, &[(&cursor_vertex_buffer, 0)]);
                    render_pass.draw(0..cursor_vertex_data.len() as u32, 0..1);
                }

                // Render the text
                {
                    glyph_brush.queue(Section {
                        text: editor_view.contents(),
                        screen_position: (0.0, 0.0),
                        color: [1.0, 1.0, 1.0, 1.0],
                        scale: font_scale,
                        bounds: (sc_descriptor.width as f32, std::f32::INFINITY),
                        ..Section::default()
                    });

                    let view_proj: [f32; 16] = {
                        let mut arr: [f32; 16] = Default::default();
                        arr.copy_from_slice(editor_view.view_projection().as_slice());
                        arr
                    };

                    // Draw the text!
                    glyph_brush
                        .draw_queued_with_transform(
                            &mut device,
                            &mut encoder,
                            &frame.view,
                            view_proj,
                        )
                        .expect("Draw queued");
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
                    if y < 0.0 {
                        editor_view.scroll_down(&font_cache);
                    } else {
                        editor_view.scroll_up(&font_cache);
                    }
                    window.request_redraw();
                }
                _ => panic!("this scroll format is not yet supported"),
            },
            event::Event::DeviceEvent {
                event: event::DeviceEvent::Key(key),
                ..
            } => {
                if let None = key.virtual_keycode {
                    return;
                }

                let virtual_keycode = key.virtual_keycode.unwrap();

                if editor_view.visual_cursor.mode() == VisualCursorMode::Edit
                    && key.state == event::ElementState::Pressed
                {
                    match virtual_keycode {
                        VirtualKeyCode::A => editor_view.insert_text("a"),
                        VirtualKeyCode::B => editor_view.insert_text("b"),
                        VirtualKeyCode::C => editor_view.insert_text("c"),
                        VirtualKeyCode::D => editor_view.insert_text("d"),
                        VirtualKeyCode::E => editor_view.insert_text("e"),
                        VirtualKeyCode::F => editor_view.insert_text("f"),
                        VirtualKeyCode::G => editor_view.insert_text("g"),
                        VirtualKeyCode::H => editor_view.insert_text("h"),
                        VirtualKeyCode::I => editor_view.insert_text("i"),
                        VirtualKeyCode::J => editor_view.insert_text("j"),
                        VirtualKeyCode::K => editor_view.insert_text("k"),
                        VirtualKeyCode::L => editor_view.insert_text("l"),
                        VirtualKeyCode::M => editor_view.insert_text("m"),
                        VirtualKeyCode::N => editor_view.insert_text("n"),
                        VirtualKeyCode::O => editor_view.insert_text("o"),
                        VirtualKeyCode::P => editor_view.insert_text("p"),
                        VirtualKeyCode::Q => editor_view.insert_text("q"),
                        VirtualKeyCode::R => editor_view.insert_text("r"),
                        VirtualKeyCode::S => editor_view.insert_text("s"),
                        VirtualKeyCode::T => editor_view.insert_text("t"),
                        VirtualKeyCode::U => editor_view.insert_text("u"),
                        VirtualKeyCode::V => editor_view.insert_text("v"),
                        VirtualKeyCode::X => editor_view.insert_text("x"),
                        VirtualKeyCode::W => editor_view.insert_text("w"),
                        VirtualKeyCode::Y => editor_view.insert_text("y"),
                        VirtualKeyCode::Z => editor_view.insert_text("z"),
                        VirtualKeyCode::Key0 => editor_view.insert_text("0"),
                        VirtualKeyCode::Key1 => editor_view.insert_text("1"),
                        VirtualKeyCode::Key2 => editor_view.insert_text("2"),
                        VirtualKeyCode::Key3 => editor_view.insert_text("3"),
                        VirtualKeyCode::Key4 => editor_view.insert_text("4"),
                        VirtualKeyCode::Key5 => editor_view.insert_text("5"),
                        VirtualKeyCode::Key6 => editor_view.insert_text("6"),
                        VirtualKeyCode::Key7 => editor_view.insert_text("7"),
                        VirtualKeyCode::Key8 => editor_view.insert_text("8"),
                        VirtualKeyCode::Key9 => editor_view.insert_text("9"),
                        VirtualKeyCode::Escape => {
                            if editor_view.visual_cursor.mode() != VisualCursorMode::Normal {
                                editor_view.visual_cursor.enter_normal_mode();
                                editor_view.move_left(&mut font_cache);
                                window.request_redraw();
                            }
                        }
                        _ => (),
                    };
                    window.request_redraw();
                }

                if editor_view.visual_cursor.mode() == VisualCursorMode::Normal {
                    match virtual_keycode {
                        VirtualKeyCode::J => {
                            if key.state == event::ElementState::Pressed {
                                editor_view.move_down(&mut font_cache).map(|_| {
                                    window.request_redraw();
                                });
                            }
                        }
                        VirtualKeyCode::K => {
                            if key.state == event::ElementState::Pressed {
                                editor_view.move_up(&mut font_cache).map(|_| {
                                    window.request_redraw();
                                });
                            }
                        }
                        VirtualKeyCode::H => {
                            if key.state == event::ElementState::Pressed {
                                editor_view.move_left(&mut font_cache).map(|_| {
                                    window.request_redraw();
                                });
                            }
                        }
                        VirtualKeyCode::L => {
                            if key.state == event::ElementState::Pressed {
                                editor_view.move_right(&mut font_cache).map(|_| {
                                    window.request_redraw();
                                });
                            }
                        }
                        VirtualKeyCode::I => {
                            if key.state == event::ElementState::Pressed {
                                if editor_view.visual_cursor.mode() != VisualCursorMode::Edit {
                                    editor_view.visual_cursor.enter_edit_mode();
                                    window.request_redraw();
                                }
                            }
                        }
                        VirtualKeyCode::E => {
                            if ctrl_pressed {
                                editor_view.scroll_down(&font_cache);
                                window.request_redraw();
                            }
                        }
                        VirtualKeyCode::Y => {
                            if ctrl_pressed {
                                editor_view.scroll_up(&font_cache);
                                window.request_redraw();
                            }
                        }
                        _ => {}
                    }
                }
            }
            event::Event::WindowEvent {
                event: event::WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            _ => (),
        }
    });
}
