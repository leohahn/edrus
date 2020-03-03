extern crate edrus;
extern crate env_logger;
extern crate log;
extern crate rusttype;
extern crate shaderc;
extern crate wgpu;
extern crate wgpu_glyph;
extern crate winit;
extern crate zerocopy;

use wgpu_glyph::{GlyphBrushBuilder, Scale, Section};
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use zerocopy::AsBytes;

#[derive(Debug, Clone, AsBytes)]
#[repr(C)]
struct Vec4([f32; 4]);

impl Vec4 {
    fn new(x: f32, y: f32, z: f32) -> Vec4 {
        Vec4([x, y, z, 1.0])
    }
}

fn create_cursor() -> (Vec<Vec4>, Vec<u16>) {
    let vertex_data = [
        Vec4::new(0.0, 0.0, 0.0),
        Vec4::new(5.0, 0.0, 0.0),
        Vec4::new(5.0, 5.0, 0.0),
        Vec4::new(0.0, 5.0, 0.0),
    ];

    let index_data: &[u16] = &[0, 1, 2, 2, 3, 0];
    (vertex_data.to_vec(), index_data.to_vec())
}

fn main() {
    std::env::set_var("RUST_LOG", "edrus=debug");
    env_logger::init();

    let filepath = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            log::error!("failed to get filepath from first argument");
            std::process::exit(1);
        }
    };

    // let file_contents = std::fs::read_to_string(filepath).unwrap();
    // let mut table = edrus::text_buffer::SimplePieceTable::new(file_contents);
    let mut buffer = edrus::buffer::Buffer::new(filepath).unwrap();

    let event_loop = EventLoop::new();

    let (window, size, surface) = {
        let window = WindowBuilder::new()
            .with_title("edrus")
            .build(&event_loop)
            .unwrap();
        let size = window.inner_size();
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

    let font_data: &[u8] = include_bytes!("consola.ttf");

    let collection = rusttype::FontCollection::from_bytes(font_data).unwrap();
    let font = collection.into_font().unwrap();

    let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(font_data)
        .build(&mut device, wgpu::TextureFormat::Bgra8UnormSrgb);

    let vertex_source = r#"
    #version 450

    out gl_PerVertex {
        vec4 gl_Position;
    };

    layout(location = 0) in vec4 a_pos;

    void main() {
        gl_Position = a_pos;
    }
    "#;

    let fragment_source = r#"
    #version 450

    layout(binding = 0) uniform UBO {
        vec4 color;
    } ubo;
    layout(location = 0) out vec4 o_color;

    void main() {
        o_color = ubo.color;
    }
    "#;

    let mut compiler = shaderc::Compiler::new().unwrap();

    let vertex_shader = compiler
        .compile_into_spirv(
            vertex_source,
            shaderc::ShaderKind::Vertex,
            "vertex_shader.glsl",
            "main",
            None,
        )
        .unwrap();

    let fragment_shader = compiler
        .compile_into_spirv(
            fragment_source,
            shaderc::ShaderKind::Fragment,
            "fragment_shader.glsl",
            "main",
            None,
        )
        .unwrap();

    let vs = include_bytes!("shader.vert.spv");
    let vs_module =
        device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs[..])).unwrap());

    let fs = include_bytes!("shader.frag.spv");
    let fs_module =
        device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&fs[..])).unwrap());

    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { bindings: &[] });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
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
        vertex_buffers: &[],
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    });

    let mut sc_descriptor = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Vsync,
    };

    let glyph = font.glyph('s');

    let mut swap_chain = device.create_swap_chain(&surface, &sc_descriptor);

    let (cursor_vertex_data, cursor_index_data) = create_cursor();
    let cursor_vertex_buffer = {
        let mapped = device.create_buffer_mapped(
            cursor_vertex_data.as_bytes().len(),
            wgpu::BufferUsage::VERTEX,
        );
        mapped.data.copy_from_slice(cursor_vertex_data.as_bytes());
        mapped.finish()
    };
    let cursor_index_buffer = {
        let mapped = device
            .create_buffer_mapped(cursor_index_data.as_bytes().len(), wgpu::BufferUsage::INDEX);
        mapped.data.copy_from_slice(cursor_index_data.as_bytes());
        mapped.finish()
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
                window.request_redraw();
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

                glyph_brush.queue(Section {
                    text: buffer.contents(),
                    screen_position: (30.0, 30.0),
                    color: [1.0, 1.0, 1.0, 1.0],
                    scale: Scale { x: 15.0, y: 15.0 },
                    bounds: (size.width as f32, size.height as f32),
                    ..Section::default()
                });

                // Draw the text!
                glyph_brush
                    .draw_queued(
                        &mut device,
                        &mut encoder,
                        &frame.view,
                        // size.width.round() as u32,
                        // size.height.round() as u32,
                        sc_descriptor.width,
                        sc_descriptor.height,
                    )
                    .expect("Draw queued");

                {
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
                    render_pass.set_bind_group(0, &bind_group, &[]);
                    render_pass.set_index_buffer(&cursor_index_buffer, 0);
                    render_pass.set_vertex_buffers(0, &[(&cursor_vertex_buffer, 0)]);
                }

                queue.submit(&[encoder.finish()]);
            }
            event::Event::DeviceEvent {
                event: event::DeviceEvent::Key(key),
                ..
            } => {
                if let Some(vk) = key.virtual_keycode {
                    match vk {
                        event::VirtualKeyCode::J => {
                            buffer.move_down();
                            window.request_redraw();
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
            _ => {}
        }
    });
}

fn orthographic_projection(width: u32, height: u32) -> [f32; 16] {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    [
        2.0 / width as f32, 0.0, 0.0, 0.0,
        0.0, 2.0 / height as f32, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -1.0, -1.0, 0.0, 1.0,
    ]
}
