extern crate cgmath;
extern crate edrus;
extern crate env_logger;
extern crate log;
extern crate nalgebra as na;
extern crate rusttype;
extern crate shaderc;
extern crate wgpu;
extern crate wgpu_glyph;
extern crate winit;
extern crate zerocopy;

// use na::{Matrix3, Matrix4, Point3, Point4, Vector2, Vector3, Vector4};
use std::mem;
use wgpu_glyph::{GlyphBrushBuilder, Scale, Section};
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

// #[derive(Debug, Copy, Clone)]
// #[repr(C)]
// struct Vec2([f32; 2]);

// impl Vec2 {
//     fn new(x: f32, y: f32) -> Vec2 {
//         Vec2([x, y])
//     }
// }

// #[derive(Debug, Copy, Clone)]
// #[repr(C)]
// struct Vec4([f32; 4]);

// impl Vec4 {
//     fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
//         Vec4([x, y, z, w])
//     }
// }

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct UBO {
    color: cgmath::Vector4<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CoordinateUniformBuffer {
    view_projection: cgmath::Matrix4<f32>,
    model: cgmath::Matrix4<f32>,
}

fn create_cursor(origin: (f32, f32)) -> Vec<cgmath::Vector2<f32>> {
    let (xo, yo) = dbg!(origin);
    let width = 100.0;
    let height = 100.0;
    dbg!(vec![
        cgmath::Vector2::new(xo, yo),
        cgmath::Vector2::new(xo + width, yo),
        cgmath::Vector2::new(xo + width, yo + height),
        cgmath::Vector2::new(xo + width, yo + height),
        cgmath::Vector2::new(xo, yo + height),
        cgmath::Vector2::new(xo, yo),
    ])
}

fn create_view_projection_matrix(
    width: u32,
    height: u32,
    eye: cgmath::Point3<f32>,
) -> cgmath::Matrix4<f32> {
    dbg!(width);
    dbg!(height);
    // let target = eye + cgmath::Vector3::new(0.0, 0.0, -1.0);
    // let eye = cgmath::Point3::new(0.0, 0.0, 1.0);
    let target = eye - cgmath::Vector3::new(0.0, 0.0, 1.0);
    // let view = cgmath::Matrix4::look_at_rh(&eye, &target, &cgmath::Vector3::new(0.0, 1.0, 0.0));
    let view = cgmath::Matrix4::look_at(eye, target, cgmath::Vector3::new(0.0, 1.0, 0.0));
    let projection = cgmath::ortho(
        0.0,           // left
        width as f32,  // right
        height as f32, // bottom
        0.0,           // top
        0.1,           // znear
        100.0,         // zfar
    );
    // opengl_to_wgpu_matrix() * projection * view
    OPENGL_TO_WGPU_MATRIX * projection * view
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

    layout (location = 0) in vec2 a_pos;

    layout (set = 0, binding = 1) uniform CoordinateUniformBuffer {
        mat4 u_view_projection;
        mat4 u_model;
    };

    out gl_PerVertex {
        vec4 gl_Position;
    };

    void main() {
        gl_Position = u_view_projection * u_model * vec4(a_pos, 0.0, 1.0);
    }
    "#;

    let fragment_source = r#"
    #version 450

    layout (set = 0, binding = 0) uniform UBO {
        vec4 u_color;
    };

    layout (location = 0) out vec4 o_color;

    void main() {
        o_color = u_color;
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
        .expect("failed to compile vertex shader");

    let fragment_shader = compiler
        .compile_into_spirv(
            fragment_source,
            shaderc::ShaderKind::Fragment,
            "fragment_shader.glsl",
            "main",
            None,
        )
        .expect("failed to compile fragment shader");

    // let vertex_shader = include_bytes!("shader.vert.spv");
    let vs_module = device.create_shader_module(
        &wgpu::read_spirv(std::io::Cursor::new(vertex_shader.as_binary_u8()))
            .expect("failed to read vertex shader"),
    );

    // let fragment_shader = include_bytes!("shader.frag.spv");
    let fs_module = device.create_shader_module(
        &wgpu::read_spirv(std::io::Cursor::new(fragment_shader.as_binary_u8()))
            .expect("failed to read fragment shader"),
    );

    let cursor_vertex_data = create_cursor((130.0, 130.0));
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
        dbg!(mem::size_of::<CoordinateUniformBuffer>() as wgpu::BufferAddress);
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
            stride: mem::size_of::<cgmath::Vector2<f32>>() as wgpu::BufferAddress,
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

    let mut sc_descriptor = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Vsync,
    };

    // let glyph = font.glyph('s');

    let mut swap_chain = device.create_swap_chain(&surface, &sc_descriptor);

    let mut eye = cgmath::Point3::new(50.0, 50.0, 1.0);
    let mut view_projection_matrix =
        create_view_projection_matrix(sc_descriptor.width, sc_descriptor.height, dbg!(eye));

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
                view_projection_matrix =
                    create_view_projection_matrix(sc_descriptor.width, sc_descriptor.height, eye);
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
                // Draw the cursor before everything else, since it should be behind
                // the text.
                {
                    {
                        let ubo = UBO {
                            color: cgmath::Vector4::new(1.0, 0.0, 0.0, 1.0),
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
                        use cgmath::SquareMatrix;
                        let coordinate_ubo = CoordinateUniformBuffer {
                            view_projection: view_projection_matrix,
                            // view_projection: cgmath::Matrix4::identity(),
                            model: cgmath::Matrix4::identity(),
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

                glyph_brush.queue(Section {
                    text: buffer.contents(),
                    screen_position: (30.0, 30.0),
                    color: [1.0, 1.0, 1.0, 1.0],
                    scale: Scale { x: 15.0, y: 15.0 },
                    //bounds: (size.width as f32, size.height as f32),
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
                        // view_projection_matrix.into::<[[f32; 4]; 4]>(),
                    )
                    .expect("Draw queued");

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
                        event::VirtualKeyCode::A => {
                            eye += cgmath::Vector3::new(3.0, 0.0, 0.0);
                            view_projection_matrix = create_view_projection_matrix(
                                sc_descriptor.width,
                                sc_descriptor.height,
                                eye,
                            );
                            window.request_redraw();
                        }
                        event::VirtualKeyCode::D => {
                            eye -= cgmath::Vector3::new(3.0, 0.0, 0.0);
                            view_projection_matrix = create_view_projection_matrix(
                                sc_descriptor.width,
                                sc_descriptor.height,
                                eye,
                            );
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

// fn orthographic_projection(width: u32, height: u32) -> [f32; 16] {
//     #[cfg_attr(rustfmt, rustfmt_skip)]
//     [
//         2.0 / width as f32, 0.0, 0.0, 0.0,
//         0.0, 2.0 / height as f32, 0.0, 0.0,
//         0.0, 0.0, 1.0, 0.0,
//         -1.0, -1.0, 0.0, 1.0,
//     ]
// }
