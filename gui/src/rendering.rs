use nalgebra::Vector2;
use std::mem;
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferAddress, BufferDescriptor, CommandEncoder, Device,
    RenderPipeline, ShaderModule, ShaderStage, VertexBufferDescriptor,
};

pub struct UniformBuffer {
    buffer: Buffer,
    size: BufferAddress,
}

impl UniformBuffer {
    pub fn new<T>(device: &Device) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: mem::size_of::<T>() as BufferAddress,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });
        Self {
            buffer: buffer,
            size: mem::size_of::<T>() as BufferAddress,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn size(&self) -> BufferAddress {
        self.size
    }

    pub fn fill_buffer(&self, device: &Device, encoder: &mut CommandEncoder, buffer_object: &[u8]) {
        let temp_buf = device.create_buffer_with_data(buffer_object, wgpu::BufferUsage::COPY_SRC);
        encoder.copy_buffer_to_buffer(&temp_buf, 0, &self.buffer, 0, self.size());
    }
}

pub enum ShaderBinding<'a> {
    Uniform {
        index: u32,
        uniform_buffer: &'a UniformBuffer,
        visibility: ShaderStage,
    },
}

pub struct Shader {
    vertex: ShaderModule,
    fragment: ShaderModule,
    bind_group: BindGroup,
    bind_group_layout: BindGroupLayout,
    vertex_buffer_descriptor: VertexBufferDescriptor<'static>,
}

impl Shader {
    pub fn new(
        device: &Device,
        vertex_bytes: &[u8],
        fragment_bytes: &[u8],
        bindings: &[ShaderBinding],
    ) -> Self {
        let vertex = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(&vertex_bytes[..])).expect("failed to read vertex shader"),
        );
        let fragment = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(&fragment_bytes[..]))
                .expect("failed to read fragment shader"),
        );

        let mut descriptor_bindings = Vec::new();
        let mut bind_group_bindings = Vec::new();

        for binding in bindings {
            match binding {
                ShaderBinding::Uniform {
                    index,
                    uniform_buffer,
                    visibility,
                } => {
                    descriptor_bindings.push(wgpu::BindGroupLayoutEntry {
                        binding: *index,
                        visibility: *visibility,
                        ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                    });

                    bind_group_bindings.push(wgpu::Binding {
                        binding: *index,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: uniform_buffer.buffer(),
                            range: 0..uniform_buffer.size(),
                        },
                    });
                }
            }
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            bindings: &descriptor_bindings[..],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            bindings: &bind_group_bindings[..],
        });

        Self {
            vertex: vertex,
            fragment: fragment,
            bind_group_layout: bind_group_layout,
            bind_group: bind_group,
            vertex_buffer_descriptor: wgpu::VertexBufferDescriptor {
                stride: mem::size_of::<Vector2<f32>>() as BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                // TODO: remove hardcoded shader layout information.
                attributes: &[wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float2,
                }],
            },
        }
    }

    pub fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }
}

pub struct WidgetPipeline {
    render_pipeline: RenderPipeline,
}

impl WidgetPipeline {
    pub fn new(device: &Device, shader: &Shader) -> Self {
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&shader.bind_group_layout],
            }),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &shader.vertex,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &shader.fragment,
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
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[shader.vertex_buffer_descriptor.clone()],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self {
            render_pipeline: render_pipeline,
        }
    }

    pub fn render_pipeline(&self) -> &RenderPipeline {
        &self.render_pipeline
    }
}
