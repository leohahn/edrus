use super::Brush;
use core::hash::BuildHasher;
use glyph_brush::delegate_glyph_brush_builder_fns;
use glyph_brush::rusttype::{Error, Font, SharedBytes};
use glyph_brush::DefaultSectionHasher;

pub struct BrushBuilder<'a, D, H = DefaultSectionHasher> {
    inner: glyph_brush::GlyphBrushBuilder<'a, H>,
    texture_filter_method: wgpu::FilterMode,
    depth: D,
}

impl<'a, H> From<glyph_brush::GlyphBrushBuilder<'a, H>> for BrushBuilder<'a, (), H> {
    fn from(inner: glyph_brush::GlyphBrushBuilder<'a, H>) -> Self {
        BrushBuilder {
            inner,
            texture_filter_method: wgpu::FilterMode::Linear,
            depth: (),
        }
    }
}

impl<'a> BrushBuilder<'a, ()> {
    /// Specifies the default font data used to render glyphs.
    /// Referenced with `FontId(0)`, which is default.
    #[inline]
    pub fn using_font_bytes<B: Into<SharedBytes<'a>>>(font_0_data: B) -> Result<Self, Error> {
        let font = Font::from_bytes(font_0_data)?;
        Ok(Self::using_font(font))
    }

    #[inline]
    pub fn using_fonts_bytes<B, V>(font_data: V) -> Result<Self, Error>
    where
        B: Into<SharedBytes<'a>>,
        V: Into<Vec<B>>,
    {
        let fonts = font_data
            .into()
            .into_iter()
            .map(Font::from_bytes)
            .collect::<Result<Vec<Font>, Error>>()?;

        Ok(Self::using_fonts(fonts))
    }

    /// Specifies the default font used to render glyphs.
    /// Referenced with `FontId(0)`, which is default.
    #[inline]
    pub fn using_font(font_0: Font<'a>) -> Self {
        Self::using_fonts(vec![font_0])
    }

    pub fn using_fonts<V: Into<Vec<Font<'a>>>>(fonts: V) -> Self {
        BrushBuilder {
            inner: glyph_brush::GlyphBrushBuilder::using_fonts(fonts),
            texture_filter_method: wgpu::FilterMode::Linear,
            depth: (),
        }
    }
}

impl<'a, D, H: BuildHasher> BrushBuilder<'a, D, H> {
    delegate_glyph_brush_builder_fns!(inner);

    /// Sets the texture filtering method.
    pub fn texture_filter_method(mut self, filter_method: wgpu::FilterMode) -> Self {
        self.texture_filter_method = filter_method;
        self
    }

    /// Sets the section hasher. `GlyphBrush` cannot handle absolute section
    /// hash collisions so use a good hash algorithm.
    ///
    /// This hasher is used to distinguish sections, rather than for hashmap
    /// internal use.
    ///
    /// Defaults to [seahash](https://docs.rs/seahash).
    pub fn section_hasher<T: BuildHasher>(self, section_hasher: T) -> BrushBuilder<'a, D, T> {
        BrushBuilder {
            inner: self.inner.section_hasher(section_hasher),
            texture_filter_method: self.texture_filter_method,
            depth: self.depth,
        }
    }

    /// Sets the depth stencil.
    pub fn depth_stencil_state(
        self,
        depth_stencil_state: wgpu::DepthStencilStateDescriptor,
    ) -> BrushBuilder<'a, wgpu::DepthStencilStateDescriptor, H> {
        BrushBuilder {
            inner: self.inner,
            texture_filter_method: self.texture_filter_method,
            depth: depth_stencil_state,
        }
    }
}

impl<'a, H: BuildHasher> BrushBuilder<'a, (), H> {
    /// Builds a `GlyphBrush` using the given `wgpu::Device` that can render
    /// text for texture views with the given `render_format`.
    pub fn build(
        self,
        device: &wgpu::Device,
        render_format: wgpu::TextureFormat,
    ) -> Brush<'a, (), H> {
        Brush::<(), H>::new(
            device,
            self.texture_filter_method,
            render_format,
            self.inner,
        )
    }
}

impl<'a, H: BuildHasher> BrushBuilder<'a, wgpu::DepthStencilStateDescriptor, H> {
    /// Builds a `GlyphBrush` using the given `wgpu::Device` that can render
    /// text for texture views with the given `render_format`.
    pub fn build(
        self,
        device: &wgpu::Device,
        render_format: wgpu::TextureFormat,
    ) -> Brush<'a, wgpu::DepthStencilStateDescriptor, H> {
        Brush::<wgpu::DepthStencilStateDescriptor, H>::new(
            device,
            self.texture_filter_method,
            render_format,
            self.depth,
            self.inner,
        )
    }
}
