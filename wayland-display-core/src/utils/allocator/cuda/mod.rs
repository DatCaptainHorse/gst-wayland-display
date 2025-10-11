use crate::cuda_call;
use ffi::{CUDA_SUCCESS, CUgraphicsResource, cuda_result_to_string};
use ffi::{GstCudaContext, PFN_eglDestroyImageKHR, eglGetProcAddress};
use gst::glib::ffi as glib_ffi;
use gst::glib::translate::ToGlibPtr;
use gst::query::Allocation;
use gst::{Buffer as GstBuffer, Element, QueryRef};
use gst_video::{VideoFormat, VideoInfoDmaDrm, VideoMeta};
use smithay::backend::allocator::Buffer;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::egl::ffi::egl::types::{EGLDisplay, EGLImageKHR, EGLint};
use std::os::fd::AsRawFd;
use std::os::raw::{c_char, c_uint};
use std::ptr;

mod ffi;

// Helper to load EGL extension functions
#[derive(Debug, Clone)]
pub struct EglExtensions {
    pub create_image: ffi::PFN_eglCreateImageKHR,
    pub destroy_image: PFN_eglDestroyImageKHR,
}

impl EglExtensions {
    unsafe fn load() -> Option<Self> {
        let create_image_name = b"eglCreateImageKHR\0";
        let destroy_image_name = b"eglDestroyImageKHR\0";

        let create_image_ptr =
            unsafe { eglGetProcAddress(create_image_name.as_ptr() as *const c_char) };
        let destroy_image_ptr =
            unsafe { eglGetProcAddress(destroy_image_name.as_ptr() as *const c_char) };

        if create_image_ptr.is_null() || destroy_image_ptr.is_null() {
            return None;
        }

        Some(EglExtensions {
            create_image: unsafe { std::mem::transmute(create_image_ptr) },
            destroy_image: unsafe { std::mem::transmute(destroy_image_ptr) },
        })
    }

    pub fn new() -> Option<Self> {
        unsafe { EglExtensions::load() }
    }
}

#[derive(Debug, Clone)]
pub struct EGLImage {
    image: EGLImageKHR,
    destroy_fn: PFN_eglDestroyImageKHR,
}

impl EGLImage {
    pub fn from(
        dmabuf: &Dmabuf,
        egl_display: &EGLDisplay,
        egl_ext: &EglExtensions,
    ) -> Result<Self, String> {
        // Get dmabuf properties
        let width = dmabuf.width();
        let height = dmabuf.height();
        let fourcc = dmabuf.format().code as u32;

        // Get modifier if available
        let modifier: u64 = dmabuf.format().modifier.into();
        let modifier_lo = (modifier & 0xFFFFFFFF) as EGLint;
        let modifier_hi = ((modifier >> 32) & 0xFFFFFFFF) as EGLint;

        // Build EGL attribute list for DMA-BUF import
        let mut attribs = [
            ffi::EGL_WIDTH,
            width as EGLint,
            ffi::EGL_HEIGHT,
            height as EGLint,
            ffi::EGL_LINUX_DRM_FOURCC_EXT,
            fourcc as EGLint,
        ]
        .to_vec();

        let offsets = dmabuf.offsets().map(|o| o as usize).collect::<Vec<_>>();

        let strides = dmabuf.strides().map(|s| s as i32).collect::<Vec<_>>();

        for (idx, handle) in dmabuf.handles().enumerate() {
            let fd = handle.as_raw_fd();
            // Add to attribs the current plane data
            if idx == 0 {
                attribs.extend_from_slice(&[
                    ffi::EGL_DMA_BUF_PLANE0_FD_EXT,
                    fd,
                    ffi::EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                    offsets[idx] as EGLint,
                    ffi::EGL_DMA_BUF_PLANE0_PITCH_EXT,
                    strides[idx],
                    ffi::EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                    modifier_lo,
                    ffi::EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                    modifier_hi,
                ]);
            } else if idx == 1 {
                attribs.extend_from_slice(&[
                    ffi::EGL_DMA_BUF_PLANE1_FD_EXT,
                    fd,
                    ffi::EGL_DMA_BUF_PLANE1_OFFSET_EXT,
                    offsets[idx] as EGLint,
                    ffi::EGL_DMA_BUF_PLANE1_PITCH_EXT,
                    strides[idx],
                    ffi::EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
                    modifier_lo,
                    ffi::EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
                    modifier_hi,
                ]);
            }
        }

        attribs.push(ffi::EGL_NONE);

        let egl_image = unsafe {
            (egl_ext.create_image)(
                *egl_display,
                ptr::null_mut(),
                ffi::EGL_LINUX_DMA_BUF_EXT,
                ptr::null_mut(),
                attribs.as_ptr(),
            )
        };
        if egl_image != ffi::EGL_NO_IMAGE_KHR {
            Ok(EGLImage {
                image: egl_image,
                destroy_fn: egl_ext.destroy_image,
            })
        } else {
            Err("Failed to create EGLImage".into())
        }
    }
}

impl Drop for EGLImage {
    fn drop(&mut self) {
        unsafe {
            (self.destroy_fn)(ffi::eglGetCurrentDisplay(), self.image);
        }
    }
}

pub const CAPS_FEATURE_MEMORY_CUDA_MEMORY: &str = "memory:CUDAMemory"; // TODO: get it from FFI from gstcudamemory.h (https://github.com/GStreamer/gstreamer/blob/9d6abcc18cc9a60a212966a2daaf4a1af243f5da/subprojects/gst-plugins-bad/gst-libs/gst/cuda/gstcudamemory.h#L113-L121)

pub fn init_cuda() -> Result<(), String> {
    static mut INITIALIZED: bool = false;
    if !unsafe { INITIALIZED } {
        cuda_call!(ffi::cuInit(0))?;
        unsafe {
            ffi::gst_cuda_load_library();
            ffi::gst_cuda_memory_init_once();
            INITIALIZED = true;
        }
        Ok(())
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub struct CUDAContext {
    ptr: *mut GstCudaContext,
    stream: Option<StreamHandle>,
}

impl Drop for CUDAContext {
    fn drop(&mut self) {
        unsafe {
            gst::ffi::gst_object_unref(self.ptr as *mut gst::ffi::GstObject);
        }
    }
}

impl Clone for CUDAContext {
    fn clone(&self) -> Self {
        unsafe {
            gst::ffi::gst_object_ref(self.ptr as *mut gst::ffi::GstObject);
        }
        CUDAContext {
            ptr: self.ptr,
            stream: self.stream.clone(),
        }
    }
}

#[derive(Debug)]
pub struct StreamHandle {
    stream: ffi::GstCudaStreamHandle,
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        unsafe {
            ffi::gst_cuda_stream_unref(self.stream);
        }
    }
}

impl Clone for StreamHandle {
    fn clone(&self) -> Self {
        unsafe {
            ffi::gst_cuda_stream_ref(self.stream);
        }
        StreamHandle {
            stream: self.stream,
        }
    }
}

#[derive(Debug)]
pub struct CUDABufferPool {
    pool: ffi::GstBufferPool,
}

impl CUDABufferPool {
    pub fn new(cudacontext: &CUDAContext) -> Result<Self, String> {
        let pool = unsafe { ffi::gst_cuda_buffer_pool_new(cudacontext.ptr) };
        if pool.is_null() {
            Err("Failed to create CUDA buffer pool".into())
        } else {
            Ok(CUDABufferPool { pool })
        }
    }

    pub fn configure(
        &self,
        caps: &gst::Caps,
        stream_handle: &StreamHandle,
        size: u32,
        min_buffers: u32,
        max_buffers: u32,
    ) -> Result<(), String> {
        let config = unsafe {
            gst::ffi::gst_buffer_pool_get_config(self.pool as *mut gst::ffi::GstBufferPool)
        };
        if config.is_null() {
            return Err("Failed to get buffer pool config".into());
        }

        // TODO: support getting the stream handler too here
        //       https://github.com/GStreamer/gstreamer/blob/c5a470e5164ce7fa8fd5fa80650d9ee35ce214d8/subprojects/gst-plugins-bad/sys/nvcodec/gstcudaconvertscale.c#L1293-L1300
        //       ultimately it doesn't matter for us because we are creating a new pool every time

        // Configure the pool
        unsafe {
            // Set the CUDA stream in the config
            ffi::gst_buffer_pool_config_set_cuda_stream(config, stream_handle.stream);

            gst::ffi::gst_buffer_pool_config_add_option(
                config,
                ffi::GST_BUFFER_POOL_OPTION_VIDEO_META.as_ptr() as *const c_char,
            );
            gst::ffi::gst_buffer_pool_config_set_params(
                config,
                caps.to_glib_none().0,
                size,
                min_buffers,
                max_buffers,
            );
        }

        // Set the configuration
        let result = unsafe {
            gst::ffi::gst_buffer_pool_set_config(self.pool as *mut gst::ffi::GstBufferPool, config)
        };
        if result == glib_ffi::GFALSE {
            Err("Failed to set buffer pool config".into())
        } else {
            Ok(())
        }
    }

    pub fn get_updated_size(&self) -> Result<u32, String> {
        let config = unsafe {
            gst::ffi::gst_buffer_pool_get_config(self.pool as *mut gst::ffi::GstBufferPool)
        };
        if config.is_null() {
            return Err("Failed to get buffer pool config".into());
        }

        let mut size = 0;
        unsafe {
            gst::ffi::gst_buffer_pool_config_get_params(
                config,
                ptr::null_mut(),
                &mut size,
                ptr::null_mut(),
                ptr::null_mut(),
            );
        }
        Ok(size)
    }

    pub fn activate(&self) -> Result<(), String> {
        let result = unsafe {
            gst::ffi::gst_buffer_pool_set_active(
                self.pool as *mut gst::ffi::GstBufferPool,
                glib_ffi::GTRUE,
            )
        };
        if result == glib_ffi::GFALSE {
            Err("Failed to activate buffer pool".into())
        } else {
            Ok(())
        }
    }

    pub fn set_nth_allocation_pool(
        &self,
        query: &mut Allocation,
        idx: u32,
        size: u32,
        min_buffers: u32,
        max_buffers: u32,
    ) {
        unsafe {
            gst::ffi::gst_query_set_nth_allocation_pool(
                query.as_mut_ptr(),
                idx,
                self.pool as *mut gst::ffi::GstBufferPool,
                size,
                min_buffers,
                max_buffers,
            );
        }
    }

    pub fn add_allocation_pool(
        &self,
        query: &mut Allocation,
        size: u32,
        min_buffers: u32,
        max_buffers: u32,
    ) {
        unsafe {
            gst::ffi::gst_query_add_allocation_pool(
                query.as_mut_ptr(),
                self.pool as *mut gst::ffi::GstBufferPool,
                size,
                min_buffers,
                max_buffers,
            )
        }
    }
}

impl Drop for CUDABufferPool {
    fn drop(&mut self) {
        unsafe {
            // TODO: if this is the last reference we should call:
            //      ffi::gst_buffer_pool_set_active(self.pool, glib_ffi::GFALSE);
            gst::glib::gobject_ffi::g_object_unref(
                self.pool as *mut gst::glib::gobject_ffi::GObject,
            );
        }
    }
}

impl Clone for CUDABufferPool {
    fn clone(&self) -> Self {
        unsafe {
            gst::glib::gobject_ffi::g_object_ref(self.pool as *mut gst::glib::gobject_ffi::GObject);
        }
        CUDABufferPool { pool: self.pool }
    }
}

unsafe impl Send for CUDAContext {}
unsafe impl Sync for CUDAContext {}
unsafe impl Send for CUDABufferPool {}
unsafe impl Sync for CUDABufferPool {}
unsafe impl Send for StreamHandle {}
unsafe impl Sync for StreamHandle {}

impl CUDAContext {
    pub fn new(device_id: c_uint) -> Result<Self, String> {
        let ptr = unsafe { ffi::gst_cuda_context_new(device_id) };
        if ptr.is_null() {
            return Err("Failed to create CUDA context".into());
        }

        // Create a CUDA stream
        let stream = unsafe { ffi::gst_cuda_stream_new(ptr) };

        Ok(CUDAContext {
            ptr,
            stream: if stream.is_null() {
                None
            } else {
                Some(StreamHandle { stream })
            },
        })
    }

    pub fn as_ptr(&self) -> *mut GstCudaContext {
        self.ptr
    }

    pub fn stream(&self) -> Option<&StreamHandle> {
        self.stream.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct CUDAImage {
    cuda_graphic_resource: CUgraphicsResource,
}

impl CUDAImage {
    pub fn from(egl_image: &EGLImage, cuda_context: &CUDAContext) -> Result<Self, String> {
        let _cuda_context_guard = ffi::CudaContextGuard::new(cuda_context)?;

        // Let's import the EGLImage into CUDA
        let mut cuda_resource: CUgraphicsResource = ptr::null_mut();
        cuda_call!(ffi::cuGraphicsEGLRegisterImage(
            &mut cuda_resource,
            egl_image.image,
            0, // flags (0 = read/write)
        ))?;
        Ok(CUDAImage {
            cuda_graphic_resource: cuda_resource,
        })
    }

    pub fn to_gst_buffer(
        &self,
        dma_video_info: VideoInfoDmaDrm,
        cuda_context: &CUDAContext,
        buffer_pool: Option<&CUDABufferPool>,
    ) -> Result<GstBuffer, Box<dyn std::error::Error>> {
        tracing::info!("to_gst_buffer called with pool: {}", buffer_pool.is_some());

        let _cuda_context_guard = ffi::CudaContextGuard::new(cuda_context)?;

        let mut egl_frame = unsafe { std::mem::zeroed() };
        cuda_call!(ffi::cuGraphicsResourceGetMappedEglFrame(
            &mut egl_frame,
            self.cuda_graphic_resource,
            0,
            0
        ))?;

        // Create Gstreamer memory
        let gst_memory = ffi::alloc_copy_gst_memory(
            egl_frame,
            cuda_context,
            dma_video_info.clone(),
            buffer_pool.map(|p| p.pool),
        )?;

        // Create the buffer using GStreamer Rust bindings
        let mut buffer = gst::Buffer::new();
        {
            let buffer_ref = buffer.get_mut().unwrap();
            buffer_ref.append_memory(gst_memory);

            let video_format = match VideoFormat::from_fourcc(dma_video_info.fourcc()) {
                VideoFormat::Unknown => {
                    tracing::debug!(
                        "Failed to convert fourcc to video format: {:?}",
                        dma_video_info.fourcc()
                    );
                    VideoFormat::Bgrx // Fallback
                }
                format => format,
            };

            VideoMeta::add(
                buffer_ref,
                gst_video::VideoFrameFlags::empty(),
                video_format,
                dma_video_info.width(),
                dma_video_info.height(),
                // TODO: Add stride and offset metadata here
            )?;
        }

        Ok(buffer)
    }
}

impl Drop for CUDAImage {
    fn drop(&mut self) {
        unsafe {
            // Unmap before unregister (if it was ever mapped)
            // This might fail if not mapped, but that's ok
            let _ = ffi::cuGraphicsUnmapResources(
                1,
                &self.cuda_graphic_resource as *const _ as *mut _,
                ptr::null_mut()
            );

            ffi::cuGraphicsUnregisterResource(self.cuda_graphic_resource);
        }
    }
}

pub fn gst_cuda_handle_context_query_wrapped(
    element: &Element,
    query: &mut QueryRef,
    cuda_context: &CUDAContext,
) -> bool {
    let result = unsafe {
        ffi::gst_cuda_handle_context_query(
            element.to_glib_none().0,
            query.as_mut_ptr(),
            cuda_context.ptr,
        )
    };
    result == glib_ffi::GTRUE
}
