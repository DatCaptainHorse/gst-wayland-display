#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::utils::allocator::cuda::CUDAContext;
use gst::ffi as gst_ffi;
use gst::ffi::{GstElement, GstQuery};
use gst::glib::ffi as glib_ffi;
use gst_video::VideoInfoDmaDrm;
use gst_video::glib::translate::ToGlibPtr;
use smithay::backend::egl::ffi::egl::types::{EGLDisplay, EGLImageKHR, EGLint};
use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;

pub(crate) type GstCudaContext = *mut c_void;

#[macro_export]
macro_rules! cuda_call {
    ($expression:expr) => {{
        let result = unsafe { $expression };
        if result != CUDA_SUCCESS {
            Err(format!(
                "CUDA error: {} (code: {})",
                cuda_result_to_string(result),
                result
            ))
        } else {
            Ok(())
        }
    }};
}

type GstCudaStream = *mut c_void;
pub(crate) type GstBufferPool = *mut c_void;
pub(crate) type GstCudaStreamHandle = *mut c_void;

#[repr(C)]
pub(crate) struct CUeglFrame {
    pub(crate) frame: CUeglFrameUnion,
    pub(crate) width: c_uint,
    pub(crate) height: c_uint,
    pub(crate) depth: c_uint,
    pub(crate) pitch: c_uint,
    pub(crate) plane_count: c_uint,
    pub(crate) num_channels: c_uint,
    // Followings are ENUMS
    pub(crate) frame_type: c_uint,
    pub(crate) egl_color_format: c_uint,
    pub(crate) cu_format: c_uint,
}

#[repr(C)]
pub(crate) union CUeglFrameUnion {
    pub p_array: [CUarray; MAX_PLANES],
    pub p_pitch: [*mut c_void; MAX_PLANES],
}

const MAX_PLANES: usize = 3;

// CUDA driver API types
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUstream = *mut c_void;
type CUdeviceptr = u64;
type CUarray = *mut c_void;
pub(crate) type CUgraphicsResource = *mut c_void;
type CUresult = c_uint;

type CUtexObject = u64;
type CUfunction = *mut c_void;
type CUmodule = *mut c_void;

const CU_TR_FILTER_MODE_POINT: c_uint = 0;
const CU_TR_ADDRESS_MODE_CLAMP: c_uint = 1;

#[repr(C)]
struct CUDA_RESOURCE_DESC {
    resType: c_uint,
    res: CUDA_RESOURCE_DESC_RES,
    flags: c_uint,
}

#[repr(C)]
union CUDA_RESOURCE_DESC_RES {
    array: CUarray,
    _padding: [u64; 16], // Large enough for any union member
}

#[repr(C)]
struct CUDA_TEXTURE_DESC {
    addressMode: [c_uint; 3],
    filterMode: c_uint,
    flags: c_uint,
    maxAnisotropy: c_uint,
    mipmapFilterMode: c_uint,
    mipmapLevelBias: f32,
    minMipmapLevelClamp: f32,
    maxMipmapLevelClamp: f32,
    borderColor: [f32; 4],
    _reserved: [c_int; 12],
}

// CUDA constants
pub(crate) const CUDA_SUCCESS: CUresult = 0;

// EGL constants
pub(crate) const EGL_NO_IMAGE_KHR: EGLImageKHR = ptr::null_mut();
pub(crate) const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
pub(crate) const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
pub(crate) const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
pub(crate) const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;
pub(crate) const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EGLint = 0x3443;
pub(crate) const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EGLint = 0x3444;
pub(crate) const EGL_DMA_BUF_PLANE1_FD_EXT: EGLint = 0x3275;
pub(crate) const EGL_DMA_BUF_PLANE1_OFFSET_EXT: EGLint = 0x3276;
pub(crate) const EGL_DMA_BUF_PLANE1_PITCH_EXT: EGLint = 0x3277;
pub(crate) const EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT: EGLint = 0x3445;
pub(crate) const EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT: EGLint = 0x3446;
pub(crate) const EGL_WIDTH: EGLint = 0x3057;
pub(crate) const EGL_HEIGHT: EGLint = 0x3056;
pub(crate) const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
pub(crate) const EGL_NONE: EGLint = 0x3038;

#[link(name = "cuda")]
unsafe extern "C" {
    // CUDA Driver API
    pub(crate) fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
    fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;
    fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;

    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn CuMemcpy2DAsync(pCopy: *const CUDA_MEMCPY2D, stream: CUstream) -> CUresult;

    // CUDA-EGL Interop
    pub(crate) fn cuGraphicsEGLRegisterImage(
        pCudaResource: *mut CUgraphicsResource,
        image: EGLImageKHR,
        flags: c_uint,
    ) -> CUresult;

    pub(crate) fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> CUresult;

    pub(crate) fn cuGraphicsResourceGetMappedEglFrame(
        pEglFrame: *mut CUeglFrame,
        resource: CUgraphicsResource,
        index: c_uint,
        mipLevel: c_uint,
    ) -> CUresult;

    fn cuStreamSynchronize(stream: CUstream) -> CUresult;

    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char)
    -> CUresult;

    fn cuTexObjectCreate(
        pTexObject: *mut CUtexObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
        pTexDesc: *const CUDA_TEXTURE_DESC,
        pResViewDesc: *const c_void,
    ) -> CUresult;

    fn cuTexObjectDestroy(texObject: CUtexObject) -> CUresult;

    fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;

    fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
}

fn gst_dma_video_info_to_video_info(
    dma_video_info: &VideoInfoDmaDrm,
) -> Result<gst_video::ffi::GstVideoInfo, String> {
    let mut video_info: gst_video::ffi::GstVideoInfo = unsafe { std::mem::zeroed() };
    unsafe { gst_video::ffi::gst_video_info_init(&mut video_info) };

    let result = unsafe {
        gst_video::ffi::gst_video_info_dma_drm_to_video_info(
            dma_video_info.to_glib_none().0,
            &mut video_info,
        )
    };
    if result == glib_ffi::GFALSE {
        return Err("Failed to convert DMA-BUF video info to GStreamer video info".into());
    }

    Ok(video_info)
}

#[link(name = "EGL")]
unsafe extern "C" {
    pub(crate) fn eglGetCurrentDisplay() -> EGLDisplay;
    pub(crate) fn eglGetProcAddress(procname: *const c_char) -> *mut c_void;
}

// EGLImage extension function pointers
pub(crate) type PFN_eglCreateImageKHR = unsafe extern "C" fn(
    dpy: EGLDisplay,
    ctx: *mut c_void,
    target: u32,
    buffer: *mut c_void,
    attrib_list: *const EGLint,
) -> EGLImageKHR;

pub(crate) type PFN_eglDestroyImageKHR =
    unsafe extern "C" fn(dpy: EGLDisplay, image: EGLImageKHR) -> c_int;

// CUDA memcpy2D structure
#[repr(C)]
struct CUDA_MEMCPY2D {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: c_uint,
    pub srcHost: *const c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: c_uint,
    pub dstHost: *mut c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}

#[allow(dead_code)]
const CU_MEMORYTYPE_HOST: c_uint = 1;
#[allow(dead_code)]
const CU_MEMORYTYPE_DEVICE: c_uint = 2;
#[allow(dead_code)]
const CU_MEMORYTYPE_ARRAY: c_uint = 3;
#[allow(dead_code)]
const CU_MEMORYTYPE_UNIFIED: c_uint = 4;

unsafe extern "C" {
    // gstcudaloader
    pub(crate) fn gst_cuda_load_library() -> glib_ffi::gboolean;

    // GstCudaContext functions
    pub(crate) fn gst_cuda_context_new(device_id: c_uint) -> *mut GstCudaContext;
    fn gst_cuda_context_get_handle(context: *mut GstCudaContext) -> CUcontext;
    fn gst_cuda_context_push(context: *mut GstCudaContext) -> glib_ffi::gboolean;
    fn gst_cuda_context_pop(pctx: *mut CUcontext) -> glib_ffi::gboolean;

    // GstCudaStream functions
    pub(crate) fn gst_cuda_stream_new(context: *mut GstCudaContext) -> GstCudaStreamHandle;
    pub(crate) fn gst_cuda_stream_ref(stream: GstCudaStreamHandle);
    pub(crate) fn gst_cuda_stream_unref(stream: GstCudaStreamHandle);

    // GstCudaMemory functions
    fn gst_cuda_allocator_alloc(
        allocator: *mut gst_ffi::GstAllocator,
        context: *mut GstCudaContext,
        stream: GstCudaStream,
        info: *const gst_video::ffi::GstVideoInfo,
    ) -> *mut gst_ffi::GstMemory;

    fn gst_cuda_allocator_alloc_wrapped(
        allocator: *mut gst_ffi::GstAllocator,
        context: *mut GstCudaContext,
        stream: GstCudaStream,
        info: *const gst_video::ffi::GstVideoInfo,
        dev_ptr: *mut CUdeviceptr,
        user_data: *mut c_void,
        notify: Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> *mut gst_ffi::GstMemory;

    fn gst_is_cuda_memory(mem: *mut gst_ffi::GstMemory) -> glib_ffi::gboolean;

    pub(crate) fn gst_cuda_memory_init_once() -> c_void;

    pub(crate) fn gst_cuda_buffer_pool_new(context: *mut GstCudaContext) -> GstBufferPool;
    pub(crate) fn gst_buffer_pool_config_set_cuda_stream(
        config: *mut gst_ffi::GstStructure,
        stream: GstCudaStreamHandle,
    );

    fn gst_cuda_stream_get_handle(stream: GstCudaStream) -> CUstream;

    fn gst_cuda_memory_get_stream(mem: *mut gst_ffi::GstMemory) -> GstCudaStream;

    pub(crate) fn gst_cuda_handle_context_query(
        element: *mut GstElement,
        query: *mut GstQuery,
        gst_cuda_context: *mut GstCudaContext,
    ) -> glib_ffi::gboolean;
}

pub(crate) struct CudaContextGuard;

impl CudaContextGuard {
    pub fn new(cuda_context: &CUDAContext) -> Result<Self, String> {
        // CHECK BEFORE PUSH
        unsafe {
            let mut dummy_ctx: CUcontext = ptr::null_mut();
            let err = cuCtxGetCurrent(&mut dummy_ctx);
            tracing::info!(
                "BEFORE gst_cuda_context_push, error state: {}",
                cuda_result_to_string(err)
            );
        }

        if unsafe { gst_cuda_context_push(cuda_context.ptr) } == glib_ffi::GFALSE {
            return Err("Failed to push CUDA context".into());
        }

        // CHECK AFTER PUSH
        unsafe {
            let mut dummy_ctx: CUcontext = ptr::null_mut();
            let err = cuCtxGetCurrent(&mut dummy_ctx);
            tracing::info!(
                "AFTER gst_cuda_context_push, error state: {}",
                cuda_result_to_string(err)
            );
        }

        Ok(CudaContextGuard)
    }
}

impl Drop for CudaContextGuard {
    fn drop(&mut self) {
        unsafe {
            gst_cuda_context_pop(ptr::null_mut());
        }
    }
}

pub(crate) const GST_BUFFER_POOL_OPTION_VIDEO_META: &[u8] = b"GstBufferPoolOptionVideoMeta\0";
const GST_MAP_CUDA: u32 = gst_ffi::GST_MAP_FLAG_LAST << 1;

struct CudaKernel {
    module: CUmodule,
    function: CUfunction,
}

unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

// Load kernel once during initialization
static COPY_KERNEL: std::sync::OnceLock<CudaKernel> = std::sync::OnceLock::new();

const NVRTC_SUCCESS: c_int = 0;

#[link(name = "nvrtc")]
unsafe extern "C" {
    fn nvrtcCreateProgram(
        prog: *mut *mut c_void,
        src: *const c_char,
        name: *const c_char,
        numHeaders: c_int,
        headers: *const *const c_char,
        includeNames: *const *const c_char,
    ) -> c_int;

    fn nvrtcCompileProgram(
        prog: *mut c_void,
        numOptions: c_int,
        options: *const *const c_char,
    ) -> c_int;

    fn nvrtcGetPTXSize(prog: *mut c_void, ptxSizeRet: *mut usize) -> c_int;

    fn nvrtcGetPTX(prog: *mut c_void, ptx: *mut c_char) -> c_int;

    fn nvrtcDestroyProgram(prog: *mut *mut c_void) -> c_int;

    fn nvrtcGetErrorString(result: c_int) -> *const c_char;
}

fn get_copy_kernel() -> Result<CUfunction, String> {
    let kernel = COPY_KERNEL.get_or_init(|| {
        // CUDA C source embedded as string
        let kernel_src = b"
extern \"C\" __global__ void copy_array_to_linear(
    cudaTextureObject_t src_tex,
    unsigned char* dst,
    int width,
    int height,
    int dst_pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Use float4 and normalize, works for any internal format
        float4 pixel = tex2D<float4>(src_tex, (float)x + 0.5f, (float)y + 0.5f);
        int dst_idx = y * dst_pitch + x * 4;

        // Convert from [0,1] to [0,255]
        dst[dst_idx + 0] = (unsigned char)(__saturatef(pixel.x) * 255.0f);
        dst[dst_idx + 1] = (unsigned char)(__saturatef(pixel.y) * 255.0f);
        dst[dst_idx + 2] = (unsigned char)(__saturatef(pixel.z) * 255.0f);
        dst[dst_idx + 3] = (unsigned char)(__saturatef(pixel.w) * 255.0f);
    }
}\0";

        // Compile at runtime
        let mut prog: *mut c_void = ptr::null_mut();
        let src_name = b"copy_kernel.cu\0";

        let result = unsafe {
            nvrtcCreateProgram(
                &mut prog,
                kernel_src.as_ptr() as *const c_char,
                src_name.as_ptr() as *const c_char,
                0,
                ptr::null(),
                ptr::null(),
            )
        };

        if result != NVRTC_SUCCESS {
            panic!("Failed to create NVRTC program: {}", unsafe {
                std::ffi::CStr::from_ptr(nvrtcGetErrorString(result)).to_string_lossy()
            });
        }

        // Compile with compute capability detection
        let result = unsafe { nvrtcCompileProgram(prog, 0, ptr::null()) };

        if result != NVRTC_SUCCESS {
            // Get compilation log
            let mut log_size: usize = 0;
            let mut log: Vec<u8> = Vec::new();

            // NVRTC has nvrtcGetProgramLogSize and nvrtcGetProgramLog
            unsafe extern "C" {
                fn nvrtcGetProgramLogSize(prog: *mut c_void, logSizeRet: *mut usize) -> c_int;
                fn nvrtcGetProgramLog(prog: *mut c_void, log: *mut c_char) -> c_int;
            }

            unsafe {
                nvrtcGetProgramLogSize(prog, &mut log_size);
            }
            if log_size > 0 {
                log.resize(log_size, 0);
                unsafe {
                    nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char);
                }
                eprintln!("NVRTC compilation log:\n{}", String::from_utf8_lossy(&log));
            }

            panic!("Failed to compile kernel: {}", unsafe {
                std::ffi::CStr::from_ptr(nvrtcGetErrorString(result)).to_string_lossy()
            });
        }

        // Get PTX
        let mut ptx_size: usize = 0;
        unsafe {
            nvrtcGetPTXSize(prog, &mut ptx_size);
        }

        let mut ptx = vec![0u8; ptx_size];
        unsafe {
            nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut c_char);
        }

        unsafe {
            nvrtcDestroyProgram(&mut prog);
        }

        // Load PTX into CUDA
        let mut module: CUmodule = ptr::null_mut();
        cuda_call!(cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void))
            .expect("Failed to load compiled PTX");

        let mut function: CUfunction = ptr::null_mut();
        let func_name = b"copy_array_to_linear\0";
        cuda_call!(cuModuleGetFunction(
            &mut function,
            module,
            func_name.as_ptr() as *const c_char
        ))
        .expect("Failed to get kernel function");

        CudaKernel { module, function }
    });

    Ok(kernel.function)
}

pub(crate) fn alloc_copy_gst_memory(
    egl_frame: CUeglFrame,
    cuda_context: &CUDAContext,
    dma_video_info: VideoInfoDmaDrm,
    buffer_pool: Option<GstBufferPool>,
) -> Result<gst::memory::Memory, Box<dyn std::error::Error>> {
    let mut video_info = gst_dma_video_info_to_video_info(&dma_video_info)?;

    let (gst_memory, stream) = if let Some(pool) = buffer_pool {
        tracing::info!("Acquiring buffer from pool...");
        let mut gst_buffer: *mut gst_ffi::GstBuffer = ptr::null_mut();

        let result = unsafe {
            gst::ffi::gst_buffer_pool_acquire_buffer(
                pool as *mut gst::ffi::GstBufferPool,
                &mut gst_buffer,
                ptr::null_mut(),
            )
        };

        tracing::info!(
            "Pool acquire result: {} (GST_FLOW_OK={})",
            result,
            gst_ffi::GST_FLOW_OK
        );

        if result != gst_ffi::GST_FLOW_OK {
            // Check pool state
            let is_active = unsafe {
                gst::ffi::gst_buffer_pool_is_active(pool as *mut gst::ffi::GstBufferPool)
            };
            tracing::error!("Pool acquire failed! Pool active: {}", is_active != 0);
            return Err(format!("Failed to acquire buffer from pool: {}", result).into());
        }

        if gst_buffer.is_null() {
            return Err("Acquired buffer is null".into());
        }

        // Get the memory from the buffer
        let gst_memory = unsafe { gst_ffi::gst_buffer_peek_memory(gst_buffer, 0) };
        if gst_memory.is_null() {
            unsafe { gst_ffi::gst_buffer_unref(gst_buffer) };
            return Err("Failed to get memory from pooled buffer".into());
        }

        // We need to ref the memory since we're returning it separately
        unsafe { gst_ffi::gst_memory_ref(gst_memory) };

        // Unref the buffer (the memory is still valid since we ref'd it)
        unsafe { gst_ffi::gst_buffer_unref(gst_buffer) };

        let cuda_stream = cuda_context
            .stream
            .clone()
            .expect("Cuda context without a stream");

        (gst_memory, cuda_stream.stream)
    } else {
        // Fallback to direct allocation
        let stream = unsafe { std::mem::zeroed() };
        let gst_memory = unsafe {
            gst_cuda_allocator_alloc(ptr::null_mut(), cuda_context.ptr, stream, &mut video_info)
        };
        if gst_memory.is_null() {
            return Err("Failed to allocate GST CUDA memory".into());
        }
        (gst_memory, stream)
    };

    // CHECK 1: After getting stream
    unsafe {
        let mut dummy_ctx: CUcontext = ptr::null_mut();
        let err = cuCtxGetCurrent(&mut dummy_ctx);
        tracing::info!(
            "After getting stream, error state: {}",
            cuda_result_to_string(err)
        );
    }

    let stream_handle = unsafe { gst_cuda_stream_get_handle(stream) };

    // CHECK 2: After gst_cuda_stream_get_handle
    unsafe {
        let mut dummy_ctx: CUcontext = ptr::null_mut();
        let err = cuCtxGetCurrent(&mut dummy_ctx);
        tracing::info!(
            "After gst_cuda_stream_get_handle, error state: {}",
            cuda_result_to_string(err)
        );
    }

    // Map the GStreamer memory to get destination device pointer
    let mut map_info: gst_ffi::GstMapInfo = unsafe { std::mem::zeroed() };
    let map_success = unsafe {
        gst_ffi::gst_memory_map(
            gst_memory,
            &mut map_info,
            gst_ffi::GST_MAP_WRITE | GST_MAP_CUDA,
        )
    };

    // CHECK 3: After memory map
    unsafe {
        let mut dummy_ctx: CUcontext = ptr::null_mut();
        let err = cuCtxGetCurrent(&mut dummy_ctx);
        tracing::info!(
            "After gst_memory_map, error state: {}",
            cuda_result_to_string(err)
        );
    }

    if map_success == glib_ffi::GFALSE {
        unsafe { gst_ffi::gst_memory_unref(gst_memory) };
        return Err("Failed to map GStreamer CUDA memory".into());
    }

    let dst_device_ptr = map_info.data as CUdeviceptr;

    // Copy from EGL frame to GStreamer memory for each plane
    let _cuda_context_guard = CudaContextGuard::new(cuda_context)?;

    // CHECK 4: After context guard (push)
    unsafe {
        let mut dummy_ctx: CUcontext = ptr::null_mut();
        let err = cuCtxGetCurrent(&mut dummy_ctx);
        tracing::info!(
            "After CudaContextGuard::new (context push), error state: {}",
            cuda_result_to_string(err)
        );
    }

    for plane in 0..egl_frame.plane_count as usize {
        tracing::info!(
            "Processing plane {}, frame_type: {}",
            plane,
            egl_frame.frame_type
        );

        if egl_frame.frame_type == 0 {
            // Array type - use GPU kernel for detiling
            let src_array = unsafe { egl_frame.frame.p_array[plane] };
            tracing::info!("src_array ptr: {:?}", src_array);

            // Create texture object from CUDA array
            let res_desc = CUDA_RESOURCE_DESC {
                resType: 0, // CU_RESOURCE_TYPE_ARRAY
                res: CUDA_RESOURCE_DESC_RES { array: src_array },
                flags: 0,
            };

            let tex_desc = CUDA_TEXTURE_DESC {
                addressMode: [CU_TR_ADDRESS_MODE_CLAMP; 3],
                filterMode: CU_TR_FILTER_MODE_POINT,
                flags: 0,
                maxAnisotropy: 1,
                mipmapFilterMode: 0,
                mipmapLevelBias: 0.0,
                minMipmapLevelClamp: 0.0,
                maxMipmapLevelClamp: 0.0,
                borderColor: [0.0; 4],
                _reserved: [0; 12],
            };

            tracing::info!("Creating texture object...");
            let mut tex_obj: CUtexObject = 0;
            let tex_result =
                unsafe { cuTexObjectCreate(&mut tex_obj, &res_desc, &tex_desc, ptr::null()) };

            if tex_result != CUDA_SUCCESS {
                tracing::error!(
                    "cuTexObjectCreate failed: {} (code: {})",
                    cuda_result_to_string(tex_result),
                    tex_result
                );
                return Err(format!(
                    "Failed to create texture object: {} (code: {})",
                    cuda_result_to_string(tex_result),
                    tex_result
                )
                .into());
            }
            tracing::info!("Texture object created: {}", tex_obj);

            // Get destination pointer
            let dst_ptr = dst_device_ptr + video_info.offset[plane] as u64;
            let dst_pitch = video_info.stride[plane] as i32;

            let height = match plane {
                0 => dma_video_info.height() as i32,
                _ => match dma_video_info.format().to_string().as_str() {
                    "NV12" | "NV21" | "I420" | "YV12" => dma_video_info.height() as i32 / 2,
                    _ => dma_video_info.height() as i32,
                },
            };
            let width = dma_video_info.width() as i32;

            tracing::info!(
                "Kernel params: width={}, height={}, dst_pitch={}",
                width,
                height,
                dst_pitch
            );

            // Launch kernel
            let kernel = get_copy_kernel()?;
            let block_dim = (16, 16);
            let grid_dim = (
                (width + block_dim.0 - 1) / block_dim.0,
                (height + block_dim.1 - 1) / block_dim.1,
            );

            tracing::info!("Grid: {:?}, Block: {:?}", grid_dim, block_dim);

            let mut args: [*mut c_void; 5] = [
                &tex_obj as *const _ as *mut c_void,
                &dst_ptr as *const _ as *mut c_void,
                &width as *const _ as *mut c_void,
                &height as *const _ as *mut c_void,
                &dst_pitch as *const _ as *mut c_void,
            ];

            // Before launching kernel, check for sticky errors
            tracing::info!("Checking for previous CUDA errors...");
            unsafe {
                let error = cuCtxGetCurrent(ptr::null_mut());
                if error != CUDA_SUCCESS {
                    tracing::error!(
                        "Previous CUDA error detected: {}",
                        cuda_result_to_string(error)
                    );
                }
            }

            tracing::info!("Launching kernel...");
            match cuda_call!(cuLaunchKernel(
                kernel,
                grid_dim.0 as c_uint,
                grid_dim.1 as c_uint,
                1,
                block_dim.0 as c_uint,
                block_dim.1 as c_uint,
                1,
                0,
                stream_handle,
                args.as_mut_ptr(),
                ptr::null_mut(),
            )) {
                Ok(_) => tracing::info!("Kernel launched successfully"),
                Err(e) => {
                    tracing::error!("cuLaunchKernel failed: {}", e);
                    cuda_call!(cuTexObjectDestroy(tex_obj))?;
                    return Err(e.into());
                }
            }

            tracing::info!("Checking for kernel errors...");
            // Clean up texture object
            let sync_result = unsafe { cuStreamSynchronize(stream_handle) };
            if sync_result != CUDA_SUCCESS {
                tracing::error!(
                    "Kernel execution failed: {}",
                    cuda_result_to_string(sync_result)
                );
                cuda_call!(cuTexObjectDestroy(tex_obj))?;
                return Err(format!(
                    "Kernel execution error: {}",
                    cuda_result_to_string(sync_result)
                )
                .into());
            }
            tracing::info!("Kernel executed successfully");
        } else {
            // Pitched pointer - use regular copy
            let mut copy_params: CUDA_MEMCPY2D = unsafe { std::mem::zeroed() };

            // Set up source (from EGL frame)
            unsafe {
                match egl_frame.frame_type {
                    0 => {
                        // Array type
                        copy_params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                        copy_params.srcArray = egl_frame.frame.p_array[plane];
                    }
                    1 => {
                        // Pitched pointer type
                        copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                        copy_params.srcDevice = egl_frame.frame.p_pitch[plane] as CUdeviceptr;
                        copy_params.srcPitch = egl_frame.pitch as usize;
                    }
                    _ => {
                        return Err("Unsupported EGL frame type".into());
                    }
                }
            }

            copy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy_params.dstDevice = dst_device_ptr + video_info.offset[plane] as u64;
            copy_params.dstPitch = video_info.stride[plane] as usize;

            // Set copy dimensions
            copy_params.WidthInBytes = video_info.stride[plane] as usize;
            copy_params.Height = match plane {
                0 => dma_video_info.height() as usize, // Y plane (or single plane)
                _ => {
                    // For YUV formats, UV planes are typically half height
                    let plane_height = match dma_video_info.format().to_string().as_str() {
                        "NV12" | "NV21" | "I420" | "YV12" => dma_video_info.height() as usize / 2,
                        _ => dma_video_info.height() as usize, // For other formats, assume same height
                    };
                    plane_height
                }
            };

            cuda_call!(CuMemcpy2DAsync(&copy_params, stream_handle))?;
        }
    }

    tracing::info!("Synchronizing stream...");
    match cuda_call!(cuStreamSynchronize(stream_handle)) {
        Ok(_) => {
            tracing::info!("Stream synchronized");
            unsafe { gst_ffi::gst_memory_unmap(gst_memory, &mut map_info) };
            Ok(unsafe { gst::Memory::from_glib_full(gst_memory) })
        }
        Err(error) => {
            tracing::error!("cuStreamSynchronize failed: {}", error);
            unsafe { gst_ffi::gst_memory_unmap(gst_memory, &mut map_info) };
            unsafe { gst_ffi::gst_memory_unref(gst_memory) };
            Err(format!("Failed to synchronize CUDA stream: {}", error).into())
        }
    }
}

pub(crate) fn cuda_result_to_string(result: CUresult) -> &'static str {
    match result {
        0 => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        5 => "CUDA_ERROR_PROFILER_DISABLED",
        6 => "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
        7 => "CUDA_ERROR_PROFILER_ALREADY_STARTED",
        8 => "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        200 => "CUDA_ERROR_INVALID_IMAGE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        202 => "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
        205 => "CUDA_ERROR_MAP_FAILED",
        206 => "CUDA_ERROR_UNMAP_FAILED",
        207 => "CUDA_ERROR_ARRAY_IS_MAPPED",
        208 => "CUDA_ERROR_ALREADY_MAPPED",
        209 => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        210 => "CUDA_ERROR_ALREADY_ACQUIRED",
        211 => "CUDA_ERROR_NOT_MAPPED",
        212 => "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
        213 => "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
        214 => "CUDA_ERROR_ECC_UNCORRECTABLE",
        215 => "CUDA_ERROR_UNSUPPORTED_LIMIT",
        216 => "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
        217 => "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
        218 => "CUDA_ERROR_INVALID_PTX",
        219 => "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
        300 => "CUDA_ERROR_INVALID_SOURCE",
        301 => "CUDA_ERROR_FILE_NOT_FOUND",
        302 => "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
        303 => "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
        304 => "CUDA_ERROR_OPERATING_SYSTEM",
        400 => "CUDA_ERROR_INVALID_HANDLE",
        500 => "CUDA_ERROR_NOT_FOUND",
        600 => "CUDA_ERROR_NOT_READY",
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        701 => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        702 => "CUDA_ERROR_LAUNCH_TIMEOUT",
        703 => "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
        704 => "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
        705 => "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
        708 => "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
        709 => "CUDA_ERROR_CONTEXT_IS_DESTROYED",
        710 => "CUDA_ERROR_ASSERT",
        711 => "CUDA_ERROR_TOO_MANY_PEERS",
        712 => "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
        713 => "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
        714 => "CUDA_ERROR_HARDWARE_STACK_ERROR",
        715 => "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        716 => "CUDA_ERROR_MISALIGNED_ADDRESS",
        717 => "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        718 => "CUDA_ERROR_INVALID_PC",
        719 => "CUDA_ERROR_LAUNCH_FAILED",
        800 => "CUDA_ERROR_NOT_PERMITTED",
        801 => "CUDA_ERROR_NOT_SUPPORTED",
        999 => "CUDA_ERROR_UNKNOWN",
        _ => "CUDA_ERROR_UNKNOWN_CODE",
    }
}
