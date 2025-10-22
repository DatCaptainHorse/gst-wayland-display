use once_cell::sync::Lazy;
use smithay::backend::allocator::gbm::GbmDevice;
use smithay::backend::drm::DrmNode;
use smithay::backend::egl::ffi::egl as ffi_egl;
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::utils::DeviceFd;
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::OwnedFd;
use std::os::raw::c_int;
use std::sync::{Arc, Mutex, Weak};

static EGL_DISPLAYS: Lazy<Mutex<HashMap<Option<DrmNode>, Weak<EGLDisplay>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn setup_renderer(render_node: Option<DrmNode>) -> GlesRenderer {
    let mut displays = EGL_DISPLAYS.lock().unwrap();
    let maybe_display = displays
        .get(&render_node)
        .and_then(|weak_display| weak_display.upgrade());

    let egl = match maybe_display {
        Some(display) => display,
        None => {
            let display = if let Some(render_node) = render_node.as_ref() {
                // Normal GPU with render node
                let path = render_node.dev_path().expect("Failed to get device path");
                let file = File::options()
                    .read(true)
                    .write(true)
                    .open(path)
                    .expect("Failed to open render node");
                let fd = DeviceFd::from(Into::<OwnedFd>::into(file));
                let gbm = GbmDevice::new(fd).expect("Failed to create GBM device");
                unsafe { EGLDisplay::new(gbm).expect("Failed to create EGLDisplay") }
            } else {
                // MIG instance - use surfaceless platform with raw EGL
                tracing::info!("No render node, using surfaceless EGL for MIG");

                unsafe {
                    let _dp_extensions = smithay::backend::egl::ffi::make_sure_egl_is_loaded()
                        .expect("Failed to load EGL");

                    let egl_display = smithay::backend::egl::wrap_egl_call_ptr(|| {
                        ffi_egl::GetPlatformDisplayEXT(
                            0x31DD, // EGL_PLATFORM_SURFACELESS_MESA
                            ffi_egl::DEFAULT_DISPLAY as *mut std::ffi::c_void,
                            std::ptr::null(),
                        )
                    })
                    .expect("Failed to get surfaceless display");

                    if egl_display == ffi_egl::NO_DISPLAY {
                        panic!("Surfaceless platform not supported");
                    }

                    let mut major = 0;
                    let mut minor = 0;
                    if ffi_egl::Initialize(egl_display, &mut major, &mut minor)
                        == ffi_egl::FALSE as u32
                    {
                        panic!("Failed to initialize EGL");
                    }

                    tracing::info!("EGL {}.{} initialized", major, minor);

                    ffi_egl::BindAPI(ffi_egl::OPENGL_ES_API);

                    // Choose config properly
                    let config_attribs = [
                        ffi_egl::SURFACE_TYPE as c_int,
                        ffi_egl::PBUFFER_BIT as c_int,
                        ffi_egl::RENDERABLE_TYPE as c_int,
                        ffi_egl::OPENGL_ES2_BIT as c_int,
                        ffi_egl::RED_SIZE as c_int,
                        1,
                        ffi_egl::GREEN_SIZE as c_int,
                        1,
                        ffi_egl::BLUE_SIZE as c_int,
                        1,
                        ffi_egl::ALPHA_SIZE as c_int,
                        0,
                        ffi_egl::DEPTH_SIZE as c_int,
                        0,
                        ffi_egl::STENCIL_SIZE as c_int,
                        0,
                        ffi_egl::NONE as c_int,
                    ];

                    let mut configs: Vec<*const std::ffi::c_void> = vec![std::ptr::null(); 10];
                    let mut num_configs = 0;

                    if ffi_egl::ChooseConfig(
                        egl_display,
                        config_attribs.as_ptr(),
                        configs.as_mut_ptr() as *mut _,
                        10,
                        &mut num_configs,
                    ) == ffi_egl::FALSE as u32
                    {
                        panic!("ChooseConfig failed");
                    }

                    if num_configs == 0 {
                        panic!("No matching EGL configs found");
                    }

                    let config = configs[0];
                    tracing::info!("Chose EGL config: {:p}", config);

                    EGLDisplay::from_raw(egl_display, config).expect("Failed to wrap EGL display")
                }
            };

            let display_arc = Arc::new(display);
            displays.insert(render_node, Arc::downgrade(&display_arc));
            display_arc
        }
    };

    let context = EGLContext::new(&egl).expect("Failed to initialize EGL context");
    let renderer = unsafe { GlesRenderer::new(context) }.expect("Failed to initialize renderer");
    renderer
}
