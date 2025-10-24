use conv::{
    from_u64_bits, map_adapter_type, map_backend_type, map_bind_group_entry,
    map_bind_group_layout_entry, map_device_descriptor, map_instance_backend_flags,
    map_instance_descriptor, map_pipeline_layout_descriptor, map_query_set_descriptor,
    map_query_set_index, map_shader_module, map_surface, map_surface_configuration,
    map_texture_view_descriptor, CreateSurfaceParams,
};
use core::slice;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    error,
    fmt::Display,
    mem,
    num::NonZeroU64,
    sync::{atomic, Arc},
    thread,
};
use utils::{
    get_base_device_limits_from_adapter_limits, make_slice, str_into_string_view,
    string_view_into_str, texture_format_has_depth,
};

pub mod conv;
pub mod logging;
pub mod unimplemented;
pub mod utils;

pub mod native {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub type WGPUAdapterImpl = wgpu::Adapter;
pub type WGPUBindGroupImpl = wgpu::BindGroup;
pub type WGPUBindGroupLayoutImpl = wgpu::BindGroupLayout;
pub type WGPUBufferImpl = wgpu::Buffer;
pub type WGPUCommandBufferImpl = wgpu::CommandBuffer;
pub type WGPUCommandEncoderImpl = wgpu::CommandEncoder;
pub struct WGPUComputePassEncoderImpl {
    compute_pass_encoder: Box<wgpu::ComputePass<'static>>,
}

// ComputePassEncoder is thread-unsafe
unsafe impl Send for WGPUComputePassEncoderImpl {}
unsafe impl Sync for WGPUComputePassEncoderImpl {}

pub type WGPUComputePipelineImpl = wgpu::ComputePipeline;

pub struct WGPUDeviceImpl {
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl Drop for WGPUDeviceImpl {
    fn drop(&mut self) {
        if !thread::panicking() {
            match self.device.poll(wgt::PollType::Wait) {
                Ok(_) => (),
                Err(err) => handle_error_fatal(err, "WGPUDeviceImpl::drop"),
            }
        }
    }
}

pub type WGPUInstanceImpl = wgpu::Instance;
pub type WGPUPipelineLayoutImpl = wgpu::PipelineLayout;

struct QuerySetData {
    query_type: native::WGPUQueryType,
    query_count: u32,
}

pub struct WGPUQuerySetImpl {
    query_set: wgpu::QuerySet,
    data: QuerySetData,
}

pub type WGPUQueueImpl = wgpu::Queue;

pub type WGPURenderBundleImpl = wgpu::RenderBundle;
pub struct WGPURenderBundleEncoderImpl {
    render_bundle_encoder: Box<wgpu::RenderBundleEncoder<'static>>,
}

// RenderBundleEncoder is thread-unsafe
unsafe impl Send for WGPURenderBundleEncoderImpl {}
unsafe impl Sync for WGPURenderBundleEncoderImpl {}

pub struct WGPURenderPassEncoderImpl {
    render_pass_encoder: Box<wgpu::RenderPass<'static>>,
}

// RenderPassEncodee is thread-unsafe
unsafe impl Send for WGPURenderPassEncoderImpl {}
unsafe impl Sync for WGPURenderPassEncoderImpl {}

pub type WGPURenderPipelineImpl = wgpu::RenderPipeline;
pub type WGPUSamplerImpl = wgpu::Sampler;
pub type WGPUShaderModuleImpl = wgpu::ShaderModule;

pub struct WGPUSurfaceImpl {
    surface: Box<wgpu::Surface<'static>>,
}

pub type WGPUTextureImpl = wgpu::Texture;
pub type WGPUTextureViewImpl = wgpu::TextureView;

const NULL_FUTURE: native::WGPUFuture = native::WGPUFuture { id: 0 };
const EMPTY_STRING: native::WGPUStringView = native::WGPUStringView {
    length: 0,
    data: std::ptr::null(),
};

struct DeviceCallback<T> {
    callback: T,
    userdata: utils::Userdata,
}
unsafe impl<T> Send for DeviceCallback<T> {}

type UncapturedErrorCallback = DeviceCallback<native::WGPUUncapturedErrorCallback>;
type DeviceLostCallback = DeviceCallback<native::WGPUDeviceLostCallback>;

unsafe extern "C" fn default_uncaptured_error_handler(
    _device: *const native::WGPUDevice,
    _typ: native::WGPUErrorType,
    message: native::WGPUStringView,
    _userdata1: *mut ::std::os::raw::c_void,
    _userdata2: *mut ::std::os::raw::c_void,
) {
    let message = string_view_into_str(message).unwrap_or("");
    log::warn!("Handling wgpu uncaptured errors as fatal by default");
    panic!("wgpu uncaptured error:\n{message}\n");
}
const DEFAULT_UNCAPTURED_ERROR_HANDLER: UncapturedErrorCallback = UncapturedErrorCallback {
    callback: Some(default_uncaptured_error_handler),
    userdata: utils::Userdata::NULL,
};

unsafe extern "C" fn default_device_lost_handler(
    _device: *const native::WGPUDevice,
    _reason: native::WGPUDeviceLostReason,
    message: native::WGPUStringView,
    _userdata1: *mut ::std::os::raw::c_void,
    _userdata2: *mut ::std::os::raw::c_void,
) {
    let message = string_view_into_str(message).unwrap_or("");
    log::warn!("Handling wgpu device lost errors as fatal by default");
    panic!("wgpu device lost error:\n{message}\n");
}
const DEFAULT_DEVICE_LOST_HANDLER: DeviceLostCallback = DeviceLostCallback {
    callback: Some(default_device_lost_handler),
    userdata: utils::Userdata::NULL,
};

#[derive(Debug)]
pub enum Error {
    DeviceLost {
        source: Box<dyn error::Error + Send + 'static>,
    },
    OutOfMemory {
        source: Box<dyn error::Error + Send + 'static>,
    },
    Validation {
        source: Box<dyn error::Error + Send + 'static>,
        description: String,
    },
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::DeviceLost { source } => Some(source.as_ref()),
            Error::OutOfMemory { source } => Some(source.as_ref()),
            Error::Validation { source, .. } => Some(source.as_ref()),
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DeviceLost { .. } => f.write_str("Device lost"),
            Error::OutOfMemory { .. } => f.write_str("Out of Memory"),
            Error::Validation { description, .. } => f.write_str(description),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
pub enum ErrorFilter {
    /// Catch only out-of-memory errors.
    OutOfMemory,
    /// Catch only validation errors.
    Validation,
}

type ErrorSink = Arc<Mutex<ErrorSinkRaw>>;

struct ErrorScope {
    error: Option<crate::Error>,
    filter: crate::ErrorFilter,
}

struct ErrorSinkRaw {
    scopes: Vec<ErrorScope>,
    uncaptured_handler: UncapturedErrorCallback,
    device_lost_handler: DeviceLostCallback,
    device: Option<native::WGPUDevice>,
}

impl ErrorSinkRaw {
    fn new(device_lost_handler: DeviceLostCallback) -> ErrorSinkRaw {
        ErrorSinkRaw {
            scopes: Vec::new(),
            uncaptured_handler: DEFAULT_UNCAPTURED_ERROR_HANDLER,
            device_lost_handler,
            device: None,
        }
    }

    fn handle_error(&mut self, err: crate::Error) {
        let (typ, filter) = match err {
            crate::Error::DeviceLost { .. } => {
                // handle device lost error early
                if let Some(callback) = self.device_lost_handler.callback {
                    let userdata = &self.device_lost_handler.userdata;
                    let msg = err.to_string();
                    unsafe {
                        callback(
                            &self.device.unwrap(),
                            native::WGPUDeviceLostReason_Destroyed,
                            str_into_string_view(&msg),
                            userdata.get_1(),
                            userdata.get_2(),
                        );
                    };
                }
                return;
            }
            crate::Error::OutOfMemory { .. } => (
                native::WGPUErrorType_OutOfMemory,
                crate::ErrorFilter::OutOfMemory,
            ),
            crate::Error::Validation { .. } => (
                native::WGPUErrorType_Validation,
                crate::ErrorFilter::Validation,
            ),
        };

        match self
            .scopes
            .iter_mut()
            .rev()
            .find(|scope| scope.filter == filter)
        {
            Some(scope) => {
                if scope.error.is_none() {
                    scope.error = Some(err);
                }
            }
            None => {
                if let Some(callback) = self.uncaptured_handler.callback {
                    let userdata = &self.uncaptured_handler.userdata;
                    let msg = err.to_string();
                    unsafe {
                        callback(
                            &self.device.unwrap(),
                            typ,
                            str_into_string_view(&msg),
                            userdata.get_1(),
                            userdata.get_2(),
                        )
                    };
                }
            }
        }
    }
}

fn format_error(err: &(impl error::Error + 'static)) -> String {
    // TODO:

    let mut output = String::new();
    // let mut level = 1;

    // fn print_tree(output: &mut String, level: &mut usize, e: &(dyn error::Error + 'static)) {
    //     let mut print = |e: &(dyn error::Error + 'static)| {
    //         use std::fmt::Write;
    //         writeln!(output, "{}{}", " ".repeat(*level * 2), e).unwrap();

    //         if let Some(e) = e.source() {
    //             *level += 1;
    //             print_tree(output, level, e);
    //             *level -= 1;
    //         }
    //     };
    //     if let Some(multi) = e.downcast_ref::<wgt::error::MultiError>() {
    //         for e in multi.errors() {
    //             print(e);
    //         }
    //     } else {
    //         print(e);
    //     }
    // }

    // print_tree(&mut output, &mut level, err);

    format!("Validation Error\n\nCaused by:\n{}", output)
}

fn handle_error_fatal(
    cause: impl error::Error + Send + Sync + 'static,
    operation: &'static str,
) -> ! {
    panic!("Error in {operation}: {f}", f = format_error(&cause));
}

fn handle_error(
    sink_mutex: &Mutex<ErrorSinkRaw>,
    source: impl error::Error + Send + Sync + 'static,
    label: &str,
    fn_ident: &'static str,
) {
    // TODO:
    //
    // let error = wgc::error::ContextError {
    //     fn_ident,
    //     source: Box::new(source),
    //     label: label.unwrap_or_default().to_string(),
    // };
    // let sink = sink_mutex.lock();
    // let mut source_opt: Option<&(dyn error::Error + 'static)> = Some(&error);
    // while let Some(source) = source_opt {
    //     match source.downcast_ref::<wgc::device::DeviceError>() {
    //         Some(wgc::device::DeviceError::Lost) => {
    //             panic!()
    //             // return sink.handle_error(crate::Error::DeviceLost {
    //             //     source: Box::new(error),
    //             // });
    //         }
    //         Some(wgc::device::DeviceError::OutOfMemory) => {
    //             panic!()
    //             // return sink.handle_error(crate::Error::OutOfMemory {
    //             //     source: Box::new(error),
    //             // });
    //         }
    //         _ => (),
    //     }
    //     source_opt = source.source();
    // }

    // Otherwise, it is a validation error
    // sink.handle_error(crate::Error::Validation {
    //     description: format_error(&error),
    //     source: Box::new(error),
    // });
}

// webgpu.h functions

#[no_mangle]
pub unsafe extern "C" fn wgpuCreateInstance(
    descriptor: Option<&native::WGPUInstanceDescriptor>,
) -> native::WGPUInstance {
    let instance_desc = match descriptor {
        Some(descriptor) => {
            if descriptor.features.timedWaitAnyEnable != 0
                || descriptor.features.timedWaitAnyMaxCount > 0
            {
                panic!("Unsupported timed WaitAny features specified");
            }

            follow_chain!(map_instance_descriptor(
                (descriptor),
                WGPUSType_InstanceExtras => native::WGPUInstanceExtras
            ))
        }
        None => wgt::InstanceDescriptor::default(),
    };

    Arc::into_raw(Arc::new(wgpu::Instance::new(&instance_desc)))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuGetInstanceCapabilities(
    capabilities: Option<&mut native::WGPUInstanceCapabilities>,
) -> native::WGPUStatus {
    let capabilities = capabilities.expect("invalid return pointer \"capabilities\"");
    // WaitAny is currently completely unsupported, so...
    capabilities.timedWaitAnyEnable = false as native::WGPUBool;
    capabilities.timedWaitAnyMaxCount = 0;
    native::WGPUStatus_Success
}

// Adapter methods

#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterGetFeatures(
    adapter: native::WGPUAdapter,
    features: Option<&mut native::WGPUSupportedFeatures>,
) -> native::WGPUStatus {
    let adapter = &adapter.as_ref().expect("invalid adapter");
    let adapter_features = adapter.features();
    let features = features.expect("invalid return pointer \"features\"");

    return_features(features, adapter_features);

    native::WGPUStatus_Success
}

fn return_features(native: &mut native::WGPUSupportedFeatures, features: wgt::Features) {
    let temp = conv::features_to_native(features);
    let mut temp = temp.into_boxed_slice();

    native.featureCount = temp.len();
    native.features = temp.as_mut_ptr();

    mem::forget(temp);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterGetLimits(
    adapter: native::WGPUAdapter,
    limits: Option<&mut native::WGPULimits>,
) -> native::WGPUBool {
    let adapter = &adapter.as_ref().expect("invalid adapter");
    let limits = limits.expect("invalid return pointer \"limits\"");

    let wgt_limits = adapter.limits();
    conv::write_limits_struct(wgt_limits, limits);

    true as native::WGPUBool // indicates that we can fill WGPUChainedStructOut
}

#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterGetInfo(
    adapter: native::WGPUAdapter,
    info: Option<&mut native::WGPUAdapterInfo>,
) -> native::WGPUStatus {
    let adapter = &adapter.as_ref().expect("invalid adapter");
    let info = info.expect("invalid return pointer \"info\"");

    let result = adapter.get_info();

    info.vendor = utils::str_into_owned_string_view(&result.driver);
    info.architecture = EMPTY_STRING; // TODO(webgpu.h)
    info.device = utils::str_into_owned_string_view(&result.name);
    info.description = utils::str_into_owned_string_view(&result.driver_info);
    info.backendType = map_backend_type(result.backend);
    info.adapterType = map_adapter_type(result.device_type);
    info.vendorID = result.vendor;
    info.deviceID = result.device;

    native::WGPUStatus_Success
}

#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterHasFeature(
    adapter: native::WGPUAdapter,
    feature: native::WGPUFeatureName,
) -> native::WGPUBool {
    let adapter = &adapter.as_ref().expect("invalid adapter");
    let adapter_features = adapter.features();

    let feature = match conv::map_feature(feature) {
        Some(feature) => feature,
        None => return false as native::WGPUBool,
    };

    adapter_features.contains(feature) as native::WGPUBool
}

#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterInfoFreeMembers(adapter_info: native::WGPUAdapterInfo) {
    utils::drop_string_view(adapter_info.vendor);
    utils::drop_string_view(adapter_info.architecture);
    utils::drop_string_view(adapter_info.device);
    utils::drop_string_view(adapter_info.description);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterAddRef(adapter: native::WGPUAdapter) {
    assert!(!adapter.is_null(), "invalid adapter");
    Arc::increment_strong_count(adapter);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterRelease(adapter: native::WGPUAdapter) {
    assert!(!adapter.is_null(), "invalid adapter");
    Arc::decrement_strong_count(adapter);
}

// BindGroup methods

#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupAddRef(bind_group: native::WGPUBindGroup) {
    assert!(!bind_group.is_null(), "invalid bind group");
    Arc::increment_strong_count(bind_group);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupRelease(bind_group: native::WGPUBindGroup) {
    assert!(!bind_group.is_null(), "invalid bind group");
    Arc::decrement_strong_count(bind_group);
}

// BindGroupLayout methods

#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupLayoutAddRef(bind_group_layout: native::WGPUBindGroupLayout) {
    assert!(!bind_group_layout.is_null(), "invalid bind group layout");
    Arc::increment_strong_count(bind_group_layout);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupLayoutRelease(
    bind_group_layout: native::WGPUBindGroupLayout,
) {
    assert!(!bind_group_layout.is_null(), "invalid bind group layout");
    Arc::decrement_strong_count(bind_group_layout);
}

// Buffer methods

#[no_mangle]
pub unsafe extern "C" fn wgpuBufferDestroy(buffer: native::WGPUBuffer) {
    let buffer = buffer.as_ref().expect("invalid buffer");
    buffer.destroy();
}

#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetConstMappedRange(
    buffer: native::WGPUBuffer,
    offset: usize,
    size: usize,
) -> *const u8 {
    let buffer = buffer.as_ref().expect("invalid buffer");
    let offset = offset as wgt::BufferAddress;
    let buf = buffer.get_mapped_range(match size as usize {
        conv::WGPU_WHOLE_MAP_SIZE => offset..offset + buffer.size(),
        _ => offset..offset + size as wgt::BufferAddress,
    });

    buf.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetMappedRange(
    buffer: native::WGPUBuffer,
    offset: usize,
    size: usize,
) -> *mut u8 {
    let buffer = buffer.as_ref().expect("invalid buffer");
    let offset = offset as wgt::BufferAddress;
    let mut buf = buffer.get_mapped_range_mut(match size as usize {
        conv::WGPU_WHOLE_MAP_SIZE => offset..offset + buffer.size(),
        _ => offset..offset + size as wgt::BufferAddress,
    });

    buf.as_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetSize(buffer: native::WGPUBuffer) -> u64 {
    let buffer = buffer.as_ref().expect("invalid buffer");
    buffer.size()
}

#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetUsage(buffer: native::WGPUBuffer) -> native::WGPUBufferUsage {
    let buffer = buffer.as_ref().expect("invalid buffer");
    buffer.usage().bits() as native::WGPUBufferUsage
}

#[no_mangle]
pub unsafe extern "C" fn wgpuBufferMapAsync(
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
    callback_info: native::WGPUBufferMapCallbackInfo,
) -> native::WGPUFuture {
    let buffer = buffer.as_ref().expect("invalid buffer");
    let callback = callback_info.callback.expect("invalid callback");
    let userdata = new_userdata!(callback_info);

    // TODO:
    // buffer.map_async(mode, bounds, callback);

    // let operation = wgc::resource::BufferMapOperation {
    //     host: match mode as native::WGPUMapMode {
    //         native::WGPUMapMode_Write => wgc::device::HostMap::Write,
    //         native::WGPUMapMode_Read => wgc::device::HostMap::Read,
    //         _ => panic!("invalid map mode"),
    //     },
    //     callback: Some(Box::new(move |result: resource::BufferAccessResult| {
    //         let (status, message) = match result {
    //             Ok(()) => (native::WGPUMapAsyncStatus_Success, String::default()),
    //             Err(cause) => {
    //                 let code = match cause {
    //                     resource::BufferAccessError::MapAborted => {
    //                         native::WGPUMapAsyncStatus_Aborted
    //                     }
    //                     _ => native::WGPUMapAsyncStatus_Error,
    //                 };

    //                 (code, format_error(&cause))
    //             }
    //         };

    //         callback(
    //             status,
    //             str_into_string_view(&message),
    //             userdata.get_1(),
    //             userdata.get_2(),
    //         );
    //     })),
    // };

    // if let Err(cause) = context.buffer_map_async(
    //     buffer_id,
    //     offset as wgt::BufferAddress,
    //     Some(size as wgt::BufferAddress),
    //     operation,
    // ) {
    //     handle_error(error_sink, cause, None, "wgpuBufferMapAsync");
    // };

    // TODO: Properly handle futures.
    NULL_FUTURE
}

#[no_mangle]
pub unsafe extern "C" fn wgpuBufferUnmap(buffer: native::WGPUBuffer) {
    let buffer = buffer.as_ref().expect("invalid buffer");
    buffer.unmap();
}

#[no_mangle]
pub unsafe extern "C" fn wgpuBufferAddRef(buffer: native::WGPUBuffer) {
    assert!(!buffer.is_null(), "invalid buffer");
    Arc::increment_strong_count(buffer);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferRelease(buffer: native::WGPUBuffer) {
    assert!(!buffer.is_null(), "invalid buffer");
    Arc::decrement_strong_count(buffer);
}

// CommandBuffer methods

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandBufferAddRef(command_buffer: native::WGPUCommandBuffer) {
    assert!(!command_buffer.is_null(), "invalid command buffer");
    Arc::increment_strong_count(command_buffer);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandBufferRelease(command_buffer: native::WGPUCommandBuffer) {
    assert!(!command_buffer.is_null(), "invalid command buffer");
    Arc::decrement_strong_count(command_buffer);
}

// CommandEncoder methods

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderBeginComputePass(
    command_encoder: native::WGPUCommandEncoder,
    descriptor: Option<&native::WGPUComputePassDescriptor>,
) -> native::WGPUComputePassEncoder {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };

    // let timestamp_writes = descriptor.and_then(|descriptor| {
    //     descriptor.timestampWrites.as_ref().map(|timestamp_write| {
    //         wgc::command::PassTimestampWrites {
    //             query_set: timestamp_write
    //                 .querySet
    //                 .as_ref()
    //                 .expect("invalid query set in timestamp writes")
    //                 .id,
    //             beginning_of_pass_write_index: map_query_set_index(
    //                 timestamp_write.beginningOfPassWriteIndex,
    //             ),
    //             end_of_pass_write_index: map_query_set_index(timestamp_write.endOfPassWriteIndex),
    //         }
    //     })
    // });

    // let desc = match descriptor {
    //     Some(descriptor) => wgc::command::ComputePassDescriptor {
    //         label: string_view_into_label(descriptor.label),
    //         timestamp_writes,
    //     },
    //     None => wgc::command::ComputePassDescriptor {
    //         label: Label::default(),
    //         timestamp_writes,
    //     },
    // };

    // let (pass, err) = context.command_encoder_begin_compute_pass(command_encoder_id, &desc);
    // if let Some(cause) = err {
    //     handle_error(
    //         error_sink,
    //         cause,
    //         desc.label,
    //         "wgpuCommandEncoderBeginComputePass",
    //     );
    // }
    // Arc::into_raw(Arc::new(WGPUComputePassEncoderImpl {
    //     context: context.clone(),
    //     encoder: Box::into_raw(Box::new(pass)),
    //     error_sink: error_sink.clone(),
    // }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderBeginRenderPass(
    command_encoder: native::WGPUCommandEncoder,
    descriptor: Option<&native::WGPURenderPassDescriptor>,
) -> native::WGPURenderPassEncoder {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    // };
    // let descriptor = descriptor.expect("invalid descriptor");

    // let depth_stencil_attachment = descriptor.depthStencilAttachment.as_ref().map(|desc| {
    //     wgc::command::RenderPassDepthStencilAttachment {
    //         view: desc
    //             .view
    //             .as_ref()
    //             .expect("invalid texture view for depth stencil attachment")
    //             .id,
    //         depth: wgc::command::PassChannel {
    //             load_op: conv::map_load_op(desc.depthLoadOp, Some(desc.depthClearValue))
    //                 .or(Some(wgc::command::LoadOp::Load)),
    //             store_op: Some(
    //                 conv::map_store_op(desc.depthStoreOp).unwrap_or(wgc::command::StoreOp::Store),
    //             ),
    //             read_only: desc.depthReadOnly != 0,
    //         },
    //         stencil: wgc::command::PassChannel {
    //             load_op: conv::map_load_op(desc.stencilLoadOp, Some(desc.stencilClearValue))
    //                 .or(Some(wgc::command::LoadOp::Load)),
    //             store_op: Some(
    //                 conv::map_store_op(desc.stencilStoreOp).unwrap_or(wgc::command::StoreOp::Store),
    //             ),
    //             read_only: desc.stencilReadOnly != 0,
    //         },
    //     }
    // });

    // let timestamp_writes = descriptor.timestampWrites.as_ref().map(|timestamp_write| {
    //     wgc::command::PassTimestampWrites {
    //         query_set: timestamp_write
    //             .querySet
    //             .as_ref()
    //             .expect("invalid query set in timestamp writes")
    //             .id,
    //         beginning_of_pass_write_index: map_query_set_index(
    //             timestamp_write.beginningOfPassWriteIndex,
    //         ),
    //         end_of_pass_write_index: map_query_set_index(timestamp_write.endOfPassWriteIndex),
    //     }
    // });

    // let desc = wgc::command::RenderPassDescriptor {
    //     label: string_view_into_label(descriptor.label),
    //     color_attachments: Cow::Owned(
    //         make_slice(descriptor.colorAttachments, descriptor.colorAttachmentCount)
    //             .iter()
    //             .map(|color_attachment| {
    //                 if color_attachment.depthSlice != native::WGPU_DEPTH_SLICE_UNDEFINED {
    //                     log::warn!("Depth slice on color attachments is not implemented");
    //                 }

    //                 color_attachment.view.as_ref().map(|view| {
    //                     wgc::command::RenderPassColorAttachment {
    //                         view: view.id,
    //                         resolve_target: color_attachment.resolveTarget.as_ref().map(|v| v.id),
    //                         load_op: conv::map_load_op(
    //                             color_attachment.loadOp,
    //                             conv::map_color(&color_attachment.clearValue),
    //                         )
    //                         .expect("invalid load op for render pass color attachment"),
    //                         store_op: conv::map_store_op(color_attachment.storeOp)
    //                             .expect("invalid store op for render pass color attachment"),
    //                         // TODO: IDK
    //                         depth_slice: (color_attachment.depthSlice
    //                             != native::WGPU_DEPTH_SLICE_UNDEFINED)
    //                             .then_some(color_attachment.depthSlice),
    //                     }
    //                 })
    //             })
    //             .collect(),
    //     ),
    //     depth_stencil_attachment: depth_stencil_attachment.as_ref(),
    //     timestamp_writes: timestamp_writes.as_ref(),
    //     occlusion_query_set: descriptor.occlusionQuerySet.as_ref().map(|v| v.id),
    // };

    // let (pass, err) = context.command_encoder_begin_render_pass(command_encoder_id, &desc);
    // if let Some(cause) = err {
    //     handle_error(
    //         error_sink,
    //         cause,
    //         desc.label,
    //         "wgpuCommandEncoderBeginRenderPass",
    //     );
    // }
    // Arc::into_raw(Arc::new(WGPURenderPassEncoderImpl {
    //     context: context.clone(),
    //     encoder: Box::into_raw(Box::new(pass)),
    //     error_sink: error_sink.clone(),
    // }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderClearBuffer(
    command_encoder: native::WGPUCommandEncoder,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    // TODO:
    todo!()
    // let command_encoder = command_encoder.as_ref().expect("invalid command encoder");

    // let buffer_id = buffer.as_ref().expect("invalid buffer").id;

    // if let Err(cause) = context.command_encoder_clear_buffer(
    //     command_encoder_id,
    //     buffer_id,
    //     offset,
    //     match size {
    //         0 => panic!("invalid size"),
    //         conv::WGPU_WHOLE_SIZE => None,
    //         _ => Some(size),
    //     },
    // ) {
    //     handle_error(error_sink, cause, None, "wgpuCommandEncoderClearBuffer");
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyBufferToBuffer(
    command_encoder: native::WGPUCommandEncoder,
    source: native::WGPUBuffer,
    source_offset: u64,
    destination: native::WGPUBuffer,
    destination_offset: u64,
    size: u64,
) {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };
    // let source_buffer_id = source.as_ref().expect("invalid source").id;
    // let destination_buffer_id = destination.as_ref().expect("invalid destination").id;

    // if let Err(cause) = context.command_encoder_copy_buffer_to_buffer(
    //     command_encoder_id,
    //     source_buffer_id,
    //     source_offset,
    //     destination_buffer_id,
    //     destination_offset,
    //     Some(size),
    // ) {
    //     handle_error(
    //         error_sink,
    //         cause,
    //         None,
    //         "wgpuCommandEncoderCopyBufferToBuffer",
    //     );
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyBufferToTexture(
    command_encoder: native::WGPUCommandEncoder,
    source: Option<&native::WGPUTexelCopyBufferInfo>,
    destination: Option<&native::WGPUTexelCopyTextureInfo>,
    copy_size: Option<&native::WGPUExtent3D>,
) {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };

    // if let Err(cause) = context.command_encoder_copy_buffer_to_texture(
    //     command_encoder_id,
    //     &conv::map_image_copy_buffer(source.expect("invalid source")),
    //     &conv::map_image_copy_texture(destination.expect("invalid destination")),
    //     &conv::map_extent3d(copy_size.expect("invalid copy size")),
    // ) {
    //     handle_error(
    //         error_sink,
    //         cause,
    //         None,
    //         "wgpuCommandEncoderCopyBufferToTexture",
    //     );
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyTextureToBuffer(
    command_encoder: native::WGPUCommandEncoder,
    source: Option<&native::WGPUTexelCopyTextureInfo>,
    destination: Option<&native::WGPUTexelCopyBufferInfo>,
    copy_size: Option<&native::WGPUExtent3D>,
) {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };

    // if let Err(cause) = context.command_encoder_copy_texture_to_buffer(
    //     command_encoder_id,
    //     &conv::map_image_copy_texture(source.expect("invalid source")),
    //     &conv::map_image_copy_buffer(destination.expect("invalid destination")),
    //     &conv::map_extent3d(copy_size.expect("invalid copy size")),
    // ) {
    //     handle_error(
    //         error_sink,
    //         cause,
    //         None,
    //         "wgpuCommandEncoderCopyTextureToBuffer",
    //     );
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyTextureToTexture(
    command_encoder: native::WGPUCommandEncoder,
    source: Option<&native::WGPUTexelCopyTextureInfo>,
    destination: Option<&native::WGPUTexelCopyTextureInfo>,
    copy_size: Option<&native::WGPUExtent3D>,
) {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };

    // if let Err(cause) = context.command_encoder_copy_texture_to_texture(
    //     command_encoder_id,
    //     &conv::map_image_copy_texture(source.expect("invalid source")),
    //     &conv::map_image_copy_texture(destination.expect("invalid destination")),
    //     &conv::map_extent3d(copy_size.expect("invalid copy size")),
    // ) {
    //     handle_error(
    //         error_sink,
    //         cause,
    //         None,
    //         "wgpuCommandEncoderCopyTextureToTexture",
    //     );
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderFinish(
    command_encoder: native::WGPUCommandEncoder,
    descriptor: Option<&native::WGPUCommandBufferDescriptor>,
) -> native::WGPUCommandBuffer {
    // TODO:
    todo!()
    // let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    // let (command_encoder_id, context, error_sink) = (
    //     command_encoder.id,
    //     &command_encoder.context,
    //     &command_encoder.error_sink,
    // );
    // command_encoder.open.store(false, atomic::Ordering::SeqCst);

    // let desc = match descriptor {
    //     Some(descriptor) => wgt::CommandBufferDescriptor {
    //         label: string_view_into_label(descriptor.label),
    //     },
    //     None => wgt::CommandBufferDescriptor::default(),
    // };

    // let (command_buffer_id, error) = context.command_encoder_finish(command_encoder_id, &desc);
    // if let Some(cause) = error {
    //     handle_error(error_sink, cause, None, "wgpuCommandEncoderFinish");
    // }

    // Arc::into_raw(Arc::new(WGPUCommandBufferImpl {
    //     context: context.clone(),
    //     id: command_buffer_id,
    //     open: atomic::AtomicBool::new(false),
    // }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderInsertDebugMarker(
    command_encoder: native::WGPUCommandEncoder,
    marker_label: native::WGPUStringView,
) {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };

    // if let Err(cause) = context.command_encoder_insert_debug_marker(
    //     command_encoder_id,
    //     string_view_into_str(marker_label).unwrap_or(""),
    // ) {
    //     handle_error(
    //         error_sink,
    //         cause,
    //         None,
    //         "wgpuCommandEncoderInsertDebugMarker",
    //     );
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderPopDebugGroup(
    command_encoder: native::WGPUCommandEncoder,
) {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };

    // if let Err(cause) = context.command_encoder_pop_debug_group(command_encoder_id) {
    //     handle_error(error_sink, cause, None, "wgpuCommandEncoderPopDebugGroup");
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderPushDebugGroup(
    command_encoder: native::WGPUCommandEncoder,
    group_label: native::WGPUStringView,
) {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };

    // if let Err(cause) = context.command_encoder_push_debug_group(
    //     command_encoder_id,
    //     string_view_into_str(group_label).unwrap_or(""),
    // ) {
    //     handle_error(error_sink, cause, None, "wgpuCommandEncoderPushDebugGroup");
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderResolveQuerySet(
    command_encoder: native::WGPUCommandEncoder,
    query_set: native::WGPUQuerySet,
    first_query: u32,
    query_count: u32,
    destination: native::WGPUBuffer,
    destination_offset: u64,
) {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };
    // let query_set_id = query_set.as_ref().expect("invalid query set").id;
    // let destination_buffer_id = destination.as_ref().expect("invalid destination").id;

    // if let Err(cause) = context.command_encoder_resolve_query_set(
    //     command_encoder_id,
    //     query_set_id,
    //     first_query,
    //     query_count,
    //     destination_buffer_id,
    //     destination_offset,
    // ) {
    //     handle_error(error_sink, cause, None, "wgpuCommandEncoderResolveQuerySet");
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderWriteTimestamp(
    command_encoder: native::WGPUCommandEncoder,
    query_set: native::WGPUQuerySet,
    query_index: u32,
) {
    // TODO:
    todo!()
    // let (command_encoder_id, context, error_sink) = {
    //     let command_encoder = command_encoder.as_ref().expect("invalid command encoder");
    //     (
    //         command_encoder.id,
    //         &command_encoder.context,
    //         &command_encoder.error_sink,
    //     )
    // };
    // let query_set_id = query_set.as_ref().expect("invalid query set").id;

    // if let Err(cause) =
    //     context.command_encoder_write_timestamp(command_encoder_id, query_set_id, query_index)
    // {
    //     handle_error(error_sink, cause, None, "wgpuCommandEncoderWriteTimestamp");
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderAddRef(command_encoder: native::WGPUCommandEncoder) {
    assert!(!command_encoder.is_null(), "invalid command encoder");
    Arc::increment_strong_count(command_encoder);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderRelease(command_encoder: native::WGPUCommandEncoder) {
    assert!(!command_encoder.is_null(), "invalid command encoder");
    Arc::decrement_strong_count(command_encoder);
}

// ComputePassEncoder methods

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderDispatchWorkgroups(
    pass: native::WGPUComputePassEncoder,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    workgroup_count_z: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.compute_pass_dispatch_workgroups(
    //     encoder,
    //     workgroup_count_x,
    //     workgroup_count_y,
    //     workgroup_count_z,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderDispatchWorkgroups",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderDispatchWorkgroupsIndirect(
    pass: native::WGPUComputePassEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let indirect_buffer_id = indirect_buffer
    //     .as_ref()
    //     .expect("invalid indirect buffer")
    //     .id;

    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.compute_pass_dispatch_workgroups_indirect(
    //     encoder,
    //     indirect_buffer_id,
    //     indirect_offset,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderDispatchWorkgroupsIndirect",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderEnd(pass: native::WGPUComputePassEncoder) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.compute_pass_end(encoder) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(&pass.error_sink, cause, None, "wgpuComputePassEncoderEnd"),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderInsertDebugMarker(
    pass: native::WGPUComputePassEncoder,
    marker_label: native::WGPUStringView,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.compute_pass_insert_debug_marker(
    //     encoder,
    //     string_view_into_str(marker_label).unwrap_or(""),
    //     0,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderInsertDebugMarker",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderPopDebugGroup(pass: native::WGPUComputePassEncoder) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.compute_pass_pop_debug_group(encoder) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderPopDebugGroup",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderPushDebugGroup(
    pass: native::WGPUComputePassEncoder,
    group_label: native::WGPUStringView,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.compute_pass_push_debug_group(
    //     encoder,
    //     string_view_into_str(group_label).unwrap_or(""),
    //     0,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderPushDebugGroup",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderSetBindGroup(
    pass: native::WGPUComputePassEncoder,
    group_index: u32,
    bind_group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    // TODO:
    todo!()
    //let pass = pass.as_ref().expect("invalid compute pass");
    ////TODO: as per webgpu.h bindgroup is nullable
    //let bind_group_id = bind_group.as_ref().expect("invalid bind group").id;
    //let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    //match pass.context.compute_pass_set_bind_group(
    //    encoder,
    //    group_index,
    //    Some(bind_group_id),
    //    make_slice(dynamic_offsets, dynamic_offset_count),
    //) {
    //    Ok(()) => (),
    //    Err(cause) => handle_error(
    //        &pass.error_sink,
    //        cause,
    //        None,
    //        "wgpuComputePassEncoderSetBindGroup",
    //    ),
    //}
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderSetPipeline(
    pass: native::WGPUComputePassEncoder,
    compute_pipeline: native::WGPUComputePipeline,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let compute_pipeline_id = compute_pipeline
    //     .as_ref()
    //     .expect("invalid compute pipeline")
    //     .id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .compute_pass_set_pipeline(encoder, compute_pipeline_id)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderSetPipeline",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderAddRef(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    assert!(
        !compute_pass_encoder.is_null(),
        "invalid command pass encoder"
    );
    Arc::increment_strong_count(compute_pass_encoder);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderRelease(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    assert!(
        !compute_pass_encoder.is_null(),
        "invalid command pass encoder"
    );
    Arc::decrement_strong_count(compute_pass_encoder);
}

// ComputePipeline methods

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineGetBindGroupLayout(
    pipeline: native::WGPUComputePipeline,
    group_index: u32,
) -> native::WGPUBindGroupLayout {
    // TODO:
    todo!()
    // let (pipeline_id, context, error_sink) = {
    //     let pipeline = pipeline.as_ref().expect("invalid pipeline");
    //     (pipeline.id, &pipeline.context, &pipeline.error_sink)
    // };

    // let (bind_group_layout_id, error) =
    //     context.compute_pipeline_get_bind_group_layout(pipeline_id, group_index, None);
    // if let Some(cause) = error {
    //     handle_error(
    //         error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePipelineGetBindGroupLayout",
    //     );
    // }

    // Arc::into_raw(Arc::new(WGPUBindGroupLayoutImpl {
    //     context: context.clone(),
    //     id: bind_group_layout_id,
    // }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineAddRef(compute_pipeline: native::WGPUComputePipeline) {
    assert!(!compute_pipeline.is_null(), "invalid command pipeline");
    Arc::increment_strong_count(compute_pipeline);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineRelease(compute_pipeline: native::WGPUComputePipeline) {
    assert!(!compute_pipeline.is_null(), "invalid command pipeline");
    Arc::decrement_strong_count(compute_pipeline);
}

// Device methods

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBindGroup(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUBindGroupDescriptor>,
) -> native::WGPUBindGroup {
    let device = device.as_ref().expect("invalid device").device;
    let descriptor = descriptor.expect("invalid descriptor");
    let bind_group_layout = descriptor
        .layout
        .as_ref()
        .expect("invalid bind group layout for bind group descriptor");

    let entries = make_slice(descriptor.entries, descriptor.entryCount)
        .iter()
        .map(|entry| {
            follow_chain!(map_bind_group_entry((entry),
                WGPUSType_BindGroupEntryExtras => native::WGPUBindGroupEntryExtras)
            )
        })
        .collect::<Vec<_>>();

    let desc = wgpu::BindGroupDescriptor {
        label: string_view_into_str(descriptor.label),
        layout: bind_group_layout,
        entries: &entries,
    };
    let bind_group = device.create_bind_group(&desc);

    Arc::into_raw(Arc::new(bind_group))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBindGroupLayout(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUBindGroupLayoutDescriptor>,
) -> native::WGPUBindGroupLayout {
    let device = device.as_ref().expect("invalid device").device;
    let descriptor = descriptor.expect("invalid descriptor");

    let entries = make_slice(descriptor.entries, descriptor.entryCount)
        .iter()
        .map(|entry| {
            follow_chain!(map_bind_group_layout_entry((entry),
                WGPUSType_BindGroupLayoutEntryExtras => native::WGPUBindGroupLayoutEntryExtras)
            )
        })
        .collect::<Vec<_>>();

    let desc = wgpu::BindGroupLayoutDescriptor {
        label: string_view_into_str(descriptor.label),
        entries: &entries,
    };

    let bind_group_layout = device.create_bind_group_layout(&desc);

    Arc::into_raw(Arc::new(bind_group_layout))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBuffer(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUBufferDescriptor>,
) -> native::WGPUBuffer {
    let device = &device.as_ref().expect("invalid device").device;
    let descriptor = descriptor.expect("invalid descriptor");

    let desc = wgt::BufferDescriptor {
        label: string_view_into_str(descriptor.label),
        size: descriptor.size,
        usage: from_u64_bits(descriptor.usage).expect("invalid buffer usage"),
        mapped_at_creation: descriptor.mappedAtCreation != 0,
    };

    let buffer = device.create_buffer(&desc);

    Arc::into_raw(Arc::new(buffer))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateCommandEncoder(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUCommandEncoderDescriptor>,
) -> native::WGPUCommandEncoder {
    let device = &device.as_ref().expect("invalid device").device;
    let desc = match descriptor {
        Some(descriptor) => wgt::CommandEncoderDescriptor {
            label: string_view_into_str(descriptor.label),
        },
        None => wgt::CommandEncoderDescriptor::default(),
    };
    let command_encoder = device.create_command_encoder(&desc);

    Arc::into_raw(Arc::new(command_encoder))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateComputePipeline(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUComputePipelineDescriptor>,
) -> native::WGPUComputePipeline {
    // TODO:
    todo!()
    // let (device_id, context, error_sink) = {
    //     let device = device.as_ref().expect("invalid device");
    //     (device.id, &device.context, &device.error_sink)
    // };
    // let descriptor = descriptor.expect("invalid descriptor");

    // let desc = wgc::pipeline::ComputePipelineDescriptor {
    //     label: string_view_into_label(descriptor.label),
    //     layout: descriptor.layout.as_ref().map(|v| v.id),
    //     stage: wgc::pipeline::ProgrammableStageDescriptor {
    //         module: descriptor
    //             .compute
    //             .module
    //             .as_ref()
    //             .expect("invalid fragment shader module for render pipeline descriptor")
    //             .id
    //             .expect("invalid fragment shader module for render pipeline descriptor"),
    //         entry_point: string_view_into_label(descriptor.compute.entryPoint),
    //         constants: make_slice(
    //             descriptor.compute.constants,
    //             descriptor.compute.constantCount,
    //         )
    //         .iter()
    //         .map(|entry| {
    //             (
    //                 string_view_into_str(entry.key).unwrap_or("").to_string(),
    //                 entry.value,
    //             )
    //         })
    //         .collect(),
    //         // TODO(wgpu.h)
    //         zero_initialize_workgroup_memory: false,
    //     },
    //     // TODO(wgpu.h)
    //     cache: None,
    // };

    // let (compute_pipeline_id, error) =
    //     context.device_create_compute_pipeline(device_id, &desc, None, None);
    // if let Some(cause) = error {
    //     if let wgc::pipeline::CreateComputePipelineError::Internal(ref error) = cause {
    //         log::warn!(
    //             "Shader translation error for stage {:?}: {}",
    //             wgt::ShaderStages::COMPUTE,
    //             error
    //         );
    //         log::warn!("Please report it to https://github.com/gfx-rs/wgpu");
    //     }
    //     handle_error(
    //         error_sink,
    //         cause,
    //         desc.label,
    //         "wgpuDeviceCreateComputePipeline",
    //     );
    // }

    // Arc::into_raw(Arc::new(WGPUComputePipelineImpl {
    //     context: context.clone(),
    //     id: compute_pipeline_id,
    //     error_sink: error_sink.clone(),
    // }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreatePipelineLayout(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUPipelineLayoutDescriptor>,
) -> native::WGPUPipelineLayout {
    let device = device.as_ref().expect("invalid device").device;
    let descriptor = descriptor.expect("invalid descriptor");

    let desc = follow_chain!(
        map_pipeline_layout_descriptor(
            (descriptor),
            WGPUSType_PipelineLayoutExtras => native::WGPUPipelineLayoutExtras)
    );
    let pipeline_layout = device.create_pipeline_layout(&desc);

    Arc::into_raw(Arc::new(pipeline_layout))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateQuerySet(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUQuerySetDescriptor>,
) -> native::WGPUQuerySet {
    let device = device.as_ref().expect("invalid device");
    let descriptor = descriptor.expect("invalid query set descriptor");

    let desc = follow_chain!(
        map_query_set_descriptor(
            (descriptor),
            WGPUSType_QuerySetDescriptorExtras => native::WGPUQuerySetDescriptorExtras)
    );

    let query_set = device.create_query_set(&desc);

    Arc::into_raw(Arc::new(query_set))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateRenderBundleEncoder(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPURenderBundleEncoderDescriptor>,
) -> native::WGPURenderBundleEncoder {
    // TODO:
    todo!()
    // let (device_id, context) = {
    //     let device = device.as_ref().expect("invalid device");
    //     (device.id, &device.context)
    // };
    // let descriptor = descriptor.expect("invalid descriptor");

    // let desc = wgc::command::RenderBundleEncoderDescriptor {
    //     label: string_view_into_label(descriptor.label),
    //     color_formats: make_slice(descriptor.colorFormats, descriptor.colorFormatCount)
    //         .iter()
    //         .map(|format| conv::map_texture_format(*format))
    //         .collect(),
    //     depth_stencil: conv::map_texture_format(descriptor.depthStencilFormat).map(|format| {
    //         wgt::RenderBundleDepthStencil {
    //             format,
    //             depth_read_only: descriptor.depthReadOnly != 0,
    //             stencil_read_only: descriptor.stencilReadOnly != 0,
    //         }
    //     }),
    //     sample_count: descriptor.sampleCount,
    //     multiview: None,
    // };

    // match wgc::command::RenderBundleEncoder::new(&desc, device_id, None) {
    //     Ok(encoder) => Arc::into_raw(Arc::new(WGPURenderBundleEncoderImpl {
    //         context: context.clone(),
    //         encoder: Box::into_raw(Box::new(Some(Box::into_raw(Box::new(encoder))))),
    //     })),
    //     Err(cause) => {
    //         handle_error_fatal(cause, "wgpuDeviceCreateRenderBundleEncoder");
    //     }
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateRenderPipeline(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPURenderPipelineDescriptor>,
) -> native::WGPURenderPipeline {
    // TODO:
    todo!()
    // let (device_id, context, error_sink) = {
    //     let device = device.as_ref().expect("invalid device");
    //     (device.id, &device.context, &device.error_sink)
    // };
    // let descriptor = descriptor.expect("invalid descriptor");

    // let desc = wgc::pipeline::RenderPipelineDescriptor {
    //     label: string_view_into_label(descriptor.label),
    //     layout: descriptor.layout.as_ref().map(|v| v.id),
    //     vertex: wgc::pipeline::VertexState {
    //         stage: wgc::pipeline::ProgrammableStageDescriptor {
    //             module: descriptor
    //                 .vertex
    //                 .module
    //                 .as_ref()
    //                 .expect("invalid vertex shader module for vertex state")
    //                 .id
    //                 .expect("invalid vertex shader module for vertex state"),
    //             entry_point: string_view_into_label(descriptor.vertex.entryPoint),
    //             constants: make_slice(descriptor.vertex.constants, descriptor.vertex.constantCount)
    //                 .iter()
    //                 .map(|entry| {
    //                     (
    //                         string_view_into_str(entry.key).unwrap_or("").to_string(),
    //                         entry.value,
    //                     )
    //                 })
    //                 .collect(),
    //             // TODO(wgpu.h)
    //             zero_initialize_workgroup_memory: false,
    //         },
    //         buffers: Cow::Owned(
    //             make_slice(descriptor.vertex.buffers, descriptor.vertex.bufferCount)
    //                 .iter()
    //                 .map(|buffer| wgc::pipeline::VertexBufferLayout {
    //                     array_stride: buffer.arrayStride,
    //                     step_mode: match buffer.stepMode {
    //                         native::WGPUVertexStepMode_Vertex => wgt::VertexStepMode::Vertex,
    //                         native::WGPUVertexStepMode_Instance => wgt::VertexStepMode::Instance,
    //                         _ => panic!("invalid vertex step mode for vertex buffer layout"),
    //                     },
    //                     attributes: Cow::Owned(
    //                         make_slice(buffer.attributes, buffer.attributeCount)
    //                             .iter()
    //                             .map(|attribute| wgt::VertexAttribute {
    //                                 format: conv::map_vertex_format(attribute.format)
    //                                     .expect("invalid vertex format for vertex attribute"),
    //                                 offset: attribute.offset,
    //                                 shader_location: attribute.shaderLocation,
    //                             })
    //                             .collect(),
    //                     ),
    //                 })
    //                 .collect(),
    //         ),
    //     },
    //     primitive: wgt::PrimitiveState {
    //         topology: conv::map_primitive_topology(descriptor.primitive.topology)
    //             .unwrap_or(wgt::PrimitiveTopology::TriangleList),
    //         strip_index_format: conv::map_index_format(descriptor.primitive.stripIndexFormat).ok(),
    //         front_face: match descriptor.primitive.frontFace {
    //             native::WGPUFrontFace_CCW | native::WGPUFrontFace_Undefined => wgt::FrontFace::Ccw,
    //             native::WGPUFrontFace_CW => wgt::FrontFace::Cw,
    //             _ => panic!("invalid front face for primitive state"),
    //         },
    //         cull_mode: match descriptor.primitive.cullMode {
    //             native::WGPUCullMode_None | native::WGPUCullMode_Undefined => None,
    //             native::WGPUCullMode_Front => Some(wgt::Face::Front),
    //             native::WGPUCullMode_Back => Some(wgt::Face::Back),
    //             _ => panic!("invalid cull mode for primitive state"),
    //         },
    //         unclipped_depth: descriptor.primitive.unclippedDepth != 0,
    //         polygon_mode: wgt::PolygonMode::Fill,
    //         conservative: false,
    //     },
    //     depth_stencil: descriptor.depthStencil.as_ref().map(|desc| {
    //         let format = conv::map_texture_format(desc.format)
    //             .expect("invalid texture format for depth stencil state");

    //         // Validation per spec.
    //         if texture_format_has_depth(format) {
    //             if desc.depthWriteEnabled == native::WGPUOptionalBool_Undefined {
    //                 panic!("Depth write not specified for depth format")
    //             }
    //         } else {
    //             if desc.depthWriteEnabled == native::WGPUOptionalBool_True {
    //                 panic!("Depth write enabled for non-depth format")
    //             }
    //         }

    //         wgt::DepthStencilState {
    //             format,
    //             depth_write_enabled: desc.depthWriteEnabled == native::WGPUOptionalBool_True,
    //             // TODO: Is validation correct if we return always for undefined depth compare?
    //             depth_compare: conv::map_compare_function(desc.depthCompare)
    //                 .expect("invalid depth compare function for depth stencil state")
    //                 .unwrap_or(wgt::CompareFunction::Always),
    //             stencil: wgt::StencilState {
    //                 front: conv::map_stencil_face_state(desc.stencilFront, "front"),
    //                 back: conv::map_stencil_face_state(desc.stencilBack, "back"),
    //                 read_mask: desc.stencilReadMask,
    //                 write_mask: desc.stencilWriteMask,
    //             },
    //             bias: wgt::DepthBiasState {
    //                 constant: desc.depthBias,
    //                 slope_scale: desc.depthBiasSlopeScale,
    //                 clamp: desc.depthBiasClamp,
    //             },
    //         }
    //     }),
    //     multisample: wgt::MultisampleState {
    //         count: descriptor.multisample.count,
    //         mask: descriptor.multisample.mask as u64,
    //         alpha_to_coverage_enabled: descriptor.multisample.alphaToCoverageEnabled != 0,
    //     },
    //     fragment: descriptor
    //         .fragment
    //         .as_ref()
    //         .map(|fragment| wgc::pipeline::FragmentState {
    //             stage: wgc::pipeline::ProgrammableStageDescriptor {
    //                 module: fragment
    //                     .module
    //                     .as_ref()
    //                     .expect("invalid fragment shader module for render pipeline descriptor")
    //                     .id
    //                     .expect("invalid fragment shader module for render pipeline descriptor"),
    //                 entry_point: string_view_into_label(fragment.entryPoint),
    //                 constants: make_slice(fragment.constants, fragment.constantCount)
    //                     .iter()
    //                     .map(|entry| {
    //                         (
    //                             string_view_into_str(entry.key).unwrap_or("").to_string(),
    //                             entry.value,
    //                         )
    //                     })
    //                     .collect(),
    //                 // TODO(wgpu.h)
    //                 zero_initialize_workgroup_memory: false,
    //             },
    //             targets: Cow::Owned(
    //                 make_slice(fragment.targets, fragment.targetCount)
    //                     .iter()
    //                     .map(|color_target| {
    //                         conv::map_texture_format(color_target.format).map(|format| {
    //                             wgt::ColorTargetState {
    //                                 format,
    //                                 blend: color_target.blend.as_ref().map(|blend| {
    //                                     wgt::BlendState {
    //                                         color: conv::map_blend_component(blend.color),
    //                                         alpha: conv::map_blend_component(blend.alpha),
    //                                     }
    //                                 }),
    //                                 write_mask: from_u64_bits(color_target.writeMask).unwrap(),
    //                             }
    //                         })
    //                     })
    //                     .collect(),
    //             ),
    //         }),
    //     // TODO(wgpu.h)
    //     multiview: None,
    //     // TODO(wgpu.h)
    //     cache: None,
    // };

    // let (render_pipeline_id, error) =
    //     context.device_create_render_pipeline(device_id, &desc, None, None);
    // if let Some(cause) = error {
    //     if let wgc::pipeline::CreateRenderPipelineError::Internal { stage, ref error } = cause {
    //         log::error!("Shader translation error for stage {:?}: {}", stage, error);
    //         log::error!("Please report it to https://github.com/gfx-rs/wgpu");
    //     }
    //     handle_error(
    //         error_sink,
    //         cause,
    //         desc.label,
    //         "wgpuDeviceCreateRenderPipeline",
    //     );
    // }

    // Arc::into_raw(Arc::new(WGPURenderPipelineImpl {
    //     context: context.clone(),
    //     id: render_pipeline_id,
    //     error_sink: error_sink.clone(),
    // }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateSampler(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUSamplerDescriptor>,
) -> native::WGPUSampler {
    let device = &device.as_ref().expect("invalid device").device;

    let desc = match descriptor {
        Some(descriptor) => wgpu::SamplerDescriptor {
            label: string_view_into_str(descriptor.label),
            address_mode_u: conv::map_address_mode(descriptor.addressModeU)
                .unwrap_or(wgt::AddressMode::ClampToEdge),
            address_mode_v: conv::map_address_mode(descriptor.addressModeV)
                .unwrap_or(wgt::AddressMode::ClampToEdge),
            address_mode_w: conv::map_address_mode(descriptor.addressModeW)
                .unwrap_or(wgt::AddressMode::ClampToEdge),
            mag_filter: conv::map_filter_mode(descriptor.magFilter)
                .unwrap_or(wgt::FilterMode::Nearest),
            min_filter: conv::map_filter_mode(descriptor.minFilter)
                .unwrap_or(wgt::FilterMode::Nearest),
            mipmap_filter: conv::map_mipmap_filter_mode(descriptor.mipmapFilter)
                .unwrap_or(wgt::FilterMode::Nearest),
            lod_min_clamp: descriptor.lodMinClamp,
            lod_max_clamp: descriptor.lodMaxClamp,
            compare: conv::map_compare_function(descriptor.compare)
                .expect("Invalid compare function"),
            anisotropy_clamp: descriptor.maxAnisotropy,
            // TODO(wgpu.h)
            border_color: None,
        },
        // wgpu-core doesn't have Default implementation for SamplerDescriptor,
        // use defaults from spec.
        // ref: https://gpuweb.github.io/gpuweb/#GPUSamplerDescriptor
        None => wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgt::AddressMode::ClampToEdge,
            address_mode_v: wgt::AddressMode::ClampToEdge,
            address_mode_w: wgt::AddressMode::ClampToEdge,
            mag_filter: wgt::FilterMode::Nearest,
            min_filter: wgt::FilterMode::Nearest,
            mipmap_filter: wgt::FilterMode::Nearest,
            lod_min_clamp: 0f32,
            lod_max_clamp: 32f32,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        },
    };

    let sampler = device.create_sampler(&desc);

    Arc::into_raw(Arc::new(sampler))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateShaderModule(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUShaderModuleDescriptor>,
) -> native::WGPUShaderModule {
    let device = device.as_ref().expect("invalid device").device;
    let descriptor = descriptor.expect("invalid descriptor");
    let desc_label = string_view_into_str(descriptor.label);

    let source = match follow_chain!(
        map_shader_module((descriptor),
        WGPUSType_ShaderSourceSPIRV => native::WGPUShaderSourceSPIRV,
        WGPUSType_ShaderSourceWGSL => native::WGPUShaderSourceWGSL,
        WGPUSType_ShaderSourceGLSL => native::WGPUShaderSourceGLSL)
    ) {
        Ok(source) => source,
        Err(cause) => {
            // TODO:
            todo!()
            // handle_error(
            //     error_sink,
            //     cause,
            //     desc_label,
            //     "wgpuDeviceCreateShaderModule",
            // );

            // return Arc::into_raw(Arc::new(WGPUShaderModuleImpl {
            //     context: context.clone(),
            //     id: None,
            // }));
        }
    };

    let desc = wgpu::ShaderModuleDescriptor {
        label: desc_label,
        source,
    };

    let shader_module = device.create_shader_module(desc);

    Arc::into_raw(Arc::new(shader_module))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateTexture(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUTextureDescriptor>,
) -> native::WGPUTexture {
    let device = &device.as_ref().expect("invalid device").device;
    let descriptor = descriptor.expect("invalid descriptor");

    let view_formats = make_slice(descriptor.viewFormats, descriptor.viewFormatCount)
        .iter()
        .map(|v| conv::map_texture_format(*v).expect("invalid view format for texture descriptor"))
        .collect::<Vec<_>>();

    let desc = wgt::TextureDescriptor {
        label: string_view_into_str(descriptor.label),
        size: conv::map_extent3d(&descriptor.size),
        mip_level_count: descriptor.mipLevelCount,
        sample_count: descriptor.sampleCount,
        dimension: conv::map_texture_dimension(descriptor.dimension)
            .unwrap_or(wgt::TextureDimension::D2),
        format: conv::map_texture_format(descriptor.format)
            .expect("invalid texture format for texture descriptor"),
        usage: from_u64_bits(descriptor.usage)
            .expect("invalid texture usage for texture descriptor"),
        view_formats: view_formats.as_slice(),
    };

    let texture = device.create_texture(&desc);

    Arc::into_raw(Arc::new(texture))
}

#[no_mangle]
pub extern "C" fn wgpuDeviceDestroy(_device: native::WGPUDevice) {
    //TODO: needs to be implemented in wgpu-core
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetFeatures(
    device: native::WGPUDevice,
    features: Option<&mut native::WGPUSupportedFeatures>,
) -> native::WGPUStatus {
    let device = &device.as_ref().expect("invalid device").device;
    let device_features = device.features();
    let features = features.expect("invalid return pointer \"features\"");

    return_features(features, device_features);

    native::WGPUStatus_Success
}

#[no_mangle]
pub unsafe extern "C" fn wgpuSupportedFeaturesFreeMembers(
    supported_features: native::WGPUSupportedFeatures,
) {
    if !supported_features.features.is_null() && supported_features.featureCount > 0 {
        drop(Box::from_raw(slice::from_raw_parts_mut(
            supported_features.features as *mut native::WGPUFeatureName,
            supported_features.featureCount,
        )))
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetLimits(
    device: native::WGPUDevice,
    limits: Option<&mut native::WGPULimits>,
) -> native::WGPUBool {
    let device = &device.as_ref().expect("invalid device").device;
    let limits = limits.expect("invalid return pointer \"limits\"");

    let wgt_limits = device.limits();
    conv::write_limits_struct(wgt_limits, limits);

    true as native::WGPUBool // indicates that we can fill WGPUChainedStructOut
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetQueue(device: native::WGPUDevice) -> native::WGPUQueue {
    let queue = device.as_ref().expect("invalid device").queue.clone();

    Arc::into_raw(Arc::new(queue))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceHasFeature(
    device: native::WGPUDevice,
    feature: native::WGPUFeatureName,
) -> native::WGPUBool {
    let device = &device.as_ref().expect("invalid device").device;
    let device_features = device.features();

    let feature = match conv::map_feature(feature) {
        Some(feature) => feature,
        None => return false as native::WGPUBool,
    };

    device_features.contains(feature) as native::WGPUBool
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDevicePopErrorScope(
    device: native::WGPUDevice,
    callback_info: native::WGPUPopErrorScopeCallbackInfo,
) -> native::WGPUFuture {
    // TODO:
    //
    // let device = device.as_ref().expect("invalid device");
    // let callback = callback_info.callback.expect("invalid callback");
    // let mut error_sink = device.error_sink.lock();
    // let scope = error_sink.scopes.pop().unwrap();

    // match scope.error {
    //     Some(error) => {
    //         let typ = match error {
    //             crate::Error::OutOfMemory { .. } => native::WGPUErrorType_OutOfMemory,
    //             crate::Error::Validation { .. } => native::WGPUErrorType_Validation,
    //             // We handle device lost error early in ErrorSinkRaw::handle_error
    //             // so we should never get device lost error here.
    //             crate::Error::DeviceLost { .. } => unreachable!(),
    //         };

    //         let msg = error.to_string();
    //         unsafe {
    //             callback(
    //                 native::WGPUPopErrorScopeStatus_Success,
    //                 typ,
    //                 str_into_string_view(&msg),
    //                 callback_info.userdata1,
    //                 callback_info.userdata2,
    //             );
    //         };
    //     }
    //     None => {
    //         unsafe {
    //             callback(
    //                 native::WGPUPopErrorScopeStatus_Success,
    //                 native::WGPUErrorType_NoError,
    //                 EMPTY_STRING,
    //                 callback_info.userdata1,
    //                 callback_info.userdata2,
    //             );
    //         };
    //     }
    // };

    NULL_FUTURE
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDevicePushErrorScope(
    device: native::WGPUDevice,
    filter: native::WGPUErrorFilter,
) {
    // TODO:
    //
    // let device = device.as_ref().expect("invalid device");
    // let mut error_sink = device.error_sink.lock();
    // error_sink.scopes.push(ErrorScope {
    //     error: None,
    //     filter: match filter {
    //         native::WGPUErrorFilter_Validation => ErrorFilter::Validation,
    //         native::WGPUErrorFilter_OutOfMemory => ErrorFilter::OutOfMemory,
    //         _ => panic!("invalid error filter"),
    //     },
    // });
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceAddRef(device: native::WGPUDevice) {
    assert!(!device.is_null(), "invalid device");
    Arc::increment_strong_count(device);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceRelease(device: native::WGPUDevice) {
    assert!(!device.is_null(), "invalid device");
    Arc::decrement_strong_count(device);
}

// Instance methods

#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceCreateSurface(
    instance: native::WGPUInstance,
    descriptor: Option<&native::WGPUSurfaceDescriptor>,
) -> native::WGPUSurface {
    let instance = &instance.as_ref().expect("invalid instance");
    let descriptor = descriptor.expect("invalid descriptor");

    let create_surface_params = follow_chain!(
        map_surface((descriptor),
            WGPUSType_SurfaceSourceWindowsHWND => native::WGPUSurfaceSourceWindowsHWND,
            WGPUSType_SurfaceSourceXCBWindow => native::WGPUSurfaceSourceXCBWindow,
            WGPUSType_SurfaceSourceXlibWindow => native::WGPUSurfaceSourceXlibWindow,
            WGPUSType_SurfaceSourceWaylandSurface => native::WGPUSurfaceSourceWaylandSurface,
            WGPUSType_SurfaceSourceMetalLayer => native::WGPUSurfaceSourceMetalLayer,
            WGPUSType_SurfaceSourceAndroidNativeWindow => native::WGPUSurfaceSourceAndroidNativeWindow,
            WGPUSType_SurfaceSourceSwapChainPanel => native::WGPUSurfaceSourceSwapChainPanel)
    );

    let surface = match create_surface_params {
        CreateSurfaceParams::Raw((rdh, rwh)) => {
            match instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: rdh,
                raw_window_handle: rwh,
            }) {
                Ok(surface) => surface,
                Err(cause) => handle_error_fatal(cause, "wgpuInstanceCreateSurface"),
            }
        }
        // TODO:
        #[cfg(all(any(target_os = "ios", target_os = "macos"), feature = "metal"))]
        CreateSurfaceParams::Metal(layer) => {
            match context.instance_create_surface_metal(layer, None) {
                Ok(surface) => surface,
                Err(cause) => handle_error_fatal(cause, "wgpuInstanceCreateSurface"),
            }
        }
        #[cfg(all(target_os = "windows", feature = "dx12"))]
        CreateSurfaceParams::SwapChainPanel(panel) => {
            match context.instance_create_surface_from_swap_chain_panel(panel, None) {
                Ok(surface) => surface,
                Err(cause) => handle_error_fatal(cause, "wgpuInstanceCreateSurface"),
            }
        }
    };

    Arc::into_raw(Arc::new(WGPUSurfaceImpl {
        surface: Box::new(surface),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceProcessEvents(instance: native::WGPUInstance) {
    let instance = &instance.as_ref().expect("invalid instance");

    instance.poll_all(false);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceRequestAdapter(
    instance: native::WGPUInstance,
    options: Option<&native::WGPURequestAdapterOptions>,
    callback_info: native::WGPURequestAdapterCallbackInfo,
) -> native::WGPUFuture {
    let instance = instance.as_ref().expect("invalid instance");
    let callback = callback_info.callback.expect("invalid callback");

    let (desc, inputs) = match options {
        Some(options) => (
            wgt::RequestAdapterOptions {
                power_preference: match options.powerPreference {
                    native::WGPUPowerPreference_LowPower => wgt::PowerPreference::LowPower,
                    native::WGPUPowerPreference_HighPerformance => {
                        wgt::PowerPreference::HighPerformance
                    }
                    _ => wgt::PowerPreference::default(),
                },
                force_fallback_adapter: options.forceFallbackAdapter != 0,
                compatible_surface: options.compatibleSurface.as_ref(),
            },
            match options.backendType {
                native::WGPUBackendType_Undefined => wgt::Backends::all(),
                native::WGPUBackendType_Null => wgt::Backends::empty(),
                native::WGPUBackendType_WebGPU => wgt::Backends::BROWSER_WEBGPU,
                native::WGPUBackendType_D3D12 => wgt::Backends::DX12,
                native::WGPUBackendType_Metal => wgt::Backends::METAL,
                native::WGPUBackendType_Vulkan => wgt::Backends::VULKAN,
                native::WGPUBackendType_OpenGL => wgt::Backends::GL,
                native::WGPUBackendType_OpenGLES => wgt::Backends::GL,
                native::WGPUBackendType_D3D11 => {
                    callback(
                        native::WGPURequestAdapterStatus_Error,
                        std::ptr::null_mut(),
                        str_into_string_view("unsupported backend type: d3d11"),
                        callback_info.userdata1,
                        callback_info.userdata2,
                    );
                    return NULL_FUTURE;
                }
                backend_type => panic!("invalid backend type: 0x{backend_type:08X}"),
            },
        ),
        None => (wgt::RequestAdapterOptions::default(), wgt::Backends::all()),
    };

    // TODO
    // match instance.request_adapter(&desc) {
    //     Ok(adapter_id) => {
    //         callback(
    //             native::WGPURequestAdapterStatus_Success,
    //             Arc::into_raw(Arc::new(WGPUAdapterImpl {
    //                 context: context.clone(),
    //                 id: adapter_id,
    //             })),
    //             EMPTY_STRING,
    //             callback_info.userdata1,
    //             callback_info.userdata2,
    //         );
    //     }
    //     Err(err) => {
    //         let message = format_error(&err);
    //         callback(
    //             match err {
    //                 wgt::RequestAdapterError::NotFound {
    //                     active_backends: _,
    //                     requested_backends: _,
    //                     supported_backends: _,
    //                     no_fallback_backends: _,
    //                     no_adapter_backends: _,
    //                     incompatible_surface_backends: _,
    //                 } => native::WGPURequestAdapterStatus_Unavailable,
    //                 _ => native::WGPURequestAdapterStatus_Unknown,
    //             },
    //             std::ptr::null_mut(),
    //             str_into_string_view(&message),
    //             callback_info.userdata1,
    //             callback_info.userdata2,
    //         );
    //     }
    // };

    NULL_FUTURE
}

#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceEnumerateAdapters(
    instance: native::WGPUInstance,
    options: Option<&native::WGPUInstanceEnumerateAdapterOptions>,
    adapters: *mut native::WGPUAdapter,
) -> usize {
    let instance = &instance.as_ref().expect("invalid instance");

    let inputs = match options {
        Some(options) => {
            map_instance_backend_flags(options.backends as native::WGPUInstanceBackend)
        }
        None => wgt::Backends::all(),
    };

    let result = instance.enumerate_adapters(inputs);
    let count = result.len();

    if !adapters.is_null() {
        let temp = std::slice::from_raw_parts_mut(adapters, count);

        result.into_iter().enumerate().for_each(|(i, adapter)| {
            // It's users responsibility to drop the adapters they
            // don't need.

            temp[i] = Arc::into_raw(Arc::new(adapter));
        });
    }

    count
}

#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceAddRef(instance: native::WGPUInstance) {
    assert!(!instance.is_null(), "invalid instance");
    Arc::increment_strong_count(instance);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceRelease(instance: native::WGPUInstance) {
    assert!(!instance.is_null(), "invalid instance");
    Arc::decrement_strong_count(instance);
}

// PipelineLayout methods

#[no_mangle]
pub unsafe extern "C" fn wgpuPipelineLayoutAddRef(pipeline_layout: native::WGPUPipelineLayout) {
    assert!(!pipeline_layout.is_null(), "invalid pipeline layout");
    Arc::increment_strong_count(pipeline_layout);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuPipelineLayoutRelease(pipeline_layout: native::WGPUPipelineLayout) {
    assert!(!pipeline_layout.is_null(), "invalid pipeline layout");
    Arc::decrement_strong_count(pipeline_layout);
}

// QuerySet methods

#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetDestroy(_query_set: native::WGPUQuerySet) {
    //TODO: needs to be implemented in wgpu-core
}

#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetGetCount(query_set: native::WGPUQuerySet) -> u32 {
    let query_set = query_set.as_ref().expect("invalid query set");
    query_set.data.query_type
}

#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetGetType(
    query_set: native::WGPUQuerySet,
) -> native::WGPUQueryType {
    let query_set = query_set.as_ref().expect("invalid query set");
    query_set.data.query_count
}

#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetAddRef(query_set: native::WGPUQuerySet) {
    assert!(!query_set.is_null(), "invalid query set");
    Arc::increment_strong_count(query_set);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetRelease(query_set: native::WGPUQuerySet) {
    assert!(!query_set.is_null(), "invalid query set");
    Arc::decrement_strong_count(query_set);
}

// Queue methods

#[no_mangle]
pub unsafe extern "C" fn wgpuQueueOnSubmittedWorkDone(
    queue: native::WGPUQueue,
    callback_info: native::WGPUQueueWorkDoneCallbackInfo,
) -> native::WGPUFuture {
    let queue = &queue.as_ref().expect("invalid queue");
    let callback = callback_info.callback.expect("invalid callback");
    let userdata = new_userdata!(callback_info);

    let closure = move || {
        callback(
            native::WGPUQueueWorkDoneStatus_Success,
            userdata.get_1(),
            userdata.get_2(),
        );
    };

    queue.on_submitted_work_done(closure);

    // TODO: Properly handle futures.
    NULL_FUTURE
}

#[no_mangle]
pub unsafe extern "C" fn wgpuQueueSubmit(
    queue: native::WGPUQueue,
    command_count: usize,
    commands: *const native::WGPUCommandBuffer,
) {
    let queue = queue.as_ref().expect("invalid queue");

    let command_buffers = make_slice(commands, command_count)
        .into_iter()
        .map(|command_buffer| {
            // TODO: NOTE somewhere that `commands` cannot be reused.
            let command_buffer = unsafe { std::ptr::read(*command_buffer) };
            command_buffer
        })
        .collect::<SmallVec<[_; 4]>>();

    queue.submit(command_buffers);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuQueueWriteBuffer(
    queue: native::WGPUQueue,
    buffer: native::WGPUBuffer,
    buffer_offset: u64,
    data: *const u8, // TODO: Check - this might not follow the header
    data_size: usize,
) {
    let queue = queue.as_ref().expect("invalid queue");
    let buffer = buffer.as_ref().expect("invalid buffer");

    queue.write_buffer(buffer, buffer_offset, make_slice(data, data_size))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuQueueWriteTexture(
    queue: native::WGPUQueue,
    destination: Option<&native::WGPUTexelCopyTextureInfo>,
    data: *const u8, // TODO: Check - this might not follow the header
    data_size: usize,
    data_layout: Option<&native::WGPUTexelCopyBufferLayout>,
    write_size: Option<&native::WGPUExtent3D>,
) {
    let queue = queue.as_ref().expect("invalid queue");

    queue.write_texture(
        &conv::map_image_copy_texture(destination.expect("invalid destination")),
        make_slice(data, data_size),
        conv::map_texture_data_layout(data_layout.expect("invalid data layout")),
        conv::map_extent3d(write_size.expect("invalid write size")),
    )
}

#[no_mangle]
pub unsafe extern "C" fn wgpuQueueAddRef(queue: native::WGPUQueue) {
    assert!(!queue.is_null(), "invalid queue");
    Arc::increment_strong_count(queue);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueRelease(queue: native::WGPUQueue) {
    assert!(!queue.is_null(), "invalid queue");
    Arc::decrement_strong_count(queue);
}

// RenderBundle methods

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleAddRef(render_bundle: native::WGPURenderBundle) {
    assert!(!render_bundle.is_null(), "invalid render bundle");
    Arc::increment_strong_count(render_bundle);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleRelease(render_bundle: native::WGPURenderBundle) {
    assert!(!render_bundle.is_null(), "invalid render bundle");
    Arc::decrement_strong_count(render_bundle);
}

// RenderBundleEncoder methods

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDraw(
    bundle: native::WGPURenderBundleEncoder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    // TODO:
    todo!()
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let encoder = bundle.render_bundle_encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_draw(
    //     encoder,
    //     vertex_count,
    //     instance_count,
    //     first_vertex,
    //     first_instance,
    // );
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDrawIndexed(
    bundle: native::WGPURenderBundleEncoder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    // TODO:
    todo!()
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_draw_indexed(
    //     encoder,
    //     index_count,
    //     instance_count,
    //     first_index,
    //     base_vertex,
    //     first_instance,
    // );
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDrawIndexedIndirect(
    bundle: native::WGPURenderBundleEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    // TODO:
    todo!()
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let indirect_buffer_id = indirect_buffer
    //     .as_ref()
    //     .expect("invalid indirect buffer")
    //     .id;
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_draw_indexed_indirect(
    //     encoder,
    //     indirect_buffer_id,
    //     indirect_offset,
    // );
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDrawIndirect(
    bundle: native::WGPURenderBundleEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    // TODO:
    todo!()
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let indirect_buffer_id = indirect_buffer
    //     .as_ref()
    //     .expect("invalid indirect buffer")
    //     .id;
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_draw_indirect(encoder, indirect_buffer_id, indirect_offset);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderFinish(
    bundle: native::WGPURenderBundleEncoder,
    descriptor: Option<&native::WGPURenderBundleDescriptor>,
) -> native::WGPURenderBundle {
    // TODO:
    //
    todo!()
    //
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let encoder = bundle.render_bundle_encoder.as_mut();

    // let desc = match descriptor {
    //     Some(descriptor) => wgt::RenderBundleDescriptor {
    //         label: string_view_into_str(descriptor.label),
    //     },
    //     None => wgt::RenderBundleDescriptor::default(),
    // };

    // let render_bundle = encoder.finish(&desc);

    // Arc::into_raw(Arc::new(render_bundle))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderInsertDebugMarker(
    _bundle: native::WGPURenderBundleEncoder,
    _marker_label: native::WGPUStringView,
) {
    // These functions are not implemented in wgpu-core, and the API is incompatible with the new WGPUStringView.
    // Commenting out until it's actually implemented.

    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_insert_debug_marker(encoder, marker_label);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderPopDebugGroup(
    _bundle: native::WGPURenderBundleEncoder,
) {
    // These functions are not implemented in wgpu-core, and the API is incompatible with the new WGPUStringView.
    // Commenting out until it's actually implemented.

    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_pop_debug_group(encoder);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderPushDebugGroup(
    _bundle: native::WGPURenderBundleEncoder,
    _group_label: native::WGPUStringView,
) {
    // These functions are not implemented in wgpu-core, and the API is incompatible with the new WGPUStringView.
    // Commenting out until it's actually implemented.

    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_push_debug_group(encoder, group_label);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetBindGroup(
    bundle: native::WGPURenderBundleEncoder,
    group_index: u32,
    group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    // TODO:
    todo!()
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // // TODO: as per webgpu.h bindgroup is nullable
    // let bind_group_id = group.as_ref().expect("invalid bind group").id;
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_set_bind_group(
    //     encoder,
    //     group_index,
    //     Some(bind_group_id),
    //     dynamic_offsets,
    //     dynamic_offset_count,
    // );
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetIndexBuffer(
    bundle: native::WGPURenderBundleEncoder,
    buffer: native::WGPUBuffer,
    format: native::WGPUIndexFormat,
    offset: u64,
    size: u64,
) {
    // TODO:
    todo!()
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let buffer_id = buffer.as_ref().expect("invalid buffer").id;
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_set_index_buffer(
    //     encoder,
    //     buffer_id,
    //     conv::map_index_format(format).expect("invalid index format"),
    //     offset,
    //     match size {
    //         0 => panic!("invalid size"),
    //         conv::WGPU_WHOLE_SIZE => None,
    //         _ => Some(NonZeroU64::new_unchecked(size)),
    //     },
    // );
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetPipeline(
    bundle: native::WGPURenderBundleEncoder,
    pipeline: native::WGPURenderPipeline,
) {
    // TODO:
    todo!()
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let pipeline_id = pipeline.as_ref().expect("invalid render pipeline").id;
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_set_pipeline(encoder, pipeline_id);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetVertexBuffer(
    bundle: native::WGPURenderBundleEncoder,
    slot: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    // TODO:
    todo!()
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // // TODO: as per webgpu.h buffer is nullable
    // let buffer_id = buffer.as_ref().expect("invalid buffer").id;
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_set_vertex_buffer(
    //     encoder,
    //     slot,
    //     buffer_id,
    //     offset,
    //     match size {
    //         0 => panic!("invalid size"),
    //         conv::WGPU_WHOLE_SIZE => None,
    //         _ => Some(NonZeroU64::new_unchecked(size)),
    //     },
    // );
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderAddRef(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
) {
    assert!(
        !render_bundle_encoder.is_null(),
        "invalid render bundle encoder"
    );
    Arc::increment_strong_count(render_bundle_encoder);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderRelease(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
) {
    assert!(
        !render_bundle_encoder.is_null(),
        "invalid render bundle encoder"
    );
    Arc::decrement_strong_count(render_bundle_encoder);
}

// RenderPassEncoder methods

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderBeginOcclusionQuery(
    pass: native::WGPURenderPassEncoder,
    query_index: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.render_pass_encoder.as_mut();

    // encoder.begin_occlusion_query(query_index);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDraw(
    pass: native::WGPURenderPassEncoder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_draw(
    //     encoder,
    //     vertex_count,
    //     instance_count,
    //     first_vertex,
    //     first_instance,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(&pass.error_sink, cause, None, "wgpuRenderPassEncoderDraw"),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDrawIndexed(
    pass: native::WGPURenderPassEncoder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_draw_indexed(
    //     encoder,
    //     index_count,
    //     instance_count,
    //     first_index,
    //     base_vertex,
    //     first_instance,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderDrawIndexed",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDrawIndexedIndirect(
    pass: native::WGPURenderPassEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let indirect_buffer_id = indirect_buffer
    //     .as_ref()
    //     .expect("invalid indirect buffer")
    //     .id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_draw_indexed_indirect(
    //     encoder,
    //     indirect_buffer_id,
    //     indirect_offset,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderDrawIndexedIndirect",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDrawIndirect(
    pass: native::WGPURenderPassEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let indirect_buffer_id = indirect_buffer
    //     .as_ref()
    //     .expect("invalid indirect buffer")
    //     .id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_draw_indirect(encoder, indirect_buffer_id, indirect_offset)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderDrawIndexedIndirect",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderEnd(pass: native::WGPURenderPassEncoder) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_end(encoder) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(&pass.error_sink, cause, None, "wgpuRenderPassEncoderEnd"),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderEndOcclusionQuery(
    pass: native::WGPURenderPassEncoder,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_end_occlusion_query(encoder) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderEndOcclusionQuery",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderExecuteBundles(
    pass: native::WGPURenderPassEncoder,
    bundle_count: usize,
    bundles: *const native::WGPURenderBundle,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass").render_pass_encoder;
    // let bundle_ids = make_slice(bundles, bundle_count)
    //     .iter()
    //     .map(|v| v.as_ref().expect("invalid render bundle"))
    //     .collect::<SmallVec<[_; 4]>>();
    // let encoder = pass.render_pass_encoder.as_mut().expect("invalid compute pass encoder");

    // pass.execute_bundles(bundle_ids);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderInsertDebugMarker(
    pass: native::WGPURenderPassEncoder,
    marker_label: native::WGPUStringView,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_insert_debug_marker(
    //     encoder,
    //     string_view_into_str(marker_label).unwrap_or(""),
    //     0,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderInsertDebugMarker",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderPopDebugGroup(pass: native::WGPURenderPassEncoder) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_pop_debug_group(encoder) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderPopDebugGroup",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderPushDebugGroup(
    pass: native::WGPURenderPassEncoder,
    group_label: native::WGPUStringView,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_push_debug_group(
    //     encoder,
    //     string_view_into_str(group_label).unwrap_or(""),
    //     0,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderPushDebugGroup",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetBindGroup(
    pass: native::WGPURenderPassEncoder,
    group_index: u32,
    bind_group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // // TODO: as per webgpu.h bindgroup is nullable
    // let bind_group_id = bind_group.as_ref().expect("invalid bind group").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_set_bind_group(
    //     encoder,
    //     group_index,
    //     Some(bind_group_id),
    //     make_slice(dynamic_offsets, dynamic_offset_count),
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderSetBindGroup",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetBlendConstant(
    pass: native::WGPURenderPassEncoder,
    color: Option<&native::WGPUColor>,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_set_blend_constant(encoder, conv::map_color(color.expect("invalid color")))
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderSetBlendConstant",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetIndexBuffer(
    pass: native::WGPURenderPassEncoder,
    buffer: native::WGPUBuffer,
    index_format: native::WGPUIndexFormat,
    offset: u64,
    size: u64,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let buffer_id = buffer.as_ref().expect("invalid buffer").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_set_index_buffer(
    //     encoder,
    //     buffer_id,
    //     conv::map_index_format(index_format).expect("invalid index format"),
    //     offset,
    //     match size {
    //         0 => panic!("invalid size"),
    //         conv::WGPU_WHOLE_SIZE => None,
    //         _ => Some(NonZeroU64::new_unchecked(size)),
    //     },
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderSetIndexBuffer",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetPipeline(
    pass: native::WGPURenderPassEncoder,
    render_pipeline: native::WGPURenderPipeline,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let render_pipeline_id = render_pipeline
    //     .as_ref()
    //     .expect("invalid render pipeline")
    //     .id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_set_pipeline(encoder, render_pipeline_id)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderSetPipeline",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetScissorRect(
    pass: native::WGPURenderPassEncoder,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_set_scissor_rect(encoder, x, y, width, height)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderSetPipeline",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetStencilReference(
    pass: native::WGPURenderPassEncoder,
    reference: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_set_stencil_reference(encoder, reference)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderSetStencilReference",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetVertexBuffer(
    pass: native::WGPURenderPassEncoder,
    slot: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // // TODO: as per webgpu.h buffer is nullable
    // let buffer_id = buffer.as_ref().expect("invalid buffer").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_set_vertex_buffer(
    //     encoder,
    //     slot,
    //     buffer_id,
    //     offset,
    //     match size {
    //         0 => panic!("invalid size"),
    //         conv::WGPU_WHOLE_SIZE => None,
    //         _ => Some(NonZeroU64::new_unchecked(size)),
    //     },
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderSetVertexBuffer",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetViewport(
    pass: native::WGPURenderPassEncoder,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_set_viewport(encoder, x, y, width, height, min_depth, max_depth)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderSetViewport",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderAddRef(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    assert!(
        !render_pass_encoder.is_null(),
        "invalid render pass encoder"
    );
    Arc::increment_strong_count(render_pass_encoder);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderRelease(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    assert!(
        !render_pass_encoder.is_null(),
        "invalid render pass encoder"
    );
    Arc::decrement_strong_count(render_pass_encoder);
}

// RenderPipeline methods

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineGetBindGroupLayout(
    render_pipeline: native::WGPURenderPipeline,
    group_index: u32,
) -> native::WGPUBindGroupLayout {
    let render_pipeline = render_pipeline.as_ref().expect("invalid render pipeline");
    let render_pipeline = render_pipeline.get_bind_group_layout(group_index);
    Arc::into_raw(Arc::new(render_pipeline))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineAddRef(render_pipeline: native::WGPURenderPipeline) {
    assert!(!render_pipeline.is_null(), "invalid render pipeline");
    Arc::increment_strong_count(render_pipeline);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineRelease(render_pipeline: native::WGPURenderPipeline) {
    assert!(!render_pipeline.is_null(), "invalid render pipeline");
    Arc::decrement_strong_count(render_pipeline);
}

// Sampler methods

#[no_mangle]
pub unsafe extern "C" fn wgpuSamplerAddRef(sampler: native::WGPUSampler) {
    assert!(!sampler.is_null(), "invalid sampler");
    Arc::increment_strong_count(sampler);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuSamplerRelease(sampler: native::WGPUSampler) {
    assert!(!sampler.is_null(), "invalid sampler");
    Arc::decrement_strong_count(sampler);
}

// ShaderModule methods

#[no_mangle]
pub unsafe extern "C" fn wgpuShaderModuleAddRef(shader_module: native::WGPUShaderModule) {
    assert!(!shader_module.is_null(), "invalid shader module");
    Arc::increment_strong_count(shader_module);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuShaderModuleRelease(shader_module: native::WGPUShaderModule) {
    assert!(!shader_module.is_null(), "invalid shader module");
    Arc::decrement_strong_count(shader_module);
}

// Surface methods

#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceConfigure(
    surface: native::WGPUSurface,
    config: Option<&native::WGPUSurfaceConfiguration>,
) {
    let surface = &surface.as_ref().expect("invalid surface").surface;
    let config = config.expect("invalid config");
    let device = config
        .device
        .as_ref()
        .expect("invalid device for surface configuration");
    let device = &device.device;

    let surface_config = follow_chain!(map_surface_configuration(
        (config),
        WGPUSType_SurfaceConfigurationExtras => native::WGPUSurfaceConfigurationExtras
    ));

    surface.configure(device, &surface_config);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceGetCapabilities(
    surface: native::WGPUSurface,
    adapter: native::WGPUAdapter,
    capabilities: Option<&mut native::WGPUSurfaceCapabilities>,
) -> native::WGPUStatus {
    let adapter = &adapter.as_ref().expect("invalid adapter");
    let surface = &surface.as_ref().expect("invalid surface").surface;
    let capabilities = capabilities.expect("invalid return pointer \"capabilities\"");

    let caps = surface.get_capabilities(adapter);

    capabilities.usages =
        conv::to_native_texture_usage_flags(caps.usages) as native::WGPUTextureUsage;

    let formats = caps
        .formats
        .iter()
        // some texture formats are not in webgpu.h and
        // conv::to_native_texture_format returns None for them.
        // so, filter them out.
        .filter_map(|f| conv::to_native_texture_format(*f))
        .collect::<Vec<_>>();

    if !formats.is_empty() {
        let mut array = formats.into_boxed_slice();
        capabilities.formats = array.as_mut_ptr();
        capabilities.formatCount = array.len();
        mem::forget(array);
    } else {
        capabilities.formats = std::ptr::null_mut();
        capabilities.formatCount = 0;
    }

    let present_modes = caps
        .present_modes
        .iter()
        .filter_map(|f| conv::to_native_present_mode(*f))
        .collect::<Vec<_>>();

    if !present_modes.is_empty() {
        let mut array = present_modes.into_boxed_slice();
        capabilities.presentModes = array.as_mut_ptr();
        capabilities.presentModeCount = array.len();
        mem::forget(array);
    } else {
        capabilities.presentModes = std::ptr::null_mut();
        capabilities.presentModeCount = 0;
    }

    let alpha_modes = caps
        .alpha_modes
        .iter()
        .map(|f| conv::to_native_composite_alpha_mode(*f))
        .collect::<Vec<_>>();

    if !alpha_modes.is_empty() {
        let mut array = alpha_modes.into_boxed_slice();
        capabilities.alphaModes = array.as_mut_ptr();
        capabilities.alphaModeCount = array.len();
        mem::forget(array);
    } else {
        capabilities.alphaModes = std::ptr::null_mut();
        capabilities.alphaModeCount = 0;
    }

    native::WGPUStatus_Success
}

#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceGetCurrentTexture(
    surface: native::WGPUSurface,
    surface_texture: Option<&mut native::WGPUSurfaceTexture>,
) {
    let surface = surface.as_ref().expect("invalid surface");
    let surface_texture = surface_texture.expect("invalid return pointer \"surface_texture\"");

    let surface_data_guard = surface.data.lock();
    let surface_data = match surface_data_guard.as_ref() {
        Some(surface_data) => surface_data,
        None => handle_error_fatal(
            wgc::present::SurfaceError::NotConfigured,
            "wgpuSurfaceGetCurrentTexture",
        ),
    };

    match context.surface_get_current_texture(surface.id, None) {
        Ok(wgc::present::SurfaceOutput { status, texture }) => {
            surface
                .has_surface_presented
                .store(false, atomic::Ordering::SeqCst);
            surface_texture.status = match status {
                wgt::SurfaceStatus::Good => {
                    native::WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
                }
                wgt::SurfaceStatus::Suboptimal => {
                    native::WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal
                }
                wgt::SurfaceStatus::Timeout => native::WGPUSurfaceGetCurrentTextureStatus_Timeout,
                wgt::SurfaceStatus::Outdated => native::WGPUSurfaceGetCurrentTextureStatus_Outdated,
                wgt::SurfaceStatus::Lost => native::WGPUSurfaceGetCurrentTextureStatus_Lost,
                // TODO add some logs to provide more context
                wgt::SurfaceStatus::Unknown => native::WGPUSurfaceGetCurrentTextureStatus_Error,
            };
            surface_texture.texture = match texture {
                Some(texture_id) => Arc::into_raw(Arc::new(WGPUTextureImpl {
                    context: context.clone(),
                    id: texture_id,
                    error_sink: surface_data.error_sink.clone(),
                    data: surface_data.texture_data,
                    surface_id: Some(surface.id),
                    has_surface_presented: surface.has_surface_presented.clone(),
                })),
                None => std::ptr::null_mut(),
            };
        }
        Err(cause) => handle_error_fatal(cause, "wgpuSurfaceGetCurrentTexture"),
    };
}

#[no_mangle]
pub unsafe extern "C" fn wgpuSurfacePresent(surface: native::WGPUSurface) -> native::WGPUStatus {
    let surface = surface.as_ref().expect("invalid surface");
    let context = &surface.context;

    let _status = match context.surface_present(surface.id) {
        Ok(status) => status,
        Err(cause) => {
            log::warn!("Presentation error: {}", cause);
            return native::WGPUStatus_Error;
        }
    };

    surface
        .has_surface_presented
        .store(true, atomic::Ordering::SeqCst);

    native::WGPUStatus_Success
}

#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceUnconfigure(surface: native::WGPUSurface) {
    let surface = surface.as_ref().expect("invalid surface");
    let mut surface_data_guard = surface.data.lock();
    let _ = surface_data_guard.take(); // drop SurfaceData
    surface
        .has_surface_presented
        .store(false, atomic::Ordering::SeqCst);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceAddRef(surface: native::WGPUSurface) {
    assert!(!surface.is_null(), "invalid surface");
    Arc::increment_strong_count(surface);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceRelease(surface: native::WGPUSurface) {
    assert!(!surface.is_null(), "invalid surface");
    Arc::decrement_strong_count(surface);
}

// SurfaceCapabilities methods

#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceCapabilitiesFreeMembers(
    capabilities: native::WGPUSurfaceCapabilities,
) {
    if !capabilities.formats.is_null() && capabilities.formatCount > 0 {
        drop(Vec::from_raw_parts(
            capabilities.formats as *mut native::WGPUTextureFormat,
            capabilities.formatCount,
            capabilities.formatCount,
        ));
    }
    if !capabilities.presentModes.is_null() && capabilities.presentModeCount > 0 {
        drop(Vec::from_raw_parts(
            capabilities.presentModes as *mut native::WGPUPresentMode,
            capabilities.presentModeCount,
            capabilities.presentModeCount,
        ));
    }
    if !capabilities.alphaModes.is_null() && capabilities.alphaModeCount > 0 {
        drop(Vec::from_raw_parts(
            capabilities.alphaModes as *mut native::WGPUCompositeAlphaMode,
            capabilities.alphaModeCount,
            capabilities.alphaModeCount,
        ));
    }
}

// Texture methods

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureCreateView(
    texture: native::WGPUTexture,
    descriptor: Option<&native::WGPUTextureViewDescriptor>,
) -> native::WGPUTextureView {
    let (texture_id, context, error_sink) = {
        let texture = texture.as_ref().expect("invalid texture");
        (texture.id, &texture.context, &texture.error_sink)
    };

    let desc = match descriptor {
        Some(descriptor) => {
            follow_chain!(map_texture_view_descriptor((descriptor),
                WGPUSType_TextureViewDescriptorExtras => native::WGPUTextureViewDescriptorExtras)
            )
        }
        None => wgc::resource::TextureViewDescriptor::default(),
    };

    let (texture_view_id, error) = context.texture_create_view(texture_id, &desc, None);
    if let Some(cause) = error {
        handle_error(error_sink, cause, None, "wgpuTextureCreateView");
    }

    Arc::into_raw(Arc::new(WGPUTextureViewImpl {
        context: context.clone(),
        id: texture_view_id,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureDestroy(texture: native::WGPUTexture) {
    let (texture_id, context) = {
        let texture = texture.as_ref().expect("invalid texture");
        (texture.id, &texture.context)
    };

    // Per spec, no error to report. Even calling destroy multiple times is valid.
    let _ = context.texture_destroy(texture_id);
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetDepthOrArrayLayers(texture: native::WGPUTexture) -> u32 {
    let texture = texture.as_ref().expect("invalid texture");
    texture.data.size.depthOrArrayLayers
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetDimension(
    texture: native::WGPUTexture,
) -> native::WGPUTextureDimension {
    let texture = texture.as_ref().expect("invalid texture");
    texture.data.dimension
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetFormat(
    texture: native::WGPUTexture,
) -> native::WGPUTextureFormat {
    let texture = texture.as_ref().expect("invalid texture");
    texture.data.format
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetHeight(texture: native::WGPUTexture) -> u32 {
    let texture = texture.as_ref().expect("invalid texture");
    texture.data.size.height
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetMipLevelCount(texture: native::WGPUTexture) -> u32 {
    let texture = texture.as_ref().expect("invalid texture");
    texture.data.mip_level_count
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetSampleCount(texture: native::WGPUTexture) -> u32 {
    let texture = texture.as_ref().expect("invalid texture");
    texture.data.sample_count
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetUsage(
    texture: native::WGPUTexture,
) -> native::WGPUTextureUsage {
    let texture = texture.as_ref().expect("invalid texture");
    texture.data.usage
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetWidth(texture: native::WGPUTexture) -> u32 {
    let texture = texture.as_ref().expect("invalid texture");
    texture.data.size.width
}

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureAddRef(texture: native::WGPUTexture) {
    assert!(!texture.is_null(), "invalid texture");
    Arc::increment_strong_count(texture);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureRelease(texture: native::WGPUTexture) {
    assert!(!texture.is_null(), "invalid texture");
    Arc::decrement_strong_count(texture);
}

// TextureView methods

#[no_mangle]
pub unsafe extern "C" fn wgpuTextureViewAddRef(texture_view: native::WGPUTextureView) {
    assert!(!texture_view.is_null(), "invalid texture");
    Arc::increment_strong_count(texture_view);
}
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureViewRelease(texture_view: native::WGPUTextureView) {
    assert!(!texture_view.is_null(), "invalid texture");
    Arc::decrement_strong_count(texture_view);
}

// wgpu.h functions

#[no_mangle]
pub unsafe extern "C" fn wgpuGenerateReport(
    instance: native::WGPUInstance,
    native_report: Option<&mut native::WGPUGlobalReport>,
) {
    // TODO
    // let context = &instance.as_ref().expect("invalid instance").context;
    // let native_report = native_report.expect("invalid return pointer \"native_report\"");
    // conv::write_global_report(native_report, &context.generate_report());
}

#[no_mangle]
pub unsafe extern "C" fn wgpuQueueSubmitForIndex(
    queue: native::WGPUQueue,
    command_count: usize,
    commands: *const native::WGPUCommandBuffer,
) -> native::WGPUSubmissionIndex {
    let (queue_id, context) = {
        let queue = queue.as_ref().expect("invalid queue");
        (queue.queue.id, &queue.queue.context)
    };

    let command_buffers = make_slice(commands, command_count)
        .iter()
        .map(|command_buffer| {
            let command_buffer = command_buffer.as_ref().expect("invalid command buffer");
            command_buffer.open.store(true, atomic::Ordering::SeqCst);
            command_buffer.id
        })
        .collect::<SmallVec<[_; 4]>>();

    match context.queue_submit(queue_id, &command_buffers) {
        Ok(submission_index) => submission_index,
        Err(cause) => handle_error_fatal(cause.1, "wgpuQueueSubmitForIndex"),
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDevicePoll(
    device: native::WGPUDevice,
    wait: bool,
    submission_index: Option<&native::WGPUSubmissionIndex>,
) -> bool {
    let (device_id, context) = {
        let device = device.as_ref().expect("invalid device");
        (device.id, &device.context)
    };

    let maintain = match wait {
        true => match submission_index {
            Some(index) => wgt::PollType::WaitForSubmissionIndex(*index),
            None => wgt::PollType::Wait,
        },
        false => wgt::PollType::Poll,
    };

    match context.device_poll(device_id, maintain) {
        Ok(wgt::PollStatus::QueueEmpty) => true,
        Ok(_) => false,
        Err(cause) => {
            handle_error_fatal(cause, "wgpuDevicePoll");
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateShaderModuleSpirV(
    device: native::WGPUDevice,
    descriptor: Option<&native::WGPUShaderModuleDescriptorSpirV>,
) -> native::WGPUShaderModule {
    let (device_id, context, error_sink) = {
        let device = device.as_ref().expect("invalid device");
        (device.id, &device.context, &device.error_sink)
    };
    let descriptor = descriptor.expect("invalid descriptor");

    let source = Cow::Borrowed(make_slice(
        descriptor.source,
        descriptor.sourceSize as usize,
    ));

    let desc_label = string_view_into_label(descriptor.label);

    let desc =
        wgc::pipeline::ShaderModuleDescriptorPassthrough::SpirV(wgt::ShaderModuleDescriptorSpirV {
            label: desc_label.clone(),
            source,
        });

    let (shader_module_id, error) =
        context.device_create_shader_module_passthrough(device_id, &desc, None);
    if let Some(cause) = error {
        handle_error(
            error_sink,
            cause,
            desc_label,
            "wgpuDeviceCreateShaderModuleSpirV",
        );
    }

    Arc::into_raw(Arc::new(WGPUShaderModuleImpl {
        context: context.clone(),
        id: Some(shader_module_id),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetPushConstants(
    pass: native::WGPURenderPassEncoder,
    stages: native::WGPUShaderStage,
    offset: u32,
    size_bytes: u32,
    data: *const u8,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_set_push_constants(
    //     encoder,
    //     from_u64_bits(stages).expect("invalid shader stage"),
    //     offset,
    //     make_slice(data, size_bytes as usize),
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderSetPushConstants",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderSetPushConstants(
    pass: native::WGPUComputePassEncoder,
    offset: u32,
    size_bytes: u32,
    data: *const u8,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.compute_pass_set_push_constants(
    //     encoder,
    //     offset,
    //     make_slice(data, size_bytes as usize),
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderSetPushConstants",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetPushConstants(
    bundle: native::WGPURenderBundleEncoder,
    stages: native::WGPUShaderStage,
    offset: u32,
    size_bytes: u32,
    data: *const u8,
) {
    // TODO:
    todo!()
    // let bundle = bundle.as_ref().expect("invalid render bundle");
    // let encoder = bundle.encoder.as_mut().expect("invalid render bundle");
    // let encoder = encoder.expect("invalid render bundle");
    // let encoder = encoder.as_mut().unwrap();

    // bundle_ffi::wgpu_render_bundle_set_push_constants(
    //     encoder,
    //     wgt::ShaderStages::from_bits(stages.try_into().unwrap()).expect("invalid shader stage"),
    //     offset,
    //     size_bytes,
    //     data,
    // );
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderMultiDrawIndirect(
    pass: native::WGPURenderPassEncoder,
    buffer: native::WGPUBuffer,
    offset: u64,
    count: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let buffer_id = buffer.as_ref().expect("invalid buffer").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_multi_draw_indirect(encoder, buffer_id, offset, count)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderMultiDrawIndirect",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderMultiDrawIndexedIndirect(
    pass: native::WGPURenderPassEncoder,
    buffer: native::WGPUBuffer,
    offset: u64,
    count: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let buffer_id = buffer.as_ref().expect("invalid buffer").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_multi_draw_indexed_indirect(encoder, buffer_id, offset, count)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderMultiDrawIndexedIndirect",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderMultiDrawIndirectCount(
    pass: native::WGPURenderPassEncoder,
    buffer: native::WGPUBuffer,
    offset: u64,
    count_buffer: native::WGPUBuffer,
    count_buffer_offset: u64,
    max_count: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let buffer_id = buffer.as_ref().expect("invalid buffer").id;
    // let count_buffer_id = count_buffer.as_ref().expect("invalid count buffer").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_multi_draw_indirect_count(
    //     encoder,
    //     buffer_id,
    //     offset,
    //     count_buffer_id,
    //     count_buffer_offset,
    //     max_count,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderMultiDrawIndirectCount",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderMultiDrawIndexedIndirectCount(
    pass: native::WGPURenderPassEncoder,
    buffer: native::WGPUBuffer,
    offset: u64,
    count_buffer: native::WGPUBuffer,
    count_buffer_offset: u64,
    max_count: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let buffer_id = buffer.as_ref().expect("invalid buffer").id;
    // let count_buffer_id = count_buffer.as_ref().expect("invalid count buffer").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_multi_draw_indexed_indirect_count(
    //     encoder,
    //     buffer_id,
    //     offset,
    //     count_buffer_id,
    //     count_buffer_offset,
    //     max_count,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderMultiDrawIndexedIndirectCount",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderBeginPipelineStatisticsQuery(
    pass: native::WGPUComputePassEncoder,
    query_set: native::WGPUQuerySet,
    query_index: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let query_set_id = query_set.as_ref().expect("invalid query set").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.compute_pass_begin_pipeline_statistics_query(
    //     encoder,
    //     query_set_id,
    //     query_index,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderBeginPipelineStatisticsQuery",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderEndPipelineStatisticsQuery(
    pass: native::WGPUComputePassEncoder,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .compute_pass_end_pipeline_statistics_query(encoder)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderEndPipelineStatisticsQuery",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderBeginPipelineStatisticsQuery(
    pass: native::WGPURenderPassEncoder,
    query_set: native::WGPUQuerySet,
    query_index: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let query_set_id = query_set.as_ref().expect("invalid query set").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass.context.render_pass_begin_pipeline_statistics_query(
    //     encoder,
    //     query_set_id,
    //     query_index,
    // ) {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderBeginPipelineStatisticsQuery",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderEndPipelineStatisticsQuery(
    pass: native::WGPURenderPassEncoder,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_end_pipeline_statistics_query(encoder)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderEndPipelineStatisticsQuery",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderWriteTimestamp(
    pass: native::WGPUComputePassEncoder,
    query_set: native::WGPUQuerySet,
    query_index: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid compute pass");
    // let query_set_id = query_set.as_ref().expect("invalid query set").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .compute_pass_write_timestamp(encoder, query_set_id, query_index)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuComputePassEncoderWriteTimestamp",
    //     ),
    // }
}

#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderWriteTimestamp(
    pass: native::WGPURenderPassEncoder,
    query_set: native::WGPUQuerySet,
    query_index: u32,
) {
    // TODO:
    todo!()
    // let pass = pass.as_ref().expect("invalid render pass");
    // let query_set_id = query_set.as_ref().expect("invalid query set").id;
    // let encoder = pass.encoder.as_mut().expect("invalid compute pass encoder");

    // match pass
    //     .context
    //     .render_pass_write_timestamp(encoder, query_set_id, query_index)
    // {
    //     Ok(()) => (),
    //     Err(cause) => handle_error(
    //         &pass.error_sink,
    //         cause,
    //         None,
    //         "wgpuRenderPassEncoderWriteTimestamp",
    //     ),
    // }
}
