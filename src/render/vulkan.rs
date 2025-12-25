use std::ffi::CString;
use std::ptr;

use anyhow::{anyhow, Context, Result};
use ash::vk;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use shaderc::ShaderKind;
use winit::window::Window;

use crate::render::config::{choose_extent, choose_present_mode, choose_surface_format};

#[allow(dead_code)]
pub struct VulkanApp {
    entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue_family_index: u32,
    queue: vk::Queue,
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    render_pass: vk::RenderPass,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    texture_extent: vk::Extent2D,
    texture_staging_buffer: vk::Buffer,
    texture_staging_memory: vk::DeviceMemory,
    texture_layout: vk::ImageLayout,
    texture_upload_command_buffer: vk::CommandBuffer,
    texture_upload_semaphore: vk::Semaphore,
    texture_upload_fence: vk::Fence,
    upload_pending: bool,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

impl VulkanApp {
    pub fn new(window: &Window, texture_size: (u32, u32)) -> Result<Self> {
        let entry = unsafe { ash::Entry::load().context("load Vulkan entry")? };
        let app_name = CString::new("nav-stokes-sim")?;
        let engine_name = CString::new("nav-stokes")?;
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&engine_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_0);
        let extension_names =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .context("enumerate required extensions")?;
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(extension_names);
        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .context("create Vulkan instance")?
        };
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .context("create surface")?
        };
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        let (physical_device, queue_family_index) =
            pick_physical_device(&instance, &surface_loader, surface)?;
        let queue_priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)
            .build()];
        let device_extensions = [ash::extensions::khr::Swapchain::name().as_ptr()];
        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions);
        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .context("create logical device")?
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) = create_swapchain(
            &swapchain_loader,
            &surface_loader,
            surface,
            physical_device,
            window,
            queue_family_index,
        )?;
        let swapchain_image_views =
            create_image_views(&device, &swapchain_images, swapchain_format)?;
        let render_pass = create_render_pass(&device, swapchain_format)?;
        let descriptor_set_layout = create_descriptor_set_layout(&device)?;
        let pipeline_layout = create_pipeline_layout(&device, descriptor_set_layout)?;
        let pipeline = create_pipeline(
            &device,
            render_pass,
            pipeline_layout,
            swapchain_extent,
        )?;
        let command_pool = create_command_pool(&device, queue_family_index)?;
        let (
            texture_image,
            texture_image_memory,
            texture_staging_buffer,
            texture_staging_memory,
            texture_extent,
        ) = create_texture_resources(
            &instance,
            &device,
            physical_device,
            command_pool,
            queue,
            texture_size,
        )?;
        let texture_image_view = create_image_view(&device, texture_image, vk::Format::R8_UNORM)?;
        let texture_sampler = create_sampler(&device)?;
        let descriptor_pool = create_descriptor_pool(&device)?;
        let descriptor_set = create_descriptor_set(
            &device,
            descriptor_pool,
            descriptor_set_layout,
            texture_image_view,
            texture_sampler,
        )?;
        let swapchain_framebuffers = create_framebuffers(
            &device,
            render_pass,
            &swapchain_image_views,
            swapchain_extent,
        )?;
        let command_buffers = create_command_buffers(
            &device,
            command_pool,
            render_pass,
            &swapchain_framebuffers,
            swapchain_extent,
            pipeline,
            pipeline_layout,
            descriptor_set,
        )?;
        let (image_available, render_finished, in_flight_fence) = create_sync_objects(&device)?;
        let texture_upload_command_buffer = create_upload_command_buffer(&device, command_pool)?;
        let (texture_upload_semaphore, texture_upload_fence) = create_upload_sync(&device)?;
        Ok(Self {
            entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            queue_family_index,
            queue,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            render_pass,
            swapchain_framebuffers,
            command_pool,
            command_buffers,
            descriptor_pool,
            descriptor_set,
            texture_image,
            texture_image_memory,
            texture_image_view,
            texture_sampler,
            texture_extent,
            texture_staging_buffer,
            texture_staging_memory,
            texture_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            texture_upload_command_buffer,
            texture_upload_semaphore,
            texture_upload_fence,
            upload_pending: false,
            image_available,
            render_finished,
            in_flight_fence,
        })
    }

    pub fn update_texture(&mut self, data: &[u8]) -> Result<()> {
        let expected = (self.texture_extent.width * self.texture_extent.height) as usize;
        if data.len() != expected {
            return Err(anyhow!(
                "texture data size mismatch: expected {expected}, got {}",
                data.len()
            ));
        }
        unsafe {
            match self.device.get_fence_status(self.texture_upload_fence) {
                Ok(false) => {
                    self.device
                        .wait_for_fences(&[self.texture_upload_fence], true, u64::MAX)
                        .context("wait for texture upload fence")?;
                }
                Ok(true) => {}
                Err(err) => return Err(anyhow!("texture upload fence status: {err}")),
            }
            self.device
                .reset_fences(&[self.texture_upload_fence])
                .context("reset texture upload fence")?;
            self.device
                .reset_command_buffer(
                    self.texture_upload_command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .context("reset texture upload command buffer")?;
        }
        upload_texture_data(
            &self.device,
            self.texture_upload_command_buffer,
            self.queue,
            self.texture_image,
            self.texture_staging_buffer,
            self.texture_staging_memory,
            self.texture_extent,
            self.texture_layout,
            self.texture_upload_semaphore,
            self.texture_upload_fence,
            data,
        )?;
        self.texture_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        self.upload_pending = true;
        Ok(())
    }

    pub fn render(&mut self) -> Result<()> {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
                .context("wait for in-flight fence")?;
            self.device
                .reset_fences(&[self.in_flight_fence])
                .context("reset in-flight fence")?;
        }
        let image_index = unsafe {
            match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            ) {
                Ok((index, _)) => index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(()),
                Err(err) => return Err(anyhow!("acquire next image: {err}")),
            }
        };
        let mut wait_semaphores = Vec::with_capacity(2);
        let mut wait_stages = Vec::with_capacity(2);
        wait_semaphores.push(self.image_available);
        wait_stages.push(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
        if self.upload_pending {
            wait_semaphores.push(self.texture_upload_semaphore);
            wait_stages.push(vk::PipelineStageFlags::FRAGMENT_SHADER);
        }
        let signal_semaphores = [self.render_finished];
        let command_buffers = [self.command_buffers[image_index as usize]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info.build()], self.in_flight_fence)
                .context("queue submit")?;
        }
        self.upload_pending = false;
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        unsafe {
            match self
                .swapchain_loader
                .queue_present(self.queue, &present_info)
            {
                Ok(_) => Ok(()),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => Ok(()),
                Err(err) => Err(anyhow!("queue present: {err}")),
            }
        }
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device
                .destroy_semaphore(self.render_finished, None);
            self.device
                .destroy_semaphore(self.image_available, None);
            self.device.destroy_command_pool(self.command_pool, None);
            for framebuffer in &self.swapchain_framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_sampler(self.texture_sampler, None);
            self.device.destroy_image_view(self.texture_image_view, None);
            self.device.destroy_image(self.texture_image, None);
            self.device
                .free_memory(self.texture_image_memory, None);
            self.device
                .destroy_buffer(self.texture_staging_buffer, None);
            self.device
                .free_memory(self.texture_staging_memory, None);
            self.device
                .destroy_semaphore(self.texture_upload_semaphore, None);
            self.device
                .destroy_fence(self.texture_upload_fence, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for view in &self.swapchain_image_views {
                self.device.destroy_image_view(*view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .context("enumerate physical devices")?
    };
    devices
        .iter()
        .copied()
        .find_map(|device| {
            find_queue_family(instance, surface_loader, surface, device)
                .map(|queue_family_index| (device, queue_family_index))
        })
        .ok_or_else(|| anyhow!("no compatible Vulkan physical device found"))
}

fn find_queue_family(
    instance: &ash::Instance,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    device: vk::PhysicalDevice,
) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    families.iter().enumerate().find_map(|(index, family)| {
        let supports_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let supports_present = unsafe {
            surface_loader
                .get_physical_device_surface_support(device, index as u32, surface)
                .ok()?
        };
        if supports_graphics && supports_present {
            Some(index as u32)
        } else {
            None
        }
    })
}

fn create_swapchain(
    swapchain_loader: &ash::extensions::khr::Swapchain,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    window: &Window,
    queue_family_index: u32,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)> {
    let capabilities = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .context("surface capabilities")?
    };
    let formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .context("surface formats")?
    };
    let present_modes = unsafe {
        surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface)
            .context("present modes")?
    };
    let surface_format = choose_surface_format(&formats);
    let present_mode = choose_present_mode(&present_modes);
    let window_size = window.inner_size();
    let extent = choose_extent(&capabilities, (window_size.width, window_size.height));
    let image_count = desired_image_count(&capabilities);
    let indices = [queue_family_index];
    let create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&indices)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&create_info, None)
            .context("create swapchain")?
    };
    let images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .context("swapchain images")?
    };
    Ok((swapchain, images, surface_format.format, extent))
}

fn desired_image_count(capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
    let preferred = capabilities.min_image_count + 1;
    if capabilities.max_image_count > 0 {
        preferred.min(capabilities.max_image_count)
    } else {
        preferred
    }
}

fn create_image_views(
    device: &ash::Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    images
        .iter()
        .map(|image| create_image_view(device, *image, format))
        .collect()
}

fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView> {
    let components = vk::ComponentMapping::default();
    let subresource = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(components)
        .subresource_range(subresource);
    unsafe {
        device
            .create_image_view(&create_info, None)
            .context("create image view")
    }
}

fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    let color_attachment_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_attachment_ref));
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    let attachments = [color_attachment.build()];
    let subpasses = [subpass.build()];
    let dependencies = [dependency.build()];
    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .context("create render pass")
    }
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build();
    let bindings = [binding];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .context("create descriptor set layout")
    }
}

fn create_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let layouts = [descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
    unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .context("create pipeline layout")
    }
}

fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    extent: vk::Extent2D,
) -> Result<vk::Pipeline> {
    const VERT_SRC: &str = r#"
#version 450
layout(location = 0) out vec2 v_uv;
vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);
vec2 uvs[3] = vec2[](
    vec2(0.0, 0.0),
    vec2(2.0, 0.0),
    vec2(0.0, 2.0)
);
void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    v_uv = uvs[gl_VertexIndex];
}
"#;
    const FRAG_SRC: &str = r#"
#version 450
layout(set = 0, binding = 0) uniform sampler2D u_tex;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;
void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec2 texel = 1.0 / vec2(textureSize(u_tex, 0));
    float h = texture(u_tex, uv).r;
    float hL = texture(u_tex, uv - vec2(texel.x, 0.0)).r;
    float hR = texture(u_tex, uv + vec2(texel.x, 0.0)).r;
    float hD = texture(u_tex, uv - vec2(0.0, texel.y)).r;
    float hU = texture(u_tex, uv + vec2(0.0, texel.y)).r;
    vec2 grad = vec2(hR - hL, hU - hD);
    vec3 normal = normalize(vec3(-grad * 4.0, 1.0));
    vec3 light_dir = normalize(vec3(0.4, 0.6, 1.0));
    float diff = clamp(dot(normal, light_dir), 0.0, 1.0);
    float height = h - 0.5;
    float depth = clamp(0.5 - height * 1.5, 0.0, 1.0);
    vec3 deep = vec3(0.02, 0.07, 0.14);
    vec3 shallow = vec3(0.10, 0.45, 0.75);
    vec3 base = mix(shallow, deep, depth);
    vec3 view_dir = vec3(0.0, 0.0, 1.0);
    vec3 half_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, half_dir), 0.0), 64.0);
    float foam = smoothstep(0.04, 0.12, length(grad) * 2.0);
    vec3 color = base * (0.55 + 0.45 * diff) + vec3(0.85) * spec;
    color = mix(color, vec3(0.9, 0.95, 1.0), foam * 0.4);
    float air_mask = smoothstep(0.02, 0.08, h);
    vec3 air = vec3(0.03, 0.05, 0.08);
    color = mix(air, color, air_mask);
    o_color = vec4(color, 1.0);
}
"#;

    let vert_spv = compile_shader(VERT_SRC, ShaderKind::Vertex, "fullscreen.vert")?;
    let frag_spv = compile_shader(FRAG_SRC, ShaderKind::Fragment, "fullscreen.frag")?;
    let vert_module = create_shader_module(device, &vert_spv)?;
    let frag_module = create_shader_module(device, &frag_spv)?;

    let entry = CString::new("main")?;
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(&entry)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(&entry)
            .build(),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };
    let viewports = [viewport];
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
        blend_enable: 0,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    };
    let color_blend_attachments = [color_blend_attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipelines = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
            .map_err(|err| anyhow!("create graphics pipeline: {err:?}"))?
    };
    let pipeline = pipelines[0];
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    Ok(pipeline)
}

fn compile_shader(source: &str, kind: ShaderKind, name: &str) -> Result<Vec<u32>> {
    let compiler = shaderc::Compiler::new().ok_or_else(|| anyhow!("shader compiler missing"))?;
    let mut options =
        shaderc::CompileOptions::new().ok_or_else(|| anyhow!("shader options missing"))?;
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    let binary = compiler
        .compile_into_spirv(source, kind, name, "main", Some(&options))
        .map_err(|err| anyhow!("compile shader {name}: {err}"))?;
    Ok(binary.as_binary().to_vec())
}

fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
    let create_info = vk::ShaderModuleCreateInfo::builder().code(code);
    unsafe {
        device
            .create_shader_module(&create_info, None)
            .context("create shader module")
    }
}

fn create_texture_resources(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    texture_size: (u32, u32),
) -> Result<(
    vk::Image,
    vk::DeviceMemory,
    vk::Buffer,
    vk::DeviceMemory,
    vk::Extent2D,
)> {
    let extent = vk::Extent2D {
        width: texture_size.0,
        height: texture_size.1,
    };
    let data = generate_texture_data(extent.width, extent.height);
    let buffer_size = data.len() as vk::DeviceSize;
    let (staging_buffer, staging_memory) = create_buffer(
        instance,
        device,
        physical_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let (image, image_memory) = create_image(
        instance,
        device,
        physical_device,
        extent.width,
        extent.height,
        vk::Format::R8_UNORM,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    upload_texture_data_blocking(
        device,
        command_pool,
        queue,
        image,
        staging_buffer,
        staging_memory,
        extent,
        vk::ImageLayout::UNDEFINED,
        &data,
    )?;
    Ok((image, image_memory, staging_buffer, staging_memory, extent))
}

fn create_sampler(device: &ash::Device) -> Result<vk::Sampler> {
    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .max_anisotropy(1.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK);
    unsafe {
        device
            .create_sampler(&sampler_info, None)
            .context("create sampler")
    }
}

fn create_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(1);
    unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .context("create descriptor pool")
    }
}

fn create_descriptor_set(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
) -> Result<vk::DescriptorSet> {
    let layouts = [descriptor_set_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(&alloc_info)
            .context("allocate descriptor set")?[0]
    };
    let image_info = vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(image_view)
        .sampler(sampler);
    let write = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(&image_info));
    unsafe {
        device.update_descriptor_sets(&[write.build()], &[]);
    }
    Ok(descriptor_set)
}

fn create_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .context("create buffer")?
    };
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type = find_memory_type(instance, physical_device, mem_requirements, properties)?;
    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .context("allocate buffer memory")?
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .context("bind buffer memory")?;
    }
    Ok((buffer, memory))
}

fn create_upload_command_buffer(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .context("allocate texture upload command buffer")?[0]
    };
    Ok(command_buffer)
}

fn create_upload_sync(device: &ash::Device) -> Result<(vk::Semaphore, vk::Fence)> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
    let semaphore = unsafe {
        device
            .create_semaphore(&semaphore_info, None)
            .context("create texture upload semaphore")?
    };
    let fence = unsafe {
        device
            .create_fence(&fence_info, None)
            .context("create texture upload fence")?
    };
    Ok((semaphore, fence))
}

#[allow(clippy::too_many_arguments)]
fn create_image(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(tiling)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe {
        device
            .create_image(&image_info, None)
            .context("create image")?
    };
    let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(instance, physical_device, mem_requirements, properties)?;
    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .context("allocate image memory")?
    };
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .context("bind image memory")?;
    }
    Ok((image, memory))
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    requirements: vk::MemoryRequirements,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let mem_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for (index, mem_type) in mem_properties.memory_types.iter().enumerate() {
        let supported = requirements.memory_type_bits & (1 << index) != 0;
        if supported && mem_type.property_flags.contains(properties) {
            return Ok(index as u32);
        }
    }
    Err(anyhow!("no suitable memory type found"))
}

fn begin_single_time_commands(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .context("allocate single-use command buffer")?[0]
    };
    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .context("begin single-use command buffer")?;
    }
    Ok(command_buffer)
}

fn end_single_time_commands(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    unsafe {
        device
            .end_command_buffer(command_buffer)
            .context("end single-use command buffer")?;
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
        device
            .queue_submit(queue, &[submit_info.build()], vk::Fence::null())
            .context("submit single-use command buffer")?;
        device.queue_wait_idle(queue).context("wait for queue")?;
        device.free_command_buffers(command_pool, &command_buffers);
    }
    Ok(())
}

fn transition_image_layout(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;
    let (src_stage, dst_stage, src_access, dst_access) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        ),
        _ => {
            return Err(anyhow!(
                "unsupported layout transition {old_layout:?} -> {new_layout:?}"
            ))
        }
    };
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier.build()],
        );
    }
    end_single_time_commands(device, command_pool, queue, command_buffer)
}

fn record_texture_upload_commands(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    staging_buffer: vk::Buffer,
    extent: vk::Extent2D,
    old_layout: vk::ImageLayout,
) -> Result<()> {
    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .context("begin texture upload command buffer")?;
    }
    let (src_stage, dst_stage, src_access, dst_access) =
        layout_transition_params(old_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier.build()],
        );
    }
    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        });
    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region.build()],
        );
    }
    let (src_stage, dst_stage, src_access, dst_access) = layout_transition_params(
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier.build()],
        );
        device
            .end_command_buffer(command_buffer)
            .context("end texture upload command buffer")?;
    }
    Ok(())
}

fn layout_transition_params(
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<(
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::AccessFlags,
    vk::AccessFlags,
)> {
    match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => Ok((
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        )),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => Ok((
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        )),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => Ok((
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        )),
        _ => Err(anyhow!(
            "unsupported layout transition {old_layout:?} -> {new_layout:?}"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn upload_texture_data(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    queue: vk::Queue,
    image: vk::Image,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    extent: vk::Extent2D,
    old_layout: vk::ImageLayout,
    signal_semaphore: vk::Semaphore,
    fence: vk::Fence,
    data: &[u8],
) -> Result<()> {
    let buffer_size = data.len() as vk::DeviceSize;
    unsafe {
        let mapped = device
            .map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
            .context("map staging buffer")?;
        ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
        device.unmap_memory(staging_memory);
    }
    record_texture_upload_commands(
        device,
        command_buffer,
        image,
        staging_buffer,
        extent,
        old_layout,
    )?;
    let signal_semaphores = [signal_semaphore];
    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores);
    unsafe {
        device
            .queue_submit(queue, &[submit_info.build()], fence)
            .context("submit texture upload")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn upload_texture_data_blocking(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    extent: vk::Extent2D,
    old_layout: vk::ImageLayout,
    data: &[u8],
) -> Result<()> {
    let buffer_size = data.len() as vk::DeviceSize;
    unsafe {
        let mapped = device
            .map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
            .context("map staging buffer")?;
        ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
        device.unmap_memory(staging_memory);
    }
    transition_image_layout(
        device,
        command_pool,
        queue,
        image,
        old_layout,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    )?;
    copy_buffer_to_image(
        device,
        command_pool,
        queue,
        staging_buffer,
        image,
        extent.width,
        extent.height,
    )?;
    transition_image_layout(
        device,
        command_pool,
        queue,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;
    Ok(())
}
fn copy_buffer_to_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;
    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });
    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region.build()],
        );
    }
    end_single_time_commands(device, command_pool, queue, command_buffer)
}

fn generate_texture_data(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / (width - 1) as f32;
            let v = y as f32 / (height - 1) as f32;
            let value = ((u * v) * 255.0) as u8;
            let idx = (y * width + x) as usize;
            data[idx] = value;
        }
    }
    data
}

fn create_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    image_views
        .iter()
        .map(|view| {
            let attachments = [*view];
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe {
                device
                    .create_framebuffer(&framebuffer_info, None)
                    .context("create framebuffer")
            }
        })
        .collect()
}

fn create_command_pool(device: &ash::Device, queue_family_index: u32) -> Result<vk::CommandPool> {
    let pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    unsafe {
        device
            .create_command_pool(&pool_info, None)
            .context("create command pool")
    }
}

#[allow(clippy::too_many_arguments)]
fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    render_pass: vk::RenderPass,
    framebuffers: &[vk::Framebuffer],
    extent: vk::Extent2D,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set: vk::DescriptorSet,
) -> Result<Vec<vk::CommandBuffer>> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(framebuffers.len() as u32);
    let command_buffers = unsafe {
        device
            .allocate_command_buffers(&allocate_info)
            .context("allocate command buffers")?
    };
    for (command_buffer, framebuffer) in command_buffers.iter().zip(framebuffers.iter()) {
        let begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe {
            device
                .begin_command_buffer(*command_buffer, &begin_info)
                .context("begin command buffer")?;
        }
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.02, 0.02, 0.04, 1.0],
            },
        }];
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(*framebuffer)
            .render_area(render_area)
            .clear_values(&clear_values);
        unsafe {
            device.cmd_begin_render_pass(
                *command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_draw(*command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(*command_buffer);
            device
                .end_command_buffer(*command_buffer)
                .context("end command buffer")?;
        }
    }
    Ok(command_buffers)
}

fn create_sync_objects(device: &ash::Device) -> Result<(vk::Semaphore, vk::Semaphore, vk::Fence)> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
    let image_available = unsafe {
        device
            .create_semaphore(&semaphore_info, None)
            .context("create image available semaphore")?
    };
    let render_finished = unsafe {
        device
            .create_semaphore(&semaphore_info, None)
            .context("create render finished semaphore")?
    };
    let in_flight_fence = unsafe {
        device
            .create_fence(&fence_info, None)
            .context("create in-flight fence")?
    };
    Ok((image_available, render_finished, in_flight_fence))
}
