use ash::vk;

pub fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .copied()
        .find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first().copied())
        .unwrap_or(vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        })
}

pub fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    modes
        .iter()
        .copied()
        .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
        .or_else(|| modes.iter().copied().find(|mode| *mode == vk::PresentModeKHR::FIFO))
        .or_else(|| modes.first().copied())
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

pub fn choose_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    window_size: (u32, u32),
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }
    let width = clamp_u32(
        window_size.0,
        capabilities.min_image_extent.width,
        capabilities.max_image_extent.width,
    );
    let height = clamp_u32(
        window_size.1,
        capabilities.min_image_extent.height,
        capabilities.max_image_extent.height,
    );
    vk::Extent2D { width, height }
}

fn clamp_u32(value: u32, min: u32, max: u32) -> u32 {
    value.max(min).min(max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_surface_format_prefers_srgb() {
        let formats = [
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];
        let chosen = choose_surface_format(&formats);
        assert_eq!(chosen.format, vk::Format::B8G8R8A8_SRGB);
    }

    #[test]
    fn choose_present_mode_prefers_mailbox_then_fifo() {
        let modes = [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::FIFO];
        assert_eq!(choose_present_mode(&modes), vk::PresentModeKHR::FIFO);
        let modes = [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
        ];
        assert_eq!(choose_present_mode(&modes), vk::PresentModeKHR::MAILBOX);
    }

    #[test]
    fn choose_extent_uses_current_when_available() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 800,
                height: 600,
            },
            ..Default::default()
        };
        let extent = choose_extent(&capabilities, (1280, 720));
        assert_eq!(extent.width, 800);
        assert_eq!(extent.height, 600);
    }

    #[test]
    fn choose_extent_clamps_window_size() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 320,
                height: 200,
            },
            max_image_extent: vk::Extent2D {
                width: 1024,
                height: 768,
            },
            ..Default::default()
        };
        let extent = choose_extent(&capabilities, (2000, 100));
        assert_eq!(extent.width, 1024);
        assert_eq!(extent.height, 200);
    }
}
