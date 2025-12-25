pub const GLYPH_WIDTH: usize = 3;
pub const GLYPH_HEIGHT: usize = 5;
pub const GLYPH_SPACING: usize = 1;
pub const LINE_SPACING: usize = 2;

#[allow(clippy::too_many_arguments)]
pub fn overlay_text(
    buffer: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    value: u8,
    flip_y: bool,
) {
    let mut cursor_x = x;
    for ch in text.chars() {
        draw_glyph(buffer, width, height, cursor_x, y, ch, value, flip_y);
        cursor_x = cursor_x.saturating_add(GLYPH_WIDTH + GLYPH_SPACING);
        if cursor_x >= width {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_glyph(
    buffer: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    ch: char,
    value: u8,
    flip_y: bool,
) {
    let rows = glyph_rows(ch);
    for (row, bits) in rows.iter().enumerate() {
        for col in 0..GLYPH_WIDTH {
            if bits & (1 << (GLYPH_WIDTH - 1 - col)) == 0 {
                continue;
            }
            let tx = x + col;
            let ty = y + row;
            if tx >= width || ty >= height {
                continue;
            }
            let ty = if flip_y { height - 1 - ty } else { ty };
            let idx = ty * width + tx;
            if let Some(cell) = buffer.get_mut(idx) {
                *cell = (*cell).max(value);
            }
        }
    }
}

fn glyph_rows(ch: char) -> [u8; GLYPH_HEIGHT] {
    match ch {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'C' => [0b111, 0b100, 0b100, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        ' ' => [0b000, 0b000, 0b000, 0b000, 0b000],
        _ => [0b000, 0b000, 0b000, 0b000, 0b000],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_text_sets_expected_pixels() {
        let width = 8;
        let height = 8;
        let mut buffer = vec![0u8; width * height];
        overlay_text(&mut buffer, width, height, 1, 1, "0", 200, false);
        let top_left = 1 + 1 * width;
        let top_right = 3 + 1 * width;
        let middle = 2 + 2 * width;
        assert_eq!(buffer[top_left], 200);
        assert_eq!(buffer[top_right], 200);
        assert_eq!(buffer[middle], 0);
    }
}
