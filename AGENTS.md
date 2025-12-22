# Agent Guidelines

## Language and Style
- Use Rust in a functional style.
- Prefer higher-order functions (map/filter/fold).
- Avoid `unsafe` unless explicitly requested or needed for optimizations, but ask.

## Testing
- Tests are required for all new behavior.
- Prefer unit tests close to the module they cover.
- Add simple, deterministic fixtures and avoid random inputs unless seeded.

## General
- Keep code small, composable, and side-effect free where possible.
- Document non-obvious logic with short comments.
