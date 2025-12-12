def clamp_to_limits(val, min_val=-5.0, max_val=5.0):
    return max(min_val, min(max_val, val))


def parse_and_clamp(text, fallback=0.0, min_val=-5.0, max_val=5.0):
    try:
        val = float(text)
    except (TypeError, ValueError):
        val = fallback
    return clamp_to_limits(val, min_val, max_val)


def update_target_value(visualizer, axis, val):
    """
    Update a target (x/y/z) from slider changes.
    Keeps text in sync and triggers visualization.
    """
    if visualizer._syncing_input:
        return

    clamped = clamp_to_limits(val)
    attr_target = f"target_{axis}"
    attr_text = f"text_{axis}"

    visualizer._syncing_input = True
    try:
        text = getattr(visualizer, attr_text, None)
        if text is not None:
            try:
                text.set_val(f"{clamped:.2f}")
            except Exception:
                pass
    finally:
        visualizer._syncing_input = False

    setattr(visualizer, attr_target, clamped)
    visualizer.update_visualization()


def submit_target_value(visualizer, axis, text_val):
    """
    Handle manual text submission for a target (x/y/z).
    Syncs slider/text and triggers visualization.
    """
    attr_target = f"target_{axis}"
    attr_slider = f"slider_{axis}"
    attr_text = f"text_{axis}"

    fallback = getattr(visualizer, attr_target, 0.0)
    val = parse_and_clamp(text_val, fallback=fallback)

    visualizer._syncing_input = True
    try:
        slider = getattr(visualizer, attr_slider, None)
        if slider is not None:
            slider.set_val(val)
        text = getattr(visualizer, attr_text, None)
        if text is not None:
            text.set_val(f"{val:.2f}")
    finally:
        visualizer._syncing_input = False

    setattr(visualizer, attr_target, val)
    visualizer.update_visualization()




