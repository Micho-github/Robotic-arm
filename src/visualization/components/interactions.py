def connect_mouse_events(visualizer):
    """
    Wire mouse events to set targets via dragging on the 3D plot.
    Expects visualizer to have: ax, fig, is_dragging, update_target_from_mouse.
    """
    ax = visualizer.ax
    fig = visualizer.fig

    def on_mouse_press(event):
        if event.inaxes == ax and event.button == 1:
            visualizer.is_dragging = True
            visualizer.update_target_from_mouse(event)

    def on_mouse_release(event):
        if event.button == 1:
            visualizer.is_dragging = False

    def on_mouse_motion(event):
        if visualizer.is_dragging and event.inaxes == ax:
            visualizer.update_target_from_mouse(event)

    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)










