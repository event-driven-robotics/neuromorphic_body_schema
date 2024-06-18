import glfw

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW can't be initialized")

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(640, 480, "Hello World", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

# Make the window's context current
glfw.make_context_current(window)

# Main loop
while not glfw.window_should_close(window):
    # Render here
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
