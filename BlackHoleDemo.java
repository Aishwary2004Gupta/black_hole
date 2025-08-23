// Build instructions (Gradle - Kotlin DSL)
// ---------------------------------------
// 1) Create a Gradle project, replace build.gradle.kts with the block below,
// 2) Put this file at src/main/java/BlackHoleDemo.java
// 3) Run: gradle run
/*
-------------------------------- build.gradle.kts --------------------------------
plugins {
    application
}

repositories {
    mavenCentral()
}

val lwjglVersion = "3.3.4"
val jomlVersion = "1.10.7"

val osName = org.gradle.internal.os.OperatingSystem.current()
val lwjglNatives = when {
    osName.isWindows -> "natives-windows"
    osName.isMacOsX -> "natives-macos"
    osName.isLinux -> "natives-linux"
    else -> error("Unsupported OS")
}

dependencies {
    implementation("org.lwjgl:lwjgl:$lwjglVersion")
    implementation("org.lwjgl:lwjgl-glfw:$lwjglVersion")
    implementation("org.lwjgl:lwjgl-opengl:$lwjglVersion")
    implementation("org.joml:joml:$jomlVersion")

    runtimeOnly("org.lwjgl:lwjgl::$lwjglNatives")
    runtimeOnly("org.lwjgl:lwjgl-glfw::$lwjglNatives")
    runtimeOnly("org.lwjgl:lwjgl-opengl::$lwjglNatives")
}

application {
    mainClass.set("BlackHoleDemo")
}
----------------------------------------------------------------------------------
*/

import org.lwjgl.*;
import org.lwjgl.glfw.*;
import org.lwjgl.opengl.*;
import org.lwjgl.system.*;
import org.joml.*;

import java.nio.*;
import java.util.*;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL31.*;
import static org.lwjgl.opengl.GL32.*;
import static org.lwjgl.opengl.GL33.*;
import static org.lwjgl.opengl.GL42.*;
import static org.lwjgl.opengl.GL43.*;
import static org.lwjgl.system.MemoryStack.*;
import static org.lwjgl.system.MemoryUtil.*;

public class BlackHoleDemo {
    // Physical constants
    static final double C = 299_792_458.0;
    static final double G = 6.67430e-11;

    static class Camera {
        Vector3f target = new Vector3f(0, 0, 0);
        float radius = 6.34194e10f;
        float minRadius = 1e10f, maxRadius = 1e12f;
        float azimuth = 0.0f;
        float elevation = (float) Math.PI / 2f;
        float orbitSpeed = 0.01f;
        double zoomSpeed = 25e9f;
        boolean dragging = false;
        boolean panning = false;
        boolean moving = false;
        double lastX = 0.0, lastY = 0.0;

        Vector3f position() {
            float el = Math.max(0.01f, Math.min((float) Math.PI - 0.01f, elevation));
            return new Vector3f(
                    (float) (radius * Math.sin(el) * Math.cos(azimuth)),
                    (float) (radius * Math.cos(el)),
                    (float) (radius * Math.sin(el) * Math.sin(azimuth)));
        }

        void update() {
            target.set(0, 0, 0);
            moving = dragging || panning;
        }
    }

    static class BlackHole {
        Vector3f position;
        double mass;
        double r_s;

        BlackHole(Vector3f p, double m) {
            position = p;
            mass = m;
            r_s = 2.0 * G * mass / (C * C);
        }
    }

    static class ObjectData {
        Vector4f posRadius; // xyz + radius
        Vector4f color; // rgb + a
        double mass;
        Vector3f velocity = new Vector3f();

        ObjectData(Vector4f pr, Vector4f c, double m) {
            posRadius = pr;
            color = c;
            mass = m;
        }
    }

    static class Engine {
        long window;
        int WIDTH = 800, HEIGHT = 600;
        int COMPUTE_WIDTH = 200, COMPUTE_HEIGHT = 150;
        int quadVAO, quadVBO, gridVAO, gridVBO, gridEBO;
        int screenProgram, gridProgram, computeProgram, screenTexture;
        int cameraUBO, diskUBO, objectsUBO;
        int gridIndexCount;

        void initGLFW() {
            if (!glfwInit())
                throw new IllegalStateException("GLFW init failed");
            glfwDefaultWindowHints();
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole", NULL, NULL);
            if (window == NULL)
                throw new RuntimeException("Failed to create window");
            glfwMakeContextCurrent(window);
            glfwSwapInterval(1);
        }

        void initGL() {
            GL.createCapabilities();
            System.out.println("OpenGL " + glGetString(GL_VERSION));
            screenProgram = createProgram(QUAD_VERT, QUAD_FRAG);
            gridProgram = createProgram(GRID_VERT, GRID_FRAG);
            computeProgram = createComputeProgram(COMPUTE_GLSL);
            quadVAO = createQuad();
            screenTexture = createTexture(COMPUTE_WIDTH, COMPUTE_HEIGHT);

            cameraUBO = glGenBuffers();
            glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
            glBufferData(GL_UNIFORM_BUFFER, 16 * 4 * 4, GL_DYNAMIC_DRAW); // 16 vec4 slots worth of space
            glBindBufferBase(GL_UNIFORM_BUFFER, 1, cameraUBO);

            diskUBO = glGenBuffers();
            glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
            glBufferData(GL_UNIFORM_BUFFER, 4 * 4, GL_DYNAMIC_DRAW);
            glBindBufferBase(GL_UNIFORM_BUFFER, 2, diskUBO);

            objectsUBO = glGenBuffers();
            glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
            // We'll pack as: int num + pad(3) + 16 posRadius(vec4) + 16 color(vec4) + 16
            // mass(vec4)
            int size = (1 + 3) * 4 + (16 * 16 * 2) + (16 * 16); // bytes
            glBufferData(GL_UNIFORM_BUFFER, size, GL_DYNAMIC_DRAW);
            glBindBufferBase(GL_UNIFORM_BUFFER, 3, objectsUBO);

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_DEPTH_TEST);
        }

        int createQuad() {
            float[] quad = {
                    -1, 1, 0, 1,
                    -1, -1, 0, 0,
                    1, -1, 1, 0,
                    -1, 1, 0, 1,
                    1, -1, 1, 0,
                    1, 1, 1, 1
            };
            int vao = glGenVertexArrays();
            int vbo = glGenBuffers();
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, quad, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, false, 4 * 4, 0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_FLOAT, false, 4 * 4, 2 * 4);
            glEnableVertexAttribArray(1);
            quadVBO = vbo;
            return vao;
        }

        int createTexture(int w, int h) {
            int tex = glGenTextures();
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, (ByteBuffer) null);
            return tex;
        }

        int createShader(int type, String src) {
            int sh = glCreateShader(type);
            glShaderSource(sh, src);
            glCompileShader(sh);
            if (glGetShaderi(sh, GL_COMPILE_STATUS) == GL_FALSE) {
                throw new RuntimeException("Shader compile error: " + glGetShaderInfoLog(sh));
            }
            return sh;
        }

        int createProgram(String vs, String fs) {
            int v = createShader(GL_VERTEX_SHADER, vs);
            int f = createShader(GL_FRAGMENT_SHADER, fs);
            int p = glCreateProgram();
            glAttachShader(p, v);
            glAttachShader(p, f);
            glLinkProgram(p);
            if (glGetProgrami(p, GL_LINK_STATUS) == GL_FALSE) {
                throw new RuntimeException("Link error: " + glGetProgramInfoLog(p));
            }
            glDeleteShader(v);
            glDeleteShader(f);
            return p;
        }

        int createComputeProgram(String csSrc) {
            int cs = createShader(GL_COMPUTE_SHADER, csSrc);
            int p = glCreateProgram();
            glAttachShader(p, cs);
            glLinkProgram(p);
            if (glGetProgrami(p, GL_LINK_STATUS) == GL_FALSE) {
                throw new RuntimeException("Compute link error: " + glGetProgramInfoLog(p));
            }
            glDeleteShader(cs);
            return p;
        }

        void generateGrid(List<ObjectData> objs) {
            int gridSize = 25;
            float spacing = 1e10f;
            ArrayList<Float> verts = new ArrayList<>();
            ArrayList<Integer> idx = new ArrayList<>();
            for (int z = 0; z <= gridSize; z++) {
                for (int x = 0; x <= gridSize; x++) {
                    float worldX = (x - gridSize / 2f) * spacing;
                    float worldZ = (z - gridSize / 2f) * spacing;
                    float y = 0f;
                    for (ObjectData o : objs) {
                        Vector3f objPos = new Vector3f(o.posRadius.x, o.posRadius.y, o.posRadius.z);
                        double r_s = 2.0 * G * o.mass / (C * C);
                        double dx = worldX - objPos.x;
                        double dz = worldZ - objPos.z;
                        double dist = Math.sqrt(dx * dx + dz * dz);
                        if (dist > r_s) {
                            double deltaY = 2.0 * Math.sqrt(r_s * (dist - r_s));
                            y += (float) deltaY - 3e10f;
                        } else {
                            y += 2.0f * (float) Math.sqrt(r_s * r_s) - 3e10f;
                        }
                    }
                    verts.add(worldX);
                    verts.add(y);
                    verts.add(worldZ);
                }
            }
            for (int z = 0; z < gridSize; z++) {
                for (int x = 0; x < gridSize; x++) {
                    int i = z * (gridSize + 1) + x;
                    idx.add(i);
                    idx.add(i + 1);
                    idx.add(i);
                    idx.add(i + gridSize + 1);
                }
            }
            if (gridVAO == 0) {
                gridVAO = glGenVertexArrays();
                gridVBO = glGenBuffers();
                gridEBO = glGenBuffers();
            }
            glBindVertexArray(gridVAO);
            glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
            FloatBuffer vb = memAllocFloat(verts.size());
            for (Float f : verts)
                vb.put(f);
            vb.flip();
            glBufferData(GL_ARRAY_BUFFER, vb, GL_DYNAMIC_DRAW);
            memFree(vb);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
            IntBuffer ib = memAllocInt(idx.size());
            for (Integer i : idx)
                ib.put(i);
            ib.flip();
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, ib, GL_STATIC_DRAW);
            memFree(ib);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, false, 3 * 4, 0);
            gridIndexCount = idx.size();
            glBindVertexArray(0);
        }

        void drawGrid(Matrix4f viewProj) {
            glUseProgram(gridProgram);
            try (MemoryStack stack = stackPush()) {
                FloatBuffer m = stack.mallocFloat(16);
                viewProj.get(m);
                int loc = glGetUniformLocation(gridProgram, "viewProj");
                glUniformMatrix4fv(loc, false, m);
            }
            glBindVertexArray(gridVAO);
            glDisable(GL_DEPTH_TEST);
            glDrawElements(GL_LINES, gridIndexCount, GL_UNSIGNED_INT, 0L);
            glBindVertexArray(0);
            glEnable(GL_DEPTH_TEST);
        }

        void uploadCameraUBO(Camera cam) {
            Vector3f pos = cam.position();
            Vector3f fwd = new Vector3f(cam.target).sub(pos).normalize();
            Vector3f up = new Vector3f(0, 1, 0);
            Vector3f right = new Vector3f(fwd).cross(up, new Vector3f()).normalize();
            up = new Vector3f(right).cross(fwd, new Vector3f());
            float tanHalfFov = (float) Math.tan(Math.toRadians(60.0 * 0.5));
            float aspect = (float) WIDTH / (float) HEIGHT;

            try (MemoryStack stack = stackPush()) {
                FloatBuffer buf = stack.mallocFloat(4 * 16); // 16 vec4 components (std140-safe padding)
                putVec3(buf, pos);
                buf.put(0); // pad
                putVec3(buf, right);
                buf.put(0);
                putVec3(buf, up);
                buf.put(0);
                putVec3(buf, fwd);
                buf.put(0);
                buf.put(tanHalfFov).put(aspect).put(cam.moving ? 1f : 0f).put(0f);
                buf.flip();
                glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
                glBufferSubData(GL_UNIFORM_BUFFER, 0, buf);
            }
        }

        void uploadDiskUBO(BlackHole sagA) {
            float r1 = (float) (sagA.r_s * 2.2);
            float r2 = (float) (sagA.r_s * 5.2);
            float num = 2.0f;
            float thickness = 1e9f;
            try (MemoryStack stack = stackPush()) {
                FloatBuffer buf = stack.mallocFloat(4);
                buf.put(r1).put(r2).put(num).put(thickness).flip();
                glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
                glBufferSubData(GL_UNIFORM_BUFFER, 0, buf);
            }
        }

        void uploadObjectsUBO(List<ObjectData> objs) {
            int count = Math.min(16, objs.size());
            try (MemoryStack stack = stackPush()) {
                // Layout: int num + pad(3), then 16 posRadius(vec4), 16 color(vec4), 16
                // mass(vec4)
                int totalFloats = 4 + 16 * 4 + 16 * 4 + 16 * 4;
                FloatBuffer fb = stack.mallocFloat(totalFloats);
                fb.put(Float.intBitsToFloat(count)); // will reinterpret on GLSL via int
                fb.put(0).put(0).put(0);
                for (int i = 0; i < 16; i++) {
                    if (i < count) {
                        Vector4f pr = objs.get(i).posRadius;
                        fb.put(pr.x).put(pr.y).put(pr.z).put(pr.w);
                    } else {
                        fb.put(0).put(0).put(0).put(0);
                    }
                }
                for (int i = 0; i < 16; i++) {
                    if (i < count) {
                        Vector4f c = objs.get(i).color;
                        fb.put(c.x).put(c.y).put(c.z).put(c.w);
                    } else {
                        fb.put(0).put(0).put(0).put(0);
                    }
                }
                for (int i = 0; i < 16; i++) {
                    if (i < count) {
                        float m = (float) objs.get(i).mass;
                        fb.put(m).put(0).put(0).put(0);
                    } else {
                        fb.put(0).put(0).put(0).put(0);
                    }
                }
                fb.flip();
                glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
                glBufferSubData(GL_UNIFORM_BUFFER, 0, fb);
            }
        }

        void dispatchCompute(Camera cam) {
            int cw = cam.moving ? COMPUTE_WIDTH : 200;
            int ch = cam.moving ? COMPUTE_HEIGHT : 150;
            glBindTexture(GL_TEXTURE_2D, screenTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cw, ch, 0, GL_RGBA, GL_UNSIGNED_BYTE, (ByteBuffer) null);
            glUseProgram(computeProgram);
            glBindImageTexture(0, screenTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);
            int gx = (int) Math.ceil(cw / 16.0);
            int gy = (int) Math.ceil(ch / 16.0);
            glDispatchCompute(gx, gy, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }

        void drawFullScreen() {
            glUseProgram(screenProgram);
            glBindVertexArray(quadVAO);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, screenTexture);
            int loc = glGetUniformLocation(screenProgram, "screenTexture");
            glUniform1i(loc, 0);
            glDisable(GL_DEPTH_TEST);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glEnable(GL_DEPTH_TEST);
        }

        static void putVec3(FloatBuffer buf, Vector3f v) {
            buf.put(v.x).put(v.y).put(v.z);
        }
    }

    // Shaders (embedded)
    static final String QUAD_VERT = "#version 330 core\n" +
            "layout(location=0) in vec2 aPos;\n" +
            "layout(location=1) in vec2 aUV;\n" +
            "out vec2 vUV;\n" +
            "void main(){ gl_Position = vec4(aPos,0.0,1.0); vUV = aUV; }\n";
    static final String QUAD_FRAG = "#version 330 core\n" +
            "in vec2 vUV; out vec4 FragColor; uniform sampler2D screenTexture;\n" +
            "void main(){ FragColor = texture(screenTexture, vUV); }\n";

    static final String GRID_VERT = "#version 330 core\n" +
            "layout(location=0) in vec3 aPos; uniform mat4 viewProj; void main(){ gl_Position = viewProj * vec4(aPos,1.0); }\n";
    static final String GRID_FRAG = "#version 330 core\n" +
            "out vec4 FragColor; void main(){ FragColor = vec4(0.2,0.7,1.0,0.25); }\n";

    // Compute shader: visual placeholder (stars + soft glow for objects + dark
    // disk)
    static final String COMPUTE_GLSL = "#version 430 core\n" +
            "layout(local_size_x=16, local_size_y=16) in;\n" +
            "layout(rgba8, binding=0) writeonly uniform image2D destTex;\n" +
            "layout(std140, binding=1) uniform Camera {\n" +
            "  vec3 camPos; float _p0; vec3 right; float _p1; vec3 up; float _p2; vec3 fwd; float _p3;\n" +
            "  float tanHalfFov; float aspect; float moving; float _p4;\n" +
            "};\n" +
            "layout(std140, binding=2) uniform Disk { float r1; float r2; float num; float thickness; };\n" +
            "layout(std140, binding=3) uniform Objects {\n" +
            "  int num; vec3 _pad; vec4 posRadius[16]; vec4 color[16]; vec4 mass[16];\n" +
            "};\n" +
            "float hash(vec2 p){ return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453); }\n" +
            "void main(){\n" +
            "  ivec2 pix = ivec2(gl_GlobalInvocationID.xy);\n" +
            "  ivec2 size = imageSize(destTex);\n" +
            "  vec2 uv = (vec2(pix)+0.5)/vec2(size);\n" +
            "  vec3 col = vec3(0.0);\n" +
            "  // starfield\n" +
            "  float n = hash(uv*vec2(size));\n" +
            "  if (n>0.995) col += vec3(1.0);\n" +
            "  // object glows\n" +
            "  for (int i=0;i<num;i++){\n" +
            "    vec3 p = posRadius[i].xyz; float r = posRadius[i].w;\n" +
            "    // project to a fake screen plane facing camera\n" +
            "    vec3 to = p - camPos; float dist = length(to);\n" +
            "    vec3 dir = normalize(to); float d = max(0.0, 1.0 - length(dir - fwd)*3.0);\n" +
            "    float glow = d * clamp(r/(dist*0.5), 0.0, 1.0);\n" +
            "    col += color[i].rgb * glow;\n" +
            "  }\n" +
            "  // simple vignette\n" +
            "  float vig = smoothstep(1.0, 0.5, length(uv-0.5));\n" +
            "  col *= vig;\n" +
            "  imageStore(destTex, pix, vec4(col,1.0));\n" +
            "}\n";

    public static void main(String[] args) {
        Camera camera = new Camera();
        BlackHole sagA = new BlackHole(new Vector3f(0, 0, 0), 8.54e36);
        List<ObjectData> objects = new ArrayList<>();
        objects.add(new ObjectData(new Vector4f(4e11f, 0, 0, 4e10f), new Vector4f(1, 1, 0, 1), 1.98892e30));
        objects.add(new ObjectData(new Vector4f(0, 0, 4e11f, 4e10f), new Vector4f(1, 0, 0, 1), 1.98892e30));
        objects.add(new ObjectData(new Vector4f(0, 0, 0, (float) (2.0 * G * sagA.mass / (C * C))),
                new Vector4f(0, 0, 0, 1), sagA.mass));

        Engine eng = new Engine();
        eng.initGLFW();
        // Callbacks
        glfwSetWindowUserPointer(eng.window, camera);
        glfwSetMouseButtonCallback(eng.window, (win, button, action, mods) -> {
            if ((button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_MIDDLE)) {
                if (action == GLFW_PRESS) {
                    camera.dragging = true;
                    camera.panning = false;
                    try (MemoryStack s = stackPush()) {
                        DoubleBuffer xb = s.mallocDouble(1), yb = s.mallocDouble(1);
                        glfwGetCursorPos(win, xb, yb);
                        camera.lastX = xb.get(0);
                        camera.lastY = yb.get(0);
                    }
                } else if (action == GLFW_RELEASE) {
                    camera.dragging = false;
                    camera.panning = false;
                }
            }
        });
        glfwSetCursorPosCallback(eng.window, (win, x, y) -> {
            double dx = x - camera.lastX, dy = y - camera.lastY;
            if (camera.dragging && !camera.panning) {
                camera.azimuth += (float) (dx * camera.orbitSpeed);
                camera.elevation -= (float) (dy * camera.orbitSpeed);
                camera.elevation = Math.max(0.01f, Math.min((float) Math.PI - 0.01f, camera.elevation));
            }
            camera.lastX = x;
            camera.lastY = y;
            camera.update();
        });
        glfwSetScrollCallback(eng.window, (win, xo, yo) -> {
            camera.radius -= yo * camera.zoomSpeed;
            camera.radius = Math.max(camera.minRadius, Math.min(camera.maxRadius, camera.radius));
            camera.update();
        });
        glfwSetKeyCallback(eng.window, (win, key, sc, action, mods) -> {
            if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
                glfwSetWindowShouldClose(win, true);
        });

        eng.initGL();

        double lastTime = glfwGetTime();
        boolean gravity = false; // toggle with right mouse if you want; kept simple here

        while (!glfwWindowShouldClose(eng.window)) {
            glViewport(0, 0, eng.WIDTH, eng.HEIGHT);
            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            double now = glfwGetTime();
            double dt = now - lastTime;
            lastTime = now;

            // Simple gravity integration (Euler) when enabled
            if (gravity) {
                for (int i = 0; i < objects.size(); i++) {
                    ObjectData a = objects.get(i);
                    for (int j = 0; j < objects.size(); j++)
                        if (i != j) {
                            ObjectData b = objects.get(j);
                            float dx = b.posRadius.x - a.posRadius.x;
                            float dy = b.posRadius.y - a.posRadius.y;
                            float dz = b.posRadius.z - a.posRadius.z;
                            float dist = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
                            if (dist > 0) {
                                float inv = 1f / dist;
                                double F = (G * a.mass * b.mass) / (dist * dist);
                                double acc = F / a.mass;
                                a.velocity.x += (float) (dx * inv * acc);
                                a.velocity.y += (float) (dy * inv * acc);
                                a.velocity.z += (float) (dz * inv * acc);
                                a.posRadius.x += a.velocity.x;
                                a.posRadius.y += a.velocity.y;
                                a.posRadius.z += a.velocity.z;
                            }
                        }
                }
            }

            // Grid
            eng.generateGrid(objects);
            Matrix4f view = new Matrix4f().lookAt(camera.position(), camera.target, new Vector3f(0, 1, 0));
            Matrix4f proj = new Matrix4f().perspective((float) Math.toRadians(60.0),
                    (float) eng.WIDTH / (float) eng.HEIGHT, 1e9f, 1e14f);
            Matrix4f vp = new Matrix4f();
            proj.mul(view, vp);
            eng.drawGrid(vp);

            // Upload UBOs then compute
            eng.uploadCameraUBO(camera);
            eng.uploadDiskUBO(sagA);
            eng.uploadObjectsUBO(objects);
            eng.dispatchCompute(camera);
            eng.drawFullScreen();

            glfwSwapBuffers(eng.window);
            glfwPollEvents();
        }
        glfwDestroyWindow(eng.window);
        glfwTerminate();
    }
}
