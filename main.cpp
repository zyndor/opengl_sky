#include "xr/VertexFormats.hpp"
#include "xr/Device.hpp"
#include "xr/Input.hpp"
#include "xr/Gfx.hpp"
#include "xr/Timer.hpp"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GLSL(str) (const char*)"#version 330\n" #str

using namespace xr;

// Sky Shaders

const char* skyVertShader = GLSL(
  out vec3 pos;
  out vec3 fsun;
  uniform mat4 P;
  uniform mat4 V;
  uniform vec4 time;	// x only

  const vec2 data[4] = vec2[](
    vec2(-1.0,  1.0), vec2(-1.0, -1.0),
    vec2( 1.0,  1.0), vec2( 1.0, -1.0));

  void main()
  {
    gl_Position = vec4(data[gl_VertexID], 0.0, 1.0);
    pos = transpose(mat3(V)) * (inverse(P) * gl_Position).xyz;
    fsun = vec3(0.0, sin(time.x * 0.01), cos(time.x * 0.01));
  }
);

const char* skyFragShader = GLSL(
  in vec3 pos;
  in vec3 fsun;
  out vec4 color;
  uniform vec4 time;	// x only
  const float cirrus = 0.4;
  const float cumulus = 0.8;

  const float Br = 0.0025;
  const float Bm = 0.0003;
  const float g =  0.9800;
  const vec3 nitrogen = vec3(0.650, 0.570, 0.475);
  const vec3 Kr = Br / pow(nitrogen, vec3(4.0));
  const vec3 Km = Bm / pow(nitrogen, vec3(0.84));

  float hash(float n)
  {
    return fract(sin(n) * 43758.5453123);
  }

  float noise(vec3 x)
  {
    vec3 f = fract(x);
    float n = dot(floor(x), vec3(1.0, 157.0, 113.0));
    return mix(mix(mix(hash(n +   0.0), hash(n +   1.0), f.x),
                   mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
               mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                   mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
  }

  const mat3 m = mat3(0.0, 1.60,  1.20, -1.6, 0.72, -0.96, -1.2, -0.96, 1.28);
  float fbm(vec3 p)
  {
    float f = 0.0;
    f += noise(p) / 2; p = m * p * 1.1;
    f += noise(p) / 4; p = m * p * 1.2;
    f += noise(p) / 6; p = m * p * 1.3;
    f += noise(p) / 12; p = m * p * 1.4;
    f += noise(p) / 24;
    return f;
  }

  void main()
  {
    if (pos.y < 0)
      discard;

    // Atmosphere Scattering
    float mu = dot(normalize(pos), normalize(fsun));
    vec3 extinction = mix(exp(-exp(-((pos.y + fsun.y * 4.0) * (exp(-pos.y * 16.0) + 0.1) / 80.0) / Br) * (exp(-pos.y * 16.0) + 0.1) * Kr / Br) * exp(-pos.y * exp(-pos.y * 8.0 ) * 4.0) * exp(-pos.y * 2.0) * 4.0, vec3(1.0 - exp(fsun.y)) * 0.2, -fsun.y * 0.2 + 0.5);
    color.rgb = 3.0 / (8.0 * 3.14) * (1.0 + mu * mu) * (Kr + Km * (1.0 - g * g) / (2.0 + g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5)) / (Br + Bm) * extinction;

    // Cirrus Clouds
    float density = smoothstep(1.0 - cirrus, 1.0, fbm(pos.xyz / pos.y * 2.0 + time.x * 0.05)) * 0.3;
    color.rgb = mix(color.rgb, extinction * 4.0, density * max(pos.y, 0.0));

    // Cumulus Clouds
    for (int i = 0; i < 3; i++)
    {
      float density = smoothstep(1.0 - cumulus, 1.0, fbm((0.7 + float(i) * 0.01) * pos.xyz / pos.y + time.x * 0.3));
      color.rgb = mix(color.rgb, extinction * density * 5.0, min(density, 1.0) * max(pos.y, 0.0));
    }

    // Dithering Noise
    color.rgb += noise(pos * 1000) * 0.01;
  }
);

// Post-Processing Shaders

const char *postVertShader = GLSL(
  out vec2 UV;

  const vec2 data[4] = vec2[](
    vec2(-1.0,  1.0), vec2(-1.0, -1.0),
    vec2( 1.0,  1.0), vec2( 1.0, -1.0));

  void main()
  {
    gl_Position = vec4(data[gl_VertexID], 0.0, 1.0);
    UV = gl_Position.xy * 0.5 + 0.5;
  }
);

const char *postFragShader = GLSL(
  in vec2 UV;
  out vec4 color;
  uniform sampler2D tex[2];

  void main()
  {
    color = texture(tex[0], UV);
    float depth = texture(tex[1], UV).r;

    // Ambient Occlusion
    vec2 r = 4.0 / textureSize(tex[0], 0);
    float occlusion = 0.0;
    for (int i = -2; i < 3; i++)
    {
      for (int j = -2; j < 3; j++)
      {
        occlusion += 1.0 / (1.0 + pow(10.0 * min(depth - texture(tex[1], UV + vec2(i, j) * r).r, 0.0), 2.0)) / 24.0;
      }
    }
    color.rgb *= occlusion;

    // Gamma Correction
    color.rgb = pow(1.0 - exp(-1.3 * color.rgb), vec3(1.3));
  }
);


// Structures

typedef struct { float x, y, z; } vector;
typedef struct { float m[16]; } matrix;

typedef struct { float x, y, z, r, r2, px, py; } gamestate;

struct entity { unsigned int depth_test;
  entity* input;

  Gfx::VertexBufferHandle buffer;

  Gfx::TextureHandle rtts[2];
  Gfx::FrameBufferHandle fb;

  Gfx::ProgramHandle program;
};
typedef struct { entity** entities; unsigned int entity_count; gamestate state; } scene;

// Globals

Gfx::UniformHandle uV;
Gfx::UniformHandle uP;
Gfx::UniformHandle uTex;
Gfx::UniformHandle uTime;

// Math Functions

matrix getProjectionMatrix(int w, int h)
{
  float fov = 65.0f;
  float aspect = (float)w / (float)h;
  float near = 1.0f;
  float far = 1000.0f;

  return matrix { {
    1.0f / (aspect * tanf(fov * 3.14f / 180.0f / 2.0f)),
    0.f,
    0.f,
    0.f,
    0.f,
    1.0f / tanf(fov * 3.14f / 180.0f / 2.0f),
    0.f,
    0.f,
    0.f,
    0.f,
    -(far + near) / (far - near),
    -1.0f,
    0.f,
    0.f,
    -(2.0f * far * near) / (far - near),
    0.f,
  }};
}

matrix getViewMatrix(float x, float y, float z, float a, float p)
{
  float cosy = cosf(a), siny = sinf(a), cosp = cosf(p), sinp = sinf(p);

  return matrix { {
    cosy,
    siny * sinp,
    siny * cosp,
    0.f,
    0.f,
    cosp,
    -sinp,
    0.f,
    -siny,
    cosy * sinp,
    cosp * cosy,
    0.f,
    -(cosy * x - siny * z),
    -(siny * sinp * x + cosp * y + cosy * sinp * z),
    -(siny * cosp * x - sinp * y + cosp * cosy * z),
    1.0f,
  }};
}

// OpenGL Helpers

Gfx::ShaderHandle makeShader(const char* code, Gfx::ShaderType shaderType)
{
  return Gfx::CreateShader(shaderType, Buffer::FromArray(strlen(code), code));;
}

Gfx::ProgramHandle makeProgram(const char* vertexShaderSource, const char* fragmentShaderSource)
{
  auto vsh = makeShader(vertexShaderSource, Gfx::ShaderType::Vertex);
  auto fsh = makeShader(fragmentShaderSource, Gfx::ShaderType::Fragment);
  auto program = Gfx::CreateProgram(vsh, fsh);
  Gfx::Release(vsh);
  Gfx::Release(fsh);
  return program;
}

Gfx::TextureHandle blankTexture(int w, int h, Gfx::TextureFormat format)
{
  return Gfx::CreateTexture(format, w, h, 0, Gfx::F_TEXTURE_NONE);
}

Gfx::FrameBufferHandle makeFramebuffer(Gfx::TextureHandle* renderTexture, Gfx::TextureHandle* depthTexture, int w, int h)
{
  *renderTexture = blankTexture(w, h, Gfx::TextureFormat::RGB16F);
  *depthTexture = blankTexture(w, h, Gfx::TextureFormat::D32);

  Gfx::TextureHandle textures[] = { *renderTexture, *depthTexture };
  return Gfx::CreateFrameBuffer(2, textures);
}

// Entities

entity* makeEntity(scene *s, const char* vs, const char* fs, int is_framebuffer, unsigned int depth_test,
  int w, int h, entity* input)
{
  entity* e = new entity{ depth_test, input };

  // Create Buffer
  e->buffer = Gfx::CreateVertexBuffer(Vertex::Formats::GetHandle<Vertex::Format<Vertex::Pos<>>>(), Buffer{ 0, nullptr });

  // Load Program
  e->program = makeProgram(vs, fs);

  // Create a framebuffer if applicable
  if (is_framebuffer)
    e->fb = makeFramebuffer(&e->rtts[0], &e->rtts[1], w, h);
  else
    e->fb = Gfx::GetDefaultFrameBuffer();

  s->entities = static_cast<entity**>(realloc(s->entities, ++s->entity_count * sizeof(entity)));
  s->entities[s->entity_count - 1] = e;

  return e;
}

void renderEntity(entity* e, matrix P, matrix V, float time)
{
  Gfx::SetFrameBuffer(e->fb);
  Gfx::Clear(Gfx::F_CLEAR_COLOR | Gfx::F_CLEAR_DEPTH, Color(0.f, 0.f, 0.f, 1.f));

  Gfx::SetProgram(e->program);
  if (e->input) {
    auto stage = 0;
    for (auto& r: e->input->rtts) {
      Gfx::SetTexture(r, stage);
      ++stage;
    }
  }

  Gfx::SetUniform(uP, P.m);
  Gfx::SetUniform(uV, V.m);

  float timeBuf[4] = { time };
  Gfx::SetUniform(uTime, timeBuf);

  Gfx::SetState(Gfx::F_STATE_DEPTH_TEST * (e->depth_test != 0));

  Gfx::Draw(e->buffer, Primitive::TriangleStrip, 0, 4);
}

void deleteEntity(entity* e)
{
  Gfx::Release(e->buffer);
  Gfx::Release(e->program);

  if (e->fb != Gfx::GetDefaultFrameBuffer()) {
    Gfx::Release(e->fb);
    Gfx::Release(e->rtts[0]);
    Gfx::Release(e->rtts[1]);
  }

  delete e;
}

// Scene

scene makeScene()
{
  scene s = { nullptr, 0, { 0.0f, 2.0f, -3.0f, 3.14f, 0.0f } };
  return s;
}

void renderScene(scene* s, int w, int h)
{
  matrix p = getProjectionMatrix(w, h);
  matrix v = getViewMatrix(s->state.x, s->state.y, s->state.z, s->state.r, s->state.r2);
  for (unsigned int i = 0; i < s->entity_count; i++)
    if (s->entities[i]->fb.IsValid())
      renderEntity(s->entities[i], p, v, (float)Timer::GetUST() * 2e-4f - 0.0f);
  for (unsigned int i = 0; i < s->entity_count; i++)
    if (!s->entities[i]->fb.IsValid())
      renderEntity(s->entities[i], p, v, 0.0f);
}

void deleteScene(scene* s)
{
  for (unsigned int i = 0; i < s->entity_count; i++)
  {
    deleteEntity(s->entities[i]);
  }
  free(s->entities);
}

// Main Loop

int main()
{
  _putenv("XR_DISPLAY_WIDTH=800");
  _putenv("XR_DISPLAY_HEIGHT=600");

  Device::Init("sky");
  Input::Init();
  Gfx::Init(Device::GetGfxContext());

  uV = Gfx::CreateUniform("V", Gfx::UniformType::Mat4);
  uP = Gfx::CreateUniform("P", Gfx::UniformType::Mat4);

  uTex = Gfx::CreateUniform("tex", Gfx::UniformType::Int1, 2);
  int textureStages[] { 0, 1 };
  Gfx::SetUniform(uTex, textureStages);

  uTime = Gfx::CreateUniform("time", Gfx::UniformType::Vec4);

  scene s = makeScene();
  makeEntity(&s, skyVertShader, skyFragShader, 1, 0, 800, 600, nullptr);
  makeEntity(&s, postVertShader, postFragShader, 0, 0, 0, 0, s.entities[0]);

  auto m = Input::GetMousePosition();
  s.state.px = (float)m.x;
  s.state.py = (float)m.y;
  while(!Device::IsQuitting())
  {
    // Move Cursor
    m = Input::GetMousePosition();
    s.state.r -= (m.x - s.state.px) * 0.01f;
    s.state.r2 -= (m.y - s.state.py) * 0.01f;
    s.state.px = (float)m.x;
    s.state.py = (float)m.y;

    // Render the Scene
    renderScene(&s, 800, 600);

    // Swap
    Gfx::Present();

    Device::YieldOS(0);
  }

  deleteScene(&s);

  Gfx::Release(uP);
  Gfx::Release(uV);
  Gfx::Release(uTex);
  Gfx::Release(uTime);

  Gfx::Shutdown();
  Input::Shutdown();
  Device::Shutdown();
  return 0;
}
