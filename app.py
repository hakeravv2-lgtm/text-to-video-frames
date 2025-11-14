import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import imageio
from PIL import Image
import tempfile
import os

# --- Загрузка модели (один раз) ---
print("Загрузка модели Stable Diffusion...")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
).to(device)
pipe.enable_attention_slicing()  # экономия памяти
print("Модель загружена!")

# --- Генерация изображения ---
def generate_image(prompt, seed=42):
    generator = torch.Generator(device).manual_seed(seed)
    with torch.autocast(device):
        image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]
    return image

# --- Создание видео из двух кадров ---
def interpolate_images(img1, img2, steps=30, fps=15):
    frames = []
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    for i in range(steps):
        alpha = i / (steps - 1)
        frame = (1 - alpha) * img1_np + alpha * img2_np
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)

    # Сохраняем видео
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        writer = imageio.get_writer(f.name, fps=fps, codec="libx264", pixelformat="yuv420p")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return f.name

# --- Основная функция ---
def create_video_from_text(description):
    # Первый кадр
    prompt_start = f"Начало сцены: {description}, высокое качество, кинематографично, детализация"
    img_start = generate_image(prompt_start, seed=100)

    # Последний кадр
    prompt_end = f"Конец сцены: {description}, драматическое изменение, эволюция, кинематографично"
    img_end = generate_image(prompt_end, seed=200)

    # Видео
    video_path = interpolate_images(img_start, img_end, steps=40, fps=12)

    return img_start, img_end, video_path

# --- Интерфейс Gradio ---
with gr.Blocks(title="Текст → Кадры → Видео") as demo:
    gr.Markdown("# Генерация видео по описанию")
    gr.Markdown("Опиши сцену — получишь **первый и последний кадр + анимацию между ними**")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Описание сцены",
                placeholder="Пример: 'Кот прыгает на стол за мышкой'",
                lines=3
            )
            btn = gr.Button("Сгенерировать", variant="primary")

        with gr.Column(scale=1):
            out_start = gr.Image(label="Первый кадр", height=300)
            out_end = gr.Image(label="Последний кадр", height=300)

    out_video = gr.Video(label="Видео (интерполяция)", height=400)

    btn.click(
        fn=create_video_from_text,
        inputs=input_text,
        outputs=[out_start, out_end, out_video]
    )

    gr.Examples(
        examples=[
            ["Солнце встаёт над горами, туман рассеивается"],
            ["Девочка рисует мелом на асфальте, потом идёт дождь"],
            ["Робот включает свет в тёмной комнате"],
        ],
        inputs=input_text
    )

# Запуск
demo.launch()
