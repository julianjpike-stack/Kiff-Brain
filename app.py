import gradio as gr
import shutil
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path
from tribev2.demo_utils import TribeModel
from tribev2.plotting import PlotBrain
import os

# Load model once at startup
CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(exist_ok=True)

print("Loading TRIBE v2 model...")
model = TribeModel.from_pretrained(
    "facebook/tribev2".replace("\\", "/"),
    cache_folder=CACHE_FOLDER,
)
plotter = PlotBrain(mesh="fsaverage5")
print("Model loaded!")

def run_prediction(input_file, n_seconds=10, n_timesteps=15):
    input_path = Path(input_file.name)
    work_folder = CACHE_FOLDER / input_path.stem
    work_folder.mkdir(exist_ok=True)

    if input_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        print(f"Image detected — converting to video...")
        img = np.array(Image.open(input_path).convert("RGB"))
        video_path = work_folder / (input_path.stem + "_converted.mp4")
        writer = imageio.get_writer(str(video_path), fps=1, macro_block_size=1)
        for _ in range(n_seconds):
            writer.append_data(img)
        writer.close()
    else:
        video_path = input_path

    df = model.get_events_dataframe(video_path=video_path)
    preds, segments = model.predict(events=df)

    fig = plotter.plot_timesteps(
        preds[:n_timesteps],
        segments=segments[:n_timesteps],
        cmap="fire",
        norm_percentile=99,
        vmin=.6,
        alpha_cmap=(0, .2),
        show_stimuli=True
    )

    output_path = str(work_folder / "brain_output.png")
    fig.savefig(output_path)
    return output_path

with gr.Blocks(title="Kiff Brain — TRIBE v2") as app:
    gr.Markdown("# 🧠 Kiff Brain")
    gr.Markdown("Upload a video or image to predict brain responses using TRIBE v2.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload file (PNG, JPG, MP4)",
                file_types=[".png", ".jpg", ".jpeg", ".mp4"]
            )
            n_seconds = gr.Slider(5, 30, value=10, step=1, label="Seconds per image (images only)")
            n_timesteps = gr.Slider(5, 30, value=15, step=1, label="Timesteps to visualize")
            run_btn = gr.Button("Run Prediction", variant="primary")

        with gr.Column():
            output_img = gr.Image(label="Brain Activity Output")

    run_btn.click(
        fn=run_prediction,
        inputs=[file_input, n_seconds, n_timesteps],
        outputs=output_img
    )

app.launch()
