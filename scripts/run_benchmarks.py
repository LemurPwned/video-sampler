import glob
import os

from tabulate import tabulate
from tqdm import tqdm

from video_sampler.sampler import SamplerConfig, VideoSampler

clip_gate = dict(
    type="clip",
    pos_samples=["a cat"],
    neg_samples=[
        "an empty background",
        "text on screen",
        "blurry image",
        "a forest with no animals",
    ],
    model_name="ViT-B-32",
    batch_size=32,
    pos_margin=0.2,
    neg_margin=0.3,
)
pass_gate = dict(type="pass")
blur_gate_laplacian = dict(type="blur", method="laplacian", threshold=120)
blur_gate_fft = dict(type="blur", method="fft", threshold=20)


def run_benchmarks(gate_config: dict, target_size: int = 256, debug: bool = False):
    configs = [
        SamplerConfig(
            buffer_config=dict(
                type="grid",
                hash_size=4,
                size=30,
                debug=debug,
                grid_x=4,
                grid_y=4,
                max_hits=1,
            ),
        ),
        SamplerConfig(
            buffer_config=dict(type="hash", hash_size=4, size=30, debug=debug)
        ),
        SamplerConfig(
            buffer_config=dict(type="hash", hash_size=4, size=30, debug=debug),
            gate_config=gate_config,
        ),
    ]
    table = []
    for cfg in tqdm(configs, desc="Creating gifs..."):
        sampler = VideoSampler(cfg)
        gate_type = cfg.gate_config["type"]
        model_type = (
            f"{cfg.buffer_config['type']}_{cfg.buffer_config['hash_size']}_{gate_type}"
        )
        for video_fn in glob.glob("videos/*.mp4"):
            frames = []
            timestamps = []
            for res in sampler.sample(video_path=video_fn):
                for frame_obj in res:
                    if frame_obj.frame is None:
                        continue
                    frames.append(frame_obj.frame)
                    timestamps.append(float(frame_obj.metadata["frame_time"]))

            # sort by the timestamps
            frames = [
                x.resize((target_size, target_size))
                for _, x in sorted(zip(timestamps, frames))
            ]
            first_frame = frames[0]
            bsn = os.path.basename(video_fn)

            savename = f"assets/{bsn}_{model_type}.gif"
            first_frame.save(
                savename,
                format="GIF",
                append_images=frames,
                save_all=True,
                duration=250,
                loop=0,
            )
            stats = sampler.stats
            table.append(
                [
                    bsn,
                    cfg.buffer_config["type"],
                    gate_type,
                    stats["decoded"],
                    stats["produced"],
                    stats["gated"],
                ]
            )
    print(
        tabulate(
            table,
            headers=["video", "buffer", "gate", "decoded", "produced", "gated"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    run_benchmarks(gate_config=clip_gate, debug=False)
    run_benchmarks(gate_config=pass_gate, debug=False)
    run_benchmarks(gate_config=blur_gate_fft, debug=False)
