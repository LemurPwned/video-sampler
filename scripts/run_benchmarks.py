import glob
import os

from tabulate import tabulate
from tqdm import tqdm

from video_sampler.sampler import SamplerConfig, VideoSampler


def run_benchmarks(target_size: int = 256, debug: bool = False):
    gate_def = dict(
        type="clip",
        pos_samples=["a cat"],
        neg_samples=[
            "an empty background",
            "text on screen",
            "a forest with no animals",
        ],
        model_name="ViT-B-32",
        batch_size=32,
        pos_margin=0.2,
        neg_margin=0.3,
    )
    configs = [
        SamplerConfig(
            buffer_config=dict(type="hash", hash_size=4, size=30, debug=debug)
        ),
        SamplerConfig(
            buffer_config=dict(type="hash", hash_size=4, size=30, debug=debug),
            gate_config=gate_def,
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
                for frame, meta in res:
                    if frame is None:
                        continue
                    frames.append(frame)
                    timestamps.append(float(meta["frame_time"]))

            # sort by the timestamps
            frames = [
                x.resize((target_size, target_size))
                for _, x in sorted(zip(timestamps, frames))
            ]
            duration = 0.05 * len(frames)
            first_frame = frames[0]
            bsn = os.path.basename(video_fn)

            savename = f"assets/{bsn}_{model_type}.gif"
            first_frame.save(
                savename,
                format="GIF",
                append_images=frames,
                save_all=True,
                duration=duration,
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
    run_benchmarks()
