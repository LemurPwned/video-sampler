import subprocess


def on_startup(command, dirty, **kwargs):
    print("Generating CLI help...")
    with open("docs/cli_help.md", "w") as f:
        f.write("# CLI Help\n\n")
        for subcommand in [
            "",
            "config",
            "hash",
            "buffer",
            "clip",
        ]:
            header = subcommand or "Main"
            header = header.capitalize()
            f.write(f"## {header} \n\n")
            args = (
                ["python", "-m", "video_sampler", subcommand, "--help"]
                if subcommand
                else ["python", "-m", "video_sampler", "--help"]
            )
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
            )
            f.write("```text\n")
            f.write(result.stdout)
            f.write("\n```")
            f.write("\n\n")
