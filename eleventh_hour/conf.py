import os
import yaml
import readline as rl

from pathlib import Path


def read_conf_file(conf_path: str | Path, prompt_string: str):
    conf_file_name = select_file(dir_path=conf_path, prompt_string=prompt_string)
    conf_file_path = conf_path / conf_file_name

    with open(conf_file_path, "+r") as conf_file:
        conf = yaml.safe_load(stream=conf_file)

    return conf


def select_file(dir_path: str | Path, prompt_string: str):
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    os.chdir(dir_path)

    # configure tab completion
    rl.set_completer_delims(" \t\n=")
    rl.parse_and_bind("tab: complete")

    file_name = input(prompt_string)

    return file_name
