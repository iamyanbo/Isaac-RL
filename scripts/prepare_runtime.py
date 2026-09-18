"""Make a private training installation from the user's installed game.

Only a save-directory string is changed in the private executable. No gameplay,
Steam licensing, or original files are modified. Packed resources are copied.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import shutil
import psutil


ROOT = Path(__file__).resolve().parents[1]


def prepare(source: Path, instance=0):
    if instance < 0 or instance > 16:
        raise ValueError("Instance must be between 0 and 16")
    runtime = ROOT / "runtime" if instance == 0 else ROOT / "runtime" / f"worker-{instance:02d}"
    dest = runtime / "game"
    for proc in psutil.process_iter(["exe"]):
        if proc.info["exe"] and Path(proc.info["exe"]).resolve() == (dest/"isaac-ng.exe").resolve():
            raise RuntimeError(f"Instance {instance} is running as PID {proc.pid}; cannot overwrite its installation")
    dest.mkdir(parents=True, exist_ok=True)
    executable = (source / "isaac-ng.exe").read_bytes()
    # A null-terminated path literal, distinct from the game's display name.
    candidates = [b"Binding of Isaac Repentance+/\0", b"Binding of Isaac Repentance/\0"]
    old = next((s for s in candidates if executable.count(s) == 1), None)
    if old is None:
        raise RuntimeError("Unrecognized save-path literal; refusing to launch against personal saves")
    title = "Isaac RL Training" if instance == 0 else f"Isaac RL Worker {instance:02d}"
    folder = title.ljust(len(old) - 2, "_")
    new = folder.encode() + b"/\0"
    assert len(old) == len(new)
    private_exe = executable.replace(old, new)
    (dest / "isaac-ng.exe").write_bytes(private_exe)
    for path in source.iterdir():
        if path.is_file() and (path.suffix == ".dll" or path.name == "steam_appid.txt"):
            # External injectors are not part of the vanilla training environment.
            if path.name.lower() not in {"dbghelp.dll", "dsound.dll", "dinput8.dll", "version.dll"}:
                shutil.copy2(path, dest / path.name)
    shutil.copytree(source / "resources", dest / "resources", dirs_exist_ok=True)
    shutil.copytree(ROOT / "mod", dest / "mods" / "isaac_rl_bridge", dirs_exist_ok=True)
    # Seed metadata for subscribed mods with disable markers before first launch.
    # Steam can copy their assets, but they must not load into worker episodes.
    base_mods = ROOT / "runtime" / "game" / "mods"
    if instance and base_mods.exists():
        for entry in base_mods.iterdir():
            if entry.is_dir() and entry.name != "isaac_rl_bridge":
                private_mod = dest/"mods"/entry.name
                private_mod.mkdir(parents=True,exist_ok=True)
                if (entry/"metadata.xml").exists():
                    shutil.copy2(entry/"metadata.xml",private_mod/"metadata.xml")
                (private_mod/"disable.it").touch()
    # Isaac builds its own path; the shell's Documents path can differ.
    # Use the original game's authoritative informational path, changing only
    # its final directory name, before any private instance starts.
    save_lines = (source / "savedatapath.txt").read_text().splitlines()
    old_save = next(line.split(": ", 1)[1] for line in save_lines if line.startswith("Save Data Path: "))
    saves = Path(old_save).parent / folder
    saves.mkdir(parents=True, exist_ok=True)
    options_path = saves / "options.ini"
    options = configparser.ConfigParser()
    options.optionxform = str
    if options_path.exists():
        options.read(options_path)
    if not options.has_section("Options"):
        options.add_section("Options")
    for key, value in {
        "SteamCloud": 0, "TryImportSave": 0, "EnableMods": 1,
        "EnableDebugConsole": 1, "PauseOnFocusLost": 0, "Fullscreen": 0,
        "VSync": 0, "MusicVolume": 0, "SFXVolume": 0,
        "WindowWidth": 960 if instance == 0 else 480,
        "WindowHeight": 540 if instance == 0 else 270, "MaxRenderScale": 1,
        "ControllerHotplug": 0,
    }.items():
        options["Options"][key] = str(value)
    with options_path.open("w") as stream:
        options.write(stream, space_around_delimiters=False)
    # Leave subscribed copies disabled so Steam will not recreate enabled copies.
    # The runtime log records which scripts actually execute.
    for entry in (dest / "mods").iterdir():
        if entry.is_dir() and entry.name != "isaac_rl_bridge":
            if entry.resolve().parent != (dest / "mods").resolve():
                raise RuntimeError("Unexpected mod path")
            (entry / "disable.it").touch()
    manifest = {
        "source": str(source), "game": str(dest), "saves": str(saves),"instance":instance,"port":9999+instance,
        "original_sha256": hashlib.sha256(executable).hexdigest(),
        "private_sha256": hashlib.sha256(private_exe).hexdigest(),
        "save_path_literal_original": old[:-1].decode(),
        "save_path_literal_private": new[:-1].decode(),
    }
    (runtime / "installation.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(r"C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth"))
    parser.add_argument("--instance",type=int,default=0)
    args = parser.parse_args()
    prepare(args.source,args.instance)
