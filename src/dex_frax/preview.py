"""Render synchronized kinematic playback; no physics simulation is implied."""
import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import mujoco
from .paths import ASSETS


def render(paths, output, fps=25):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = [pickle.loads(Path(p).read_bytes()) for p in paths]
    if any(d["quat_order"] != "xyzw" or float(d["fps"]) != 50. for d in data):
        raise ValueError("Preview requires 50 Hz XYZW motion PKLs")
    if len({len(d["dof_pos"]) for d in data}) != 1 or not 1 <= fps <= 50:
        raise ValueError("Synchronized comparison needs equal lengths and 1 <= fps <= 50")
    spec = mujoco.MjSpec.from_file(str(ASSETS / "mjcf/dex_29.xml"))
    spec.geom("floor").rgba = [.22, .25, .29, 1.]
    spec.worldbody.add_light(pos=[2., -3., 5.], dir=[-.2, .3, -1.], diffuse=[.9, .9, .9])
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 640
    state = mujoco.MjData(model)
    manifest = json.loads((ASSETS / "model.json").read_text())
    if any(d["joint_names"] != manifest["joint_names"] for d in data):
        raise ValueError("Preview joint order mismatch")
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in manifest["joint_names"]]
    renderer = mujoco.Renderer(model, height=640, width=640)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = option.geomgroup[2] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 135., -12., 3.6
    camera.lookat[:] = [0., 0., .9]
    center = np.median(np.concatenate([d["root_pos"] for d in data]), axis=0)
    camera.lookat[:2] = center[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = np.arange(0, len(data[0]["dof_pos"]), 50. / fps).astype(int)
    labels = [Path(p).parent.name for p in paths]
    gif_path = output.with_suffix(".gif")
    snapshots = []
    try:
        with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
            for index, frame in enumerate(frames):
                panels = []
                for d, label in zip(data, labels):
                    state.qpos[:3] = d["root_pos"][frame]
                    state.qpos[3:7] = d["root_rot"][frame, [3, 0, 1, 2]]
                    state.qpos[qadr] = d["dof_pos"][frame]
                    mujoco.mj_forward(model, state)
                    renderer.update_scene(state, camera, scene_option=option)
                    im = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(im)
                    draw.rectangle((0, 0, 640, 48), fill=(20, 23, 28))
                    draw.text((12, 8), f"{label}  |  {frame/50:.2f}s", fill="white")
                    draw.text((12, 26), "Kinematic playback - no dynamics validation", fill=(200, 205, 210))
                    panels.append(np.asarray(im))
                image = np.concatenate(panels, axis=1)
                writer.append_data(image)
                # Whole clip GIF at half preview rate and half resolution.
                if index % 2 == 0:
                    snapshots.append(np.asarray(Image.fromarray(image).resize((image.shape[1]//2, 320))))
                if index in {0, len(frames)//2, len(frames)-1}:
                    Image.fromarray(image).save(output.with_name(f"{output.stem}_{frame:04d}.png"))
        imageio.mimsave(gif_path, snapshots, duration=2000/fps, loop=0)
    finally:
        renderer.close()
    print(f"Saved {output} and {gif_path}")
