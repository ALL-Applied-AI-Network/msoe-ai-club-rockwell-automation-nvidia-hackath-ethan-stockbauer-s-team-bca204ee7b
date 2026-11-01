"""Render one image from a camera of the digital twin with Isaac Sim (headless).

    ./python.sh tools/render_example.py --camera CapSealCam --out cap_seal.png
    ./python.sh tools/render_example.py --camera FillCam --frame 301 --out fill.png
    ./python.sh tools/render_example.py --camera CapSealCam --stage stations/tray_inspection/tray_inspection.usdc --out cap_seal.png
    ./python.sh tools/render_example.py --camera FillCam --stage stations/fill_station/fill_station.usda --out fill.png

Run with Isaac Sim's python (python.sh in the Isaac Sim install). By default it opens the full line
(csi_line.usda); --stage opens a single station file instead. The camera settings come from cameras/<name>.json.
CapSealCam is rendered with overscan and then given the real lens distortion (tools/apply_lens.py).
"""
import argparse, json, os, subprocess, sys, traceback
from pathlib import Path

from render_batch1 import DEFAULT_LIQUID_HEIGHT_SCALE, DEFAULT_LIQUID_OPACITY
from render_geometry import align_stream_to_liquid_surface

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--camera", required=True, choices=["CapSealCam", "FillCam"],
                help="camera name; settings come from cameras/<name>.json")
ap.add_argument("--stage", default=os.path.join(HERE, "csi_line.usda"), help="csi_line.usda or one station file")
ap.add_argument("--frame", type=int, default=1, help="time code (fill station animation: 1-686 at 30 fps)")
ap.add_argument("--out", default="render.png")
ap.add_argument("--subframes", type=int, default=32, help="path-tracing subframes (quality)")
ap.add_argument("--liquid-color", type=float, nargs=3, metavar=("R", "G", "B"),
                help="recipe liquid color in sRGB [0,1]; overrides the bound Fill_liquid material")
ap.add_argument("--stream-color", type=float, nargs=3, metavar=("R", "G", "B"),
                help="currently dispensing tank color in sRGB [0,1]; overrides the nozzle stream")
ap.add_argument("--liquid-opacity", type=float, default=DEFAULT_LIQUID_OPACITY,
                help="liquid opacity; lower values make the dyed liquid translucent")
ap.add_argument("--liquid-height-scale", type=float, default=DEFAULT_LIQUID_HEIGHT_SCALE)
ap.add_argument("--stream-opacity", type=float, default=0.42,
                help="dispensing-stream opacity")
ap.add_argument("--kit-arg", action="append", default=[], help="extra Isaac Sim / Kit argument (repeatable)")
a = ap.parse_args()

cam = json.load(open(os.path.join(HERE, "cameras", a.camera + ".json")))
stage_path = Path(a.stage).expanduser()
if not stage_path.is_absolute():
    stage_path = Path(HERE) / stage_path
stage_path = str(stage_path.resolve())
if not os.path.isfile(stage_path):
    ap.error(f"stage does not exist: {stage_path}")
if a.frame < 1 or (a.camera == "FillCam" and a.frame > 686):
    ap.error("use a positive time code; the fill animation covers 1-686")
if a.subframes < 1:
    ap.error("--subframes must be positive")
if not 0 <= a.liquid_opacity <= 1 or not 0 <= a.stream_opacity <= 1:
    ap.error("opacity values must be between 0 and 1")
if a.liquid_height_scale <= 0:
    ap.error("--liquid-height-scale must be positive")
Path(a.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
a.out = str(Path(a.out).expanduser().resolve())
rw, rh = cam["render_px"]

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": rw, "height": rh, "extra_args": a.kit_arg})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep

    ctx = omni.usd.get_context()
    if not ctx.open_stage(stage_path):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    prim = next((p for p in [cam["prim_in_csi_line"], cam["prim_in_station_file"]]
                 if stage.GetPrimAtPath(p).IsValid()), None)
    if prim is None:
        raise RuntimeError(f"camera {a.camera} not found in {stage_path}")
    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(a.frame / stage.GetTimeCodesPerSecond())
    kit.update()

    # RTX's first sync of a scaled camera (CapSealCam carries the 0.0254 inch scale) misreads its
    # clippingRange and renders black. Changing and restoring the value in the session layer forces a
    # correct re-sync; the authored camera in the USD file is untouched.
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade
    if a.liquid_height_scale != 1.0:
        liquid_roots = [
            p for p in stage.Traverse()
            if p.GetName() == "Fill_liquid" and p.IsA(UsdGeom.Xform)
        ]
        if not liquid_roots and a.camera == "FillCam":
            raise RuntimeError("no Fill_liquid Xform found in the FillCam stage")
        if liquid_roots:
            stage.SetEditTarget(stage.GetSessionLayer())
            for liquid_root in liquid_roots:
                height_scale = UsdGeom.Xformable(liquid_root).AddScaleOp(
                    UsdGeom.XformOp.PrecisionFloat, "renderHeight"
                )
                height_scale.Set(Gf.Vec3f(1.0, 1.0, a.liquid_height_scale))
    if not any(prim.HasAPI(UsdLux.LightAPI) for prim in stage.Traverse()):
        stage.SetEditTarget(stage.GetSessionLayer())
        dome = UsdLux.DomeLight.Define(stage, "/Render/StationDomeLight")
        dome.CreateIntensityAttr(1000.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    if a.liquid_color is not None or a.stream_color is not None:
        colors = {
            "Fill_liquid": a.liquid_color,
            "Fill_dispensing_stream": a.stream_color if a.stream_color is not None else a.liquid_color,
        }
        for name, color in colors.items():
            if color is not None and any(channel < 0 or channel > 1 for channel in color):
                raise ValueError(f"invalid {name} color channel")
        stage.SetEditTarget(stage.GetSessionLayer())
        materials_by_root = {}
        for liquid_root in stage.Traverse():
            color = colors.get(liquid_root.GetName())
            if color is None or not liquid_root.IsA(UsdGeom.Imageable):
                continue
            materials = materials_by_root.setdefault(liquid_root.GetName(), {})
            for liquid_prim in Usd.PrimRange(liquid_root):
                material, _ = UsdShade.MaterialBindingAPI(liquid_prim).ComputeBoundMaterial()
                if material:
                    materials[str(material.GetPath())] = material

        updated_by_root = {name: [] for name, color in colors.items() if color is not None}
        for root_name, materials in materials_by_root.items():
            color = colors[root_name]
            for material in materials.values():
                for shader_prim in Usd.PrimRange(material.GetPrim()):
                    shader = UsdShade.Shader(shader_prim)
                    shader_id = shader.GetIdAttr().Get() if shader else None
                    if shader_id != "UsdPreviewSurface":
                        continue
                    color_input = shader.GetInput("diffuseColor")
                    if not color_input:
                        continue
                    def to_linear(channel):
                        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
                    color_input.Set(Gf.Vec3f(*(to_linear(c) for c in color)))
                    opacity_input = shader.GetInput("opacity")
                    opacity = (
                        a.liquid_opacity
                        if root_name == "Fill_liquid"
                        else a.stream_opacity
                    )
                    if opacity_input:
                        opacity_input.Set(opacity)
                    updated_by_root[root_name].append(
                        f"{material.GetPath()}:{shader_prim.GetPath()}"
                    )

        missing_roots = [name for name, shaders in updated_by_root.items() if not shaders]
        if missing_roots:
            raise RuntimeError(
                "could not recolor requested material root(s): "
                + ", ".join(missing_roots)
            )
        for root_name, shaders in updated_by_root.items():
            print(
                f"applied {root_name} sRGB {colors[root_name]} to "
                f"{len(shaders)} shader(s): {shaders}",
                flush=True,
            )

    if a.camera == "FillCam":
        alignment = align_stream_to_liquid_surface(stage, a.frame)
        if alignment is not None:
            print(
                f"aligned stream to liquid surface at frame {a.frame}: "
                f"gap {alignment[0]:.4f} m -> {alignment[1]:.4f} m",
                flush=True,
            )
            kit.update()

    rp = rep.create.render_product(prim, (rw, rh))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    clip = UsdGeom.Camera(stage.GetPrimAtPath(prim)).GetClippingRangeAttr()
    near_far = clip.Get()
    if near_far is not None:
        stage.SetEditTarget(stage.GetSessionLayer())
        clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
        kit.update()
        clip.Set(near_far)
        kit.update()

    data = None
    for attempt in range(20):
        rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
        kit.update()
        data = np.asarray(ann.get_data())
        if data.size == 0:
            continue
        if data.ndim != 3 or data.shape[:2] != (rh, rw) or data.shape[2] < 3:
            raise RuntimeError(
                f"RGB capture returned shape {data.shape}; check GPU/RTX startup errors"
            )
        break
    else:
        raise RuntimeError(
            f"RGB capture returned no data after {attempt + 1} attempts; "
            "check GPU/RTX startup errors"
        )
    img = data[..., :3].copy()

    # Save and post-process before close(): Isaac's default fast shutdown exits
    # the process and will never execute statements after kit.close().
    if cam.get("overscan", 1.0) != 1.0:
        tmp = a.out + ".pinhole.png"
        if not cv2.imwrite(tmp, img[..., ::-1]):
            raise OSError(f"could not write {tmp}")
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "tools", "apply_lens.py"), tmp,
                            os.path.join(HERE, "cameras", a.camera + ".json"), a.out], check=True)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    elif not cv2.imwrite(a.out, img[..., ::-1]):
        raise OSError(f"could not write {a.out}")
    print("wrote", a.out, flush=True)
except BaseException:
    traceback.print_exc()
    # Preserve failure status even when Kit's fast shutdown terminates Python.
    kit.close(exit_code=1)
    raise
kit.close()
