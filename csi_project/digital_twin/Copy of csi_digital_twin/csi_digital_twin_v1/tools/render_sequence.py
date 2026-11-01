"""Render a sequence of FillCam frames in one Isaac Sim process."""
import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp

from render_batch1 import (
    ANIMATION_END,
    DEFAULT_LIQUID_HEIGHT_SCALE,
    DEFAULT_LIQUID_OPACITY,
)
from render_geometry import align_stream_to_liquid_surface


def to_linear(channel):
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--camera", default="FillCam")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--subframes", type=int, default=1)
    parser.add_argument("--renderer", default="PathTracing")
    parser.add_argument(
        "--liquid-opacity", type=float, default=DEFAULT_LIQUID_OPACITY
    )
    parser.add_argument("--stream-opacity", type=float, default=0.42)
    parser.add_argument(
        "--liquid-height-scale", type=float, default=DEFAULT_LIQUID_HEIGHT_SCALE
    )
    args = parser.parse_args()
    if not 0 <= args.liquid_opacity <= 1 or not 0 <= args.stream_opacity <= 1:
        parser.error("opacity values must be between 0 and 1")
    if args.liquid_height_scale <= 0:
        parser.error("liquid height scale must be positive")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    print(f"loaded {len(plan)} frames from {args.plan}", flush=True)
    app = SimulationApp({"headless": True, "renderer": args.renderer, "width": 640, "height": 480})
    try:
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

        context = omni.usd.get_context()
        print(f"opening stage: {args.stage}", flush=True)
        if not context.open_stage(str(args.stage.resolve())):
            raise RuntimeError(f"could not open stage: {args.stage}")
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        for _ in range(10):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"stage failed to load: {args.stage}")
        print(f"stage loaded: {stage.GetRootLayer().identifier}", flush=True)
        if args.liquid_height_scale != 1.0:
            liquid_roots = [
                prim
                for prim in stage.Traverse()
                if prim.GetName() == "Fill_liquid"
                and prim.IsA(UsdGeom.Xform)
            ]
            if not liquid_roots:
                raise RuntimeError("no Fill_liquid Xform found in the stage")
            for prim in liquid_roots:
                height_scale = UsdGeom.Xformable(prim).AddScaleOp(
                    UsdGeom.XformOp.PrecisionFloat, "renderHeight"
                )
                height_scale.Set(Gf.Vec3f(1.0, 1.0, args.liquid_height_scale))
            print(
                f"scaled liquid height by {args.liquid_height_scale:g}",
                flush=True,
            )
        camera = next(
            (prim for prim in stage.Traverse()
             if prim.GetName() == args.camera and prim.IsA(UsdGeom.Camera)),
            None,
        )
        if camera is None:
            raise RuntimeError(f"camera {args.camera} not found in {args.stage}")
        if not any(prim.HasAPI(UsdLux.LightAPI) for prim in stage.Traverse()):
            stage.SetEditTarget(stage.GetSessionLayer())
            dome = UsdLux.DomeLight.Define(stage, "/Render/SequenceDomeLight")
            dome.CreateIntensityAttr(1000.0)
            dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        render_product = rep.create.render_product(camera, (640, 480))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        print(f"attached RGB annotator to {camera.GetPath()}", flush=True)
        timeline = omni.timeline.get_timeline_interface()
        stage.SetEditTarget(stage.GetSessionLayer())

        stream_root = next(
            (prim for prim in stage.Traverse()
             if prim.GetName() == "Fill_dispensing_stream"
             and prim.IsA(UsdGeom.Xform)),
            None,
        )
        if stream_root is None:
            raise RuntimeError("no Fill_dispensing_stream Xform found in the stage")
        stream_ops = {
            op.GetOpName(): op
            for op in UsdGeom.Xformable(stream_root).GetOrderedXformOps()
        }
        translate_op = stream_ops.get("xformOp:translate")
        scale_op = stream_ops.get("xformOp:scale")
        if translate_op is None or scale_op is None:
            raise RuntimeError("fill stream must have animated translate and scale ops")
        stream_transforms = {}
        for item in plan:
            frame = item["render_frame"]
            if frame not in stream_transforms:
                time = Usd.TimeCode(frame)
                stream_transforms[frame] = (
                    Gf.Vec3d(translate_op.Get(time)),
                    Gf.Vec3f(scale_op.Get(time)),
                )
        stream_gap_before_max = 0.0
        stream_frames_aligned = 0
        liquid_color_held_frames = 0

        material_cache = {}
        for root in ("Fill_liquid", "Fill_dispensing_stream"):
            materials = {}
            for prim in stage.Traverse():
                if prim.GetName() != root or not prim.IsA(UsdGeom.Imageable):
                    continue
                for child in Usd.PrimRange(prim):
                    material, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()
                    if material:
                        materials[str(material.GetPath())] = material
            material_cache[root] = list(materials.values())
        print(
            "bound materials: "
            + ", ".join(f"{root}={len(materials)}" for root, materials in material_cache.items()),
            flush=True,
        )

        if plan:
            warmup_frame = max(
                ANIMATION_END,
                stage.GetEndTimeCode(),
                max(item["render_frame"] for item in plan),
            )
            print(
                f"priming animation at time code {warmup_frame}",
                flush=True,
            )
            timeline.set_current_time(warmup_frame / stage.GetTimeCodesPerSecond())
            for _ in range(3):
                app.update()
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
            timeline.set_current_time(
                plan[0]["render_frame"] / stage.GetTimeCodesPerSecond()
            )
            for _ in range(3):
                app.update()
            print("animation primed", flush=True)

        for index, item in enumerate(plan):
            print(
                f"capturing frame {index + 1}/{len(plan)} "
                f"at time code {item['render_frame']}",
                flush=True,
            )
            frame_time = Usd.TimeCode(item["render_frame"])
            stream_visible = (
                UsdGeom.Imageable(stream_root).ComputeVisibility(frame_time)
                != UsdGeom.Tokens.invisible
            )
            if not stream_visible:
                liquid_color_held_frames += 1
            colors = {
                "Fill_liquid": (
                    item.get("liquid_color") if stream_visible else None
                ),
                "Fill_dispensing_stream": item.get("stream_color"),
            }
            for root, color in colors.items():
                if color is None:
                    continue
                for material in material_cache[root]:
                    for shader_prim in Usd.PrimRange(material.GetPrim()):
                        shader = UsdShade.Shader(shader_prim)
                        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
                            continue
                        color_input = shader.GetInput("diffuseColor")
                        if color_input:
                            color_input.Set(Gf.Vec3f(*(to_linear(c) for c in color)))
                        opacity_input = shader.GetInput("opacity")
                        if opacity_input:
                            opacity_input.Set(
                                args.liquid_opacity
                                if root == "Fill_liquid"
                                else args.stream_opacity
                            )

            timeline.set_current_time(
                item["render_frame"] / stage.GetTimeCodesPerSecond()
            )
            for _ in range(3):
                app.update()
            alignment = align_stream_to_liquid_surface(
                stage,
                item["render_frame"],
                stream_transforms[item["render_frame"]],
            )
            if alignment is not None:
                stream_gap_before_max = max(stream_gap_before_max, alignment[0])
                stream_frames_aligned += 1
                app.update()
            image = None
            for attempt in range(20):
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
                app.update()
                image = np.asarray(annotator.get_data())
                if image.size == 0:
                    continue
                if image.ndim != 3 or image.shape[:2] != (480, 640):
                    raise RuntimeError(f"unexpected RGB capture shape: {image.shape}")
                if np.any(image[..., :3]):
                    break
            else:
                if image is None or image.size == 0:
                    raise RuntimeError(
                        f"received no RGB capture at render frame "
                        f"{item['render_frame']} after {attempt + 1} attempts"
                    )
                raise RuntimeError(
                    f"received an all-black capture at render frame "
                    f"{item['render_frame']} after {attempt + 1} attempts"
                )
            output = Path(item["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), image[..., :3][..., ::-1]):
                raise OSError(f"could not write {output}")
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(plan):
                print(f"rendered {index + 1}/{len(plan)} frames", flush=True)
        print(
            f"aligned stream to liquid surface in {stream_frames_aligned} frames; "
            f"maximum original gap {stream_gap_before_max:.4f} m; "
            f"held liquid color during {liquid_color_held_frames} no-flow frames",
            flush=True,
        )
    except Exception as error:
        print(f"render sequence failed: {error!r}", flush=True)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
