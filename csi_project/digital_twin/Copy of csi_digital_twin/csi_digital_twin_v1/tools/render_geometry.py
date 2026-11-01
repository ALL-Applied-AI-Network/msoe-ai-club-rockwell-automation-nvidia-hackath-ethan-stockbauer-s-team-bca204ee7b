"""USD geometry adjustments used by the fill-station renderers."""

STREAM_SURFACE_ALIGNMENT_MODEL = "nozzle_anchored_vertical_stretch_v1"


def align_stream_to_liquid_surface(stage, time_code, original_transform=None):
    from pxr import Gf, Usd, UsdGeom

    session_layer = stage.GetSessionLayer()
    stage.SetEditTarget(session_layer)
    session_layer.timeCodesPerSecond = stage.GetTimeCodesPerSecond()
    time = Usd.TimeCode(time_code)
    roots = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.GetName() in ("Fill_dispensing_stream", "Fill_liquid")
        and prim.IsA(UsdGeom.Xform)
    }
    if "Fill_dispensing_stream" not in roots or "Fill_liquid" not in roots:
        raise RuntimeError("fill stream or liquid geometry is missing from the stage")

    stream_root = roots["Fill_dispensing_stream"]
    if UsdGeom.Imageable(stream_root).ComputeVisibility(time) == UsdGeom.Tokens.invisible:
        return None

    def find_mesh(root, name):
        meshes = [
            prim for prim in Usd.PrimRange(root)
            if prim.IsA(UsdGeom.Mesh)
        ]
        if len(meshes) != 1:
            raise RuntimeError(f"expected one mesh below {name}, found {len(meshes)}")
        return meshes[0]

    stream_mesh = find_mesh(stream_root, "Fill_dispensing_stream")
    liquid_mesh = find_mesh(roots["Fill_liquid"], "Fill_liquid")
    stream_points = UsdGeom.Mesh(stream_mesh).GetPointsAttr().Get(time)
    if not stream_points:
        raise RuntimeError("fill stream mesh has no points")
    local_top = max(float(point[2]) for point in stream_points)

    transform_ops = {
        op.GetOpName(): op
        for op in UsdGeom.Xformable(stream_root).GetOrderedXformOps()
    }
    translate_op = transform_ops.get("xformOp:translate")
    scale_op = transform_ops.get("xformOp:scale")
    if translate_op is None or scale_op is None:
        raise RuntimeError("fill stream must have animated translate and scale ops")

    if original_transform is not None:
        original_translate, original_scale = original_transform
        translate_op.Set(original_translate, time)
        scale_op.Set(original_scale, time)

    cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_])
    stream_bounds = cache.ComputeWorldBound(stream_mesh).ComputeAlignedBox()
    liquid_bounds = cache.ComputeWorldBound(liquid_mesh).ComputeAlignedBox()
    stream_bottom = float(stream_bounds.GetMin()[2])
    stream_top = float(stream_bounds.GetMax()[2])
    liquid_surface = float(liquid_bounds.GetMax()[2])
    if stream_top <= liquid_surface:
        raise RuntimeError(
            f"fill stream top ({stream_top:.6f} m) is below the liquid surface "
            f"({liquid_surface:.6f} m) at frame {time_code}"
        )
    if stream_top <= stream_bottom:
        raise RuntimeError(f"fill stream has no vertical extent at frame {time_code}")

    translate = translate_op.Get(time)
    scale = scale_op.Get(time)
    new_scale_z = float(scale[2]) * (
        (stream_top - liquid_surface) / (stream_top - stream_bottom)
    )
    new_translate_z = float(translate[2]) + (
        float(scale[2]) - new_scale_z
    ) * local_top
    translate_op.Set(Gf.Vec3d(translate[0], translate[1], new_translate_z), time)
    scale_op.Set(Gf.Vec3f(scale[0], scale[1], new_scale_z), time)

    aligned_bounds = UsdGeom.BBoxCache(
        time, [UsdGeom.Tokens.default_]
    ).ComputeWorldBound(stream_mesh).ComputeAlignedBox()
    gap_after = abs(float(aligned_bounds.GetMin()[2]) - liquid_surface)
    top_drift = abs(float(aligned_bounds.GetMax()[2]) - stream_top)
    if gap_after > 1e-4:
        raise RuntimeError(
            f"could not align fill stream at frame {time_code}: "
            f"surface gap is {gap_after:.6f} m"
        )
    if top_drift > 1e-4:
        raise RuntimeError(
            f"fill stream nozzle end moved at frame {time_code}: "
            f"{top_drift:.6f} m"
        )
    return abs(stream_bottom - liquid_surface), gap_after
