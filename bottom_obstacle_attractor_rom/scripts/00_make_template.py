#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    ROOT,
    control_dict_text,
    ensure_project_dirs,
    load_config,
    openfoam_header,
    write_decompose_par,
    write_viscosity_files,
)


def _v(ix: int, iy: int, iz: int) -> int:
    return iz * 12 + iy * 4 + ix


def _face_x(ix: int, y0: int, y1: int, outward: str) -> str:
    if outward == "-":
        pts = [_v(ix, y0, 0), _v(ix, y0, 1), _v(ix, y1, 1), _v(ix, y1, 0)]
    else:
        pts = [_v(ix, y0, 0), _v(ix, y1, 0), _v(ix, y1, 1), _v(ix, y0, 1)]
    return "(" + " ".join(map(str, pts)) + ")"


def _face_y(iy: int, x0: int, x1: int, outward: str) -> str:
    if outward == "-":
        pts = [_v(x0, iy, 0), _v(x1, iy, 0), _v(x1, iy, 1), _v(x0, iy, 1)]
    else:
        pts = [_v(x0, iy, 0), _v(x0, iy, 1), _v(x1, iy, 1), _v(x1, iy, 0)]
    return "(" + " ".join(map(str, pts)) + ")"


def _face_z(iz: int, x0: int, x1: int, y0: int, y1: int, outward: str) -> str:
    if outward == "-":
        pts = [_v(x0, y0, iz), _v(x0, y1, iz), _v(x1, y1, iz), _v(x1, y0, iz)]
    else:
        pts = [_v(x0, y0, iz), _v(x1, y0, iz), _v(x1, y1, iz), _v(x0, y1, iz)]
    return "(" + " ".join(map(str, pts)) + ")"


def build_block_mesh_dict(cfg: dict) -> str:
    g = cfg["geometry"]
    m = cfg["mesh"]
    xs = [
        g["x_min"],
        g["x_obstacle_front"],
        g["x_obstacle_back"],
        g["x_max"],
    ]
    ys = [g["y_min"], g["y_obstacle_top"], g["y_max"]]
    zs = [g["z_min"], g["z_max"]]
    vertices = []
    for z in zs:
        for y in ys:
            for x in xs:
                vertices.append(f"    ({x:g} {y:g} {z:g})")

    grading = " ".join(str(x) for x in m.get("simple_grading", [1, 1, 1]))
    blocks = [
        # name, x0, x1, y0, y1, nx, ny
        ("upstream_lower", 0, 1, 0, 1, m["nx_upstream"], m["ny_lower"]),
        ("upstream_upper", 0, 1, 1, 2, m["nx_upstream"], m["ny_upper"]),
        ("above_obstacle", 1, 2, 1, 2, m["nx_obstacle"], m["ny_upper"]),
        ("downstream_lower", 2, 3, 0, 1, m["nx_downstream"], m["ny_lower"]),
        ("downstream_upper", 2, 3, 1, 2, m["nx_downstream"], m["ny_upper"]),
    ]
    block_lines = []
    for name, x0, x1, y0, y1, nx, ny in blocks:
        pts = [
            _v(x0, y0, 0),
            _v(x1, y0, 0),
            _v(x1, y1, 0),
            _v(x0, y1, 0),
            _v(x0, y0, 1),
            _v(x1, y0, 1),
            _v(x1, y1, 1),
            _v(x0, y1, 1),
        ]
        block_lines.append(
            "    hex ({pts}) ({nx} {ny} {nz}) simpleGrading ({grading}) // {name}".format(
                pts=" ".join(map(str, pts)),
                nx=nx,
                ny=ny,
                nz=m["nz"],
                grading=grading,
                name=name,
            )
        )

    front_back = []
    for _, x0, x1, y0, y1, *_ in blocks:
        front_back.append("        " + _face_z(0, x0, x1, y0, y1, "-"))
        front_back.append("        " + _face_z(1, x0, x1, y0, y1, "+"))

    return (
        openfoam_header("blockMeshDict", "system")
        + f"""convertToMeters 1;

vertices
(
{chr(10).join(vertices)}
);

blocks
(
{chr(10).join(block_lines)}
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            {_face_x(0, 0, 1, "-")}
            {_face_x(0, 1, 2, "-")}
        );
    }}

    outlet
    {{
        type patch;
        faces
        (
            {_face_x(3, 0, 1, "+")}
            {_face_x(3, 1, 2, "+")}
        );
    }}

    topWall
    {{
        type wall;
        faces
        (
            {_face_y(2, 0, 1, "+")}
            {_face_y(2, 1, 2, "+")}
            {_face_y(2, 2, 3, "+")}
        );
    }}

    bottomWall
    {{
        type wall;
        faces
        (
            {_face_y(0, 0, 1, "-")}
            {_face_y(0, 2, 3, "-")}
        );
    }}

    obstacle
    {{
        type wall;
        faces
        (
            {_face_x(1, 0, 1, "+")}
            {_face_y(1, 1, 2, "-")}
            {_face_x(2, 0, 1, "-")}
        );
    }}

    frontAndBack
    {{
        type empty;
        faces
        (
{chr(10).join(front_back)}
        );
    }}
);

mergePatchPairs
(
);
"""
    )


def build_fv_schemes() -> str:
    return (
        openfoam_header("fvSchemes", "system")
        + """ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss limitedLinearV 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""
    )


def build_fv_solution() -> str:
    return (
        openfoam_header("fvSolution", "system")
        + """solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.01;
        smoother        GaussSeidel;
    }

    pFinal
    {
        $p;
        tolerance       1e-7;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.05;
    }

    UFinal
    {
        $U;
        relTol          0;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
    residualControl
    {
        p               1e-5;
        U               1e-6;
    }
}

PIMPLE
{
    momentumPredictor yes;
    nOuterCorrectors  1;
    nCorrectors       2;
    nNonOrthogonalCorrectors 0;
    pRefCell          0;
    pRefValue         0;
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}
"""
    )


def build_u_field() -> str:
    return (
        openfoam_header("U", "0", "volVectorField")
        + """dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (1 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    outlet
    {
        type            zeroGradient;
    }
    topWall
    {
        type            noSlip;
    }
    bottomWall
    {
        type            noSlip;
    }
    obstacle
    {
        type            noSlip;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""
    )


def build_p_field() -> str:
    return (
        openfoam_header("p", "0", "volScalarField")
        + """dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    topWall
    {
        type            zeroGradient;
    }
    bottomWall
    {
        type            zeroGradient;
    }
    obstacle
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""
    )


def write_template(cfg: dict) -> Path:
    ensure_project_dirs(ROOT)
    template = ROOT / cfg["project"]["template"]
    (template / "0").mkdir(parents=True, exist_ok=True)
    (template / "constant").mkdir(parents=True, exist_ok=True)
    (template / "system").mkdir(parents=True, exist_ok=True)

    (template / "system" / "blockMeshDict").write_text(
        build_block_mesh_dict(cfg), encoding="utf-8"
    )
    (template / "system" / "fvSchemes").write_text(build_fv_schemes(), encoding="utf-8")
    (template / "system" / "fvSolution").write_text(build_fv_solution(), encoding="utf-8")
    write_decompose_par(
        template,
        int(cfg["parallel"]["nProcs_per_case"]),
        str(cfg["parallel"]["decomposition_method"]),
    )

    mode = cfg["modes"]["test"]
    (template / "system" / "controlDict").write_text(
        control_dict_text(
            cfg,
            application="pimpleFoam",
            start_from="startTime",
            start_time=0,
            end_time=mode["endTime_retain"],
            delta_t=mode["deltaT"],
            write_control="runTime",
            write_interval=mode["writeInterval"],
            adjust_time_step=True,
            max_co=mode["maxCo"],
            max_delta_t=mode["maxDeltaT"],
        ),
        encoding="utf-8",
    )

    write_viscosity_files(template, 1.0 / 20.0)
    (template / "constant" / "turbulenceProperties").write_text(
        openfoam_header("turbulenceProperties", "constant")
        + """simulationType  laminar;
""",
        encoding="utf-8",
    )
    (template / "constant" / "momentumTransport").write_text(
        openfoam_header("momentumTransport", "constant")
        + """simulationType  laminar;
""",
        encoding="utf-8",
    )
    (template / "0" / "U").write_text(build_u_field(), encoding="utf-8")
    (template / "0" / "p").write_text(build_p_field(), encoding="utf-8")
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the OpenFOAM case template.")
    parser.add_argument("--config", default=str(ROOT / "config" / "sweep.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    template = write_template(cfg)
    print(f"Template written to {template}")


if __name__ == "__main__":
    main()
