#!/usr/bin/env python3
"""Emit an exact-scale calibration checkerboard as a PDF.

Calibration is only as good as the square size you type into --square, and the
usual way that goes wrong is the printer: "Fit to page" silently scales the
sheet by a percent or two and every focal length inherits the error. So this
writes a PDF in real units with a printed ruler on it - measure the ruler after
printing, and if it is not exactly 100 mm, the sheet is scaled and the numbers
on it are wrong.

PDF is written by hand rather than through a plotting library because the
units have to survive end to end: 1 pt is exactly 1/72 inch, so millimetres are
exact and nothing resamples.

    make_checkerboard.py -o assets/calib/checkerboard_a4_9x6_25mm.pdf

Default is 9x6 INNER corners (10x7 squares) at 25 mm on A4 landscape, which
leaves a ~23 mm white quiet zone - findChessboardCorners needs that border and
fails on a board printed edge to edge.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MM = 72.0 / 25.4          # points per millimetre
PAGES = {"a4": (297.0, 210.0), "a3": (420.0, 297.0), "letter": (279.4, 215.9)}


def build(cols: int, rows: int, sq_mm: float, page_mm: tuple[float, float]) -> bytes:
    """cols/rows are SQUARE counts; inner corners are (cols-1) x (rows-1)."""
    pw, ph = page_mm[0] * MM, page_mm[1] * MM
    s = sq_mm * MM
    bw, bh = cols * s, rows * s
    if bw > pw or bh > ph:
        raise SystemExit(
            f"{cols}x{rows} squares of {sq_mm} mm is {cols*sq_mm:.0f}x{rows*sq_mm:.0f} mm, "
            f"which does not fit {page_mm[0]:.0f}x{page_mm[1]:.0f} mm")
    x0, y0 = (pw - bw) / 2, (ph - bh) / 2
    margin = min(x0, y0) / MM
    if margin < 8:
        print(f"warning: only {margin:.0f} mm of quiet zone; detection may fail",
              file=sys.stderr)

    ops = ["0 0 0 rg"]
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ops.append(f"{x0 + c*s:.4f} {y0 + r*s:.4f} {s:.4f} {s:.4f} re f")

    # A 100 mm ruler under the board: the print-scale check.
    ry = max(y0 - 10 * MM, 6 * MM)
    ops += ["0.6 w", f"{x0:.4f} {ry:.4f} m {x0 + 100*MM:.4f} {ry:.4f} l S"]
    for i in range(11):                       # 10 mm ticks
        tx = x0 + i * 10 * MM
        h = 3.5 * MM if i % 5 == 0 else 2.0 * MM
        ops.append(f"{tx:.4f} {ry:.4f} m {tx:.4f} {ry - h:.4f} l S")
    inner = f"{cols-1}x{rows-1}"
    label = (f"{inner} inner corners, {sq_mm:g} mm squares  |  "
             f"calibrate_cameras.py --board {inner} --square {sq_mm/1000:g}")
    ops += ["BT /F1 8 Tf", f"1 0 0 1 {x0:.4f} {ry + 3*MM:.4f} Tm",
            f"({label}) Tj", "ET",
            "BT /F1 7 Tf", f"1 0 0 1 {x0 + 102*MM:.4f} {ry - 2*MM:.4f} Tm",
            "(<- this line is exactly 100 mm. Measure it. If it is not, reprint "
            "at 100% scale / Actual size.) Tj", "ET"]
    stream = "\n".join(ops).encode("latin-1")

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.4f %.4f] "
         "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
         % (pw, ph)).encode("latin-1"),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, xref))
    return bytes(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--squares", default="10x7", help="SQUARE counts, not corners")
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--page", default="a4", choices=sorted(PAGES))
    p.add_argument("-o", "--output", type=Path, required=True)
    a = p.parse_args()

    cols, rows = (int(v) for v in a.squares.lower().split("x"))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(build(cols, rows, a.square_mm, PAGES[a.page]))
    print(f"{a.output}  {cols}x{rows} squares @ {a.square_mm:g} mm "
          f"-> {cols-1}x{rows-1} inner corners, "
          f"board {cols*a.square_mm:.0f}x{rows*a.square_mm:.0f} mm on {a.page.upper()}")
    print(f"print at 100% scale, then: --board {cols-1}x{rows-1} "
          f"--square {a.square_mm/1000:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
