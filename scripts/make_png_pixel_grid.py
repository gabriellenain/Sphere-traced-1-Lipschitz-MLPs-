#!/usr/bin/env python3
"""Overlay a coordinate grid on an 8-bit RGB PNG without third-party deps."""
from __future__ import annotations

import argparse
import binascii
import struct
import zlib
from pathlib import Path


PNG_SIG = b"\x89PNG\r\n\x1a\n"


FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    ",": ("000", "000", "000", "010", "100"),
    "x": ("000", "101", "010", "101", "000"),
    "y": ("000", "101", "111", "001", "110"),
    "=": ("000", "111", "000", "111", "000"),
    " ": ("000", "000", "000", "000", "000"),
}


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_rgb_png(path: Path) -> tuple[int, int, bytearray]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIG):
        raise ValueError(f"{path} is not a PNG")
    pos = len(PNG_SIG)
    width = height = None
    idat = bytearray()
    bit_depth = color_type = interlace = None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    if (bit_depth, color_type, interlace) != (8, 2, 0):
        raise ValueError(f"expected non-interlaced 8-bit RGB PNG, got bit={bit_depth} color={color_type} interlace={interlace}")
    assert width is not None and height is not None

    raw = zlib.decompress(bytes(idat))
    bpp = 3
    stride = width * bpp
    out = bytearray(height * stride)
    prev = bytearray(stride)
    src = 0
    for y in range(height):
        filt = raw[src]
        src += 1
        scan = bytearray(raw[src:src + stride])
        src += stride
        for i in range(stride):
            left = scan[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            if filt == 1:
                scan[i] = (scan[i] + left) & 255
            elif filt == 2:
                scan[i] = (scan[i] + up) & 255
            elif filt == 3:
                scan[i] = (scan[i] + ((left + up) >> 1)) & 255
            elif filt == 4:
                scan[i] = (scan[i] + _paeth(left, up, up_left)) & 255
            elif filt != 0:
                raise ValueError(f"unsupported PNG filter {filt}")
        out[y * stride:(y + 1) * stride] = scan
        prev = scan
    return width, height, out


def _blend(buf: bytearray, width: int, x: int, y: int, rgb: tuple[int, int, int], alpha: float) -> None:
    i = (y * width + x) * 3
    inv = 1.0 - alpha
    buf[i] = int(buf[i] * inv + rgb[0] * alpha)
    buf[i + 1] = int(buf[i + 1] * inv + rgb[1] * alpha)
    buf[i + 2] = int(buf[i + 2] * inv + rgb[2] * alpha)


def draw_line_h(buf: bytearray, width: int, height: int, y: int, rgb: tuple[int, int, int], alpha: float, thickness: int) -> None:
    for yy in range(max(0, y - thickness // 2), min(height, y + (thickness + 1) // 2)):
        for x in range(width):
            _blend(buf, width, x, yy, rgb, alpha)


def draw_line_v(buf: bytearray, width: int, height: int, x: int, rgb: tuple[int, int, int], alpha: float, thickness: int) -> None:
    for xx in range(max(0, x - thickness // 2), min(width, x + (thickness + 1) // 2)):
        for y in range(height):
            _blend(buf, width, xx, y, rgb, alpha)


def draw_text(buf: bytearray, width: int, height: int, x: int, y: int, text: str, rgb: tuple[int, int, int], scale: int = 2) -> None:
    def dot(px: int, py: int, color: tuple[int, int, int]) -> None:
        if 0 <= px < width and 0 <= py < height:
            i = (py * width + px) * 3
            buf[i:i + 3] = bytes(color)

    cx = x
    for ch in text:
        glyph = FONT.get(ch, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            dot(cx + gx * scale + sx + 1, y + gy * scale + sy + 1, (0, 0, 0))
                            dot(cx + gx * scale + sx, y + gy * scale + sy, rgb)
        cx += 4 * scale


def write_rgb_png(path: Path, width: int, height: int, buf: bytearray) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    rows = bytearray()
    stride = width * 3
    for y in range(height):
        rows.append(0)
        rows.extend(buf[y * stride:(y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_SIG + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--minor", type=int, default=10)
    ap.add_argument("--major", type=int, default=100)
    args = ap.parse_args()

    width, height, buf = read_rgb_png(args.src)
    for x in range(0, width, args.minor):
        draw_line_v(buf, width, height, x, (255, 255, 255), 0.18, 1)
    for y in range(0, height, args.minor):
        draw_line_h(buf, width, height, y, (255, 255, 255), 0.18, 1)
    for x in range(0, width, args.major):
        draw_line_v(buf, width, height, x, (255, 230, 0), 0.62, 2)
    for y in range(0, height, args.major):
        draw_line_h(buf, width, height, y, (255, 230, 0), 0.62, 2)

    # Borders mark the full 1600x1200 image extent; labels are pixel coordinates.
    draw_line_v(buf, width, height, 0, (255, 64, 64), 0.9, 3)
    draw_line_v(buf, width, height, width - 1, (255, 64, 64), 0.9, 3)
    draw_line_h(buf, width, height, 0, (255, 64, 64), 0.9, 3)
    draw_line_h(buf, width, height, height - 1, (255, 64, 64), 0.9, 3)
    for x in range(0, width, args.major):
        draw_text(buf, width, height, min(x + 3, width - 70), 4, f"x={x}", (255, 245, 80), 2)
    for y in range(0, height, args.major):
        draw_text(buf, width, height, 4, min(y + 3, height - 14), f"y={y}", (255, 245, 80), 2)

    write_rgb_png(args.dst, width, height, buf)
    print(f"wrote {args.dst} ({width}x{height})")


if __name__ == "__main__":
    main()
