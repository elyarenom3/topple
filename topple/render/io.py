"""Image input and output: a minimal OpenEXR scanline writer and reader (16 bit half float or
32 bit float, no compression, several passes as channel groups) and PNG through imageio.

Only the subset of the OpenEXR format needed for this project is implemented: single part
scanline images, INCREASING_Y line order, NO_COMPRESSION, HALF or FLOAT channels. Files
written here open in standard tools (OpenEXR, Blender, Nuke) and files from those tools can be
read back as long as they are uncompressed scanline images.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = 20000630
VERSION = 2


def _attr(name: str, type_name: str, data: bytes) -> bytes:
    return name.encode() + b"\x00" + type_name.encode() + b"\x00" + struct.pack("<i", len(data)) + data


def _pass_channels(passes: dict) -> list[tuple[str, np.ndarray]]:
    """Flatten passes to (channel name, 2D array) pairs. Beauty uses R, G, B; other passes
    use pass.R or pass.Y for scalars."""
    chans = []
    for pname, arr in passes.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            chans.append((f"{pname}.Y" if pname != "beauty" else "Y", arr))
            continue
        labels = ["R", "G", "B", "A"][: arr.shape[2]]
        for k, lab in enumerate(labels):
            name = lab if pname == "beauty" else f"{pname}.{lab}"
            chans.append((name, np.ascontiguousarray(arr[:, :, k])))
    chans.sort(key=lambda c: c[0])
    return chans


def write_exr(path, passes: dict, half: bool = True) -> None:
    """Write passes (dict name -> (H, W) or (H, W, C) float arrays) as one multi channel EXR."""
    chans = _pass_channels(passes)
    H, W = chans[0][1].shape
    ptype = 1 if half else 2
    nbytes = 2 if half else 4
    chlist = b""
    for name, _ in chans:
        chlist += name.encode() + b"\x00" + struct.pack("<iBBBBii", ptype, 0, 0, 0, 0, 1, 1)
    chlist += b"\x00"
    header = b""
    header += _attr("channels", "chlist", chlist)
    header += _attr("compression", "compression", b"\x00")
    header += _attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, W - 1, H - 1))
    header += _attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, W - 1, H - 1))
    header += _attr("lineOrder", "lineOrder", b"\x00")
    header += _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\x00"
    pre = struct.pack("<ii", MAGIC, VERSION) + header
    offsets_pos = len(pre)
    line_size = 8 + len(chans) * W * nbytes
    data_start = offsets_pos + 8 * H
    offsets = np.array([data_start + y * line_size for y in range(H)], dtype="<u8")
    rows = []
    dtype = "<f2" if half else "<f4"
    stacked = np.stack([c[1] for c in chans], axis=0).astype(dtype)  # (C, H, W)
    for y in range(H):
        rows.append(struct.pack("<ii", y, len(chans) * W * nbytes) + stacked[:, y, :].tobytes())
    with open(path, "wb") as f:
        f.write(pre)
        f.write(offsets.tobytes())
        f.write(b"".join(rows))


def read_exr(path) -> dict:
    """Read an uncompressed scanline EXR into passes: dict name -> (H, W) or (H, W, C) float32."""
    with open(path, "rb") as f:
        buf = f.read()
    magic, version = struct.unpack_from("<ii", buf, 0)
    if magic != MAGIC:
        raise ValueError("not an EXR file")
    pos = 8
    attrs = {}
    while True:
        if buf[pos] == 0:
            pos += 1
            break
        end = buf.index(b"\x00", pos)
        name = buf[pos:end].decode()
        pos = end + 1
        end = buf.index(b"\x00", pos)
        tname = buf[pos:end].decode()
        pos = end + 1
        size = struct.unpack_from("<i", buf, pos)[0]
        pos += 4
        attrs[name] = (tname, buf[pos : pos + size])
        pos += size
    if attrs["compression"][1] != b"\x00":
        raise ValueError("only uncompressed EXR files are supported")
    x0, y0, x1, y1 = struct.unpack("<iiii", attrs["dataWindow"][1])
    W, H = x1 - x0 + 1, y1 - y0 + 1
    chans = []
    cl = attrs["channels"][1]
    p = 0
    while cl[p] != 0:
        end = cl.index(b"\x00", p)
        name = cl[p:end].decode()
        p = end + 1
        ptype = struct.unpack_from("<i", cl, p)[0]
        p += 16
        chans.append((name, ptype))
    offsets = np.frombuffer(buf, dtype="<u8", count=H, offset=pos)
    out = {name: np.empty((H, W), dtype=np.float32) for name, _ in chans}
    for y in range(H):
        o = int(offsets[y])
        yy, size = struct.unpack_from("<ii", buf, o)
        o += 8
        for name, ptype in chans:
            if ptype == 1:
                row = np.frombuffer(buf, dtype="<f2", count=W, offset=o).astype(np.float32)
                o += 2 * W
            elif ptype == 2:
                row = np.frombuffer(buf, dtype="<f4", count=W, offset=o)
                o += 4 * W
            else:
                row = np.frombuffer(buf, dtype="<u4", count=W, offset=o).astype(np.float32)
                o += 4 * W
            out[name][yy] = row
    # regroup channels into passes
    passes: dict = {}
    groups: dict = {}
    for name in out:
        if "." in name:
            pname, lab = name.rsplit(".", 1)
        else:
            pname, lab = "beauty", name
        groups.setdefault(pname, {})[lab] = out[name]
    for pname, d in groups.items():
        if "Y" in d and len(d) == 1:
            passes[pname] = d["Y"]
        else:
            labs = [l for l in ["R", "G", "B", "A"] if l in d]
            passes[pname] = np.stack([d[l] for l in labs], axis=-1)
    return passes


def write_png(path, rgb: np.ndarray) -> None:
    """Write an sRGB encoded image in [0, 1] (or uint8) as PNG."""
    import imageio.v3 as iio

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), arr)


def read_png(path) -> np.ndarray:
    import imageio.v3 as iio

    return np.asarray(iio.imread(str(path)))[..., :3].astype(np.float64) / 255.0
