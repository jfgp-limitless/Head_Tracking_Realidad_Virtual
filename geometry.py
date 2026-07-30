"""
geometry.py
Carga de modelos CAD reales (.stl / .obj) exportados desde SolidWorks
u otro software, más algunas formas básicas de respaldo.

Para mantener el rendimiento en equipos con pocos recursos, todo
modelo cargado pasa por una etapa de "decimación" (reducción de
vértices/aristas) antes de dibujarse como wireframe.
"""

import os
import struct
import numpy as np


# ---------------------------------------------------------------------
# Formas básicas de respaldo (por si no se selecciona ningún archivo)
# ---------------------------------------------------------------------

def cube(size=1.0):
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
    ], dtype=np.float64)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return vertices, edges


def build_room(width=9.75, depth=9.0, floor_y=4.875, ceil_y=-4.875, num_grid_x=10, num_grid_y=10, num_grid_z=9):
    """
    Genera la estructura de un cuarto 3D que ocupa todo el lienzo (canvas):
    - La boca frontal del cuarto (z = 0) coincide exactamente con los bordes del lienzo.
    - Las 4 paredes (piso, techo, izquierda, derecha) convergen hacia la pared trasera.
    """
    verts = []
    grid_edges = []
    frame_edges = []

    def add_line(p1, p2, target_edges):
        i = len(verts)
        verts.append(p1)
        verts.append(p2)
        target_edges.append((i, i + 1))

    hw = width / 2.0
    xs = np.linspace(-hw, hw, num_grid_x + 1)
    ys = np.linspace(ceil_y, floor_y, num_grid_y + 1)
    zs = np.linspace(0.0, depth, num_grid_z + 1)

    # --- piso (y = floor_y) ---
    for x in xs:
        add_line((x, floor_y, 0.0), (x, floor_y, depth), grid_edges)
    for z in zs:
        add_line((-hw, floor_y, z), (hw, floor_y, z), grid_edges)

    # --- techo (y = ceil_y) ---
    for x in xs:
        add_line((x, ceil_y, 0.0), (x, ceil_y, depth), grid_edges)
    for z in zs:
        add_line((-hw, ceil_y, z), (hw, ceil_y, z), grid_edges)

    # --- pared trasera (z = depth) ---
    for x in xs:
        add_line((x, ceil_y, depth), (x, floor_y, depth), grid_edges)
    for y in ys:
        add_line((-hw, y, depth), (hw, y, depth), grid_edges)

    # --- pared izquierda (x = -hw) ---
    for z in zs:
        add_line((-hw, ceil_y, z), (-hw, floor_y, z), grid_edges)
    for y in ys:
        add_line((-hw, y, 0.0), (-hw, y, depth), grid_edges)

    # --- pared derecha (x = hw) ---
    for z in zs:
        add_line((hw, ceil_y, z), (hw, floor_y, z), grid_edges)
    for y in ys:
        add_line((hw, y, 0.0), (hw, y, depth), grid_edges)

    # --- esquinas/marco estructural principal del cuarto ---
    # 4 vigas en profundidad (de z=0 a z=depth)
    add_line((-hw, floor_y, 0.0), (-hw, floor_y, depth), frame_edges)
    add_line((hw, floor_y, 0.0), (hw, floor_y, depth), frame_edges)
    add_line((-hw, ceil_y, 0.0), (-hw, ceil_y, depth), frame_edges)
    add_line((hw, ceil_y, 0.0), (hw, ceil_y, depth), frame_edges)

    # marco trasero (z = depth)
    add_line((-hw, floor_y, depth), (hw, floor_y, depth), frame_edges)
    add_line((-hw, ceil_y, depth), (hw, ceil_y, depth), frame_edges)
    add_line((-hw, floor_y, depth), (-hw, ceil_y, depth), frame_edges)
    add_line((hw, floor_y, depth), (hw, ceil_y, depth), frame_edges)

    # marco frontal (z = 0)
    add_line((-hw, floor_y, 0.0), (hw, floor_y, 0.0), frame_edges)
    add_line((-hw, ceil_y, 0.0), (hw, ceil_y, 0.0), frame_edges)
    add_line((-hw, floor_y, 0.0), (-hw, ceil_y, 0.0), frame_edges)
    add_line((hw, floor_y, 0.0), (hw, ceil_y, 0.0), frame_edges)

    return np.array(verts, dtype=np.float64), grid_edges, frame_edges


# ---------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------

def _center_and_scale(vertices, target=1.0):
    """Centra el modelo en el origen y lo normaliza a un tamaño manejable."""
    center = vertices.mean(axis=0)
    vertices = vertices - center
    max_extent = np.max(np.linalg.norm(vertices, axis=1))
    if max_extent > 0:
        vertices = vertices / max_extent * target
    return vertices


def decimate_mesh(vertices, edges, max_edges=2500):
    """
    Reduce la cantidad de vértices/aristas fusionando puntos cercanos en
    una grilla (quantización espacial). Es una decimación barata mucho
    más liviana que un algoritmo de simplificación de malla completo,
    ideal para equipos con pocos recursos.
    """
    edges = np.array(list(edges), dtype=np.int64)
    if len(edges) <= max_edges or len(vertices) < 4:
        return vertices, [tuple(e) for e in edges]

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    size = np.maximum(bbox_max - bbox_min, 1e-9)

    resolution = 220
    while resolution > 6:
        cell = size / resolution
        keys = np.floor((vertices - bbox_min) / cell).astype(np.int64)

        # unicidad vectorizada de las celdas ocupadas
        view = np.ascontiguousarray(keys).view(
            [('x', keys.dtype), ('y', keys.dtype), ('z', keys.dtype)]
        ).reshape(-1)
        _, inverse = np.unique(view, return_inverse=True)

        remapped = inverse[edges]
        valid = remapped[:, 0] != remapped[:, 1]
        new_edges = np.unique(np.sort(remapped[valid], axis=1), axis=0)

        if len(new_edges) <= max_edges or resolution <= 8:
            n_cells = inverse.max() + 1
            new_verts = np.zeros((n_cells, 3))
            counts = np.zeros(n_cells)
            np.add.at(new_verts, inverse, vertices)
            np.add.at(counts, inverse, 1)
            new_verts /= counts[:, None]
            return new_verts, [tuple(e) for e in new_edges]

        resolution = int(resolution * 0.6)

    return vertices, [tuple(e) for e in edges]


# ---------------------------------------------------------------------
# Carga de OBJ
# ---------------------------------------------------------------------

def load_obj(path, target=1.0):
    vertices = []
    edges = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()[1:4]
                vertices.append([float(p) for p in parts])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idxs = [int(p.split("/")[0]) - 1 for p in parts]
                n = len(idxs)
                for k in range(n):
                    a, b = idxs[k], idxs[(k + 1) % n]
                    edges.add(tuple(sorted((a, b))))

    if not vertices:
        raise ValueError("El archivo .OBJ no contiene vértices válidos.")

    vertices = _center_and_scale(np.array(vertices, dtype=np.float64), target=target)
    return vertices, list(edges)


# ---------------------------------------------------------------------
# Carga de STL (binario y ASCII)
# ---------------------------------------------------------------------

def _load_stl_binary(path, n_tri):
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("v3", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    with open(path, "rb") as f:
        f.seek(84)
        data = np.fromfile(f, dtype=dtype, count=n_tri)

    tris = np.stack([data["v1"], data["v2"], data["v3"]], axis=1)  # (n,3,3)
    vertices = tris.reshape(-1, 3).astype(np.float64)

    n = len(tris)
    base = (np.arange(n) * 3)[:, None]
    local_edges = np.array([[0, 1], [1, 2], [2, 0]])
    edges = (base[:, None, :] + local_edges[None, :, :]).reshape(-1, 2)
    return vertices, [tuple(e) for e in edges]


def _load_stl_ascii(path):
    vertices = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()[1:4]
                vertices.append([float(p) for p in parts])

    if not vertices:
        raise ValueError("El archivo .STL (ASCII) no contiene vértices válidos.")

    vertices = np.array(vertices, dtype=np.float64)
    n_tri = len(vertices) // 3
    vertices = vertices[: n_tri * 3]
    base = (np.arange(n_tri) * 3)[:, None]
    local_edges = np.array([[0, 1], [1, 2], [2, 0]])
    edges = (base[:, None, :] + local_edges[None, :, :]).reshape(-1, 2)
    return vertices, [tuple(e) for e in edges]


def load_stl(path, target=1.0):
    with open(path, "rb") as f:
        f.read(80)
        count_bytes = f.read(4)

    if len(count_bytes) == 4:
        n_tri = struct.unpack("<I", count_bytes)[0]
        expected_size = 84 + 50 * n_tri
        actual_size = os.path.getsize(path)
        if actual_size == expected_size and n_tri > 0:
            vertices, edges = _load_stl_binary(path, n_tri)
            vertices = _center_and_scale(vertices, target=target)
            return vertices, edges

    vertices, edges = _load_stl_ascii(path)
    vertices = _center_and_scale(vertices, target=target)
    return vertices, edges


# ---------------------------------------------------------------------
# Punto de entrada único: detecta formato y aplica decimación
# ---------------------------------------------------------------------

def load_cad_file(path, max_edges=2500):
    """
    Carga cualquier archivo CAD soportado (.stl o .obj), lo centra,
    normaliza y decima para que sea liviano de renderizar en tiempo real.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        vertices, edges = load_stl(path)
    elif ext == ".obj":
        vertices, edges = load_obj(path)
    else:
        raise ValueError(f"Formato no soportado: {ext} (use .stl o .obj)")

    vertices, edges = decimate_mesh(vertices, edges, max_edges=max_edges)
    return vertices, edges