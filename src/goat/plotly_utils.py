import numpy as np
import plotly.graph_objects as go


def color_str(color):
    """Plotly color string.
    Args:
        c: list [r, g, b] in [0, 1] range
    """
    color = list((np.array(color) * 255.0).astype("int"))
    return """rgb({}, {}, {})""".format(color[0], color[1], color[2])


def c_to_colors(c):
    """
    Args:
        c: np.array with shape (n,)
    """
    colors = [color_str(color) for color in c]
    return colors


def get_scatter3d(x, y, z, colors=None, name=None, size=2, opacity=1.0):
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        name=name,
        mode='markers',
        marker=dict(
            size=size,
            # set colors to that of the image
            color=colors,
            colorscale='Viridis',   # choose a colorscale
            opacity=opacity
        )
    )


def get_lines_for_axis(T, length=1.0):
    """T is the transormation to apply.
    """
    lines = []
    for i in range(3):
        point = np.zeros(4)
        point[3] = 1.0  # because homogeneous
        origin = point.copy()
        point[i] = 1.0 * length
        line_points = np.stack([origin, point], axis=0)  # 2, 4

        # transform them and remove homogeneous nature
        line_points = (T @ line_points.transpose()).transpose()[:, :3]
        lines.append(line_points)
    return lines


def get_line_segments_from_lines(lines,
                                 color=color_str((1, 0, 0)),
                                 marker_color=color_str((1, 0, 0)),
                                 colors=None,
                                 draw_marker=True,
                                 draw_line=True,
                                 marker_size=4,
                                 line_width=10):
    """
    lines - list of [np.array(2, 3), ...]
    line is defined by a (2, 3) array
    """
    data = []
    for idx, line in enumerate(lines):
        thiscolor = color if draw_line else "rgba(0, 0, 0, 0)"
        if colors is not None:
            marker_color = colors[idx]
            thiscolor = colors[idx]
        data.append(
            go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                showlegend=False,
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                    colorscale='Viridis',
                ) if draw_marker else dict(color="rgba(0, 0, 0, 0)"),
                line=dict(
                    color=thiscolor,
                    width=line_width
                )
            )
        )
    return data


def get_axis_scatters_from_transform(T, length=1.0):
    """Return the scatters needed to plot the camera pose with an axis representation.
    """
    lines = get_lines_for_axis(T, length=length)
    scatters = []
    for idx, line in enumerate(lines):
        # x = red, y = green, z = blue
        color = np.zeros(3)
        color[idx] = 1
        scatters += get_line_segments_from_lines([line], color_str(color))
    return scatters