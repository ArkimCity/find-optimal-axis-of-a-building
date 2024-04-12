"""
이 특정 파일명의 파이썬 파일이 root dir에 있어야 debuging시 import 되는 듯 함.
https://gitlab.com/fehrlich/vscode-debug-visualizer-py
"""
import json
from enum import Enum
from typing import Any, List, Tuple

import plotly
import plotly.graph_objects as go
from plotly.missing_ipywidgets import FigureWidget
from plotly.subplots import make_subplots
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry, GeometrySequence
from vscodedebugvisualizer import globalVisualizationFactory

COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS


def gen_fig(rows=1, cols=1) -> FigureWidget:
    """그림을 그릴 figure를 생성한다.

    Args:
        rows (int, optional): Subplot row 숫자. Defaults to 1.
        cols (int, optional): Subplot column 숫자. Defaults to 1.

    Returns자
        FigureWidget: figure.
    """
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes="all", shared_yaxes="all")
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def fig_to_json(fig: FigureWidget, save_name="dv.html") -> str:
    """Figure를 plotly json으로 변환한다.

    Args:
        fig (FigureWidget): figure.

    Returns:
        str: 변환된 json.
    """
    fig_json = fig.to_json()
    fig_dict = json.loads(fig_json)
    fig_dict["kind"] = {"plotly": True}
    with open(save_name, "w", encoding="utf-8") as f:
        fig.write_html(f)
    return json.dumps(fig_dict)


class FillType(Enum):
    """도형 내부를 채울지에 대한 정보."""

    NOFILL = 0
    FILL = 1
    ERASE = 2


def get_draw_items(obj: Any) -> List[Tuple[List[float], List[float], FillType]]:
    """물체의 한 붓 그리기 아이템 리스트를 구한다.
    한 붓 그리기 아이템은 x, y 좌표 리스트와 그 안을 색으로 채울지 여부이다.

    Args:
        obj (Any): 그릴 물체.

    Returns:
        List[Tuple[List[float], List[float], FillType]]: 한 붓 그리기할 아이템 리스트.
    """
    # Base case: non-collection
    if isinstance(obj, Point):
        if obj.is_empty:
            return [([], [], FillType.NOFILL)]
        return [(list(obj.coords.xy[0]), list(obj.coords.xy[1]), FillType.NOFILL)]
    elif isinstance(obj, LineString):
        if obj.is_empty:
            return [([], [], FillType.NOFILL)]
        return [(list(obj.coords.xy[0]), list(obj.coords.xy[1]), FillType.NOFILL)]
    elif isinstance(obj, Polygon):
        exterior: LineString
        if obj.is_empty:
            return [([], [], FillType.NOFILL)]
        elif isinstance(obj.boundary, LineString):
            exterior = obj.boundary
            return [
                (
                    list(exterior.coords.xy[0]),
                    list(exterior.coords.xy[1]),
                    FillType.FILL,
                )
            ]
        else:
            # 외곽선
            exterior_items: List[Tuple[List[float], List[float], FillType]]
            exterior = obj.boundary.geoms[0]
            exterior_items = [
                (
                    list(exterior.coords.xy[0]),
                    list(exterior.coords.xy[1]),
                    FillType.FILL,
                )
            ]
            # 내부 hole
            interior_items: List[Tuple[List[float], List[float], FillType]] = []
            for interior_line in obj.boundary.geoms[1:].geoms:
                item = (
                    list(interior_line.coords.xy[0]),
                    list(interior_line.coords.xy[1]),
                    FillType.ERASE,
                )
                interior_items.append(item)
            return exterior_items + interior_items

    # Base case: empty collection
    if isinstance(obj, GeometrySequence):
        obj = list(obj)
    elif isinstance(obj, BaseGeometry):
        obj = list(obj.geoms)

    if len(obj) == 0:
        return []

    # Inductive case: non-empty collection
    coords = []
    for element in obj:
        coords += get_draw_items(element)
    return coords


def plot_draw_item(
    fig,
    row: int,
    col: int,
    draw_item: Tuple[List[float], List[float], FillType],
    color,
    legendgroup: str,
    name: str,
    showlegend=True,
) -> None:
    """주어진 figure에 한 붓 그리기 아이템을 그려준다.

    Args:
        fig: figure.
        row (int): subfigure row.
        col (int): subfigure column.
        draw_item (Tuple[List[float], List[float], FillType]): 한 붓 그리기 아이템.
        color: 그려줄 색.
        legendgroup (str): 물체 그룹.
        name (str): 물체 이름.
        showlegend (bool, optional): 레전드를 그릴지 말지. Defaults to True.
    """
    fillcolor = f"rgba({color[4:-1]}, 0.3)"
    fig.add_trace(
        go.Scatter(
            x=draw_item[0],
            y=draw_item[1],
            mode="lines+markers",
            marker={"color": color},
            fill=None if draw_item[2] == FillType.NOFILL else "toself",
            fillcolor="white" if draw_item[2] == FillType.ERASE else fillcolor,
            line_color=color,
            legendgroup=legendgroup,
            showlegend=showlegend,
            name=name,
        ),
        row=row,
        col=col,
    )


def plot_floor_object(fig, floor_obj: Any, num_floors: int, idx: int) -> None:
    """주어진 figure에 층마다 있는 물체를 그려준다.

    Args:
        fig: figure.
        floor_obj (Any): 층마다 있는 물체.
        num_floors (int): 그려줄 최대 층수.
        idx (int): 이 물체의 index.
    """
    # Nothing to plot.
    if num_floors == 0 or len(floor_obj) == 0:
        return

    legendgroup = f"group{idx}"
    name = f"obj{idx}"
    showlegend = True
    for i in range(0, num_floors):
        obj = floor_obj[i]
        draw_items = get_draw_items(obj)
        if len(draw_items) == 0:
            continue

        col = i + 1
        plot_draw_item(fig, 1, col, draw_items[0], COLORS[idx % len(COLORS)], legendgroup, name, showlegend)
        showlegend = False
        for draw_item in draw_items[1:]:
            plot_draw_item(fig, 1, col, draw_item, COLORS[idx % len(COLORS)], legendgroup, name, showlegend)


def plot(*obj_list: List[Any]) -> str:
    """주어진 물체들을 figure에 그려준다.

    Example:
        >>> dv.plot(unit_space, splits)
        >>> # figure에 unit_space와 splits를 동시에 그려준다.

    Returns:
        str: plotly가 인식할 수 있는 json string.
    """
    fig = gen_fig(1, 1)
    for idx, obj in enumerate(obj_list):
        plot_floor_object(fig, [obj], 1, idx)
    return fig_to_json(fig)


def plot_floors(*floor_obj_list: List[Any], floor: int = -1, save_name="dv.html") -> str:
    """층 마다 있는 물체들를 각가의 subfigure에 그려준다.

    Example:
        >>> dv.plot_floors(floor, core)
        >>> # subfigure 1에 1층의 floor와 core를,
        >>> # subfigure 2에 2층의 floor와 core를 그려준다.

    Args:
        floor (int, optional): 특정 층만 plot할 수 있다. Defaults to -1.

    Returns:
        str: plotly가 인식할 수 있는 json string.
    """
    # 특정 층만 보고 싶은 경우
    if floor != -1:
        obj_list = [floor_obj[floor] for floor_obj in floor_obj_list]
        return plot(*obj_list)

    num_subplots = min(len(floor_obj) for floor_obj in floor_obj_list)
    fig = gen_fig(1, num_subplots)
    for idx, floor_obj in enumerate(floor_obj_list):
        plot_floor_object(fig, floor_obj, num_subplots, idx)
    return fig_to_json(fig, save_name)


class Plot:
    def __init__(self, *obj_list: List[Any]) -> None:
        self.json_str = plot(*obj_list)


class PlotFloors:
    def __init__(self, *floor_obj_list: List[Any], floor: int = -1, save_name="dv.html") -> None:
        self.json_str = plot_floors(*floor_obj_list, floor=floor, save_name=save_name)


class PlotVisualizer:
    def checkType(self, t):  # pylint: disable=invalid-name
        return isinstance(t, Plot)

    def visualize(self, p: Plot):
        return p.json_str


class PlotFloorsVisualizer:
    def checkType(self, t):  # pylint: disable=invalid-name
        return isinstance(t, PlotFloors)

    def visualize(self, p: PlotFloors):
        return p.json_str


globalVisualizationFactory.addVisualizer(PlotVisualizer())
globalVisualizationFactory.addVisualizer(PlotFloorsVisualizer())
