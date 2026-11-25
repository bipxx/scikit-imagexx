"""Record threshold panels or drive scripted selections on the Agg backend."""

from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class PanelAxis:
    transAxes = object()

    def __init__(self):
        self.images = []
        self.texts = []
        self.axison = True
        self.title = ""

    def imshow(self, array, *, cmap):
        assert cmap == "gray"
        self.images.append(SimpleNamespace(get_array=lambda: array))

    def set_title(self, title):
        self.title = title

    def get_title(self):
        return self.title

    def text(self, x, y, message, *, ha, va, transform):
        assert (x, y, ha, va, transform) == (0.5, 0.5, "center", "center", self.transAxes)
        self.texts.append(SimpleNamespace(get_text=lambda: message))

    def axis(self, setting):
        assert setting == "off"
        self.axison = False


def panel_pyplot():
    def subplots(rows, cols, *, figsize, sharex, sharey):
        assert (rows, cols, sharex, sharey) == (4, 2, True, True)
        fig = SimpleNamespace(get_size_inches=lambda: figsize, tight_layout=lambda: None)
        axes = np.array([PanelAxis() for _ in range(rows * cols)]).reshape(rows, cols)
        return fig, axes

    return SimpleNamespace(subplots=subplots, cm=SimpleNamespace(gray="gray"))


@contextmanager
def selections(mode, polygons, *, undo=False):
    from matplotlib import pyplot as plt, widgets
    from matplotlib.backend_bases import MouseEvent

    callbacks = []

    class ScriptedLasso:
        def __init__(self, ax, callback):
            callbacks.append(callback)

    def show(*, block):
        assert block is True
        fig = plt.gcf()
        ax = fig.axes[0]
        fig.canvas.draw()
        for vertices in polygons:
            if mode == "lasso":
                callbacks[-1](vertices)
            else:
                for x, y in vertices:
                    event = MouseEvent("button_press_event", fig.canvas,
                                       *ax.transData.transform((x, y)), button=1)
                    # MouseEvent quantizes display coordinates to integer pixels.
                    # Use the specified data coordinates for exact edge tests.
                    event.xdata, event.ydata = x, y
                    fig.canvas.callbacks.process(event.name, event)
                event = MouseEvent("button_press_event", fig.canvas,
                                   *ax.transData.transform(vertices[-1]), button=3)
                fig.canvas.callbacks.process(event.name, event)
        callbacks.clear()
        if undo:
            x, y = fig.axes[1].transAxes.transform((0.5, 0.5))
            for name in ("button_press_event", "button_release_event"):
                event = MouseEvent(name, fig.canvas, x, y, button=1)
                fig.canvas.callbacks.process(name, event)
        plt.close(fig)

    with ExitStack() as context:
        context.enter_context(patch.object(plt, "show", show))
        context.enter_context(patch.object(plt, "draw"))
        if mode == "lasso":
            context.enter_context(patch.object(widgets, "LassoSelector", ScriptedLasso))
        yield
