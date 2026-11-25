"""Drive a fixed repeatable Matplotlib selection sequence on the Agg backend."""

from contextlib import contextmanager, ExitStack
from unittest.mock import patch

from matplotlib import pyplot as plt, widgets
from matplotlib.backend_bases import MouseEvent


@contextmanager
def selections(mode, polygons, *, undo=False):
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
