"""Harbor adapter package for the Pico agent harness.

The only public surface is :class:`pico_adapter.pico_agent.PicoAgent`, which
Harbor resolves through ``--agent pico_adapter.pico_agent:PicoAgent``.
"""

from pico_adapter.pico_agent import PicoAgent

__all__ = ["PicoAgent"]
