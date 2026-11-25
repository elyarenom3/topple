"""State recorder at sub frame resolution (Section 10.2): the renderer reads poses at any
time by linear interpolation of positions and spherical linear interpolation of orientations."""

from __future__ import annotations

import bisect

import numpy as np

from ..core.transforms import SE3, quat_slerp
from .body import WorldState


class StateRecorder:
    def __init__(self, world=None, every: int = 1):
        self.every = max(1, int(every))
        self.times: list[float] = []
        self.states: list[WorldState] = []
        self.names: list[str] = [b.name for b in world.bodies] if world is not None else []
        self._count = 0
        if world is not None:
            world.recorder = self
            self.append(world.state())

    def append(self, state: WorldState):
        self._count += 1
        if (self._count - 1) % self.every != 0:
            return
        self.times.append(float(state.time))
        self.states.append(state.copy())

    def __len__(self):
        return len(self.states)

    def pose_at(self, body_id: int, t: float) -> SE3:
        if not self.states:
            raise ValueError("empty recorder")
        if t <= self.times[0]:
            s = self.states[0]
            return SE3(s.q[body_id], s.x[body_id])
        if t >= self.times[-1]:
            s = self.states[-1]
            return SE3(s.q[body_id], s.x[body_id])
        j = bisect.bisect_right(self.times, t)
        i = j - 1
        t0, t1 = self.times[i], self.times[j]
        u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        s0, s1 = self.states[i], self.states[j]
        q = quat_slerp(s0.q[body_id], s1.q[body_id], u)
        x = (1 - u) * s0.x[body_id] + u * s1.x[body_id]
        return SE3(q, x)

    def state_at(self, t: float) -> WorldState:
        n = self.states[0].q.shape[0]
        q = np.empty((n, 4))
        x = np.empty((n, 3))
        for b in range(n):
            T = self.pose_at(b, t)
            q[b] = T.q
            x[b] = T.t
        return WorldState(t, q, x, np.zeros((n, 3)), np.zeros((n, 3)))

    def save(self, path):
        np.savez_compressed(
            path,
            times=np.array(self.times),
            q=np.array([s.q for s in self.states]),
            x=np.array([s.x for s in self.states]),
            v=np.array([s.v for s in self.states]),
            w=np.array([s.w for s in self.states]),
            names=np.array(self.names),
            every=self.every,
        )

    @staticmethod
    def load(path) -> "StateRecorder":
        d = np.load(path, allow_pickle=False)
        r = StateRecorder(None, int(d["every"]))
        r.names = [str(s) for s in d["names"]]
        for i, t in enumerate(d["times"]):
            r.times.append(float(t))
            r.states.append(WorldState(float(t), d["q"][i], d["x"][i], d["v"][i], d["w"][i]))
        r._count = len(r.times) * r.every
        return r
