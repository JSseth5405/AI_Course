"""
Wumpus World (tiny console version)

- The world is an N x N grid.
- You place pits, one Wumpus, and one gold.
- The agent starts at (1, 1) and tries to find the gold without dying.

Percepts at the current cell:
- Breeze  -> at least one adjacent pit
- Stench  -> the Wumpus is in an adjacent cell
- Glitter -> the gold is in the current cell

This is a simple knowledge-based explorer: it marks cells as "maybe dangerous" or "ruled out"
based on what it has sensed so far, and only moves through cells it currently believes are safe.

Coordinate system:
- (1, 1) is the bottom-left.
- x increases to the right, y increases upwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable, Optional, Set, Tuple, List
import random

Cell = Tuple[int, int]


# -----------------------------
# World
# -----------------------------
@dataclass(frozen=True)
class World:
    size: int
    pits: Set[Cell]
    wumpus: Cell
    gold: Cell

    def valid(self, cell: Cell) -> bool:
        x, y = cell
        return 1 <= x <= self.size and 1 <= y <= self.size

    def neighbours(self, cell: Cell) -> List[Cell]:
        x, y = cell
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in cand if self.valid(c)]

    def percepts(self, cell: Cell) -> tuple[bool, bool, bool]:
        """Return (breeze, stench, glitter)."""
        neigh = self.neighbours(cell)
        breeze = any(n in self.pits for n in neigh)
        stench = any(n == self.wumpus for n in neigh)
        glitter = (cell == self.gold)
        return breeze, stench, glitter


# -----------------------------
# Agent
# -----------------------------
class Explorer:
    """
    A small "knowledge-based" explorer.

    It tracks four kinds of knowledge:
    - no_pit / no_wumpus: cells that have been ruled out as hazards
    - maybe_pit / maybe_wumpus: cells that could contain a hazard
    - visited: cells we've stepped on and survived
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.pos: Cell = (1, 1)

        self.visited: Set[Cell] = set()

        self.no_pit: Set[Cell] = set()
        self.no_wumpus: Set[Cell] = set()

        self.maybe_pit: Set[Cell] = set()
        self.maybe_wumpus: Set[Cell] = set()

    def neighbours(self, cell: Cell) -> List[Cell]:
        x, y = cell
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in cand if 1 <= c[0] <= self.size and 1 <= c[1] <= self.size]

    def update_knowledge(self, cell: Cell, breeze: bool, stench: bool) -> None:
        """
        Update the knowledge base using the current percepts.

        Intuition:
        - If there's NO breeze here, none of the neighbours can be pits.
        - If there IS a breeze, neighbours become "maybe pit" unless already ruled out.
        Same for stench / Wumpus.
        """
        self.visited.add(cell)

        neigh = self.neighbours(cell)

        # Pits
        if not breeze:
            for n in neigh:
                self.no_pit.add(n)
                self.maybe_pit.discard(n)
        else:
            for n in neigh:
                if n not in self.no_pit:
                    self.maybe_pit.add(n)

        # Wumpus
        if not stench:
            for n in neigh:
                self.no_wumpus.add(n)
                self.maybe_wumpus.discard(n)
        else:
            for n in neigh:
                if n not in self.no_wumpus:
                    self.maybe_wumpus.add(n)

    def safe_cells(self) -> Set[Cell]:
        """
        Cells we currently believe are safe enough to walk on.

        Rule used here (simple, not perfect logic):
        - A cell is considered safe if it is:
          - already visited, OR
          - ruled out for pits, OR
          - ruled out for Wumpus
        - Then we remove cells that are still marked as "maybe pit" or "maybe wumpus"
          (unless that hazard has been ruled out for that cell).
        """
        safe = set(self.visited) | {(1, 1)} | set(self.no_pit) | set(self.no_wumpus)

        # Drop cells that are still suspicious
        for c in list(safe):
            pit_suspect = (c in self.maybe_pit and c not in self.no_pit)
            wumpus_suspect = (c in self.maybe_wumpus and c not in self.no_wumpus)
            if pit_suspect or wumpus_suspect:
                safe.discard(c)

        # Make sure we don't keep anything outside the grid
        safe = {c for c in safe if 1 <= c[0] <= self.size and 1 <= c[1] <= self.size}
        return safe

    def choose_next_step(self) -> Optional[Cell]:
        """
        Pick the next cell to move to.

        Strategy:
        - Only walk through cells we currently believe are safe.
        - Prefer going to a safe cell we haven't visited yet.
        - If the closest unvisited safe cell isn't adjacent, use BFS through safe cells
          to find a short path and take just the first step.
        """
        allowed = self.safe_cells()
        goals = allowed - self.visited

        if not goals:
            return None

        # Easy win: if any unvisited safe neighbour exists, pick one.
        direct = [n for n in self.neighbours(self.pos) if n in goals]
        if direct:
            return random.choice(direct)

        # Otherwise, BFS through safe cells to the nearest goal.
        step = _bfs_next_step(start=self.pos, goals=goals, allowed=allowed, neighbours=self.neighbours)
        return step


def _bfs_next_step(
    start: Cell,
    goals: Set[Cell],
    allowed: Set[Cell],
    neighbours,
) -> Optional[Cell]:
    """Return the *next* cell on a shortest path from start to any goal, restricted to allowed cells."""
    if start in goals:
        return start

    q = deque([start])
    parent: dict[Cell, Optional[Cell]] = {start: None}

    while q:
        cur = q.popleft()
        for nxt in neighbours(cur):
            if nxt not in allowed:
                continue
            if nxt in parent:
                continue
            parent[nxt] = cur
            if nxt in goals:
                # Reconstruct backwards until the first step after start
                node = nxt
                while parent[node] is not None and parent[node] != start:
                    node = parent[node]  # type: ignore[assignment]
                return node
            q.append(nxt)

    return None


# -----------------------------
# Display helpers
# -----------------------------
def print_agent_view(agent: Explorer) -> None:
    legend = "A=agent  V=visited  S=known safe  ?=unknown"
    print("\nAgent view (" + legend + "):")

    safe = agent.safe_cells()

    for y in range(agent.size, 0, -1):
        row = []
        for x in range(1, agent.size + 1):
            c = (x, y)
            if c == agent.pos:
                row.append("A")
            elif c in agent.visited:
                row.append("V")
            elif c in safe:
                row.append("S")
            else:
                row.append("?")
        print(" ".join(f"{ch:>1}" for ch in row))


def print_true_world(world: World) -> None:
    legend = "P=pit  W=wumpus  G=gold  .=empty"
    print("\nTrue world (" + legend + "):")
    for y in range(world.size, 0, -1):
        row = []
        for x in range(1, world.size + 1):
            c = (x, y)
            if c in world.pits:
                row.append("P")
            elif c == world.wumpus:
                row.append("W")
            elif c == world.gold:
                row.append("G")
            else:
                row.append(".")
        print(" ".join(f"{ch:>1}" for ch in row))


# -----------------------------
# Input helpers
# -----------------------------
def ask_int(prompt: str, *, min_value: int = 2) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            n = int(raw)
            if n < min_value:
                print(f"Please enter a number >= {min_value}.")
                continue
            return n
        except ValueError:
            print("Please enter a valid integer.")


def parse_cell(raw: str) -> Optional[Cell]:
    parts = raw.strip().split()
    if len(parts) != 2:
        return None
    try:
        x = int(parts[0])
        y = int(parts[1])
    except ValueError:
        return None
    return (x, y)


def ask_cell(prompt: str, *, size: int, banned: Set[Cell]) -> Cell:
    while True:
        raw = input(prompt).strip()
        cell = parse_cell(raw)
        if cell is None:
            print("Type it like: 2 3")
            continue
        x, y = cell
        if not (1 <= x <= size and 1 <= y <= size):
            print(f"Out of bounds. Use x,y between 1 and {size}.")
            continue
        if cell in banned:
            print("That cell is already used. Pick another one.")
            continue
        return cell


def ask_pits(*, size: int, banned: Set[Cell]) -> Set[Cell]:
    pits: Set[Cell] = set()
    print("\nEnter PIT positions as: x y")
    print("Type 'done' when you're finished.\n")

    while True:
        raw = input("PIT: ").strip()
        if raw.lower() == "done":
            return pits

        cell = parse_cell(raw)
        if cell is None:
            print("Type it like: 2 3 (or 'done')")
            continue

        x, y = cell
        if not (1 <= x <= size and 1 <= y <= size):
            print(f"Out of bounds. Use x,y between 1 and {size}.")
            continue
        if cell in banned or cell in pits:
            print("That cell is already used. Pick another one.")
            continue
        pits.add(cell)


# -----------------------------
# Simulation
# -----------------------------
def run() -> None:
    print("Wumpus World (console)")
    print("I will ask you for a grid size and the positions of pits, the Wumpus, and the gold.")
    print("Coordinates are (x y) with (1, 1) at the bottom-left.\n")

    size = ask_int("Grid size N (for N x N): ", min_value=2)

    # Optional seed so runs can be reproduced
    seed_raw = input("Random seed (press Enter to skip): ").strip()
    if seed_raw:
        try:
            random.seed(int(seed_raw))
        except ValueError:
            # If it's not a number, still allow it as a string seed
            random.seed(seed_raw)

    start = (1, 1)
    banned: Set[Cell] = {start}

    pits = ask_pits(size=size, banned=banned)

    banned_all = set(banned) | pits
    wumpus = ask_cell("WUMPUS (x y): ", size=size, banned=banned_all)

    banned_all.add(wumpus)
    gold = ask_cell("GOLD (x y): ", size=size, banned=banned_all)

    world = World(size=size, pits=pits, wumpus=wumpus, gold=gold)
    agent = Explorer(size=size)

    print("\nStarting...\n")

    max_steps = size * size * 10  # small guard so we don't loop forever
    for step in range(1, max_steps + 1):
        cell = agent.pos
        breeze, stench, glitter = world.percepts(cell)

        print(f"Step {step}: I am at {cell}")
        print(f"Percepts -> Breeze={breeze}, Stench={stench}, Glitter={glitter}")

        if glitter:
            print("\nFound the gold. Done!")
            print_agent_view(agent)
            print_true_world(world)
            return

        agent.update_knowledge(cell, breeze=breeze, stench=stench)
        print_agent_view(agent)

        nxt = agent.choose_next_step()
        if nxt is None:
            print("\nI don't have any safe move left, so I'm stopping.")
            print_true_world(world)
            return

        agent.pos = nxt

        if nxt in world.pits:
            print("\nOops — I fell into a pit.")
            print_true_world(world)
            return

        if nxt == world.wumpus:
            print("\nOops — the Wumpus got me.")
            print_true_world(world)
            return

        print()  # blank line between steps

    print("\nI hit the step limit and stopped (to avoid an endless loop).")
    print_true_world(world)


if __name__ == "__main__":
    run()
