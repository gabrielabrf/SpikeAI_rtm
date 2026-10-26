"""Somente saída do chão e aterrissagem, derivadas do contato dos pés."""

import numpy as np

EVENTS = {
    "take_off": {"label": "SAIDA DO CHAO", "definition": "Último frame com apoio antes da fase aérea."},
    "landing": {"label": "ATERRISSAGEM", "definition": "Primeiro frame com apoio após a fase aérea."},
}


def runs(values):
    boundaries = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
    return [(int(a), int(b), int(values[a]))
            for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]


def detect_events(states, timestamps, min_flight_s=0.12, min_support_s=0.06,
                  max_gap_s=0.15, max_uncertainty_s=0.15, source="footskate"):
    states = np.asarray(states, dtype=np.int8).copy()
    times = np.asarray(timestamps, dtype=float)
    if len(states) != len(times) or not len(times):
        raise ValueError("Cada frame precisa de um estado e um tempo.")
    if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("Os tempos dos frames devem ser finitos e crescentes.")
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30

    def duration(a, b):
        return float(times[b] - times[a]) if b < len(times) else float(times[-1] - times[a] + dt)

    # Um apoio isolado muito breve não confirma uma transição. Não vira voo.
    for a, b, state in runs(states):
        if state == 1 and duration(a, b) + 1e-9 < min_support_s:
            states[a:b] = -1

    # Uma falha curta dentro do mesmo estado não representa outro salto.
    for a, b, state in runs(states):
        if (state == -1 and a > 0 and b < len(states)
                and states[a - 1] == states[b] and states[b] != -1
                and duration(a, b) <= max_gap_s + 1e-9):
            states[a:b] = states[b]

    flights = [(a, b) for a, b, state in runs(states)
               if state == 0 and duration(a, b) + 1e-9 >= min_flight_s]

    def event(name, frame=None, reason=None, bounds=None):
        status = "inconclusive" if frame is None else ("estimated" if source != "footskate" else "detected")
        if frame is not None and bounds is not None and bounds[0] != bounds[1]:
            status = "estimated"
        return {"event": name, "status": status,
                "frame": frame, "time_s": round(float(times[frame]), 6) if frame is not None else None,
                "frame_range": bounds,
                "time_range_s": [round(float(times[i]), 6) for i in bounds] if bounds else None,
                "source": source, "reason": reason}

    if not flights:
        reason = "Nenhuma fase aérea com evidência suficiente foi identificada."
        return [event("take_off", reason=reason), event("landing", reason=reason)], states

    # O recorte deve conter um único ataque. A maior fase aérea é a candidata.
    flights.sort(key=lambda ab: duration(*ab), reverse=True)
    if len(flights) > 1 and duration(*flights[1]) >= 0.7 * duration(*flights[0]):
        reason = "Mais de uma fase aérea comparável; não foi possível escolher um único salto."
        return [event("take_off", reason=reason), event("landing", reason=reason)], states

    def transition(name, edge, step):
        support = edge
        while 0 <= support < len(states) and states[support] == -1:
            support += step
        if not 0 <= support < len(states) or states[support] != 1:
            position = "antes da" if step < 0 else "após a"
            return event(name, reason=f"Não há apoio observado {position} fase aérea.")
        bounds = sorted([edge, support])
        if times[bounds[1]] - times[bounds[0]] > max_uncertainty_s + 1e-9:
            return event(name, bounds=bounds, reason="Intervalo de transição amplo demais para localizar o frame.")
        return event(name, int(round(sum(bounds) / 2)), bounds=bounds)

    start, end = flights[0]
    return [transition("take_off", start - 1, -1), transition("landing", end, 1)], states
