"""Vídeo com esqueleto/eventos e registros JSON + CSV."""

import csv
import json
from fractions import Fraction

import av
import cv2
import numpy as np

# Mapeamento estendido para suportar os 4 eventos do pipeline e fallbacks legados
EVENT_MAP = {
    "take_off": {"label": "Take Off", "color": (0, 255, 0), "definition": "Momento de descolamento do solo"},
    "landing": {"label": "Landing", "color": (0, 0, 255), "definition": "Momento do primeiro contato de pouso"},
    "preparacao_inicio": {"label": "Inicio Preparacao", "color": (255, 255, 0), "definition": "Início da fase de recuo do braço"},
    "preparacao_maxima_aparente": {"label": "Prep. Maxima", "color": (255, 165, 0), "definition": "Pico máximo de extensão/recuo do braço"},
    "braco_frente_inicio": {"label": "Braco a Frente", "color": (0, 255, 255), "definition": "Início do movimento do braço à frente"},
    "saida_vs_apice": {"label": "Saida / Apice", "color": (255, 0, 255), "definition": "Diferenciação entre descolamento (saída) e ápice aéreo"}
}

try:
    from algorithm_rules import EVENTS as LEGACY_EVENTS
    # Mescla mantendo prioridade nos eventos estendidos
    EVENTS = {**LEGACY_EVENTS, **EVENT_MAP}
except ImportError:
    EVENTS = EVENT_MAP

from footskate import CONTACT_NAMES, CONTACT_POINTS, COMMIT, CHECKPOINT_SHA256, MIN_CONF

CONNECTIONS = [(1, 8), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
               (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
               (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]
EVENT_FIELDS = ["event", "status", "frame", "time_s", "frame_range", "time_range_s", "source", "reason"]


def pipeline_status(events, alerts=()):
    """APROVADO se todos os eventos têm frame válido; senão PENDENTE_REVISAO_MANUAL."""
    pending = [f"{event['event']}: {event.get('reason', 'Não detectado')}" for event in events if event.get("frame") is None or event.get("status") == "INCONCLUSIVO"]
    return ("PENDENTE_REVISAO_MANUAL" if pending else "APROVADO"), [*alerts, *pending]


def save_data(video, out, athlete_id, fps, timestamps, probabilities, states, evidence_states, events,
              threshold, ground_info, alerts=()):
    status, notices = pipeline_status(events, alerts)
    
    definitions = {}
    for ev in events:
        ev_name = ev.get("event")
        if ev_name in EVENTS:
            definitions[ev_name] = EVENTS[ev_name].get("definition", "Evento do ataque")
        else:
            definitions[ev_name] = "Evento customizado do pipeline"

    data = {
        "video_id": video.stem, "pipeline_status": status, "alertas": notices,
        "video": video.name, "attacker_id": athlete_id, "frame_index_base": 0,
        "fps_average": fps, "frame_count": len(timestamps), "time_source": "video_pts",
        "time_origin": "first_video_frame",
        "ground_reference": ground_info,
        "definitions": definitions,
        "model": {"name": "Footskate Reducer / ground_detector", "commit": COMMIT,
                  "checkpoint_sha256": CHECKPOINT_SHA256, "contact_threshold": threshold,
                  "pose": "RTMPose-m HALPE26 via rtmlib BodyWithFeet"},
        "events": events,
    }
    (out / "events.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "events.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS, delimiter=";", extrasaction='ignore')
        writer.writeheader()
        writer.writerows(events)
    # Dados suficientes para conferir os estados de contato sem repetir RTMPose.
    with (out / "contacts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["frame", "time_s", *[f"p_{name}" for name in CONTACT_NAMES], "state_evidence", "state"])
        for i, (time, prob, evidence, state) in enumerate(zip(timestamps, probabilities, evidence_states, states)):
            writer.writerow([i, round(float(time), 6), *map(float, prob), int(evidence), int(state)])
    return data


def draw_overlay(image, pose, probabilities, state, frame_index, timestamp, events, threshold=0.5,
                 ground_line=None, foot_states=None, hybrid=False):
    if ground_line is not None and np.isfinite(ground_line).all() and abs(ground_line).max() < 1e6:
        a, b = ground_line.astype(int)
        cv2.line(image, tuple(a), tuple(b), (0, 200, 255), 2)
    valid = pose[:, 2] >= MIN_CONF
    for a, b in CONNECTIONS:
        if valid[a] and valid[b]:
            cv2.line(image, tuple(pose[a, :2].astype(int)), tuple(pose[b, :2].astype(int)), (255, 210, 0), 2)
    for index, (x, y, _) in enumerate(pose):
        if valid[index]:
            cv2.circle(image, (int(x), int(y)), 3, (255, 210, 0), -1)
    for index, joint in enumerate(CONTACT_POINTS):
        if valid[joint]:
            color = (0, 230, 0) if probabilities[index] >= threshold else (0, 60, 255)
            if foot_states is not None:
                color = {1: (0, 230, 0), 0: (0, 60, 255), -1: (150, 150, 150)}[int(foot_states[index])]
            if state == -1:
                color = (150, 150, 150)
            cv2.circle(image, tuple(pose[joint, :2].astype(int)), 6, color, 2)

    # Faixa separada: legível também nos clipes verticais pequenos.
    panel_width = max(520, image.shape[1])
    canvas = np.zeros((image.shape[0] + 132, panel_width, 3), dtype=np.uint8)
    x_offset = (panel_width - image.shape[1]) // 2
    canvas[132:, x_offset:x_offset + image.shape[1]] = image
    state_text = {1: "APOIO", 0: "SEM APOIO", -1: "SEM EVIDENCIA"}[int(state)]
    if state == -1 and any(
            event.get("status") == "estimated" and event.get("frame_range")
            and event["frame_range"][0] <= frame_index <= event["frame_range"][1]
            for event in events):
        state_text = "TRANSICAO ESTIMADA"
    
    lines = [f"Frame {frame_index} | {timestamp:.3f} s | {'HIBRIDO' if hybrid else 'FOOTSKATE'}: {state_text}"]
    for event in events:
        ev_key = event["event"]
        label = EVENTS.get(ev_key, {}).get("label", ev_key)
        
        if event.get("frame") is None or event.get("status") == "INCONCLUSIVO":
            reason = event.get("reason") or ""
            suffix = " / SEM POUSO OBSERVADO" if "após a fase" in reason else ""
            lines.append(f"{label}: INCONCLUSIVO{suffix}")
        else:
            marker = " >>>" if event["frame"] == frame_index else ""
            bounds = event.get("frame_range")
            estimated = "~" if event.get("status") == "estimated" else ""
            interval = f" [{bounds[0]}-{bounds[1]}]" if bounds and bounds[0] != bounds[1] else ""
            time_val = event.get('time_s') or 0.0
            lines.append(f"{label}: {estimated}{event['frame']}{interval} | {time_val:.3f} s{marker}")

    for row, text in enumerate(lines):
        highlight = row > 0 and events[row - 1].get("frame") == frame_index
        color = (0, 240, 255) if highlight else (240, 240, 240)
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
        font_scale = 0.62 * min(1.0, (panel_width - 24) / max(text_width, 1))
        cv2.putText(canvas, text, (12, 30 + row * 36), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
        
    if any(event.get("frame") == frame_index for event in events):
        cv2.rectangle(canvas, (2, 2), (panel_width - 3, canvas.shape[0] - 3), (0, 240, 255), 4)
        
    # H264 yuv420p precisa de dimensões pares.
    return cv2.copyMakeBorder(canvas, 0, canvas.shape[0] % 2, 0, panel_width % 2, cv2.BORDER_CONSTANT)


def save_video(video, output, poses, probabilities, states, timestamps, events, threshold=0.5,
               ground_lines=None, foot_states=None, hybrid=False):
    with av.open(str(video)) as source, av.open(str(output), mode="w", options={"movflags": "+faststart"}) as destination:
        original = source.streams.video[0]
        stream = destination.add_stream("libx264", rate=original.average_rate or Fraction(30))
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 90000)
        stream.options = {"crf": "20", "preset": "veryfast"}
        count = 0
        for i, frame in enumerate(source.decode(video=0)):
            if i >= len(poses):
                raise ValueError("O vídeo mudou durante o processamento.")
            image = draw_overlay(frame.to_ndarray(format="bgr24"), poses[i], probabilities[i],
                                 states[i], i, timestamps[i], events, threshold,
                                 ground_lines[i] if ground_lines is not None else None,
                                 foot_states[i] if foot_states is not None else None, hybrid)
            if i == 0:
                stream.width, stream.height = image.shape[1], image.shape[0]
            encoded_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            encoded_frame.pts = round(timestamps[i] * 90000)
            encoded_frame.time_base = Fraction(1, 90000)
            for packet in stream.encode(encoded_frame):
                destination.mux(packet)
            count += 1
        for packet in stream.encode():
            destination.mux(packet)
        if count != len(poses):
            raise ValueError("Quantidade de frames diferente entre análise e exportação.")