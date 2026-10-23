"""Escolha inicial do atacante e associação conservadora entre frames."""

import cv2
import numpy as np


def iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    area_a = np.maximum(0, a[2:] - a[:2]).prod()
    area_b = np.maximum(0, b[2:] - b[:2]).prod()
    return float(intersection / max(area_a + area_b - intersection, 1))


def choose_attacker(frame, boxes, athlete_id, interactive, selection_path):
    if len(boxes) == 0:
        raise ValueError("Nenhum atleta encontrado no primeiro frame. Use um clipe que já mostre o atacante.")
    if athlete_id is not None:
        if not 0 <= athlete_id < len(boxes):
            raise ValueError(f"Índice de atleta inválido: escolha entre 0 e {len(boxes) - 1}.")
        return athlete_id
    if len(boxes) == 1:
        return 0

    preview = frame.copy()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(preview, str(index), (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)
    cv2.imwrite(str(selection_path), preview)
    if not interactive:
        raise ValueError(f"Há vários atletas. Veja {selection_path} e repita com --atleta N, ou use --selecionar.")
    scale = min(1.0, 1200 / frame.shape[1], 800 / frame.shape[0])
    window = "Marque o atacante e pressione ENTER"
    display = cv2.resize(preview, None, fx=scale, fy=scale)
    roi = cv2.selectROI(window, display, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        raise ValueError("Seleção cancelada.")
    x, y, w, h = np.asarray(roi) / scale
    overlaps = [iou(np.array([x, y, x + w, y + h]), box) for box in boxes]
    if max(overlaps) < 0.1:
        raise ValueError("A seleção não corresponde a um atleta detectado.")
    return int(np.argmax(overlaps))


class AttackerTracker:
    def __init__(self, box, max_missing=10):
        self.box = np.asarray(box, dtype=float)
        self.velocity = np.zeros(2)
        self.missing = 0
        self.max_missing = max_missing

    def update(self, boxes):
        if self.missing >= self.max_missing:
            return None  # Não troca de pessoa depois de perder a trajetória.
        predicted = self.box + np.tile(self.velocity * (self.missing + 1), 2)
        diagonal = max(float(np.linalg.norm(self.box[2:] - self.box[:2])), 1)
        ranked = []
        for index, box in enumerate(boxes):
            distance = np.linalg.norm((box[:2] + box[2:] - predicted[:2] - predicted[2:]) / 2) / diagonal
            overlap = iou(predicted, box)
            ratio = np.prod(np.maximum(1, box[2:] - box[:2])) / np.prod(np.maximum(1, self.box[2:] - self.box[:2]))
            if distance < 0.6 and 0.4 <= ratio <= 2.5 and (overlap > 0.05 or distance < 0.2):
                ranked.append((overlap - 0.4 * distance, index))
        ranked.sort(reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.15):
            self.missing += 1
            return None
        selected = ranked[0][1]
        new_box = np.asarray(boxes[selected], dtype=float)
        displacement = (new_box[:2] + new_box[2:] - self.box[:2] - self.box[2:]) / 2
        self.velocity = 0.5 * self.velocity + 0.5 * displacement / (self.missing + 1)
        self.box = new_box
        self.missing = 0
        return selected
