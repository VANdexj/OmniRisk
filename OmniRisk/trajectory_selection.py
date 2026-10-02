import numpy as np


def add_endpoint_continuity_cost(scores, endpoint_positions,
                                 previous_action_id, continuity_weight):
    """Penalize candidates by normalized distance from the previous endpoint."""
    scores = np.asarray(scores).reshape(-1)
    if continuity_weight <= 0.0 or previous_action_id is None:
        return scores

    previous_action_id = int(previous_action_id)
    if previous_action_id < 0 or previous_action_id >= scores.size:
        return scores

    endpoint_positions = np.asarray(endpoint_positions)
    distances = np.linalg.norm(
        endpoint_positions - endpoint_positions[previous_action_id], axis=1)
    max_distance = np.max(distances)
    if max_distance == 0.0:
        return scores
    return scores + float(continuity_weight) * distances / max_distance


def select_action_with_hysteresis(scores, previous_action_id, switch_margin):
    """Keep the previous action unless another action is clearly better."""
    scores = np.asarray(scores).reshape(-1)
    best_action_id = int(np.argmin(scores))

    if switch_margin <= 0.0 or previous_action_id is None:
        return best_action_id
    previous_action_id = int(previous_action_id)
    if previous_action_id < 0 or previous_action_id >= scores.size:
        return best_action_id
    if scores[best_action_id] + switch_margin < scores[previous_action_id]:
        return best_action_id
    return previous_action_id
