"""Validate template geometry, environment normalization, and prediction-head dimensions."""

import math


def resolve_motion_prior_config(spec, *, horizon, state_dim):
    defaults = dict(
        horizon=horizon,
        state_dim=state_dim,
        components=4,
        head_settings=None,
        environments=None,
        length_scale=None,
        action_min=None,
        action_max=None,
        template_distance=None,
        template_yaw=None,
        template_z=None,
        noise_cholesky=None,
        logit_shift=None,
        noise_scale=0.25,
        max_normalized_length=8.0,
        max_abs_yaw=2 * math.pi,
    )
    defaults.update(
        {key: value for key, value in (spec or {}).items() if key in defaults}
    )
    c = defaults
    if c["horizon"] != horizon or c["state_dim"] != state_dim:
        raise ValueError("Motion prior horizon/state_dim mismatch")
    for key in ("horizon", "state_dim", "components"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"Motion prior {key} must be a positive integer")

    def numeric(value):
        return type(value) in (int, float) and math.isfinite(value)

    def vector(name, size):
        values = c[name]
        if (
            not isinstance(values, list)
            or len(values) != size
            or not all(map(numeric, values))
        ):
            raise ValueError(f"Invalid template prior {name}")

    def matrix(name, rows, width):
        values = c[name]
        if (
            not isinstance(values, list)
            or len(values) != rows
            or any(
                not isinstance(row, list)
                or len(row) != width
                or not all(map(numeric, row))
                for row in values
            )
        ):
            raise ValueError(f"Invalid template prior {name}")

    names = c["environments"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != len(names)
        or c["components"] != 4
    ):
        raise ValueError("Template prior requires unique environments and four classes")
    settings = c["head_settings"]
    allowed = {
        "hidden_size",
        "state_dim",
        "history_steps",
        "action_dim",
        "history_width",
        "widths",
        "arm",
        "history_dropout",
    }
    if (
        not isinstance(settings, dict)
        or set(settings) - allowed
        or settings.get("arm") != "token"
        or settings.get("state_dim") != state_dim
        or settings.get("action_dim") != 4
        or type(settings.get("hidden_size")) is not int
        or settings["hidden_size"] < 1
        or type(settings.get("history_steps")) is not int
        or settings["history_steps"] < 1
        or settings.get("history_dropout", 0) != 0
    ):
        raise ValueError(
            "Template prior requires a token+state head with matching dimensions"
        )
    widths = settings.get("widths", [512, 256])
    if (
        not isinstance(widths, (list, tuple))
        or len(widths) != 2
        or any(type(w) is not int or w < 1 for w in widths)
        or type(settings.get("history_width", 128)) is not int
        or settings.get("history_width", 128) < 1
    ):
        raise ValueError("Invalid template head widths")
    vector("length_scale", len(names))
    vector("logit_shift", 4)
    matrix("action_min", len(names), 3)
    matrix("action_max", len(names), 3)
    for key in ("template_distance", "template_yaw", "template_z"):
        matrix(key, 4, horizon)
    if any(s <= 0 for s in c["length_scale"]) or any(
        hi < lo
        for low, high in zip(c["action_min"], c["action_max"])
        for lo, hi in zip(low, high)
    ):
        raise ValueError("Invalid template environment normalization")
    if any(
        any(v <= 0 for v in row) or abs(sum(row) - 1) > 1e-5
        for row in c["template_distance"]
    ):
        raise ValueError(
            "Template horizontal-distance profiles must be positive and sum to one"
        )
    if any(abs(v) >= math.pi / 2 for row in c["template_yaw"] for v in row):
        raise ValueError("Template yaw must preserve the forward-body convention")
    matrices = c["noise_cholesky"]
    d = horizon * 3
    if not isinstance(matrices, list) or len(matrices) != 4:
        raise ValueError("noise_cholesky requires four matrices")
    for covariance in matrices:
        if not isinstance(covariance, list) or len(covariance) != d:
            raise ValueError("noise_cholesky dimension mismatch")
        for i, row in enumerate(covariance):
            if not isinstance(row, list) or len(row) != d:
                raise ValueError("noise_cholesky dimension mismatch")
            for j, value in enumerate(row):
                if (
                    not numeric(value)
                    or (j > i and value != 0)
                    or (j == i and value <= 0)
                ):
                    raise ValueError(
                        "noise_cholesky must be finite lower triangular with positive diagonal"
                    )
    for key in ("noise_scale", "max_normalized_length", "max_abs_yaw"):
        if not numeric(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid template prior {key}")
    return defaults
