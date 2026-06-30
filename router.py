ROUTING: dict[str, tuple[str, str]] = {
    "idea_generation": ("big_pickle", "big-pickle"),
    "platform_classification": ("big_pickle", "big-pickle"),
    "final_copy": ("opencode_zen", "big-pickle"),
    "image_generation": ("pollinations", "pollinations"),
}

VALID_PROVIDERS: set[str] = {"big_pickle", "claude", "pollinations", "opencode_zen"}


def route_task(task_type: str) -> tuple[str, str]:
    """Return (provider_name, model_id) for a given task type.

    Args:
        task_type: One of "idea_generation", "platform_classification",
                   "final_copy", "image_generation".

    Returns:
        (provider_name, model_id) tuple.

    Raises:
        ValueError: If task_type is not recognised.
    """
    entry = ROUTING.get(task_type)
    if entry is None:
        raise ValueError(f"Unknown task type: {task_type}")
    return entry


def validate_routing() -> list[str]:
    """Check that every task type maps to a valid provider.

    Returns:
        List of error messages (empty if everything is valid).
    """
    errors: list[str] = []
    for task_type, (provider, model) in ROUTING.items():
        if provider not in VALID_PROVIDERS:
            errors.append(
                f"Task '{task_type}' has unknown provider '{provider}'. "
                f"Valid providers: {sorted(VALID_PROVIDERS)}"
            )
    return errors
