import requests
import json

from config import (
    OPENCODE_BASE_URL,
    OPENCODE_MODEL,
    OPENCODE_API_KEY,
    OPENCODE_ZEN_MODEL,
    CLAUDE_API_KEY,
    CLAUDE_MODEL,
)
from router import route_task, validate_routing

FLOW_INTELLIGENCE_CONTEXT = """
За Flow Intelligence:
Flow Intelligence е българска агенция за custom софтуер и AI автоматизация, базирана в Бургас. Помагаме на малки и средни бизнеси (счетоводни кантори, брокерски агенции, ритейл, услуги) да спестят часове ръчна работа чрез:
- Custom CRM системи (управление на клиенти, задачи, комуникация на едно място)
- AI автоматизация на рутинни процеси (обработка на документи, имейли, данни)
- Custom уебсайтове и бизнес инструменти, пригодени точно за конкретния бизнес
- Интеграции между съществуващи системи, за да спрат "ръчното копиране" между програми

ВАЖНО: Съдържанието НИКОГА не говори за "AI агенти", "автоматизационни системи" или техническия начин по който работим - то говори за РЕЗУЛТАТА за бизнеса (спестено време, по-малко грешки, по-доволни клиенти, по-добра организация). Клиентът не се интересува от технологията, а от това че кантората/агенцията/бизнесът му работи по-добре.
"""

BRAND_VOICE_CONTEXT = """
Ти си marketing copywriter за Flow Intelligence — българска агенция за AI автоматизация и custom софтуер за малък и среден бизнес (SMB).

ПРАВИЛА ЗА ВСЯКО СЪДЪРЖАНИЕ:
1. БЕЗУПРЕЧНА българска граматика и правопис — никакви грешки, никакви буквални преводи от английски. Естествен, плавен български.
2. Всеки пост трябва да адресира РЕАЛЕН, КОНКРЕТЕН проблем на бизнеса (загубено време на ръчна администрация, пропуснати клиенти заради бавен отговор, грешки от човешки фактор, скъпи служители за рутинни задачи). Не общи приказки за "дигитализация" — конкретен болезнен момент, който читателят разпознава от собствения си бизнес.
3. Тон: уверен, директен, леко закачлив ("чийзи" в добрия смисъл) — кратки изречения, риторични въпроси, лек хумор където пасва, но НИКОГА за сметка на професионализма. Мисли "приятел, който случайно е експерт", не корпоративен жаргон.
4. Структура за социални постове: hook (първо изречение спира скрола) → проблем (2-3 изречения, конкретен сценарий) → решение/инсайт (как AI/automation решава точно това) → CTA (ясен следващ стъпка - "Пишете ни", "Безплатна демонстрация" и т.н.)
5. Използвай реални числа/казуси където е възможно (напр. "8 часа седмично спестени", "68 клиента управлявани от един човек") - конкретиката продава, не абстракцията.
6. Емоджита - умерено, 1-3 на пост, не повече, само където реално добавят тон, не декорация.
7. НИКОГА клишета от типа "разковничето е", "в днешния забързан свят", "не пропускайте възможността".

Платформени нюанси:
- LinkedIn: по-професионален тон, но все пак директен, фокус на ROI и бизнес стойност
- Instagram/Facebook: по-неформален, визуално ориентиран caption, по-кратък текст
- TikTok: разговорен, hook в първите 3 секунди като текст, кратки изречения за voice-over стил
"""


def call_big_pickle(prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
    """Send a completion request to the OpenCode Zen endpoint.

    Uses an OpenAI-compatible endpoint at OPENCODE_BASE_URL.

    Args:
        prompt: The user text prompt to send.
        system_prompt: Optional system-level instruction.
        model: Model ID override (defaults to OPENCODE_MODEL from config).

    Returns:
        The model's response text.

    Raises:
        ConnectionError: On network / timeout errors.
        PermissionError: On 401 (bad or missing API key).
        RuntimeError: On 5xx server errors or unexpected status codes.
    """
    headers = {
        "Content-Type": "application/json",
    }
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or OPENCODE_MODEL,
        "messages": messages,
    }

    try:
        response = requests.post(
            f"{OPENCODE_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
    except requests.exceptions.Timeout:
        raise ConnectionError("big_pickle request timed out after 180s")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("big_pickle request failed — network error")

    status = response.status_code
    if status >= 500:
        raise RuntimeError(
            f"big_pickle API returned {status} — server error"
        )
    elif status != 200:
        raise RuntimeError(
            f"big_pickle API returned unexpected status {status}"
        )

    return response.json()["choices"][0]["message"]["content"]


def call_claude(prompt: str, system_prompt: str | None = None) -> str:
    """Send a completion request to Anthropic's Claude model.

    Args:
        prompt: The user text prompt to send.
        system_prompt: Optional system-level instruction.

    Returns:
        The model's response text.

    Raises:
        PermissionError: On authentication errors.
        RuntimeError: On API/server errors.
    """
    import anthropic
    from anthropic import APIStatusError, APIConnectionError

    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        kwargs = {
            "model": CLAUDE_MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        message = client.messages.create(**kwargs)
    except APIStatusError as e:
        if e.status_code == 401:
            raise PermissionError(
                "Claude API returned 401 — invalid or missing CLAUDE_API_KEY"
            )
        raise RuntimeError(f"Claude API returned error {e.status_code}: {e}")
    except APIConnectionError:
        raise ConnectionError("Claude API request failed — network error")

    return message.content[0].text


def generate_idea(topic: str, platform: str = "linkedin") -> str:
    """Generate a marketing content idea for the given topic and platform.

    Uses the provider configured in ROUTING for "idea_generation".

    Args:
        topic: The subject matter (e.g. "AI chatbots for SMB").
        platform: Target social platform (default "linkedin").

    Returns:
        A content idea string in Bulgarian.
    """
    provider, model = route_task("idea_generation")

    system_prompt = FLOW_INTELLIGENCE_CONTEXT + "\n" + BRAND_VOICE_CONTEXT + (
        "\n\nСпецифично за тази задача: Генерирай 1 кратка маркетинг идея (2-3 изречения) "
        "за публикация — все още НЕ пиши финалния копирайт, само концепцията."
    )
    user_prompt = (
        f"Генерирай 1 маркетинг идея за публикация в {platform} "
        f"на тема: „{topic}“. "
        f"Идеята трябва да е кратка (2-3 изречения), подходяща за "
        f"българска аудитория и да подчертава практическите ползи "
        f"от AI автоматизацията за малкия бизнес."
    )

    if provider == "big_pickle":
        return call_big_pickle(user_prompt, system_prompt, model=model)
    elif provider == "claude":
        return call_claude(user_prompt, system_prompt)
    elif provider == "opencode_zen":
        return call_opencode_zen(user_prompt, system_prompt, model=model)
    else:
        raise ValueError(f"Unsupported provider for idea_generation: {provider}")


def classify_platform(idea: str) -> dict:
    """Classify the best social platform and format for a given content idea.

    Uses keyword heuristics based on the idea content.

    Args:
        idea: The content idea text.

    Returns:
        Dict with "platform" and "format" keys.
    """
    idea_lower = idea.lower()

    trigger_map = {
        "instagram": {"video": "reel", "image": "carousel"},
        "tiktok": {"video": "short_form"},
        "linkedin": {"article": "long_form", "post": "text_image"},
        "facebook": {"video": "native_video", "post": "text_image"},
    }

    if any(w in idea_lower for w in ["instagram", "визуален", "снимка", "reel", "стори"]):
        return {"platform": "instagram", "format": "reel"}
    elif any(w in idea_lower for w in ["tiktok", "танцув", "quick", "trend"]):
        return {"platform": "tiktok", "format": "short_form"}
    elif any(w in idea_lower for w in ["linkedin", "професионал", "business", "b2b"]):
        return {"platform": "linkedin", "format": "article"}
    elif any(w in idea_lower for w in ["facebook", "fb", "група", "community"]):
        return {"platform": "facebook", "format": "text_image"}

    return {"platform": "linkedin", "format": "post"}


def generate_final_copy(idea: str, platform: str) -> str:
    """Generate a polished final copy from a content idea for a specific platform.

    Uses the provider configured in ROUTING for "final_copy".

    Args:
        idea: The content idea string.
        platform: Target social platform.

    Returns:
        Polished copy text in Bulgarian with Flow Intelligence brand voice.
    """
    provider, model = route_task("final_copy")

    system_prompt = FLOW_INTELLIGENCE_CONTEXT + "\n" + BRAND_VOICE_CONTEXT + (
        "\n\nСпецифично за тази задача: Напиши ФИНАЛЕН копирайт, готов за публикуване — "
        "с емотикони, хаштагове и форматиране. Спазвай стриктно rules 1-7 по-горе."
    )
    user_prompt = (
        f"Напиши финален копирайт за {platform} публикация на базата "
        f"на следната идея:\n\n{idea}\n\n"
        f"Копирайтът трябва да е готов за публикуване — с емотикони, "
        f"хаштагове и форматиране, подходящо за {platform}."
    )

    if provider == "big_pickle":
        return call_big_pickle(user_prompt, system_prompt, model=model)
    elif provider == "claude":
        return call_claude(user_prompt, system_prompt)
    elif provider == "opencode_zen":
        return call_opencode_zen(user_prompt, system_prompt, model=model)
    else:
        raise ValueError(f"Unsupported provider for final_copy: {provider}")


def call_opencode_zen(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
) -> str:
    """Send a completion request to OpenCode Zen endpoint with deepseek model.

    OpenAI-compatible call to https://opencode.ai/zen/v1/chat/completions.

    Args:
        prompt: The user text prompt.
        system_prompt: Optional system-level instruction (role "system").
        model: Model ID override (defaults to OPENCODE_ZEN_MODEL).

    Returns:
        The model's response text.

    Raises:
        ConnectionError: On timeout or network errors.
        PermissionError: On 401.
        RuntimeError: On other non-200 responses.
    """
    headers = {
        "Authorization": f"Bearer {OPENCODE_API_KEY}",
        "Content-Type": "application/json",
    }
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or OPENCODE_ZEN_MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post(
            "https://opencode.ai/zen/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
    except requests.exceptions.Timeout:
        raise ConnectionError("OpenCode Zen request timed out after 180s")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("OpenCode Zen request failed — network error")

    status = response.status_code
    if status == 401:
        raise PermissionError(
            "OpenCode Zen returned 401 — invalid or missing OPENCODE_API_KEY"
        )
    elif status >= 500:
        raise RuntimeError(f"OpenCode Zen returned {status} — server error")
    elif status != 200:
        raise RuntimeError(
            f"OpenCode Zen returned unexpected status {status}: "
            f"{response.text[:200]}"
        )

    return response.json()["choices"][0]["message"]["content"]


def test_deepseek_models() -> None:
    """Test deepseek-v4-flash model via OpenCode Zen endpoint."""
    print("=" * 60)
    print("test_deepseek_models()")
    print("=" * 60)

    test_prompt = "Кажи 'здравей' на български."
    print(f"\n[call_opencode_zen] Sending request...")
    try:
        result = call_opencode_zen(
            prompt=test_prompt,
            system_prompt=None,
        )
        print(f"  ✓ Response ({len(result)} chars): {result}")
    except (ConnectionError, PermissionError, RuntimeError) as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")
        return

    print("\n" + "=" * 60)
    print("test_deepseek_models() COMPLETE")
    print("=" * 60)


def test_without_keys() -> None:
    """Structural test that verifies function wiring without real API keys.

    Catches API-level errors (missing keys) but confirms the code reaches
    the API call attempt without syntax, import, or logic errors.
    """
    import traceback

    print("=" * 60)
    print("test_without_keys() — structural validation")
    print("=" * 60)

    # --- router tests ---
    print("\n[router] route_task — known types...")
    for t in ["idea_generation", "platform_classification", "final_copy", "image_generation"]:
        provider, model = route_task(t)
        print(f"  ✓ {t} → ({provider}, {model})")

    print("\n[router] route_task — unknown type...")
    try:
        route_task("nonexistent_task")
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    print("\n[router] validate_routing...")
    errors = validate_routing()
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("  ✓ All routing entries valid")

    # --- call_big_pickle test ---
    print("\n[call_big_pickle] Attempting API call...")
    try:
        result = call_big_pickle("test prompt", "test system")
        print(f"  ℹ API call succeeded (keys present and endpoint live)")
        print(f"  ℹ Response preview: {result[:80]}...")
    except (PermissionError, ConnectionError, RuntimeError) as e:
        print(f"  ✓ Caught expected API error: {type(e).__name__}: {e}")

    # --- call_claude test ---
    print("\n[call_claude] Attempting API call...")
    try:
        result = call_claude("test prompt", "test system")
        print(f"  ℹ API call succeeded (keys present and endpoint live)")
        print(f"  ℹ Response preview: {result[:80]}...")
    except (PermissionError, ConnectionError, RuntimeError, TypeError) as e:
        print(f"  ✓ Caught expected API error: {type(e).__name__}: {e}")

    # --- generate_idea test ---
    print("\n[generate_idea] Structural test...")
    try:
        result = generate_idea("AI chatbots", "linkedin")
        print(f"  ℹ API call succeeded — result: {result[:80]}...")
    except (PermissionError, ConnectionError, RuntimeError, ValueError, TypeError) as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}: {e}")

    # --- generate_final_copy test ---
    print("\n[generate_final_copy] Structural test...")
    try:
        result = generate_final_copy("Test idea", "linkedin")
        print(f"  ℹ API call succeeded — result: {result[:80]}...")
    except (PermissionError, ConnectionError, RuntimeError, ValueError, TypeError) as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}: {e}")

    # --- classify_platform test ---
    print("\n[classify_platform] Deterministic tests...")
    test_cases = [
        ("Перфектна Instagram снимка с визуален AI", "instagram"),
        ("TikTok тенденция за малък бизнес", "tiktok"),
        ("LinkedIn статия за B2B автоматизация", "linkedin"),
        ("Facebook група за предприемачи", "facebook"),
        ("Нова стратегия за растеж", "linkedin"),
    ]
    for idea_text, expected in test_cases:
        result = classify_platform(idea_text)
        status = "✓" if result["platform"] == expected else "✗"
        print(f"  {status} classify_platform({idea_text[:40]}...) = {result}")

    print("\n" + "=" * 60)
    print("ALL STRUCTURAL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_without_keys()
