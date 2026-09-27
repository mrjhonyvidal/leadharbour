"""Optional Vertex draft interface. It cannot send outreach."""

from __future__ import annotations


def draft_for_review(context: str, *, consent_checked: bool, project: str,
                     location: str, model: str = "gemini-2.5-flash", client=None) -> dict:
    if not consent_checked:
        raise ValueError("A human must check consent before drafting")
    if not context.strip() or len(context) > 2000:
        raise ValueError("Provide a short, reviewed context of at most 2000 characters")
    if client is None:
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("Install optional agent support: pip install -e '.[agent]'") from error
        client = genai.Client(vertexai=True, project=project, location=location)
    response = client.models.generate_content(
        model=model,
        contents=("Draft a short, respectful outreach note for human review only. "
                  "Use only these reviewed facts. Do not invent a claim or ask for sensitive data. "
                  "No message will be sent.\n\n" + context),
    )
    return {"draft": response.text or "", "sent": False, "review_required": True}
