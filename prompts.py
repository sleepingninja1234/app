from langchain_core.prompts import ChatPromptTemplate

TONE_INSTRUCTIONS = {
    "engaging": "conversational, curiosity-driven, light energy, no emojis/hashtags",
    "authoritative": "confident, insight-led, precise, no emojis/hashtags",
    "educational": "clear, example-led, mini-structure, no emojis/hashtags",
}

post_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a LinkedIn content creator for B2B audiences. Write a single LinkedIn post only. No title, no hashtags, no emojis. Structure: strong hook (1–2 lines), tight body, and a soft CTA at the end. Tone: {tone_instruction}. Target length: {target_words} words.",
        ),
        (
            "human",
            "Topic: {topic}\n\nConstraints:\n- Keep to the tone and word target.\n- Avoid bullet lists unless clearly useful.\n- Optimize scannability with short paragraphs.\n- Assume the audience is founders/marketers.",
        ),
    ]
)

