import gradio as gr
from main import run_analysis # We will create this function in main.py

def chat_interface(message, history):
    # Include a limited window of recent history to preserve context without exceeding token limits
    # Each history item is a [user, assistant] pair from Gradio
    turns_to_include = 6  # keep small to respect token/TPM constraints
    history = history or []
    recent = history[-turns_to_include:]

    context_lines = []
    for user_msg, assistant_msg in recent:
        if user_msg:
            context_lines.append(f"User: {user_msg}")
        if assistant_msg:
            context_lines.append(f"Assistant: {assistant_msg}")

    context_block = "\n".join(context_lines)
    if context_block:
        composed = (
            "Conversation so far:\n" + context_block +
            "\n\nFollow-up question: " + message
        )
    else:
        composed = message

    return run_analysis(composed)

iface = gr.ChatInterface(
    chat_interface,
    title="Ecological Monitoring Agent",
    description="Ask me questions about your ecological database.",
    theme="soft",
    examples=[
        ["Show me the top 10 fish species by abundance"],
        ["Compare fish biomass among regions"],
        ["Calculate invertebrate diversity over time in the La Paz region"]
    ]
)

if __name__ == "__main__":
    iface.launch()
