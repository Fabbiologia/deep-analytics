import gradio as gr
from main import run_analysis # We will create this function in main.py

def chat_interface(message, history):
    return run_analysis(message)

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
