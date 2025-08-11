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
        ["What is the average biomass for the species Pocillopora verrucosa?"],
        ["Compare the coral cover between La Paz and Cabo Pulmo."],
        ["Generate a chart showing the temporal trend of coral productivity."]
    ]
)

if __name__ == "__main__":
    iface.launch()
