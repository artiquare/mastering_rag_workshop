# Import Gradio for UI
import gradio as gr
from rag_app.chains.reranking_qa_chain import qar

if __name__ == "__main__":
   
    # Function to add a new input to the chat history
    def add_text(history, text):
        # Append the new text to the history with a placeholder for the response
        history = history + [(text, None)]
        return history, ""

    # Function representing the bot's response mechanism
    def bot(history):
        # Obtain the response from the 'infer' function using the latest input
        response = infer(history[-1][0], history)
        sources = [doc.metadata.get("Section") for doc in response['source_documents']]
        src_list = '\n'.join(sources)
        print_this = response['result']+"\n\n\n Sources: \n\n\n"+src_list
        history[-1][1] = print_this 
        return history

    # Function to infer the response using the RAG model
    def infer(question, history):
            
        # Use the question and history to query the RAG model
        result = qar({"query": question, "history": history, "question": question})
        return result
        

    # CSS styling for the Gradio interface
    css = """
    #col-container {max-width: 1200px; margin-left: auto; margin-right: auto;}
    """

    # HTML content for the Gradio interface title
    title = """
    <div style="text-align:left;">
        <p>Hello, I BotTina 2.0, your intelligent AI assistant. <br />
    </div>
    """
    head_style = """
    <style>
    @media (min-width: 1536px)
    {
        .gradio-container {
            min-width: var(--size-full) !important;
        }
    }
    </style>
    """

    # Building the Gradio interface
    with gr.Blocks(theme=gr.themes.Soft(), title="RFP AI Analyzer 🤵🏻‍♂️", head=head_style) as demo:
        with gr.Column(elem_id="col-container"):
            gr.HTML()  # Add the HTML title to the interface
            chatbot = gr.Chatbot([], elem_id="chatbot",
                                        label="RFP Reranker Analyzer",
                                        bubble_full_width=False,
                                        avatar_images=(None, "https://dacodi-production.s3.amazonaws.com/store/87bc00b6727589462954f2e3ff6f531c.png"),
                                        height=600,)  # Initialize the chatbot component

            # Create a row for the question input
            with gr.Row():
                question = gr.Textbox(label="Question", show_label=False, placeholder="Type your question and hit Enter ", scale=4)
                send_btn = gr.Button(value="Send", variant="primary", scale=0)

            with gr.Row():
                clear = gr.Button("Clear")  # Add a button to clear the chat

        # Define the action when the question is submitted
        question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
            bot, chatbot, chatbot)
        send_btn.click(add_text, [chatbot, question], [chatbot, question], queue=False).then(
            bot, chatbot, chatbot)
        # Define the action for the clear button
        clear.click(lambda: None, None, chatbot, queue=False)

    # Launch the Gradio demo interface
    demo.queue().launch(share=False, debug=True, server_port=7864)