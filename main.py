from tempfile import NamedTemporaryFile

import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool

############## Initialize Agent ###############

tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key="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",
    temperature=3,
    model_name="gpt-3.5-turbo"
)

agent = initialize_agent(
    # The agent identifier or name.
    agent="chat-conversational-react-description",
    tools=tools,                               # Some tools or configuration.
    # A variable or configuration related to the language model (LLM).
    llm=llm,
    # The maximum number of iterations.
    max_iterations=5,
    # A verbosity flag (probably for logging).
    verbose=True,
    # Some kind of conversation memory or context.
    memory=conversational_memory,
    early_stoppy_method="generate"             # A method for early stopping.
)


# Set title
st.title("Ask a question to an image ")
# Set header
st.header("Please upload an Image")
# Upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])
if file:
    # display image
    st.image(file, use_column_width=True)
    # text input
    user_question = st.text_input("Ask a question about your Image")

############## compute agent response ###########

    with NamedTemporaryFile(dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name

        response = agent.run(
            '{}, this is the image path: {}'.format(user_question, image_path))
        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress ..."):
                response = agent.run(
                    '{}, this is the image path: {}'.format(user_question, image_path))

                st.write(response)
