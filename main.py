import streamlit as st
from streamlit_chat import message
from PIL import Image
import os
from my_agents.supervisor_agent import main_app

# App title
st.title("Flow Chart Creator AI")

# Initialize session state for user inputs, bot responses, and images
if "user_input" not in st.session_state:
    st.session_state["user_input"] = []

if "bot_response" not in st.session_state:
    st.session_state["bot_response"] = []

if "bot_images" not in st.session_state:
    st.session_state["bot_images"] = []

# Function to process user input with your workflow
def process_input(user_input):
    try:
        # Call your custom function and handle its response
        response = main_app.invoke({"messages": [("human", user_input)]})
        # print("Response:", response)

        bot_response = response['messages'][-1].content  # AI's textual response
        print("Bot Response:", bot_response)
        if "flowchart.png" in bot_response:
            return bot_response, "\output\flowchart.png"
        return bot_response, False

    except Exception as e:
        return f"Error: {e}", None

# Function to get user text input
def get_text():
    return st.text_input("Write your message here:", key="input")

# Capture user input
user_input = get_text()

if user_input:
    # Process the input with your function
    bot_output, bot_image_path = process_input(user_input)
    
    # Store the user input and bot response in session state
    st.session_state.user_input.append(user_input)
    st.session_state.bot_response.append(bot_output)
    st.session_state.bot_images.append(bot_image_path)

# Display chat history
if st.session_state["user_input"]:
    for i in range(len(st.session_state["user_input"])):
        # Display user input
        message(st.session_state["user_input"][i], key=f"user_{i}", avatar_style="miniavs")
        # Display bot response
        message(st.session_state["bot_response"][i], key=f"bot_{i}", avatar_style="bottts", is_user=True)

        # Display the corresponding image, if available
        if st.session_state["bot_images"][i]:
            try:
                img = Image.open(r"output/flowchart.png")
                st.image(img, caption=f"Generated Image {i+1}", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display the image: {e}")

# Note:
# Ensure that your `main_app.invoke()` function generates and provides the correct image path as part of the response.
