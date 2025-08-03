import streamlit as st
import io
from PIL import Image
import base64
import ollama

st.subheader('Input')
image_input = st.file_uploader('Upload an image', type=["png", "jpg", "jpeg"])

if image_input:
    st.image(image_input, caption="Uploaded Image", use_container_width=True)

    # Convert image to base64
    image = Image.open(image_input)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Embed image in markdown-style base64 image format
    image_markdown = f"![image](data:image/png;base64,{image_base64})"

    # Construct full prompt for description

    # prompt = (
    #     "Describe the content of this image in detail, including the objects, scenes and any visible context. "
    #     "Provide a concise and well-organised description.\n\n"
        
    # )

    prompt = ('Extract all text from this image accurately, preserving the line breaks and formatting where possible. output the text as plain text without any explanations or comments')



    # Call Ollama
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[
            {
                "role": "user",
                "content": prompt,
                'images': [image_base64]
            }
        ]
    )

    st.subheader('Output')

    # Display result
    with st.expander('Image Description'):
        st.markdown(
            f"""
            <div style="color:gold; padding:10px; line-height:1.6;">
            {response['message'].content}
            </div>
            """,
            unsafe_allow_html=True

        )

    print(response)
