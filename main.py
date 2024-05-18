# coding: utf-8
import os

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image

def load_image_from_url(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Cache image file locally.
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def load_image(uploaded_file):
    """Loads an uploaded image file."""
    img = tf.io.decode_image(uploaded_file.getvalue(), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [384, 384])
    img = img[tf.newaxis, ...]
    return img

def main():
    st.title('Style Transfer App')
    st.write('This app transfers the style from one image to another using TensorFlow Hub and Streamlit.')

    # File uploader for content and style images
    content_file = st.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
    style_file = st.file_uploader("Choose a Style Image", type=["png", "jpg", "jpeg"])

    # Load example images
    content_image = load_image_from_url('https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg')
    style_image = load_image_from_url('https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg')

    # Load images if uploaded
    if content_file is not None:
        content_image = load_image(content_file)
    if style_file is not None:
        style_image = load_image(style_file)

    # Display images
    if content_image and style_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(content_image.numpy(), caption="Content Image")
        with col2:
            st.image(style_image.numpy(), caption="Style Image")

    # Button to perform style transfer
    if st.button('Transfer Style'):
        if content_image is not None and style_image is not None:
            # Load TF Hub module
            hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
            hub_module = hub.load(hub_handle)

            # Stylize content image with given style image
            outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
            stylized_image = outputs[0]

            # Display the stylized image
            st.image(stylized_image.numpy(), caption='Stylized Image', width=300)
        else:
            st.write('Please upload both content and style images before transferring style.')

if __name__ == "__main__":
    main()