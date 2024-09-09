from audiocraft.models import MusicGen
import streamlit as st 
import torch 
import torchaudio
import os 
import base64
import tempfile  # For handling temporary file paths

# Cache the model to avoid loading it multiple times
@st.cache_resource
def load_model():
    try:
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def generate_music_tensors(description, duration: int):
    model = load_model()
    if model is None:
        return None

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]


def save_audio(samples: torch.Tensor):
    """Saves audio samples to a temporary file and returns the file path."""
    sample_rate = 32000

    # Use a temporary directory for storing audio files
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio_output.wav")

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    torchaudio.save(audio_path, samples[0], sample_rate)
    return audio_path

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon="🎵",
    page_title="Music Gen"
)

def main():
    st.title("Text to Music Generator🎵")

    with st.expander("See explanation"):
        st.write("Music Generator app built using Meta's Audiocraft library. We are using the Music Gen Small model.")

    text_area = st.text_area("Enter your description...")
    time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

    if text_area and time_slider:
        st.json({
            'Your Description': text_area,
            'Selected Time Duration (in Seconds)': time_slider
        })

        st.subheader("Generated Music")

        # Generate music tensors
        music_tensors = generate_music_tensors(text_area, time_slider)
        if music_tensors is None:
            st.error("Music generation failed. Please check the logs.")
            return
        
        # Save and display audio
        save_music_file = save_audio(music_tensors)
        audio_file = open(save_music_file, 'rb')
        audio_bytes = audio_file.read()
        
        # Streamlit audio player
        st.audio(audio_bytes, format='audio/wav')
        
        # Provide download link for the audio
        st.markdown(get_binary_file_downloader_html(save_music_file, 'Audio'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
