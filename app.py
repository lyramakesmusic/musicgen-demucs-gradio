
import gradio as gr
import numpy as np

import soundfile as sf
import tempfile
from demucs.separate import main as demucs_main
import openai

from audiocraft.models import musicgen

def mirror(x):
    return x

# prompt enhancer
def enhance_prompt(simple_prompt, api_key, n_prompts=10, use_gpt4=True):
    openai.api_key = api_key
    sys_prompt = f"""You write descriptive prompts for a text to music ai. Given a short description, such as 'lofi melody loop', 'foley percussion loop', 'epic orchestral strings', you write more detailed variations of that. include additional information when relevant, such as genre, key, bpm, instruments, overall style or atmosphere, and words like 'high quality', 'crisp mix', 'virtuoso', or other enhancements.
here's one possible way a prompt could be formatted:
'lofi melody loop, A minor, 110 bpm, jazzy chords evoking a feeling of curiosity, relaxing, vinyl recording'

Write {n_prompts} prompts for the given topic in a similar style. be descriptive! only describe the relevant elements - we don't want drums in a melody loop, nor melody or bass in a percussion loop. we also don't need to describe atmosphere for a drum loop. note: the text to music model cannot make vocals, so don't write prompts for with them. Also, for melody loops, make sure to specify 'no drums' in your prompt. I'd like them to be varied and high quality. Format them like this:

[
    'prompt 1',
    'prompt 2',
    ...
]"""
    message_history = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": simple_prompt},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4" if use_gpt4 else "gpt-3.5-turbo", 
            messages=message_history, 
            temperature=1.0
        )
        out_prompts = response.choices[0]['message']['content'].strip()
    except:
        return "Error connecting to OpenAI API. Did you paste the right API key?"

    try:
        import ast
        prompts = ast.literal_eval(out_prompts)
    except Exception as e:
        print(f"Cannot parse output as python array: \n{out_prompts}\n{e}")

    return "\n".join(prompts)
    

# musicgen
def run_musicgen(prompt, model_size='large', length=10):

    # load model
    model = musicgen.MusicGen.get_pretrained(model_size, device='cuda')
    model.set_generation_params(duration=length)

    # run model
    res = model.generate([prompt], progress=True)
    output = res.cpu()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"/outputs/{prompt.replace(' ', '_')}_{timestamp}.wav"
    sf.write(filename, output, 32000)

    # return output.squeeze().numpy()
    return filename


# demucs
def run_demucs(audio, stem_type='drums'):

    # Create a temporary file for the input audio
    input_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    if isinstance(audio, tuple):
        sf.write(input_file.name, audio[1] / 32767.0, audio[0])
    elif isinstance(audio, str):
        input_file = audio
    
    # separate
    demucs_args = f"--n htdemucs --two-stems {stem_type} {input_file.name}"
    demucs_main(shlex.split(demucs_args))

    # Load and return the separated track
    output_file = f"separated/htdemucs/{stem_type}/{stem_type}.wav"
    rate, output = sf.read(output_file)
    output = (output * 32767).astype(np.int16)
    return output.squeeze().numpy()


demo = gr.Blocks(theme='ParityError/Anime')
with demo:

    with gr.Row():
        gr.HTML("<br><br>")

    with gr.Row(equal_height=False):

        # prompt enhancer
        with gr.Column():
            gr.Markdown("## Prompt Enhancer")
            simple_prompt = gr.Textbox(label="Simple Prompt")
            with gr.Accordion(label="Example Prompts", open=False):
                gr.Examples([
                    "synthwave solo", 
                    "epic staccato strings d minor 174bpm", 
                    "music to mosh to", 
                    "hyperpop elements starter pack", 
                    "glitchy breakcore stuff", 
                    "war drums", 
                    "geese fucking up a guitar center", 
                    "sad girl hours", 
                    "lush organic pads",
                    "foley percussion loop",
                ], simple_prompt, fn=mirror)

            n_prompts = gr.Number(label="How many prompts", value=10)
            api_key_box = gr.Textbox(type="password", label="OpenAI API key")
            gpt4_checkbox = gr.Checkbox(label="Use GPT-4", value=True)
            prompts_button = gr.Button("Create Prompts")
            
            prompts_list = gr.Textbox(label="Prompts List", show_copy_button=True, interactive=False)
        
        # musicgen
        with gr.Column():
            gr.Markdown("## MusicGen")
            musicgen_prompt = gr.Textbox(label="Musicgen Prompt")
            model_size = gr.Radio(["large", "medium", "small"], value="large", label="Model Size")
            gen_length = gr.Slider(2, 30, value=10, label="Generation Length (seconds)")
            generate_button = gr.Button("Generate Audio")
            musicgen_audio = gr.Audio(type="numpy", label="Musicgen Output", interactive=False)

        # Demucs
        with gr.Column():
            gr.Markdown("## Demucs")
            demucs_in_audio = gr.Audio(type="numpy", label="Upload to Demucs")
            split_musicgen_button = gr.Button("Split MusicGen Stems")
            split_uploaded_button = gr.Button("Or Split Uploaded Stems")
            stem_type = gr.Radio(["drums", "melody"], value="drums", label="Desired Stem")
            demucs_audio = gr.Audio(type="numpy", label="Demucs Output")
        

    prompts_button.click(enhance_prompt, inputs=[simple_prompt, api_key_box, n_prompts, gpt4_checkbox], outputs=prompts_list)
    generate_button.click(run_musicgen, inputs=[musicgen_prompt, model_size, gen_length], outputs=musicgen_audio)
    split_musicgen_button.click(run_demucs, inputs=[musicgen_audio, stem_type], outputs=demucs_audio)
    split_uploaded_button.click(run_demucs, inputs=[demucs_in_audio, stem_type], outputs=demucs_audio)

    # dark mode button
    # with gr.Row(equal_height=False):
    #     toggle_dark = gr.Button(value="Toggle dark mode").style(full_width=False)
    #     toggle_dark.click(None, _js="() => { document.body.classList.toggle('dark'); }")

demo.launch(share=True)
