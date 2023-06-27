# musicgen sample farm

Gradio UI for [this colab](https://colab.research.google.com/drive/1Dlo3Jb8193GAWZZzYPF1IPG7h8fiAtKG)

I wish you luck installing this cuda environment. Here are a few of the commands I had to run (in order!) to get it to work:

```
pip install gradio openai

pip install -U git+https://github.com/facebookresearch/audiocraft#egg=audiocraft
pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs

pip install torch torchvision torchaudio --pre -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html

pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
