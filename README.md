Kudos to AI4Bharat for training hindi specific speech recognition model.
Visit: https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large


This repository aims to 
1. quantize the .nemo model.
2. remove nemo specific dependencies
3. finally use the converted onnx model. 


---

Note : We have only used the CTC version for this model. 
If you want to use RNNT version then you make trivial changes in the `onnxconversion.ipynb`
notebook.
---

There is a notebook already provided for conversion to float 16 model. 
The name of the notebook is `onnxconversion.ipynb`


# How to perform inference 

## Installation

1. Install PyTorch first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

2. Install other dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

1. Copy the environment template:
```bash
cp env.example .env
```

2. Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

## Running the Application

### Basic Inference
```bash
python inference.py
```

### Chatbot with Voice Input
```bash
streamlit run chatbot3.py
```

### Voice Transcription Only
```bash
python inference_wrapped.py
```

## Expected Output

```
Audio features shape: (1, 80, 1413), Length: [1413]
Transcription: शिवपाल की यह टिप्पणी फ़िल्म काल्या के डायलॉग से मिलतीजुलती है शिवपाल चाहते हैं कि मुलायम पारती के मुखिया फिर से बने फ़िलहाल सपा अध्यक्ष अखिलेश यादव हैं पिता से पार्ट की कमान छीनी थी
```

## Security Note

**Never commit API keys to version control.** Use environment variables as shown in the setup instructions above.
