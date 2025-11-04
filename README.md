# Malayalam Movie Review Sentiment Classifier

## 📌 Project Overview  
This project presents a web application for predicting the sentiment of Malayalam-language movie reviews. The core model is a fine-tuned BERT-based classifier trained to distinguish between **positive**, **neutral**, and **negative** sentiments in Malayalam film critiques.  
The UI is built with Streamlit, allowing users to input a Malayalam review and receive an instant sentiment prediction.

## 🧪 Supported Environment  
- **Python version:** 3.12.12  
- Ensure your interpreter is **Python 3.12.12** (or a compatible 3.12.x version) to avoid compatibility issues with dependencies.  
  :contentReference[oaicite:0]{index=0}

## 🧠 Key Features  
- Accepts **only Malayalam** text input and warns if non-Malayalam text is detected.  
- Single-field text UI – simple and intuitive.  
- Fast inference using PyTorch and GPU (if available).  
- Model weights can be loaded from a `.pth` file for easy deployment.  
- Clean structure separating backend logic (`model_backend.py`) and UI logic (`app.py`).

## 🗂 Project Structure  
```

├── app.py                              # Streamlit UI
├── model_backend.py                    # Model loading & prediction logic
├── requirements.txt                    # Python dependencies
├── indic_bert_sentiment_model.pth      # Fine-tuned model weights
├── README.md                           # This file
└── …                                   # Other files (if any)

````

## 🚀 Getting Started  
### Prerequisites  
- Python 3.12.12  
- PyTorch  
- Transformers library  
- Streamlit  
- (Optional) GPU for faster inference  

### Installation  
```bash
git clone YOUR_REPO_URL
cd YOUR_REPO_NAME
pip install -r requirements.txt
````

### Setup Model Access

If you are using a gated/private model repository on Hugging Face, ensure you have a valid token:

```bash
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

(On Windows CMD: `set HF_TOKEN=YOUR_HUGGINGFACE_TOKEN`)

### Run the App

```bash
streamlit run app.py
```

Then open the provided local URL (typically `http://localhost:8501`) in your browser and enter your Malayalam review in the text field.

## 🧮 Usage Example

**Input:**

> “സിനിമ എന്ന കലയെ അപമാനിക്കുന്നതാണ് ഈ ചിത്രം. വളരെ മോശം സംവിധാനം ആണ്…”
> **Output:**
> Predicted Sentiment: **negative**

## 🛠 Model & Prediction Details

* Model architecture: `BertForSequenceClassification`
* Tokenizer: `BertTokenizer`
* Prediction function: `predict(text)` returns one of `positive`, `neutral`, `negative` (or an error code for non-Malayalam input).
* Language enforcement: checks for Malayalam Unicode characters and warns if the input is not predominantly Malayalam.

## ✅ Deployment Notes

* `requirements.txt` lists all needed packages (e.g., `streamlit`, `torch`, `transformers`).
* For deployment (e.g., on Streamlit Community Cloud), commit the `requirements.txt`, `app.py`, `model_backend.py`, and this `README.md`.
* Keep the `.pth` model weights either in the repo (if size allows) or accessible via a secure path.

## 🤝 Contribution

Contributions are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Add features or fix bugs
4. Commit your changes (`git commit -m "Add your feature"`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

## 🙏 Acknowledgments

* Thanks to the developers of the Transformers library.
* Thanks to the Malayalam movie review datasets and the open-source community.
* Inspired by projects and templates for sentiment-analysis apps.



