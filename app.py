import streamlit as st

st.set_page_config(page_title="Deep Learning Fundamentals Suite", page_icon="🧠", layout="wide")

st.title("🧠 Deep Learning Fundamentals Suite")
st.caption("Inference-only Streamlit app. Models + preprocessors are pulled from public Hugging Face repos.")

st.markdown(
    '''
### Mini-projects (pages)
- **Bank Churn Prediction (ANN)** — 🔗 [ash001/bank-churn-ann](https://huggingface.co/ash001/bank-churn-ann)
- **NYC Taxi Fare Regression (PyTorch ANN)** — 🔗 [ash001/nyc-taxi-fare-regression-ann](https://huggingface.co/ash001/nyc-taxi-fare-regression-ann)
- **IMDB Sentiment (SimpleRNN)** — 🔗 [ash001/imdb-sentiment-simple-rnn](https://huggingface.co/ash001/imdb-sentiment-simple-rnn)
- **Hamlet Next-Word (LSTM)** — 🔗 [ash001/hamlet-nextword-lstm](https://huggingface.co/ash001/hamlet-nextword-lstm)
- **Time-Series Forecast (PyTorch LSTM)** — 🔗 [ash001/timeseries-forecast-lstm](https://huggingface.co/ash001/timeseries-forecast-lstm)
- **Cats vs Dogs (Transfer Learning CNN)** — 🔗 [ash001/cats-dogs-transferlearning-cnn](https://huggingface.co/ash001/cats-dogs-transferlearning-cnn)
    '''
)

st.info(
    "Open a page from the left sidebar. The first time you open a page, the model artifacts will download and cache."
)
