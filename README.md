# Markov-Huffman Compression Visualizer

A **Streamlit web app** for visualizing and comparing different **text compression algorithms** — including **Standard Huffman**, **Markov-Huffman**, **Run-Length Encoding (RLE)**, and **Arithmetic Encoding**.  
It provides detailed statistics, visual graphs, and compression ratios for any text or uploaded file.

---

## Features

- **Compare multiple compression algorithms:**
  - Standard Huffman Coding  
  - Markov-Huffman Coding  
  - Run-Length Encoding (RLE)  
  - Arithmetic Encoding  

- **Interactive Visualizations:**
  - Huffman tree structures  
  - Markov transition graphs  
  - Run-length distribution charts  

- **Compression metrics:**
  - Total bits used  
  - Compression ratio (%)  
  - Bits per character  

- **Automatic file type detection:**  
  Detects already-compressed files (e.g., `.jpg`, `.mp3`, `.mp4`) and explains why recompression isn’t effective.

---

## Live Demo

You can run this project locally using **Streamlit**.

```bash
# Clone this repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate       # for Windows
# or source venv/bin/activate  # for macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run markov_huffman_visualizer.py
