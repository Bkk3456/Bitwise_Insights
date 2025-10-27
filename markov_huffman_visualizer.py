# markov_huffman_visualizer.py

import heapq
from collections import defaultdict, Counter
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import mimetypes  
import math

# ------------------------ARITHMETIC----------------------------
def arithmetic_encode(data):
    if not data:
        return 0, {}

    freq = Counter(data)
    total = sum(freq.values())
    prob_ranges = {}

    # Build ranges
    cumulative = 0.0
    for char in sorted(freq):
        prob = freq[char] / total
        prob_ranges[char] = (cumulative, cumulative + prob)
        cumulative += prob

    # Perform encoding
    low = 0.0
    high = 1.0

    for char in data:
        range_ = high - low
        char_low, char_high = prob_ranges[char]
        high = low + range_ * char_high
        low = low + range_ * char_low

    # Final code interval
    final_code = (low + high) / 2
    interval_size = high - low

    # Prevent zero-bit output (when float precision fails)
    if interval_size <= 0.0:
        total_bits = len(data) * 8  # fallback to worst case
    else:
        bit_length = -math.log2(interval_size)
        total_bits = max(1, math.ceil(bit_length))  # minimum of 1 bit

    return total_bits, prob_ranges

# ----------------------------- RLE -----------------------------------
def run_length_encode(data):
    if not data:
        return []

    encoded = []
    prev_char = data[0]
    count = 1

    for char in data[1:]:
        if char == prev_char:
            count += 1
        else:
            encoded.append((prev_char, count))
            prev_char = char
            count = 1
    encoded.append((prev_char, count))
    return encoded

# ----------------------------- HUFFMAN TREE ---------------------------
class Node:
    def __init__(self, symbol=None, freq=0):  
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):  # used by heapq for comparing nodes
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    heap = [Node(sym, freq) for sym, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(None, n1.freq + n2.freq)
        merged.left, merged.right = n1, n2
        heapq.heappush(heap, merged)
    return heap[0] if heap else None

def get_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}

    if node:
        # Leaf node
        if node.symbol is not None:
            codebook[node.symbol] = prefix if prefix else "0"  # assign '0' if tree has only one symbol
        else:
            get_codes(node.left, prefix + "0", codebook)
            get_codes(node.right, prefix + "1", codebook)
    return codebook


def draw_huffman_tree(node, graph=None, parent=None, label=""):
    if graph is None:
        graph = nx.DiGraph()
    graph.add_node(id(node), label=node.symbol if node.symbol else "")
    if parent:
        graph.add_edge(parent, id(node), label=label)
    if node.left:
        draw_huffman_tree(node.left, graph, id(node), "0")
    if node.right:
        draw_huffman_tree(node.right, graph, id(node), "1")
    return graph

# ----------------------------- MARKOV MODEL ---------------------------
def build_transition_matrix(data):
    transitions = defaultdict(Counter)
    for i in range(len(data) - 1):
        curr, next_ = data[i], data[i + 1]
        transitions[curr][next_] += 1
    return transitions

def markov_huffman(data):
    transitions = build_transition_matrix(data)
    total_bits = 0
    codebooks = {}

    for context, counter in transitions.items():
        tree = build_huffman_tree(counter)
        codes = get_codes(tree)
        codebooks[context] = codes

    for i in range(len(data) - 1):
        curr, next_ = data[i], data[i + 1]
        code = codebooks[curr][next_]
        total_bits += len(code)

    return total_bits, codebooks, transitions

# ----------------------------- STANDARD HUFFMAN -----------------------
def standard_huffman(data):
    freq = Counter(data)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    total_bits = sum(len(codes[sym]) for sym in data)
    return total_bits, codes, freq, tree

# ----------------------------- STREAMLIT UI ---------------------------
st.set_page_config(page_title="Compression Visualizer", layout="centered")
st.title("🧠 Markov-Huffman Compression Visualizer")
st.markdown("Analyze and compare *Standard Huffman, **Markov-Huffman, and **Run-Length Encoding (RLE)* on text files.")

user_text = st.text_area("✍ Or enter text manually:", height=150)
uploaded_file = st.file_uploader("📂 Upload a file (any type)", type=None)

text = None
error_msg = None

if user_text.strip():
    text = user_text.strip()
elif uploaded_file:
    mimetype, _ = mimetypes.guess_type(uploaded_file.name)
    if mimetype and mimetype.startswith("text"):
        text = uploaded_file.read().decode("utf-8").replace("\n", "")
    else:
        if mimetype:
            reason = {
                "image": "📷 Images like JPEG and PNG are already compressed using specialized algorithms such as spatial prediction and entropy coding (e.g., Huffman or DEFLATE). Applying Huffman again won’t significantly reduce the size.",
                "audio": "🎵 Audio files like MP3 and AAC use advanced techniques like psychoacoustic modeling, which removes inaudible data. These formats are already highly compressed.",
                "video": "🎬 Videos in formats like MP4 or AVI use motion estimation, inter-frame compression, and entropy coding. They are already optimized for storage and streaming.",
                "application/pdf": "📄 PDFs typically include internal compression methods such as Flate (similar to ZIP) or LZW. Additional text compression won’t be effective."
            }
            main_type = mimetype.split("/")[0]
            msg = reason.get(main_type, "This file type has built-in efficient compression.")
            error_msg = f"❌ Cannot compress {uploaded_file.name}. {msg}"

if error_msg:
    st.error(error_msg)

if text:
    st.subheader("📊 Compression Summary")

    std_bits, std_codes, std_freqs, std_tree = standard_huffman(text)
    markov_bits, markov_codes, transitions = markov_huffman(text)


    def calculate_rle_bits(encoded_runs):
        total_bits = 0
        for char, count in encoded_runs:
            char_bits = 8  # 8 bits to represent the character
            count_bits = math.ceil(math.log2(count + 1)) if count > 1 else 1
            total_bits += char_bits + count_bits
        return total_bits
    
    rle_encoded = run_length_encode(text)  # This defines rle_encoded first
    rle_bits = calculate_rle_bits(rle_encoded)  # Now this works fine

    raw_bits = len(text) * 8
    length = len(text)

    std_ratio = round((1 - std_bits / raw_bits) * 100, 2)
    markov_ratio = round((1 - markov_bits / raw_bits) * 100, 2)
    rle_ratio = round((1 - rle_bits / raw_bits) * 100, 2)

    std_bpc = std_bits / length
    markov_bpc = markov_bits / length
    rle_bpc = rle_bits / length

    arith_bits, arith_probs = arithmetic_encode(text)

    df = pd.DataFrame({
        "Method": ["Raw (ASCII)", "Standard Huffman", "Markov-Huffman", "Run-Length Encoding", "Arithmetic Encoding"],
        "Total Bits": [raw_bits, std_bits, markov_bits, rle_bits, arith_bits],
        "Compression Ratio (%)": [0, std_ratio, markov_ratio, rle_ratio, round((1 - arith_bits / raw_bits) * 100, 2)],
        "Bits per Character": [8, std_bpc, markov_bpc, rle_bpc, arith_bits / length]
    })

    st.table(df)

    # Add a bar chart for compression ratio
    st.subheader("📉 Compression Ratios")
    st.bar_chart(df.set_index("Method")["Compression Ratio (%)"])

    best_method = df.loc[df["Compression Ratio (%)"].idxmax(), "Method"]
    st.success(f"✅ Best compression achieved using *{best_method}*.")

    # Frequency Graph
    st.subheader("🔢 Top Character Frequencies")
    freq_df = pd.DataFrame(std_freqs.most_common(10), columns=["Character", "Frequency"])
    st.bar_chart(freq_df.set_index("Character"))

    # Visualizations: Huffman Tree and Markov Graph
    st.subheader("🧮 Huffman & Markov Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌳 Standard Huffman Tree")
        tree_graph = draw_huffman_tree(std_tree)
        labels = nx.get_node_attributes(tree_graph, 'label')
        edge_labels = nx.get_edge_attributes(tree_graph, 'label')
        pos = nx.spring_layout(tree_graph, seed=1)
        plt.figure(figsize=(8, 6))
        nx.draw(tree_graph, pos, labels=labels, with_labels=True, node_color='lightgreen', node_size=800)
        nx.draw_networkx_edge_labels(tree_graph, pos, edge_labels=edge_labels)
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.markdown("### 🔁 Markov Transition Graph")
        G = nx.DiGraph()
        for src, dests in transitions.items():
            for dest, count in dests.items():
                G.add_edge(src, dest, weight=count)
        pos = nx.spring_layout(G, seed=42)
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        st.pyplot(plt.gcf())
        plt.clf()

    # Run-Length Encoding Visualization
    st.subheader("📏 Run-Length Encoding Visualization")
    rle_df = pd.DataFrame(rle_encoded, columns=["Character", "Run Length"])
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(rle_df)), rle_df["Run Length"], color='salmon')
    ax.set_xlabel("Run Index")
    ax.set_ylabel("Run Length")
    ax.set_title("RLE Segment Lengths")
    ax.set_xticks([])
    st.pyplot(fig)

    # Expandable Sections
    with st.expander("📄 Standard Huffman Codes"):
        st.json(std_codes)

    with st.expander("📄 Markov-Huffman Codes (context → next → code)"):
        st.json(markov_codes)

    with st.expander("📄 Arithmetic Encoding Probability Ranges"):
        st.json(arith_probs)

else:
    if not error_msg:
        st.info("👆 Upload a file or enter text above to see compression analysis.")
