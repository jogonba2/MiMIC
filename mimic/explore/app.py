from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import textstat
from datasets import load_from_disk
from wordcloud import WordCloud


def reset_index():
    st.session_state.current_index = 0


def analyze_text_complexity(text):
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "readability": textstat.flesch_reading_ease(text),
    }


def analyze_image_complexity(image):
    return {
        "image_size_after_compressing": get_image_size_after_compressing(image)
    }


def get_image_size_after_compressing(image, image_format: str = "jpeg") -> int:
    with BytesIO() as output:
        image.save(output, format=image_format)
        return output.tell()


def main():
    st.set_page_config(
        page_title="MIMIC Dataset Explorer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        .main {padding: 2rem 3rem;}
        .metric-box {padding: 1rem; border-radius: 8px; background: #f8f9fa; margin: 0.5rem 0;}
        .hover-highlight:hover {transform: scale(1.02); transition: all 0.3s ease;}
        .stTabs [data-baseweb="tab-list"] {gap: 1rem;}
        .stPlotlyChart {border-radius: 8px; overflow: hidden;}
        .prompt-box {padding: 1rem; background: #f1f1f1; border-radius: 5px; margin-bottom: 1rem;}
        .metadata-item {padding: 0.2rem 0;}
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("🔍 Navigation")
        dataset_path = st.text_input("Dataset path", value="mimic_dataset")

        if not Path(dataset_path).exists():
            st.error("Dataset path not found!")
            st.stop()

        try:
            dataset = load_from_disk(dataset_path)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

        available_splits = list(dataset.keys())
        selected_split = st.selectbox(
            "Dataset Split",
            available_splits,
            index=0,
            on_change=reset_index,
        )

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    st.title("📊 MIMIC Dataset Analyzer")
    st.markdown(
        f"**Selected Split:** {selected_split} | **Total Samples:** {len(dataset[selected_split])}"
    )
    st.divider()

    tab_samples, tab_stats = st.tabs(
        ["🔍 Sample Explorer", "📈 Dataset Analytics"]
    )

    # --- Sample Explorer Tab ---
    with tab_samples:
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            if st.button("⏮ Previous", use_container_width=True):
                st.session_state.current_index = max(
                    0, st.session_state.current_index - 1
                )
        with col2:
            if st.button("⏭ Next", use_container_width=True):
                st.session_state.current_index = min(
                    len(dataset[selected_split]) - 1,
                    st.session_state.current_index + 1,
                )
        with col3:
            st.progress(
                (st.session_state.current_index + 1)
                / len(dataset[selected_split]),
                f"Sample {st.session_state.current_index + 1} of {len(dataset[selected_split])}",
            )

        current_sample = dataset[selected_split][st.session_state.current_index]

        with st.expander("📄 Sample Metadata", expanded=True):
            meta_col1, meta_col2, meta_col3, meta_col4, meta_col5 = st.columns(
                5
            )
            with meta_col1:
                st.markdown(
                    f'<div class="metadata-item"><strong>Label:</strong> '
                    f'{current_sample["label"]}</div>',
                    unsafe_allow_html=True,
                )
            with meta_col2:
                st.markdown(
                    f'<div class="metadata-item"><strong>Text Model:</strong> '
                    f'{current_sample["text_model"]}</div>',
                    unsafe_allow_html=True,
                )
            with meta_col3:
                st.markdown(
                    f'<div class="metadata-item"><strong>Image Model:</strong> {current_sample["image_model"]}</div>',
                    unsafe_allow_html=True,
                )
            with meta_col4:
                st.markdown(
                    f'<div class="metadata-item"><strong>Original height:</strong> {current_sample["original_height"]}</div>',
                    unsafe_allow_html=True,
                )
            with meta_col5:
                st.markdown(
                    f'<div class="metadata-item"><strong>Original width:</strong> {current_sample["original_width"]}</div>',
                    unsafe_allow_html=True,
                )
        img_col, text_col = st.columns([1, 2])
        with img_col:
            st.image(current_sample["image"], use_container_width=True)
        with text_col:
            st.markdown(
                f'<div class="prompt-box">{current_sample["text"]}</div>',
                unsafe_allow_html=True,
            )
            tab1, tab2 = st.tabs(
                ["📝 Text Generation Prompt", "🖼️ Image Generation Prompt"]
            )
            with tab1:
                st.markdown(
                    f'<div class="prompt-box">{current_sample["text_generation_prompt"]}</div>',
                    unsafe_allow_html=True,
                )
            with tab2:
                st.markdown(
                    f'<div class="prompt-box">{current_sample["image_generation_prompt"]}</div>',
                    unsafe_allow_html=True,
                )
            st.download_button(
                label="📥 Download Text Content",
                data=current_sample["text"],
                file_name=f"sample_{st.session_state.current_index}_text.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # --- Dataset Analytics Tab ---
    with tab_stats:
        df = pd.DataFrame(dataset[selected_split])
        text_analysis = pd.DataFrame(
            [analyze_text_complexity(text) for text in df["text"]]
        )
        image_analysis = pd.DataFrame(
            [analyze_image_complexity(image) for image in df["image"]]
        )

        combined_df = pd.concat([df, text_analysis, image_analysis], axis=1)

        st.header("📊 Core Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(df))
        col2.metric("Unique Labels", df["label"].nunique())
        col3.metric("Text Models", df["text_model"].nunique())
        col4.metric("Image Models", df["image_model"].nunique())

        with st.expander("📚 Label Distribution", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Bar Chart")
                st.bar_chart(df["label"].value_counts())
            with col2:
                st.subheader("Pie Chart")
                fig, ax = plt.subplots()
                df["label"].value_counts().plot(
                    kind="pie", autopct="%1.1f%%", ax=ax
                )
                st.pyplot(fig)
                plt.close(fig)

        with st.expander("🤖 Model Distributions", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Text Models")
                text_model_counts = (
                    df.groupby(["label", "text_model"])
                    .size()
                    .unstack(fill_value=0)
                )
                st.bar_chart(text_model_counts)
            with col2:
                st.subheader("Image Models")
                image_model_counts = (
                    df.groupby(["label", "image_model"])
                    .size()
                    .unstack(fill_value=0)
                )
                st.bar_chart(image_model_counts)

        with st.expander("📝 Text Analysis", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Avg Word Count",
                    f"{text_analysis['word_count'].mean():.1f}",
                )
            with col2:
                st.metric("Max Word Count", text_analysis["word_count"].max())
            with col3:
                st.metric(
                    "Avg Readability",
                    f"{text_analysis['readability'].mean():.1f}",
                )

            st.subheader("Global Text Distributions")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.histplot(
                    text_analysis["word_count"], bins=30, kde=True, ax=ax
                )
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                fig, ax = plt.subplots()
                sns.boxplot(x=text_analysis["readability"], ax=ax)
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Per-Label Text Metrics")
            st.markdown("##### Word Count Distribution by Label")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                x="label",
                y="word_count",
                data=combined_df,
                ax=ax,
                palette="viridis",
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("##### Char Count Distribution by Label")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                x="label",
                y="char_count",
                data=combined_df,
                ax=ax,
                palette="viridis",
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("##### Readability Scores by Label")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                x="label",
                y="readability",
                data=combined_df,
                ax=ax,
                palette="rocket",
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("##### Image Size After Compressing by Label")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                x="label",
                y="image_size_after_compressing",
                data=combined_df,
                ax=ax,
                palette="rocket",
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("🔠 Vocabulary Analytics", expanded=True):
            st.subheader("Vocabulary Size per Label")
            label_groups = df.groupby("label")["text"].apply(list)
            vocab_word = defaultdict(set)
            vocab_char = defaultdict(set)

            for label, texts in label_groups.items():
                for text in texts:
                    vocab_word[label].update(text.lower().split())
                    vocab_char[label].update(text.lower())

            # Plot word vocabulary size per label
            labels_sorted = sorted(vocab_word.keys())
            word_vocab_sizes = [
                len(vocab_word[label]) for label in labels_sorted
            ]
            plt.figure(figsize=(8, 4))
            sns.barplot(x=labels_sorted, y=word_vocab_sizes, palette="viridis")
            plt.title("Word Vocabulary Size per Label")
            plt.xlabel("Label")
            plt.ylabel("Vocabulary Size")
            st.pyplot(plt.gcf())
            plt.close()

            # Plot character vocabulary size per label
            char_vocab_sizes = [
                len(vocab_char[label]) for label in labels_sorted
            ]
            plt.figure(figsize=(8, 4))
            sns.barplot(x=labels_sorted, y=char_vocab_sizes, palette="viridis")
            plt.title("Character Vocabulary Size per Label")
            plt.xlabel("Label")
            plt.ylabel("Vocabulary Size")
            st.pyplot(plt.gcf())
            plt.close()

            # Compute exclusive words and characters for each label
            exclusive_words = {}
            exclusive_chars = {}
            exclusive_char_counts = {}

            all_labels = list(vocab_word.keys())

            for label in all_labels:
                other_labels = [
                    label_ for label_ in all_labels if label_ != label
                ]

                # Compute exclusive words
                other_words = set().union(
                    *(vocab_word[label_] for label_ in other_labels)
                )
                exclusive_words[label] = vocab_word[label] - other_words

                # Compute exclusive characters
                other_chars = set().union(
                    *(vocab_char[label_] for label_ in other_labels)
                )
                exclusive_chars[label] = vocab_char[label] - other_chars

                # Compute character frequencies
                char_counts = Counter(
                    char
                    for word in vocab_word[label]
                    for char in word
                    if char in exclusive_chars[label]
                )
                exclusive_char_counts[label] = char_counts

            # Display WordClouds
            st.subheader("WordClouds of Exclusive Words per Label")
            for label in all_labels:
                st.markdown(f"**Exclusive words for label: {label}**")
                if exclusive_words[label]:
                    wc = WordCloud(
                        width=800, height=400, background_color="white"
                    ).generate(" ".join(exclusive_words[label]))
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt.gcf())
                    plt.close()
                else:
                    st.write("No exclusive words found for this label.")

            # Display Exclusive Characters and Their Frequencies
            st.subheader("Exclusive Characters per Label with Frequencies")
            for label in all_labels:
                st.markdown(f"**Exclusive characters for label: {label}**")
                st.json(exclusive_char_counts[label])

            st.subheader("Zipf Distribution per Label")
            plt.figure(figsize=(8, 6))
            for label in labels_sorted:
                label_texts = df[df["label"] == label]["text"].dropna().tolist()
                words = []
                for text in label_texts:
                    words.extend(text.lower().split())
                if not words:
                    continue
                freq = Counter(words)
                sorted_freqs = sorted(freq.values(), reverse=True)
                ranks = range(1, len(sorted_freqs) + 1)
                plt.loglog(
                    ranks, sorted_freqs, marker=".", linestyle="-", label=label
                )
            plt.xlabel("Rank")
            plt.ylabel("Frequency")
            plt.title("Zipf's Law per Label")
            plt.legend()
            st.pyplot(plt.gcf())
            plt.close()

            st.subheader("Heap's Law per Label")
            plt.figure(figsize=(8, 6))
            for label in labels_sorted:
                label_texts = df[df["label"] == label]["text"].dropna().tolist()
                all_text = " ".join(label_texts).lower()
                tokens = all_text.split()
                vocab_set = set()
                cumulative = []
                step = max(1, len(tokens) // 1000)
                for i, token in enumerate(tokens, 1):
                    vocab_set.add(token)
                    if i % step == 0 or i == len(tokens):
                        cumulative.append((i, len(vocab_set)))
                token_counts, vocab_sizes = zip(*cumulative)
                plt.loglog(
                    token_counts,
                    vocab_sizes,
                    marker=".",
                    linestyle="-",
                    label=label,
                )
            plt.xlabel("Total Tokens")
            plt.ylabel("Vocabulary Size")
            plt.title("Heap's Law per Label")
            plt.legend()
            st.pyplot(plt.gcf())
            plt.close()

        with st.expander("✅ Quality Report", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Missing Images", df["image"].isna().sum())
            with col2:
                empty_texts = df["text"].str.strip().eq("").sum()
                st.metric("Empty Texts", empty_texts)
            with col3:
                dupes = df.duplicated(subset=["id"]).sum()
                st.metric("Duplicates", dupes)

        with st.expander("🔍 Data Preview", expanded=False):
            st.dataframe(df[["label", "text_model", "image_model"]], height=300)


if __name__ == "__main__":
    main()
