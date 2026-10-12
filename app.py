import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision.datasets import CIFAR10

from src.inference import (
    load_models,
    predict_image,
    CLASSES,
    CNN_TRANSFORM,
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="ANN Interactive Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_ORDER = [
    "SimpleCNN",
    "Random Forest",
    "Logistic Regression",
    "Decision Tree",
    "Naive Bayes",
]

MODEL_ICONS = {
    "SimpleCNN": "🧠",
    "Random Forest": "🌲",
    "Logistic Regression": "📈",
    "Decision Tree": "🌳",
    "Naive Bayes": "📊",
}

EXPERIMENT_ACCURACY = {
    "SimpleCNN": 65.9,
    "Random Forest": 39.1,
    "Logistic Regression": 29.4,
    "Decision Tree": 22.5,
    "Naive Bayes": 29.5,
}

# ============================================================
# STYLES
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #090e1b;
    color: #e5edf7;
}
.block-container {
    max-width: 1380px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}
header[data-testid="stHeader"] {
    background: rgba(9,14,27,0.8);
}
.hero {
    background: linear-gradient(120deg,#132344,#12344a,#10413e);
    border: 1px solid #28526a;
    border-radius: 20px;
    padding: 25px 32px;
    margin-bottom: 18px;
}
.hero h1 {
    color: #f8fafc !important;
    font-size: 36px;
    margin: 0;
    padding: 0;
}
.hero p {
    color: #c1d1e0;
    margin: 10px 0 0;
}
.eyebrow {
    color: #67e8f9;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 8px;
}
.panel {
    background: #111b2e;
    border: 1px solid #26354d;
    border-radius: 16px;
    padding: 18px;
}
.prediction {
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
    background: #121d30;
    border: 1px solid #293951;
}
.prediction.good {
    border-left: 5px solid #22c55e;
}
.prediction.bad {
    border-left: 5px solid #f87171;
}
.prediction.unknown {
    border-left: 5px solid #38bdf8;
}
.pred-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    font-weight: 700;
    color: #e2e8f0;
}
.pred-answer {
    font-size: 23px;
    font-weight: 800;
    color: #fff;
    margin: 5px 0;
}
.pred-foot {
    font-size: 12px;
    color: #9fb0c7;
}
.pill {
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 700;
}
.pill.good {
    background: #123b2a;
    color: #86efac;
}
.pill.bad {
    background: #42202a;
    color: #fca5a5;
}
.pill.unknown {
    background: #12354a;
    color: #7dd3fc;
}
.explain {
    background: #102b39;
    border: 1px solid #1c5264;
    border-radius: 14px;
    padding: 18px 22px;
    color: #d6f1fa;
    margin-top: 16px;
}
.step {
    background: #111c30;
    border: 1px solid #2b3b55;
    border-radius: 13px;
    padding: 14px 10px;
    text-align: center;
    min-height: 95px;
}
.step-title {
    color: #f8fafc;
    font-weight: 800;
    font-size: 15px;
}
.step-sub {
    color: #9fb0c7;
    font-size: 11px;
    margin-top: 8px;
}
div[data-testid="stMetric"] {
    background: #111c30;
    border: 1px solid #26354d;
    border-radius: 14px;
    padding: 14px 18px;
}
div[data-testid="stMetricValue"] {
    color: #38bdf8;
}
div[data-testid="stImage"] img {
    border-radius: 12px;
}
.footer {
    text-align: center;
    color: #718096;
    font-size: 12px;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def get_models():
    return load_models()


@st.cache_resource
def get_dataset():
    return CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
    )


models = get_models()
dataset = get_dataset()

if "image_index" not in st.session_state:
    st.session_state.image_index = 0

if "presentation_mode" not in st.session_state:
    st.session_state.presentation_mode = True


def random_image():
    current = st.session_state.image_index
    if len(dataset) > 1:
        new_index = random.randrange(len(dataset) - 1)
        if new_index >= current:
            new_index += 1
        st.session_state.image_index = new_index


def get_image(source, uploaded):
    if source == "CIFAR-10 Test Set":
        idx = st.session_state.image_index
        img, label = dataset[idx]
        return img, CLASSES[label], idx

    if uploaded is not None:
        return Image.open(uploaded).convert("RGB"), None, None

    return None, None, None


def predict_status(predicted, truth):
    if truth is None:
        return "unknown", "PREDICTED"
    if predicted == truth:
        return "good", "CORRECT ✓"
    return "bad", "WRONG ✕"


def prediction_card(model_name, result, truth):
    status, badge = predict_status(result["label"], truth)
    icon = MODEL_ICONS[model_name]

    return f"""
    <div class="prediction {status}">
        <div class="pred-head">
            <span>{icon} {model_name}</span>
            <span class="pill {status}">{badge}</span>
        </div>
        <div class="pred-answer">
            {result["label"].title()}
        </div>
        <div class="pred-foot">
            Predicted-class confidence:
            {result["confidence"] * 100:.1f}%
        </div>
    </div>
    """


def feature_maps_from_cnn(image, cnn):
    """
    Capture outputs immediately after Conv2d.
    Hooks are removed after inference.
    """
    captured = []
    hooks = []

    convs = [
        layer for layer in cnn.modules()
        if isinstance(layer, nn.Conv2d)
    ]

    def create_hook(index):
        def hook(module, inputs, output):
            captured.append(
                (index, output.detach().cpu())
            )
        return hook

    try:
        for index, layer in enumerate(convs, start=1):
            hooks.append(
                layer.register_forward_hook(create_hook(index))
            )

        x = CNN_TRANSFORM(
            image.convert("RGB").resize((32, 32))
        ).unsqueeze(0)

        cnn.eval()
        with torch.no_grad():
            cnn(x)

    finally:
        for handle in hooks:
            handle.remove()

    return captured


def make_feature_grid(activation, max_channels=16):
    features = activation[0].numpy()
    count = min(max_channels, features.shape[0])

    columns = 4
    rows = (count + columns - 1) // columns
    tile = 112
    gap = 8

    canvas = Image.new(
        "RGB",
        (
            columns * tile + (columns + 1) * gap,
            rows * tile + (rows + 1) * gap,
        ),
        "#111c30",
    )

    for i in range(count):
        arr = features[i].astype(np.float32)
        lo, hi = arr.min(), arr.max()

        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)

        image_tile = Image.fromarray(
            (arr * 255).astype(np.uint8)
        ).convert("RGB")

        image_tile = image_tile.resize(
            (tile, tile),
            Image.Resampling.NEAREST,
        )

        x = gap + (i % columns) * (tile + gap)
        y = gap + (i // columns) * (tile + gap)
        canvas.paste(image_tile, (x, y))

    return canvas


@st.cache_data(show_spinner=False)
def evaluate_cnn_subset(count):
    cnn = get_models()["SimpleCNN"]
    cnn.eval()

    truth, predictions = [], []

    for i in range(count):
        img, label = dataset[i]
        tensor = CNN_TRANSFORM(img).unsqueeze(0)

        with torch.no_grad():
            pred = int(cnn(tensor).argmax(dim=1).item())

        truth.append(label)
        predictions.append(pred)

    matrix = confusion_matrix(
        truth,
        predictions,
        labels=list(range(10)),
    )

    accuracy = (
        np.mean(np.array(truth) == np.array(predictions))
        * 100
    )

    return accuracy, matrix


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">ARTIFICIAL NEURAL NETWORK DEMONSTRATION</div>
    <h1>🧠 ANN Interactive Demo</h1>
    <p>
        Can a neural network recognize images better than
        traditional machine learning?
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Display Settings")
    st.toggle(
        "Presentation Mode",
        key="presentation_mode",
        help="Hide extra technical details for classroom demos.",
    )
    st.caption("CIFAR-10 · 10 classes · 32 × 32 pixels")
    st.caption("All predictions run locally using saved models.")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("CNN Accuracy", "65.9%")
with m2:
    st.metric("Random Forest", "39.1%")
with m3:
    st.metric("Models Compared", "5")
with m4:
    st.metric("Image Classes", "10")

st.write("")

# ============================================================
# SHARED IMAGE CONTROLS
# ============================================================

st.markdown("### 🎯 Choose an Image")

source_col, input_col = st.columns([1, 2], gap="large")

with source_col:
    source = st.radio(
        "Image source",
        ["CIFAR-10 Test Set", "Upload Image"],
        horizontal=True,
    )

with input_col:
    if source == "CIFAR-10 Test Set":
        index_col, random_col = st.columns([3, 1])

        with index_col:
            st.number_input(
                "Test image index",
                min_value=0,
                max_value=len(dataset) - 1,
                step=1,
                key="image_index",
            )

        with random_col:
            st.write("")
            st.button(
                "🎲 Surprise Me",
                on_click=random_image,
                use_container_width=True,
            )

        uploaded = None
    else:
        uploaded = st.file_uploader(
            "Upload JPG or PNG",
            type=["jpg", "jpeg", "png"],
        )

image, ground_truth, image_index = get_image(
    source, uploaded
)

st.divider()

tab_battle, tab_cnn, tab_performance = st.tabs([
    "🏆 AI Battle Arena",
    "🔬 Inside the CNN",
    "📊 Why CNN Wins?",
])

# ============================================================
# TAB 1 - AI BATTLE ARENA
# ============================================================

with tab_battle:
    st.subheader("🏆 AI Battle Arena")
    st.caption(
        "Five models face the same image. "
        "Which ones recognize it correctly?"
    )

    if image is None:
        st.info("Choose an image to begin.")

    else:
        results = predict_image(image, models)

        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown("#### The Challenge")

            st.image(
                image.resize(
                    (256, 256),
                    Image.Resampling.NEAREST,
                ),
                width=260,
            )

            if ground_truth is not None:
                st.success(
                    f"ACTUAL ANSWER: {ground_truth.upper()}"
                )
            else:
                st.info(
                    "Uploaded image — true label unknown."
                )

            st.caption(
                "CIFAR-10 images contain only 32 × 32 pixels."
            )

        with right:
            st.markdown("#### The Predictions")

            for name in MODEL_ORDER:
                st.markdown(
                    prediction_card(
                        name,
                        results[name],
                        ground_truth,
                    ),
                    unsafe_allow_html=True,
                )

        if ground_truth is not None:
            correct = sum(
                results[name]["label"] == ground_truth
                for name in MODEL_ORDER
            )

            cnn_correct = (
                results["SimpleCNN"]["label"]
                == ground_truth
            )

            if cnn_correct:
                message = (
                    f"On this image, SimpleCNN predicted "
                    f"the correct class. {correct} out of "
                    f"5 models were correct."
                )
            else:
                message = (
                    f"SimpleCNN missed this image. "
                    f"{correct} out of 5 models were correct. "
                    f"Neural networks can also make mistakes."
                )

            st.markdown(
                f"""
                <div class="explain">
                    <b>💡 What happened?</b><br><br>
                    {message}
                    <br><br>
                    A single image is an example, not proof
                    of overall model performance.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("#### 🧠 What Does the CNN Think?")

        probabilities = results["SimpleCNN"][
            "probabilities"
        ]

        top_indices = np.argsort(probabilities)[::-1][:3]

        for rank, idx in enumerate(top_indices, start=1):
            label = CLASSES[idx]
            probability = float(probabilities[idx])

            label_col, bar_col, pct_col = st.columns(
                [1, 4, 1]
            )

            with label_col:
                st.write(f"**#{rank} {label.title()}**")

            with bar_col:
                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

            with pct_col:
                st.write(f"{probability * 100:.1f}%")

        if not st.session_state.presentation_mode:
            with st.expander(
                "Show all 10 class probabilities"
            ):
                for idx in np.argsort(probabilities)[::-1]:
                    st.write(
                        f"{CLASSES[idx]}: "
                        f"{probabilities[idx] * 100:.2f}%"
                    )

        st.caption(
            "Confidence values may be poorly calibrated. "
            "100% confidence does not guarantee correctness."
        )

# ============================================================
# TAB 2 - INSIDE THE CNN
# ============================================================

with tab_cnn:
    st.subheader("🔬 Inside the CNN")
    st.caption(
        "Follow the image through convolutional layers "
        "and inspect what each feature channel responds to."
    )

    if image is None:
        st.info("Choose an image first.")

    else:
        cnn = models["SimpleCNN"]
        activations = feature_maps_from_cnn(
            image, cnn
        )

        st.markdown("#### How Does the Image Travel?")

        steps = [
            ("📷 Input", "32 × 32 × 3"),
            ("🔍 Conv 1", "32 channels"),
            ("🧩 Conv 2", "64 channels"),
            ("🧠 Conv 3", "128 channels"),
            ("🎯 Output", "10 classes"),
        ]

        step_cols = st.columns(5)

        for col, (title, subtitle) in zip(
            step_cols, steps
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="step">
                        <div class="step-title">{title}</div>
                        <div class="step-sub">{subtitle}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption(
            "The displayed channel counts describe "
            "the convolutional outputs. Pooling reduces "
            "spatial resolution between blocks."
        )

        st.divider()

        if not activations:
            st.warning(
                "No Conv2d layers were found in the model."
            )
        else:
            layer_options = [
                f"Conv {idx}"
                for idx, _ in activations
            ]

            selected = st.radio(
                "Select a layer to explore",
                range(len(activations)),
                format_func=lambda i: layer_options[i],
                horizontal=True,
            )

            layer_index, activation = activations[
                selected
            ]

            _, channels, height, width = activation.shape

            descriptions = [
                (
                    "Early convolutional responses may "
                    "highlight edges, colors and local "
                    "contrast."
                ),
                (
                    "Intermediate responses combine "
                    "patterns from earlier layers."
                ),
                (
                    "Deeper responses encode more complex "
                    "representations, though individual "
                    "channels are not necessarily semantic."
                ),
            ]

            st.info(
                descriptions[
                    min(selected, len(descriptions) - 1)
                ]
            )

            original_col, features_col = st.columns(
                [1, 2],
                gap="large",
            )

            with original_col:
                st.markdown("#### Original Image")
                st.image(
                    image.resize(
                        (224, 224),
                        Image.Resampling.NEAREST,
                    ),
                    width=230,
                )
                st.caption("Original input: 32 × 32 RGB")

            with features_col:
                st.markdown(
                    f"#### Conv {layer_index} — Feature Maps"
                )

                grid = make_feature_grid(
                    activation,
                    max_channels=16,
                )

                st.image(
                    grid,
                    width=490,
                )

                st.caption(
                    f"{channels} channels · "
                    f"{height} × {width} resolution · "
                    "showing first 16 channels"
                )

            st.divider()

            st.markdown("#### Explore a Single Feature Channel")

            channel = st.slider(
                "Choose feature channel",
                0,
                int(channels) - 1,
                0,
            )

            feature = activation[
                0, channel
            ].numpy()

            heatmap = go.Figure(
                data=go.Heatmap(
                    z=feature,
                    colorscale="Viridis",
                    colorbar=dict(title="Activation"),
                )
            )

            heatmap.update_layout(
                title=(
                    f"Conv {layer_index} — "
                    f"Channel {channel}"
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(
                    showticklabels=False,
                    scaleanchor="y",
                ),
                yaxis=dict(
                    showticklabels=False,
                    autorange="reversed",
                ),
            )

            st.plotly_chart(
                heatmap,
                use_container_width=True,
            )

            st.caption(
                "Feature maps are raw Conv2d outputs, "
                "captured before BatchNorm, ReLU and pooling. "
                "Each tile in the grid is normalized "
                "independently for display."
            )

        st.divider()

        st.markdown("#### Why Is This Different from Traditional ML?")

        a, b = st.columns(2, gap="large")

        with a:
            st.markdown(
                """
                **Traditional ML on flattened pixels**

                `Image → 3072 pixel values → Classifier`

                Flattening removes the explicit
                two-dimensional neighborhood structure
                from the input representation.
                """
            )

        with b:
            st.markdown(
                """
                **Convolutional Neural Network**

                `Image → Convolution → Feature Maps → Classifier`

                Convolution uses local receptive fields
                and shared filters to learn spatial patterns.
                """
            )

# ============================================================
# TAB 3 - MODEL PERFORMANCE
# ============================================================

with tab_performance:
    st.subheader("📊 Why CNN Wins?")
    st.caption(
        "Compare the models across the experiment, "
        "not just one selected image."
    )

    st.markdown("#### Experimental Test Accuracy")

    names = sorted(
        EXPERIMENT_ACCURACY,
        key=EXPERIMENT_ACCURACY.get,
    )

    values = [
        EXPERIMENT_ACCURACY[name]
        for name in names
    ]

    colors = [
        "#22d3ee" if name == "SimpleCNN"
        else "#64748b"
        for name in names
    ]

    accuracy_fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[
                f"{value:.1f}%"
                for value in values
            ],
            textposition="outside",
        )
    )

    accuracy_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=10, r=45, t=20, b=20),
        xaxis=dict(
            title="Test Accuracy (%)",
            range=[0, 85],
        ),
        yaxis_title="",
    )

    st.plotly_chart(
        accuracy_fig,
        use_container_width=True,
    )

    st.caption(
        "Accuracy values are taken from the recorded "
        "notebook experiment. Verify the saved models "
        "match that experiment before presenting."
    )

    st.markdown("#### What Do These Results Mean?")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("CNN", "65.9%")
        st.caption(
            "Learns visual features using convolution."
        )

    with c2:
        st.metric("Best ML Baseline", "39.1%")
        st.caption(
            "Random Forest on flattened image features."
        )

    with c3:
        st.metric(
            "Observed Difference",
            "+26.8 pp",
        )
        st.caption(
            "Percentage-point difference in this experiment."
        )

    st.divider()

    st.markdown("#### CNN Confusion Matrix")

    st.write(
        "Evaluate the loaded CNN on a small subset "
        "of CIFAR-10 test images."
    )

    count = st.select_slider(
        "Test sample count",
        options=[100, 200, 500, 1000],
        value=200,
    )

    if st.button(
        "▶ Evaluate CNN",
        type="primary",
    ):
        with st.spinner(
            f"Evaluating {count} images..."
        ):
            accuracy, matrix = evaluate_cnn_subset(
                count
            )

        st.metric(
            f"Accuracy on {count} images",
            f"{accuracy:.2f}%",
        )

        confusion_fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=CLASSES,
                y=CLASSES,
                colorscale="Blues",
                text=matrix,
                texttemplate="%{text}",
                colorbar=dict(title="Count"),
            )
        )

        confusion_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=560,
            xaxis_title="Predicted Class",
            yaxis_title="True Class",
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(
            confusion_fig,
            use_container_width=True,
        )

        st.caption(
            "This is a subset evaluation, not "
            "the full test-set confusion matrix."
        )

    st.divider()

    st.markdown("#### Three Key Ideas")

    st.success(
        "**1. Local connectivity:** "
        "Convolution processes nearby pixels together."
    )

    st.info(
        "**2. Weight sharing:** "
        "The same learned filter is applied across "
        "different image locations."
    )

    st.warning(
        "**3. Hierarchical representations:** "
        "Stacked convolutional blocks progressively "
        "transform image features."
    )

    if not st.session_state.presentation_mode:
        with st.expander("Experimental limitations"):
            st.write(
                "Results depend on the selected data split, "
                "training setup, preprocessing and "
                "hyperparameters. A stronger comparison "
                "would tune each model using validation data "
                "and evaluate on an untouched test set."
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    ANN Interactive Demo · CIFAR-10 · PyTorch · scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
