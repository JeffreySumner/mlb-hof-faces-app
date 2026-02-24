"""Hall of Fame Face Predictor Streamlit application."""

from PIL import Image
import streamlit as st

from config import DEFAULT_HOF_THRESHOLD
from inference.predict import predict_hof_probability
from inference.preprocess import preprocess_for_model
from services.bbref_image import fetch_player_image
from services.model_loader import load_model
from services.player_lookup import get_team_codes, search_players


st.set_page_config(page_title="Hall of Fame Face Predictor", page_icon="⚾", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(ttl=3600, show_spinner=False)
def get_search_results(query: str, team_filter: str):
    return search_players(query=query, team_filter=team_filter, limit=40)


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_options():
    return get_team_codes()


DEFAULT_STATE = {
    "query_input": "",
    "team_filter": "",
    "force_refresh": False,
    "quick_name": "",
    "threshold_player": float(DEFAULT_HOF_THRESHOLD),
    "threshold_upload": float(DEFAULT_HOF_THRESHOLD),
    "upload_key_nonce": 0,
}


def reset_app_state():
    """Reset UI state and cached search data."""
    for key, value in DEFAULT_STATE.items():
        if key == "upload_key_nonce":
            # file_uploader values cannot be assigned directly; bump key instead.
            st.session_state[key] = int(st.session_state.get(key, 0)) + 1
        else:
            st.session_state[key] = value
    st.cache_data.clear()


st.title("⚾ Hall of Fame Face Predictor")
st.markdown(
    "Search players by ID, name, or team, fetch their Baseball Reference headshot, "
    "and run face-based Hall of Fame prediction."
)
st.info(
    "⚠️ Satirical project: appearance has nothing to do with baseball ability. "
    "This model demonstrates spurious pattern detection in image ML."
)

if st.button("Clear all / reset app", use_container_width=True):
    reset_app_state()
    st.rerun()

player_tab, upload_tab = st.tabs(["Player lookup", "Upload your own photo"])

with player_tab:
    with st.expander("Search options", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            query = st.text_input(
                "Player search",
                placeholder="bbref id, playerid, or name (e.g., aaronha01, bondsba01, Hank Aaron)",
                key="query_input",
            ).strip()
        with col2:
            team_choices = [""] + get_team_options()
            team_filter = st.selectbox(
                "Optional team filter",
                options=team_choices,
                index=0,
                key="team_filter",
                help="Lahman team codes (e.g., NYA, NYN, BOS, ATL).",
            ).strip()

    force_refresh = st.checkbox(
        "Force refresh image from Baseball Reference",
        value=False,
        key="force_refresh",
    )
    results_df = get_search_results(query=query, team_filter=team_filter) if (query or team_filter) else None

    if results_df is not None and not results_df.empty:
        st.subheader("Matching players")
        st.dataframe(
            results_df[["full_name", "bbrefid", "playerid", "teams", "debut", "finalgame"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        options = list(results_df.itertuples(index=False))

        # Quick type-ahead helper for long result sets.
        quick_name = st.text_input(
            "Quick narrow within matches",
            placeholder="Start typing player name to narrow the dropdown...",
            key="quick_name",
        ).strip().lower()
        if quick_name:
            options = [o for o in options if quick_name in o.full_name.lower()]

        if not options:
            st.warning("No players match that quick filter.")
            st.stop()

        selected = st.selectbox(
            "Select player",
            options=options,
            format_func=lambda r: f"{r.full_name} ({r.bbrefid}) - Teams: {r.teams}",
        )

        threshold = st.slider(
            "HOF threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(DEFAULT_HOF_THRESHOLD),
            step=0.01,
            key="threshold_player",
            help="If P(HOF) >= threshold, prediction label is HOF.",
        )

        if st.button("🔮 Predict", type="primary", use_container_width=True):
            try:
                with st.spinner("Fetching player image..."):
                    image_path = fetch_player_image(selected.bbrefid, force_refresh=force_refresh)
                    original_img = Image.open(image_path)

                with st.spinner("Running preprocessing and model inference..."):
                    model_input, img_gray, img_resized = preprocess_for_model(original_img)
                    prediction = predict_hof_probability(
                        model=get_model(),
                        model_input=model_input,
                        threshold=threshold,
                    )

                st.success(f"Prediction complete for **{selected.full_name}** ({selected.bbrefid})")

                st.subheader("Image pipeline")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Original**")
                    st.image(original_img, use_container_width=True)
                with c2:
                    st.markdown("**Grayscale**")
                    st.image(img_gray, use_container_width=True)
                with c3:
                    st.markdown("**128x128 model input**")
                    st.image(img_resized, use_container_width=True)

                st.subheader("Prediction")
                left, right = st.columns([2, 1])
                with left:
                    st.markdown("**Hall of Fame probability**")
                    st.progress(prediction.prob_hof)
                    st.metric("P(HOF)", f"{prediction.prob_hof * 100:.1f}%")
                    st.markdown("**Not Hall of Fame probability**")
                    st.progress(prediction.prob_not_hof)
                    st.metric("P(Not HOF)", f"{prediction.prob_not_hof * 100:.1f}%")
                with right:
                    if prediction.label == "HOF":
                        st.success("### 🏆 Predicted: HOF")
                    else:
                        st.error("### 🚫 Predicted: Not HOF")
                    st.caption(f"Threshold: {prediction.threshold:.2f}")

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    else:
        st.caption("Enter a query above to search players.")

with upload_tab:
    st.markdown(
        "Upload your own photo to see what the model predicts. "
        "This is for entertainment only."
    )
    upload_key = f"upload_photo_{st.session_state.get('upload_key_nonce', 0)}"
    uploaded = st.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png", "webp"],
        key=upload_key,
    )
    threshold_upload = st.slider(
        "HOF threshold (uploaded photo)",
        min_value=0.10,
        max_value=0.90,
        value=float(DEFAULT_HOF_THRESHOLD),
        step=0.01,
        key="threshold_upload",
        help="If P(HOF) >= threshold, prediction label is HOF.",
    )

    if uploaded is not None:
        original_upload = Image.open(uploaded)
        st.markdown("**Uploaded image preview**")
        st.image(original_upload, use_container_width=True)

        if st.button("🔮 Predict uploaded photo", type="primary", use_container_width=True):
            try:
                with st.spinner("Running preprocessing and model inference..."):
                    model_input, img_gray, img_resized = preprocess_for_model(original_upload)
                    prediction = predict_hof_probability(
                        model=get_model(),
                        model_input=model_input,
                        threshold=threshold_upload,
                    )

                st.subheader("Image pipeline")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Original**")
                    st.image(original_upload, use_container_width=True)
                with c2:
                    st.markdown("**Grayscale**")
                    st.image(img_gray, use_container_width=True)
                with c3:
                    st.markdown("**128x128 model input**")
                    st.image(img_resized, use_container_width=True)

                st.subheader("Prediction")
                left, right = st.columns([2, 1])
                with left:
                    st.markdown("**Hall of Fame probability**")
                    st.progress(prediction.prob_hof)
                    st.metric("P(HOF)", f"{prediction.prob_hof * 100:.1f}%")
                    st.markdown("**Not Hall of Fame probability**")
                    st.progress(prediction.prob_not_hof)
                    st.metric("P(Not HOF)", f"{prediction.prob_not_hof * 100:.1f}%")
                with right:
                    if prediction.label == "HOF":
                        st.success("### 🏆 Predicted: HOF")
                    else:
                        st.error("### 🚫 Predicted: Not HOF")
                    st.caption(f"Threshold: {prediction.threshold:.2f}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
