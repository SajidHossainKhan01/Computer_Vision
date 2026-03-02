import streamlit as st
import sys
import time
from pathlib import Path
from src.inference import YOLOv11Inference
from src.utils import save_metadata, load_metadata, get_unique_classes_counts

# streamlit run app.py
# Above code runs the application on port 8501

# streamlit run app.py --server.port 8080
# Above code runs the application on port 8080


# Add project root to the system path
sys.path.append(str(Path(__file__).parent))

def init_session_state():
    session_defaults = {
    "metadata" : None,
    "unique_classes" : [],
    "count_options" : {},
    "search_results" : [],
    "search_params" : {
        "search_mode" : "Any of selected classes (OR)",
        "selected_classes" : [],
        "thresholds" : {}
    } 
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

st.set_page_config(page_title="YOLOv11 Search App", layout="wide")
st.title("Computer Vision Powered Search Application")

# Main options
option = st.radio("Choose an option:",
                  ("Process new images", "Load existing metadata"),
                  horizontal=True)

if option == "Process new images":
    with st.expander("Process new images", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            image_dir = st.text_input("Image directory path:", placeholder="path/to/images")
        with col2:
            model_path = st.text_input("Model weights path:", "yolo11m.pt")

        if st.button("Start Inference"):
            if image_dir:
                try:
                    with st.spinner("Running object detection..."):
                        inferencer = YOLOv11Inference(model_path)
                        metadata = inferencer.process_directory(image_dir)
                        metadata_path = save_metadata(metadata, image_dir)
                        st.success(f"Processed {len(metadata)} images. Metadata saved to:")
                        st.code(str(metadata_path))
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
            else:
                st.warning(f"Please enter an image directory path")
else :
    with st.expander("Load Existing Metadata", expanded=True):
        metadata_path = st.text_input("Metadata file path:", placeholder="path/to/matadata.json")

        if st.button("Load Metadata"):
            if metadata_path:
                try:
                    with st.spinner("Loading Metadata..."):
                        metadata = load_metadata(metadata_path)
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                        st.success(f"Successfully loaded metadata for {len(metadata)} images.")
                except Exception as e:
                    st.error(f"Error loading metadata: {str(e)}")
            else:
                st.warning(f"Please enter a metadata file path")


                # Person, car, airplane, banana,apple
                # Person : 1,2,3,10

# st.write(f"{st.session_state.unique_classes}, {st.session_state.count_options}")

# Search Functionality
if st.session_state.metadata:
    st.header("🔍 Search Engine")

    # "search_params" : {
    #     "search_mode" : "Any of selected classes (OR)",
    #     "selected_classes" : [],
    #     "thresholds" : {}
    # } 

    with st.container():
        st.session_state.search_params["search_mode"] = st.radio("Search mode:", 
                ("Any of selected classes (OR)", "All selected classes (AND)"),
                horizontal=True
        )

        st.session_state.search_params["selected_classes"] = st.multiselect(
            "Classes to search for:", 
            options=st.session_state.unique_classes
        )

        if st.session_state.search_params["selected_classes"]:
            st.subheader("Count Thresholds (optional)")
            cols = st.columns(len(st.session_state.search_params["selected_classes"]))
            for i, cls in enumerate(st.session_state.search_params["selected_classes"]):
                with cols[i]:
                    st.session_state.search_params["thresholds"][cls] = st.selectbox(
                        f"Max count for {cls}",
                        options=["None"] + st.session_state.count_options[cls]
                    )

        if st.button("Search Images", type="primary") and st.session_state.search_params["selected_classes"]:
            results = []
            search_params = st.session_state.search_params

            for item in st.session_state.metadata:
                matches = False
                class_matches = {}

                for cls in search_params["selected_classes"]:
                    class_detections = [d for d in item['detections'] if d['class'] == cls]
                    class_count = len(class_detections)
                    # 10 person
                    class_matches[cls] = False

                    threshold = search_params["thresholds"].get(cls, "None")
                    if threshold == "None":
                        class_matches[cls] = (class_count>=1)
                    else : 
                        class_matches[cls] = (class_count>=1 and class_count<= int(threshold))
                        # example 1: 
                        # threshold = 4
                        # class_count = 8
                        # then : class_matches[cls] = False
                        # We dont want to show this image

                        # example 2: 
                        # threshold = 4
                        # class_count = 2
                        # then : class_matches[cls] = True
                        # We want to show this image

                if search_params["search_mode"] == "Any of selected classes (OR)":
                    # not work only when both are not present or False
                    matches = any(class_matches.values())
                    # 1.jpg
                    # apple : False
                    # banana : True
                    # any(False, true) --> True
                else : # AND mode
                    # only work when both are present or True
                    matches = all(class_matches.values())
                    # 1.jpg
                    # apple : True
                    # banana : True
                    # any(False, true) --> True
                
                if matches:
                    results.append(item)

            st.session_state.search_results = results

        st.write(st.session_state.search_results)


                    

        
        

