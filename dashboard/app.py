"""
Final Polished RAG Dashboard
"""

import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("📚 RAG Document Q&A Assistant")
st.caption("Hybrid RAG • Sentence-BERT • FAISS • BM25 • Confidence Scoring")

api_url = "http://127.0.0.1:8000/api"

tab1, tab2 = st.tabs(["💬 Chat", "⚠️ Review Queue"])

# ====================== CHAT TAB ======================
with tab1:
    st.header("Ask Questions About Your Documents")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "confidence" in msg:
                conf = msg["confidence"]
                if conf >= 0.7:
                    st.success(f"✅ Confidence: {conf:.3f}")
                elif conf >= 0.5:
                    st.warning(f"⚠️ Confidence: {conf:.3f}")
                else:
                    st.error(f"❌ Confidence: {conf:.3f}")
            
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Sources Used", expanded=False):
                    for i, source in enumerate(msg["sources"], 1):
                        chunk_id = source['metadata'].get('chunk_index', 'N/A')
                        st.write(f"**Source {i} (Chunk {chunk_id})**")
                        st.caption(source['text'][:500] + "..." if len(source['text']) > 500 else source['text'])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    resp = requests.post(f"{api_url}/query", json={"question": prompt}, timeout=40)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    answer = data["answer"]
                    confidence = data.get("confidence", 0.0)
                    flagged = data.get("flagged", False)
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if confidence >= 0.7:
                        st.success(f"✅ Confidence: {confidence:.3f}")
                    elif confidence >= 0.5:
                        st.warning(f"⚠️ Confidence: {confidence:.3f}")
                    else:
                        st.error(f"❌ Confidence: {confidence:.3f}")
                    
                    if flagged:
                        st.info("⚠️ This answer has been flagged for human review.")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "confidence": confidence,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Upload section
    st.subheader("📤 Upload New Documents")
    uploaded_file = st.file_uploader("PDF, TXT or DOCX", type=["pdf", "txt", "docx"])
    
    if uploaded_file and st.button("Upload & Index"):
        with st.spinner("Processing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                resp = requests.post(f"{api_url}/ingest", files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ {data['filename']} uploaded and indexed! ({data['chunks_created']} chunks)")
                else:
                    st.error(f"Upload failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ====================== REVIEW QUEUE TAB ======================
with tab2:
    st.header("⚠️ Human Review Queue")
    review_path = "data/review_queue.csv"
    
    if not os.path.exists(review_path):
        st.info("No flagged answers yet.")
    else:
        try:
            df = pd.read_csv(review_path)
            if 'reason' not in df.columns:
                df['reason'] = 'No reason recorded'
            if 'status' not in df.columns:
                df['status'] = 'pending'
            
            pending = df[df["status"] == "pending"]
            
            if pending.empty:
                st.success("🎉 All flagged answers have been reviewed!")
            else:
                st.write(f"**{len(pending)} pending reviews**")
                
                for idx, row in pending.iterrows():
                    with st.expander(f"Q: {row['question'][:70]}...", expanded=False):
                        st.write("**Answer:**")
                        st.write(row.get("answer", ""))
                        st.metric("Confidence", f"{row.get('confidence', 0):.3f}")
                        st.write(f"**Reason:** {row.get('reason', 'N/A')}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Mark Reviewed", key=f"rev_{idx}"):
                                df.loc[idx, "status"] = "reviewed"
                                df.to_csv(review_path, index=False)
                                st.success("Marked as reviewed!")
                                st.rerun()
                        with col2:
                            if st.button("🗑️ Remove", key=f"del_{idx}"):
                                df = df.drop(idx)
                                df.to_csv(review_path, index=False)
                                st.error("Removed from queue.")
                                st.rerun()
            
            st.subheader("Full History")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading review queue: {e}")

st.sidebar.caption("Built with Hybrid RAG + Sentence-BERT + FAISS + BM25")