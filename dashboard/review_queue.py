"""
Streamlit Review Queue Page - Fixed Version
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Review Queue", layout="wide")
st.title("⚠️ Human Review Queue")
st.markdown("Review and resolve low-confidence answers flagged by the system.")

review_path = "data/review_queue.csv"

if not os.path.exists(review_path):
    st.info("No flagged answers yet. The review queue is empty.")
    st.stop()

# Load the review queue with error handling
try:
    df = pd.read_csv(review_path)
except Exception as e:
    st.error(f"Error reading review queue: {e}")
    st.stop()

# Add missing columns if they don't exist (for backward compatibility)
if 'reason' not in df.columns:
    df['reason'] = 'No reason recorded'
if 'status' not in df.columns:
    df['status'] = 'pending'

# Filter pending items
pending = df[df["status"] == "pending"].copy()

if pending.empty:
    st.success("🎉 All flagged answers have been reviewed!")
    st.dataframe(df, use_container_width=True)
else:
    st.write(f"**{len(pending)} pending reviews**")

    for idx, row in pending.iterrows():
        with st.expander(f"Q: {row['question'][:80]}...", expanded=False):
            st.write("**Answer:**")
            st.write(row.get("answer", "No answer"))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{row.get('confidence', 0):.3f}")
            with col2:
                st.write(f"**Reason:** {row.get('reason', 'No reason recorded')}")
            with col3:
                st.write(f"**Time:** {str(row.get('timestamp', ''))[:16]}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Mark as Reviewed", key=f"approve_{idx}"):
                    df.loc[idx, "status"] = "reviewed"
                    df.to_csv(review_path, index=False)
                    st.success("Marked as reviewed!")
                    st.rerun()
            with col_b:
                if st.button("❌ Reject & Remove", key=f"reject_{idx}"):
                    df = df.drop(idx)
                    df.to_csv(review_path, index=False)
                    st.error("Answer removed from queue.")
                    st.rerun()

# Show full history
st.subheader("Full Review History")
st.dataframe(df, use_container_width=True)