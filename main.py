# face rec project - testing purpose
# written while learning (by Ritika)

import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import streamlit as st

# these i added later while testing
import os
import cv2


# === folder setup ===
base_dir= Path("data")
img_dir= base_dir / "images"
enc_dir= base_dir / "encodings"
excel_path= base_dir / "faces.xlsx"

# creating folders manually if not there
if not base_dir.exists():
    base_dir.mkdir()
if not img_dir.exists(): img_dir.mkdir()
if not enc_dir.exists(): enc_dir.mkdir()


# making excel file (only first time)
if not os.path.exists(excel_path):
    df= pd.DataFrame(columns=["ID", "Name", "Photo", "Date Added"])
    df.to_excel(excel_path, index=False, engine="openpyxl")
else:
    df= pd.read_excel(excel_path, engine="openpyxl")
    df.columns= df.columns.str.strip()

# extra check (i had missed this before)
if "Date Added" not in df.columns:
    df["Date Added"]= ""


# store in session state
if "df" not in st.session_state:
    st.session_state.df= df



# ==============================
# function for adding new face
# ==============================
def add_face(image_file, person_name):
    df= st.session_state.df
    image_file= Path(image_file)

    if not image_file.exists():
        return {"msg": "file not found"}

    # load image
    img= face_recognition.load_image_file(str(image_file))
    loc = face_recognition.face_locations(img)

    if len(loc) == 0:
        return {"msg": "no face found"}

    enc = face_recognition.face_encodings(img, loc)
    if len(enc) == 0:
        return {"msg": "encoding failed"}

    enc = enc[0]
    print("encoding done for", person_name)

    # check duplicate
    for i, r in df.iterrows():
        enc_file= enc_dir / f"{r['ID']}.npy"
        if enc_file.exists():
            old_enc= np.load(enc_file)
            if face_recognition.compare_faces([old_enc], enc, tolerance=0.6)[0]:
              print("Already exists:", r["Name"])
              return {"msg": "Face already exists","ID": r["ID"], "Name": r["Name"]}

    # if not duplicate, save new one
    new_id= 1 if df.empty else df["ID"].max() + 1
    clean_name= "".join(c for c in person_name if c.isalnum() or c in (" ", "_")).strip()
    fname= img_dir / f"{new_id}_{clean_name}.jpg"

    cv2.imwrite(str(fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    np.save(enc_dir / f"{new_id}.npy", enc)

    new_row = {
        "ID": new_id,
        "Name": person_name,
        "Photo": str(fname.name),
        "Date Added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df.loc[len(df)]= new_row
    df.to_excel(excel_path,index=False,engine="openpyxl")
    st.session_state.df = df

    print("added:", person_name)
    return {"msg": "Face is add successfully","Name": person_name,"Photo": str(fname)}




# ==============================
# Streamlit interface
# ==============================
st.title("Face Recognition System (local test)")

upload= st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
name_in= st.text_input("Enter person name:")

if st.button("Submit"):
    if upload and name_in.strip():
        tmp= "temp.jpg"
        with open(tmp, "wb") as f:
            f.write(upload.getbuffer())

        res= add_face(tmp, name_in)
        st.write(res["msg"])

        if "Photo" in res and Path(res["Photo"]).exists():
            st.image(res["Photo"], caption=res.get("Name", "unknown"))
    else:
        st.warning("Please upload and enter name")


# optional: view stored faces
if st.checkbox("Show saved data"):
    st.dataframe(st.session_state.df)

    for i, r in st.session_state.df.iterrows():
      photo_val= r.get("Photo","")
      if isinstance(photo_val, str) and photo_val.strip() != "":
        p = img_dir / r["Photo"]
        if p.exists():
            st.image(str(p), caption=f"{r['ID']} - {r['Name']}", width=140)
        else:
            print("photo entry for:", r.get("Name", "unknown"))
