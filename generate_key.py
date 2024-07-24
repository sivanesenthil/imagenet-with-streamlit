import pickle
from pathlib import Path
import streamlit_authenticator as stauth
#values
names = ["Administrator"]
usernames = ["admin"]
passwords = ["admin"]

#change hash
hashed_passwords= stauth.Hasher(passwords).generate()
#path to store pickle
file_path = Path(__file__).parent/"hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords,file)
