import json
import os
import re

import chromadb
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()
BIG_MODEL = "openai/gpt-oss-120b"


def tokenize(text):
    return re.findall(r"[a-z0-9-]+", text.lower())


def name_key(name):
    # "CR-V", "CR V", and "crv" all become "CRV", so names match
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


@st.cache_resource
def load_everything():
    # load the clean tables, the search tools, and the LLM client once
    recalls = pd.read_parquet("data/clean/recalls.parquet")
    cars = pd.read_parquet("data/clean/cars.parquet")
    cars["make_key"] = cars["make"].map(name_key)
    cars["model_key"] = cars["model"].map(name_key)
    texts = (recalls["part"] + ". " + recalls["summary"] + " " + recalls["consequence"] + " " + recalls["remedy"]).tolist()
    ids = recalls["recall_id"].tolist()
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    collection = chromadb.PersistentClient(path="data/chroma").get_collection("recalls")
    bm25 = BM25Okapi([tokenize(text) for text in texts])
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return recalls, cars, texts, dict(zip(ids, range(len(ids)))), embed_model, collection, bm25, groq_client


recalls, cars, texts, id_to_row, embed_model, collection, bm25, groq_client = load_everything()
if "messages" not in st.session_state:
    st.session_state.messages = []  # the chat so far
    st.session_state.last_car = None  # the car from the last question, for follow-ups


# ---- search functions from the notebook (car_ids=None means search every recall) ----
def car_recall_ids(make, model, year):
    match = cars[(cars["make"] == make) & (cars["model"] == model) & (cars["year"] == year)]
    return match["recall_id"].tolist()


def search_embeddings(question, car_ids, k=5):
    vector = embed_model.encode([question], normalize_embeddings=True).tolist()
    if car_ids is None:
        result = collection.query(query_embeddings=vector, n_results=k)
    else:
        result = collection.query(query_embeddings=vector, n_results=min(k, len(car_ids)),
                                  where={"recall_id": {"$in": car_ids}})
    return result["ids"][0]


def search_keywords(question, car_ids, k=5):
    scores = bm25.get_scores(tokenize(question))
    if car_ids is None:
        car_ids = list(id_to_row)
    ranked = sorted(car_ids, key=lambda r: scores[id_to_row[r]], reverse=True)
    return ranked[:k]


def search_hybrid(question, car_ids, k=5):
    points = {}
    for ranked in [search_embeddings(question, car_ids, 20), search_keywords(question, car_ids, 20)]:
        for rank, recall_id in enumerate(ranked):
            points[recall_id] = points.get(recall_id, 0) + 1 / (60 + rank)
    return sorted(points, key=points.get, reverse=True)[:k]


# ---- the LLM parts ----
def ask_llm(prompt):
    response = groq_client.chat.completions.create(
        model=BIG_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    return response.choices[0].message.content


def find_car(question, last_car):
    # ask the LLM which car the question names
    prompt = f"""Which car is this question about? Reply with only JSON, like {{"make": "HONDA", "model": "CR-V", "year": 2019}}.
Use null for anything the question doesn't say.

Question: {question}"""
    match = re.search(r"\{.*\}", ask_llm(prompt), re.DOTALL)
    try:
        car = json.loads(match.group())
    except (AttributeError, ValueError):
        car = {}
    if last_car and not car.get("make"):
        # a follow-up like "what about 2020?" keeps the car from the last question
        car["make"] = last_car.get("make")
        car["model"] = car.get("model") or last_car.get("model")
        car["year"] = car.get("year") or last_car.get("year")
    return car


def recall_ids_for(car):
    # the recalls that match what the question said: make, model, and/or year
    match = cars
    if car.get("make"):
        match = match[match["make_key"] == name_key(car["make"])]
    if car.get("model"):
        match = match[match["model_key"] == name_key(car["model"])]
    if car.get("year"):
        match = match[match["year"] == int(car["year"])]
    if len(match) == len(cars):
        return None  # no car named, so search every recall
    return match["recall_id"].drop_duplicates().tolist()


@st.cache_data
def chat_answer(question, last_car):
    # RAG for the chatbot: find the car, find its best recalls, then answer from them only
    car = find_car(question, last_car)
    car_text = " ".join(str(car[field]) for field in ["year", "make", "model"] if car.get(field)) or "all cars"
    car_ids = recall_ids_for(car)
    if car_ids == []:
        return f"I couldn't find any recalls for the {car_text} in my data (model years 2011 to 2026).", car
    sources = ""
    for recall_id in search_hybrid(question, car_ids):
        sources += "Recall " + recall_id + ": " + texts[id_to_row[recall_id]] + "\n\n"
    prompt = f"""You are a friendly recall bot that helps car owners understand safety recalls.
Answer the question using ONLY the recalls below, and cite the recall numbers you used.
If a car is given, every recall below covers that car, even if its text doesn't name it.
If the recalls don't answer the question, say "I don't know based on these recalls."

Car: {car_text}
Question: {question}

Recalls:
{sources}"""
    return ask_llm(prompt) + f"\n\n*Searched recalls for: {car_text}*", car


@st.cache_data
def explain(recall_id):
    prompt = "Explain this car recall to a car owner in 2 or 3 short, simple sentences:\n\n" + texts[id_to_row[recall_id]]
    return ask_llm(prompt)


def decode_vin(vin):
    url = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/" + vin + "?format=json"
    result = requests.get(url, timeout=10).json()["Results"][0]
    return result["Make"].upper(), result["Model"].upper(), int(result["ModelYear"])


def pick_car(label):
    # three dropdowns that filter each other: make, then model, then year
    col1, col2, col3 = st.columns(3)
    makes = sorted(cars["make"].unique())
    make = col1.selectbox(label + " make", makes, index=makes.index("HONDA"))
    models = sorted(cars[cars["make"] == make]["model"].unique())
    model = col2.selectbox(label + " model", models)
    years = sorted(cars[(cars["make"] == make) & (cars["model"] == model)]["year"].unique(), reverse=True)
    year = col3.selectbox(label + " year", years)
    return make, model, year


# ---- the page ----
st.title("Recall Decoder")
st.write("Chat with the recall bot, look up the safety recalls for a car, and compare two cars. "
         "Data: NHTSA recalls for model years 2011 to 2026.")
bot_tab, lookup_tab, compare_tab, scores_tab = st.tabs(["Ask the recall bot", "Look up a car", "Compare cars", "Test scores"])

with lookup_tab:
    vin = st.text_input("Enter a VIN, or leave it empty and pick a car")
    car = None
    if vin:
        try:
            car = decode_vin(vin.strip())
            st.write(f"This VIN is a **{car[2]} {car[0]} {car[1]}**.")
        except (ValueError, KeyError, requests.RequestException):
            st.error("Sorry, I couldn't read that VIN.")
    else:
        car = pick_car("Car")
    if car:
        make, model, year = car
        car_recalls = recalls[recalls["recall_id"].isin(car_recall_ids(make, model, year))]
        car_recalls = car_recalls.sort_values("report_date", ascending=False)
        st.subheader(f"{len(car_recalls)} recalls for the {year} {make} {model}")
        for i, row in car_recalls.iterrows():
            title = row["recall_id"] + " | " + row["part"] + " | " + str(row["report_date"].date())
            if row["do_not_drive"]:
                title = "DO NOT DRIVE | " + title
            with st.expander(title):
                st.write(row["summary"])
                st.write("**What can happen:** " + row["consequence"])
                st.write("**The fix:** " + row["remedy"])
                if st.button("Explain in plain words", key=row["recall_id"]):
                    st.info(explain(row["recall_id"]))
        st.caption("These recalls cover this model, not your exact car. To see if your car still needs a fix, "
                   "enter its VIN at nhtsa.gov/recalls.")

with bot_tab:
    chat = st.container()  # the messages go here, above the input box
    question = st.chat_input("Ask about any car's recalls")
    with chat:
        st.chat_message("assistant").write("Hi! I'm your recall bot. Ask me anything about car recalls, "
                                           "like \"Is there an air bag recall for the 2019 Honda CR-V?\"")
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])
        if question:
            st.chat_message("user").write(question)
            with st.spinner("Looking up recalls..."):
                reply, asked_car = chat_answer(question, st.session_state.last_car)
            st.chat_message("assistant").write(reply)
            st.session_state.messages += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
            if any(asked_car.values()):
                st.session_state.last_car = asked_car

with compare_tab:
    make1, model1, year1 = pick_car("First car")
    make2, model2, year2 = pick_car("Second car")
    first = recalls[recalls["recall_id"].isin(car_recall_ids(make1, model1, year1))]
    second = recalls[recalls["recall_id"].isin(car_recall_ids(make2, model2, year2))]
    table = pd.DataFrame({
        f"{year1} {make1} {model1}": first["part"].value_counts(),
        f"{year2} {make2} {model2}": second["part"].value_counts(),
    }).fillna(0).astype(int)
    st.write("Recalls by part:")
    st.dataframe(table)
    st.write(f"Do not drive warnings: {first['do_not_drive'].sum()} vs {second['do_not_drive'].sum()}")

with scores_tab:
    if os.path.exists("data/answer_scores.csv"):
        st.write("How often search found the right recall in its top 5:")
        st.dataframe(pd.read_csv("data/search_scores.csv"))
        st.write("How often the answers were right, wrong, or made up (%):")
        st.dataframe(pd.read_csv("data/answer_scores.csv", index_col=0))
    else:
        st.write("Run Step 6 in the notebook first.")
