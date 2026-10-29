# Recall Decoder

A chatbot that answers questions about car safety recalls, built on real NHTSA data. It finds the right recalls for your car, answers in plain words, and shows the recall numbers it used, so you can check every answer.

**Built with:** Python, pandas, Plotly, sentence-transformers, Chroma (vector database), BM25, Groq LLM API, NHTSA vPIC API, Streamlit

<!-- Screenshot: add one later, like ![Recall Decoder app](images/app.png) -->

## Results

- **Hybrid search works best.** It found the right recall in the top 5 for 84% of test questions. Keyword search alone got 80%, and meaning search alone got 76%.
- **RAG stops made-up answers.** With RAG, the LLM was right on 28 of 30 questions and made up nothing. The same LLM with no search made up recall numbers on all 30.
- **It knows when there is no recall.** On 5 trick questions about problems that have no recall, RAG said so every time.

## What the app does

| Tab | What you can do |
|---|---|
| Ask the recall bot | Chat about any car, like "Is there an air bag recall for the 2019 Honda CR-V?" The bot finds the car in your question and remembers it, so follow-ups like "what about 2020?" work. |
| Look up a car | Type a VIN, or pick a make, model, and year. See every recall, with "DO NOT DRIVE" ones flagged, and get any recall explained in plain words. |
| Compare cars | Put two cars side by side: recalls by part, and "do not drive" warnings. |
| Test scores | See the test results below. |

## How it works

This is RAG: search finds the right recalls first, then the LLM answers using only those.

The notebook `recall_decoder.ipynb` builds everything in 6 steps:

1. **Load and clean.** NHTSA's file has 246,899 rows and no header row. I keep vehicle recalls for model years 2011 to 2026 and split them into two tables: 12,512 recalls, and 70,766 rows showing which cars each recall covers.
2. **Charts.** Recalls per year, the 10 parts with the most recalls, and recalls with a "do not drive" or "park outside" warning.
3. **Search.** Meaning search: the free `BAAI/bge-small-en-v1.5` model turns each recall into an embedding (384 numbers), and Chroma stores them. Keyword search: BM25 matches exact words like "CR-V" or "inflator". Hybrid search mixes the two lists with reciprocal rank fusion. It only searches the recalls that cover the car being asked about.
4. **Answers (RAG).** The top 5 recalls go to `openai/gpt-oss-120b` on Groq, at temperature 0 for steady answers. The prompt says to use only those recalls, cite their numbers, and say "I don't know based on these recalls." when they don't answer the question.
5. **VIN lookup.** NHTSA's free vPIC API turns a VIN into a make, model, and year.
6. **Testing.** See below.

## Testing

I made 30 test questions:

- **25 real ones**, each written from a real recall on a popular car. I only used cars with at least 10 recalls, so finding the right one in the top 5 is a real test.
- **5 trick ones** about problems that have no recall, like a leaky Mustang sunroof.

I graded every answer by hand as **right**, **wrong**, or **made up**. Made up means the answer names a recall, problem, or fix that isn't in the real recall.

**Search:** did the right recall show up in the top 5? (25 real questions)

| Search | Found it |
|---|---|
| Keywords (BM25) | 80% (20 of 25) |
| Meaning (embeddings) | 76% (19 of 25) |
| **Hybrid (both)** | **84% (21 of 25)** |

**Answers** (30 questions)

| Setup | Right | Wrong | Made up |
|---|---|---|---|
| **RAG + big model** (gpt-oss-120b) | **93% (28)** | 7% (2) | **0%** |
| RAG + small model (gpt-oss-20b) | 90% (27) | 10% (3) | 0% |
| Big model alone, no search | 0% | 0% | 100% (30) |

## Run it yourself

1. Get the code: click the green **Code** button on this page, then **Download ZIP**.
2. Install the libraries (Python 3.12):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download NHTSA's recall file and put it in a new `data/raw/` folder:
   https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip
4. Get a free Groq API key at https://console.groq.com/keys. Copy `.env.example`, name the copy `.env`, and paste your key after `GROQ_API_KEY=`.
5. Run the notebook from Step 1 to Step 5. Step 3 builds the search index the app needs. Don't run Step 6: the graded test files are already in `data/`, and running it again would write over them.
6. Start the app:
   ```bash
   streamlit run app.py
   ```

## Files

```
recall_decoder.ipynb      the whole pipeline: clean, charts, search, RAG, VIN, testing
app.py                    the Streamlit app
requirements.txt          the libraries, with versions
.env.example              shows where your Groq key goes
data/clean/               the cleaned tables (recalls.parquet, cars.parquet)
data/test_questions.csv   the 30 test questions
data/test_answers.csv     every answer, with my grades and notes
data/search_scores.csv    search test results
data/answer_scores.csv    answer test results
```

The raw NHTSA file and the Chroma search index aren't here because they're big. Step 3 of the notebook builds the index again.

## Limits

- It covers vehicle recalls for model years 2011 to 2026. Tire, car seat, and equipment recalls are left out.
- A recall covers a model and year, not your exact car. To see if your car still needs a fix, enter its VIN at [nhtsa.gov/recalls](https://www.nhtsa.gov/recalls).
- The test is small (30 questions), so the scores are a first check, not a final answer.
- This is a student project, not official NHTSA advice.

## Data

Free, public [NHTSA recall data](https://www.nhtsa.gov/nhtsa-datasets-and-apis). The column guide is [RCL.txt](https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt).
