"""
Film industry corpus builder — pulls director → film → cast/studio/award
relationships from Wikidata (not raw Wikipedia links, which return noise),
batches SPARQL queries to stay fast and avoid WDQS rate limits/timeouts,
and saves both article text and relationship metadata for later use in
GraphRAG.
"""

from SPARQLWrapper import SPARQLWrapper, JSON
import wikipediaapi
import json
import time
import os

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Wikidata requires a real identifying User-Agent or it throttles hard.
# Put your actual GitHub/email here.
USER_AGENT = "rag-corpus-builder/1.0 (https://github.com/agarmeow; contact: agarwalmanyalko@gmail.com)"

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
sparql = SPARQLWrapper(WIKIDATA_ENDPOINT, agent=USER_AGENT)
wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")

DIRECTORS = [
    "Christopher Nolan", "Denis Villeneuve", "Greta Gerwig",
    "Christopher McQuarrie", "Ryan Coogler", "James Cameron",
    "Bong Joon-ho", "Sofia Coppola", "Jon Watts", "Taika Waititi",
]

# Some directors' Wikidata occupation label doesn't cleanly match
# "film director" (Q2526255) — hardcode their QIDs to skip that filter.
DIRECTOR_QID_OVERRIDES = {
    "James Cameron": "Q42574",
}

FILMS_PER_DIRECTOR = 5      # 10 x 5 ≈ 45-50 films
CAST_PER_FILM = 3           # cap so actor count stays ~25-30
RELATION_BATCH_SIZE = 10    # films per SPARQL relation query


# ---------------------------------------------------------------------
# SPARQL HELPERS
# ---------------------------------------------------------------------

def run_query(query, max_retries=6):
    """Runs a SPARQL query with exponential backoff on 429/503/504."""
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    delay = 5
    for attempt in range(max_retries):
        try:
            return sparql.query().convert()["results"]["bindings"]
        except Exception as e:
            msg = str(e)
            if "429" in msg or "503" in msg or "504" in msg:
                print(f"    Rate-limited/timeout (attempt {attempt+1}/{max_retries}), "
                      f"waiting {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 90)
                continue
            raise
    raise RuntimeError(f"Gave up after {max_retries} retries on query.")


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ---------------------------------------------------------------------
# STEP 1: DIRECTOR QIDs
# ---------------------------------------------------------------------

def get_all_director_qids(names):
    """One batched query for all directors, plus hardcoded overrides
    for anyone whose Wikidata occupation label doesn't match cleanly."""
    remaining = [n for n in names if n not in DIRECTOR_QID_OVERRIDES]
    result = dict(DIRECTOR_QID_OVERRIDES)

    if remaining:
        values = " ".join(f'"{n}"@en' for n in remaining)
        query = f"""
        SELECT ?personLabel ?person WHERE {{
          VALUES ?nameLabel {{ {values} }}
          ?person rdfs:label ?nameLabel;
                  wdt:P106 wd:Q2526255.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        rows = run_query(query)
        for r in rows:
            result[r["personLabel"]["value"]] = r["person"]["value"].split("/")[-1]

    return result


# ---------------------------------------------------------------------
# STEP 2: FILMS PER DIRECTOR
# ---------------------------------------------------------------------

def get_all_films(director_qids, limit_per_director=FILMS_PER_DIRECTOR):
    """Batched query for every director's films, now ranked by notability
    (sitelink count) so well-known films are kept over obscure ones."""
    values = " ".join(f"wd:{qid}" for qid in director_qids)
    query = f"""
    SELECT ?director ?film ?filmLabel ?article ?sitelinks WHERE {{
      VALUES ?director {{ {values} }}
      ?film wdt:P57 ?director.
      ?article schema:about ?film;
               schema:isPartOf <https://en.wikipedia.org/>.
      ?film wikibase:sitelinks ?sitelinks.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    rows = run_query(query)

    by_director = {}
    for r in rows:
        d_qid = r["director"]["value"].split("/")[-1]
        by_director.setdefault(d_qid, [])
        by_director[d_qid].append({
            "qid": r["film"]["value"].split("/")[-1],
            "title": r["filmLabel"]["value"],
            "article_title": r["article"]["value"].split("/wiki/")[-1].replace("_", " "),
            "sitelinks": int(r["sitelinks"]["value"]),
        })

    # Rank by notability (sitelink count) and take the top N per director
    for d_qid in by_director:
        by_director[d_qid].sort(key=lambda f: f["sitelinks"], reverse=True)
        by_director[d_qid] = by_director[d_qid][:limit_per_director]

    return by_director


# ---------------------------------------------------------------------
# STEP 3: RELATIONS (CAST / STUDIO / AWARD), BATCHED PER PROPERTY
# ---------------------------------------------------------------------

def get_relations_for_property(film_qids_chunk, prop, rel_name):
    """One relation type, one small batch of films — fast and light,
    avoids the 504 timeout a single giant UNION query causes."""
    values = " ".join(f"wd:{qid}" for qid in film_qids_chunk)
    query = f"""
    SELECT ?film ?entity ?entityLabel ?article WHERE {{
      VALUES ?film {{ {values} }}
      ?film wdt:{prop} ?entity.
      ?article schema:about ?entity;
               schema:isPartOf <https://en.wikipedia.org/>.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    rows = run_query(query)
    return [
        {
            "film_qid": r["film"]["value"].split("/")[-1],
            "relationship": rel_name,
            "entity_title": r["entityLabel"]["value"],
            "article_title": r["article"]["value"].split("/wiki/")[-1].replace("_", " "),
        }
        for r in rows
    ]


def get_all_film_relations(film_qids, cast_limit=CAST_PER_FILM,
                            batch_size=RELATION_BATCH_SIZE):
    property_map = [
        ("P161", "STARS"),
        ("P272", "PRODUCED_BY"),
        ("P166", "WON_AWARD"),
    ]

    all_results = []
    film_chunks = list(chunk(film_qids, batch_size))

    for prop, rel_name in property_map:
        print(f"  Fetching {rel_name} relations ({len(film_chunks)} batches)...")
        for i, batch in enumerate(film_chunks, 1):
            rows = get_relations_for_property(batch, prop, rel_name)
            all_results.extend(rows)
            print(f"    batch {i}/{len(film_chunks)} -> {len(rows)} rows")
            time.sleep(1)  # be polite between batches

    # cap cast members per film
    cast_count = {}
    filtered = []
    for r in all_results:
        if r["relationship"] == "STARS":
            cast_count[r["film_qid"]] = cast_count.get(r["film_qid"], 0) + 1
            if cast_count[r["film_qid"]] > cast_limit:
                continue
        filtered.append(r)
    return filtered


# ---------------------------------------------------------------------
# WIKIPEDIA TEXT FETCHING
# ---------------------------------------------------------------------

def fetch_text(article_title, retries=3):
    for attempt in range(retries):
        try:
            page = wiki.page(article_title)
            return page.text if page.exists() else None
        except Exception:
            time.sleep(3)
    return None


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def build_corpus():
    os.makedirs("data/raw", exist_ok=True)
    triples = []
    articles = {}

    print("Step 1/3: Looking up director QIDs...")
    director_qid_map = get_all_director_qids(DIRECTORS)
    missing = set(DIRECTORS) - set(director_qid_map.keys())
    if missing:
        print(f"  WARNING: no QID found for: {missing}")

    print("Step 2/3: Fetching all filmographies...")
    films_by_director = get_all_films(list(director_qid_map.values()))

    qid_to_name = {v: k for k, v in director_qid_map.items()}
    all_films = []
    for d_qid, films in films_by_director.items():
        d_name = qid_to_name[d_qid]
        for film in films:
            triples.append({"source": d_name, "relationship": "DIRECTED", "target": film["title"]})
            all_films.append(film)

    print(f"  Found {len(all_films)} films across {len(films_by_director)} directors.")

    print("Step 3/3: Fetching cast/studio/award relations...")
    relations = get_all_film_relations([f["qid"] for f in all_films])

    film_title_by_qid = {f["qid"]: f["title"] for f in all_films}
    for rel in relations:
        film_title = film_title_by_qid.get(rel["film_qid"])
        if not film_title:
            continue
        triples.append({
            "source": film_title,
            "relationship": rel["relationship"],
            "target": rel["entity_title"],
        })

    # Collect every article title we need text for
    article_titles = set(DIRECTORS)
    article_titles.update(f["article_title"] for f in all_films)
    article_titles.update(r["article_title"] for r in relations)

    print(f"\nDownloading text for {len(article_titles)} articles via Wikipedia API...")
    for i, title in enumerate(sorted(article_titles), 1):
        text = fetch_text(title)
        if text:
            articles[title] = text
        if i % 20 == 0:
            print(f"  {i}/{len(article_titles)} fetched...")

    for title, text in articles.items():
        safe_name = title.replace("/", "_").replace(":", "_")
        with open(f"data/raw/{safe_name}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    with open("data/relationships.json", "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(articles)} articles, {len(triples)} relationships")
    print("Saved to data/raw/ and data/relationships.json")


if __name__ == "__main__":
    build_corpus()