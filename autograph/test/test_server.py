from openai import OpenAI
import json
import pandas as pd
import jellyfish
import time
start_time = time.time()

kg_generation_prompt = """You are an expert knowledge graph constructor.  
Your task is to extract factual information from the provided text and represent it strictly as a JSON array of knowledge graph triples.  

### Output Format
- The output must be a **JSON array**.
- Each element in the array must be a **JSON object** with exactly three non-empty keys:
  - "subject": the main entity, concept, event, or attribute.  
  - "relation": a concise, descriptive phrase or verb that describes the relationship (e.g., "founded by", "started on", "is a", "has circulation of").  
  - "object": the entity, concept, value, event, or attribute that the subject has a relationship with.  

### Constraints
- **Do not include any text other than the JSON output.**
- Do not add explanations, comments, or formatting outside of the JSON array.  
- Extract **all possible and relevant triples**.  
- All keys must exist and all values must be non-empty strings.  
- The "subject" and "object" can be specific entities (e.g., "Radio City", "Football in Albania", "Echosmith") or specific values (e.g., "3 July 2001", "1,310,696").  
- If no triples can be extracted, return exactly: `[]`."""
# Initialize the OpenAI client with your server's base URL
client = OpenAI(
    base_url="http://localhost:8130/v1",  # Your server's URL
    api_key="dummy_key"  # Your server doesn't check this, but the client requires a non-empty string
)

# Prepare the triples_string as a JSON-encoded list of triples
triples = [
    {"subject": "Arthur's Magazine", "relation": "published by", "object": "T.S. Arthur"},
    {"subject": "Arthur's Magazine", "relation": "published in", "object": "Philadelphia"},
    {"subject": "Arthur's Magazine", "relation": "published during", "object": "1844–1846"},
    {"subject": "Arthur's Magazine", "relation": "merged into", "object": "Godey's Lady's Book"},
    {"subject": "Arthur's Magazine", "relation": "merged into", "object": "May 1846"},
    {"subject": "First for Women", "relation": "started in", "object": "1989"},
    {"subject": "First for Women", "relation": "publisher", "object": "Bauer Media Group"},
    {"subject": "First for Women", "relation": "based in", "object": "Englewood Cliffs, New Jersey"},
    {"subject": "First for Women", "relation": "circulation", "object": "1,310,696 copies"},
]

parsed_kg = [
    {"subject": "Radio City", "relation": "founded by", "object": "Abraham Thomas"},
    {"subject": "Radio City", "relation": "started on", "object": "3 July 2001"},
    {"subject": "Radio City", "relation": "broadcasts on", "object": "91.1 megahertz"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Mumbai"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Bengaluru"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Lucknow"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "New Delhi"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Hyderabad"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Chennai"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Visakhapatnam"},
    {"subject": "Radio City", "relation": "plays", "object": "Hindi songs"},
    {"subject": "Radio City", "relation": "plays", "object": "English songs"},
    {"subject": "Radio City", "relation": "plays", "object": "regional songs"},
    {"subject": "Radio City", "relation": "launched", "object": "PlanetRadiocity.com"},
    {"subject": "Radio City", "relation": "launched", "object": "May 2008"},
    {"subject": "Albanian Football Federation (FSHF)", "relation": "created", "object": "1930"},
    {"subject": "Albania", "relation": "first international match against", "object": "Yugoslavia"},
    {"subject": "Albania", "relation": "first international match date", "object": "1946"},
    {"subject": "Albania", "relation": "joined FIFA", "object": "1932"},
    {"subject": "Albania", "relation": "joined UEFA", "object": "1954"},
    {"subject": "Echosmith", "relation": "formed in", "object": "February 2009"},
    {"subject": "Echosmith", "relation": "formed in", "object": "Chino, California"},
    {"subject": "Echosmith", "relation": "originally consisted of", "object": "four siblings"},
    {"subject": "Echosmith", "relation": "consists of", "object": "Sydney, Noah and Graham Sierota"},
    {"subject": "Echosmith", "relation": "started as", "object": "Ready Set Go!"},
    {"subject": "Echosmith", "relation": "signed to", "object": "Warner Bros. Records"},
    {"subject": "Echosmith", "relation": "signed to", "object": "May 2012"},
    {"subject": "Echosmith", "relation": "best known for", "object": "hit song 'Cool Kids'"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "reached number 13 on the Billboard Hot 100"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "certified double platinum by the RIAA"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "certified double platinum by ARIA in Australia"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "Warner Bros. Records' fifth-biggest-selling-digital song of 2014"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "1.3 million downloads sold"},
    {"subject": "Echosmith", "relation": "debut album", "object": "Talking Dreams"},
    {"subject": "Echosmith", "relation": "debut album release date", "object": "October 8, 2013"},
    {"subject": "Salem College", "relation": "oldest female educational institution in the South", "object": "true"},
    {"subject": "Wesleyan College", "relation": "first college established specifically for women in the South", "object": "true"},
    {"subject": "First Arthur County Courthouse and Jail", "relation": "description", "object": "smallest courthouse in the United States"},
    {"subject": "First Arthur County Courthouse and Jail", "relation": "current use", "object": "museum"},
    {"subject": "Arthur's Magazine", "relation": "published by", "object": "T.S. Arthur"},
    {"subject": "Arthur's Magazine", "relation": "published in", "object": "Philadelphia"},
    {"subject": "Arthur's Magazine", "relation": "published during", "object": "1844–1846"},
    {"subject": "Arthur's Magazine", "relation": "merged into", "object": "Godey's Lady's Book"},
    {"subject": "Arthur's Magazine", "relation": "merged into", "object": "May 1846"},
    {"subject": "2014–15 Ukrainian Hockey Championship", "relation": "season number", "object": "23"},
    {"subject": "2014–15 Ukrainian Hockey Championship", "relation": "number of participating teams", "object": "4"},
    {"subject": "Generals Kiev", "relation": "participated in", "object": "2014–15 Ukrainian Hockey Championship"},
    {"subject": "Generals Kiev", "relation": "participated in", "object": "previous season"},
    {"subject": "2014–15 Ukrainian Hockey Championship", "relation": "started after", "object": "year-end of 2014"},
    {"subject": "2014–15 Ukrainian Hockey Championship", "relation": "regular season rounds", "object": "12"},
    {"subject": "ATEK Kiev", "relation": "defeated", "object": "HK Kremenchuk"},
    {"subject": "ATEK Kiev", "relation": "defeated in", "object": "final"},
    {"subject": "First for Women", "relation": "started in", "object": "1989"},
    {"subject": "First for Women", "relation": "publisher", "object": "Bauer Media Group"},
    {"subject": "First for Women", "relation": "based in", "object": "Englewood Cliffs, New Jersey"},
    {"subject": "First for Women", "relation": "circulation", "object": "1,310,696 copies"},
    {"subject": "Freeway Complex Fire", "relation": "started on", "object": "November 15, 2008"},
    {"subject": "Freeway Complex Fire", "relation": "started first", "object": "Freeway Fire"},
    {"subject": "Freeway Complex Fire", "relation": "started first", "object": "9am"},
    {"subject": "Freeway Complex Fire", "relation": "started second", "object": "Landfill Fire"},
    {"subject": "Freeway Complex Fire", "relation": "started second", "object": "approximately 2 hours later"},
    {"subject": "William Rast", "relation": "founded by", "object": "Justin Timberlake and Trace Ayala"},
    {"subject": "William Rast", "relation": "started as", "object": "denim line"},
    {"subject": "William Rast", "relation": "evolved into", "object": "men’s and women’s clothing line"}
]

parquet_path = '/data/autograph/data/musique_validation_doc_size_10_distract_True_with_mcq_False_difficulty_medium.parquet'

df = pd.read_parquet(parquet_path)

# Get the first example (row)
first_example = df.iloc[0]

decomposed_queries = first_example['extra_info']['interaction_kwargs']['sub_queries']
# triples_string = json.dumps(parsed_kg)
# generat kg here
llm = OpenAI(
    base_url="http://localhost:8129/v1",  # Your server's URL
    api_key="EMPTY"
)
prompt = first_example['prompt']

# replace example prompt with the created prompt
new_prompt = [{"role":"system","content":f"{kg_generation_prompt}"}]
for msg in prompt:
    if msg['role'] == "user":
        new_prompt.append(msg)
        # print(msg['content'])
response = llm.chat.completions.create(model = 'Qwen/Qwen2.5-7B-Instruct', messages = new_prompt, max_completion_tokens=4096)
gen_kg = response.choices[0].message.content
print(gen_kg)
# gen_kg = """[
#     {"subject": "Namibia", "relation": "adopted constitution", "object": "February 1990"},
#     {"subject": "Namibia", "relation": "incorporated protection for", "object": "human rights"},
#     {"subject": "Namibia", "relation": "compensation for", "object": "state expropriations of private property"},
#     {"subject": "Namibia", "relation": "has independent judiciary", "object": ""},
#     {"subject": "Namibia", "relation": "executive presidency", "object": ""},
#     {"subject": "Namibia", "relation": "became independent on", "object": "21 March 1990"},
#     {"subject": "Sam Nujoma", "relation": "sworn in as first President", "object": "Namibia"},
#     {"subject": "Sam Nujoma", "relation": "sworn in", "object": "watched by Nelson Mandela"},
#     {"subject": "Walvis Bay", "relation": "ceded to", "object": "Namibia"},
#     {"subject": "Walvis Bay", "relation": "ceded", "object": "1994"},
#     {"subject": "Warsaw Pact", "relation": "disbanded", "object": "25 February 1991"},
#     {"subject": "Warsaw Pact", "relation": "disbanded by", "object": "defense and foreign ministers from remaining Pact countries"},
#     {"subject": "Warsaw Pact", "relation": "disbanded in", "object": "Hungary"},
#     {"subject": "Warsaw Pact", "relation": "formally ended", "object": "1 July 1991"},
#     {"subject": "Warsaw Pact", "relation": "formally ended by", "object": "Czechoslovak President Václav Havel"},
#     {"subject": "Warsaw Pact", "relation": "formally ended in", "object": "Prague"},
#     {"subject": "Warsaw Pact", "relation": "de facto disbanded", "object": "December 1989"},
#     {"subject": "Warsaw Pact", "relation": "de facto disbanded due to", "object": "violent revolution in Romania"},
#     {"subject": "Warsaw Pact", "relation": "disestablished by", "object": "USSR"},
#     {"subject": "Warsaw Pact", "relation": "disestablished in", "object": "December 1991"},
#     {"subject": "Zeferino Martins", "relation": "is", "object": "East Timorese footballer"},
#     {"subject": "Zeferino Martins", "relation": "plays as", "object": "midfielder for Ad. Dili Oeste"},
#     {"subject": "Zeferino Martins", "relation": "plays for", "object": "Timor-Leste national team"},
#     {"subject": "Tracy Byrd", "relation": "released", "object": "The Truth About Men (album)"},
#     {"subject": "The Truth About Men (album)", "relation": "written by", "object": "Paul Overstreet, Rory Lee Feek and Tim Johnson"},
#     {"subject": "Jessi Colter", "relation": "recorded", "object": "What's Happened to Blue Eyes (song)"},
#     {"subject": "Jessi Colter", "relation": "released", "object": "What's Happened to Blue Eyes (song)"},
#     {"subject": "What's Happened to Blue Eyes (song)", "relation": "peaked at", "object": "Top 10 on Billboard Country Chart"},
#     {"subject": "Ho Chi Minh", "relation": "organized and led", "object": "Viet Minh"},
#     {"subject": "Ho Chi Minh", "relation": "organized and led to fight for", "object": "Vietnamese Independence"},
#     {"subject": "Ho Chi Minh", "relation": "declared Vietnam independence from", "object": "France"},
#     {"subject": "Ho Chi Minh", "relation": "became prime minister of", "object": "Democratic Republic of Vietnam (North Vietnam)"},
#     {"subject": "Ho Chi Minh", "relation": "became president of", "object": "Democratic Republic of Vietnam (North Vietnam)"},
#     {"subject": "New Zealand", "relation": "granted independence to", "object": "Western Samoa"},
#     {"subject": "Western Samoa", "relation": "granted independence on", "object": "1 January 1962"},
#     {"subject": "Western Samoa", "relation": "signed friendship treaty with", "object": "New Zealand"},
#     {"subject": "Western Samoa", "relation": "joined", "object": "Commonwealth of Nations"},
#     {"subject": "Western Samoa", "relation": "joined Commonwealth of Nations on", "object": "28 August 1970"},
#     {"subject": "Karedok", "relation": "is", "object": "raw vegetable salad in peanut sauce from West Java, Indonesia"},
#     {"subject": "Karedok", "relation": "is one of", "object": "Sundanese signature dishes"},
#     {"subject": "Karedok", "relation": "includes", "object": "cucumbers, bean sprouts, cabbage, legumes, Thai basil, and small green eggplant"},
#     {"subject": "Karedok", "relation": "covered in", "object": "peanut sauce dressing"},
#     {"subject": "Karedok", "relation": "similar to", "object": "gado-gado"},
#     {"subject": "Karedok", "relation": "uses", "object": "kencur, Thai basil and eggplant"},
#     {"subject": "Karedok", "relation": "served with", "object": "hot rice, tofu, tempeh, and krupuk"},
#     {"subject": "Karedok", "relation": "served from", "object": "hawker carts, stalls, and restaurants"},
#     {"subject": "Rule of law", "relation": "declared fundamental principle by", "object": "International Commission of Jurists"},
#     {"subject": "Rule of law", "relation": "declared in", "object": "Declaration of Delhi, New Delhi, 1959"},
#     {"subject": "Rule of law", "relation": "implies", "object": "certain rights and freedoms"},
#     {"subject": "Rule of law", "relation": "implies", "object": "independent judiciary"},
#     {"subject": "Rule of law", "relation": "implies", "object": "social, economic and cultural conditions conducive to human dignity"},
#     {"subject": "Patty Loveless", "relation": "recorded", "object": "The Trouble with the Truth (song)"},
#     {"subject": "Patty Loveless", "relation": "recorded", "object": "The Trouble with the Truth (album)"},
#     {"subject": "The Trouble with the Truth (song)", "relation": "written by", "object": "Gary Nicholson"},
#     {"subject": "The Trouble with the Truth (song)", "relation": "released as", "object": "fifth and final single"},
#     {"subject": "The Trouble with the Truth (song)", "relation": "released in", "object": "April 1997"},
#     {"subject": "Indonesia–Timor Leste Commission of Truth and Friendship", "relation": "established by", "object": "governments of Indonesia and East Timor"},
#     {"subject": "Indonesia–Timor Leste Commission of Truth and Friendship", "relation": "established in", "object": "August 2005"},
#     {"subject": "Indonesia–Timor Leste Commission of Truth and Friendship", "relation": "investigated", "object": "acts of violence around independence referendum in East Timor, 1999"},
#     {"subject": "Indonesia–Timor Leste Commission of Truth and Friendship", "relation": "final report submitted to", "object": "presidents of both nations"},
#     {"subject": "Indonesia–Timor Leste Commission of Truth and Friendship", "relation": "endorsed by", "object": "Indonesian President Susilo Bambang Yudhoyono"},
#     {"subject": "Estonia", "relation": "declared independence from", "object": "Russia"},
#     {"subject": "Estonia", "relation": "first competed as a nation at", "object": "1920 Summer Olympics"},
#     {"subject": "Estonia", "relation": "participated in", "object": "Olympic Games until 1940"},
#     {"subject": "Estonia", "relation": "regained independence in", "object": "1991"},
#     {"subject": "Estonia", "relation": "participated in all", "object": "Olympics"},
#     {"subject": "Estonia", "relation": "best results in medal table", "object": "1936 Summer Olympics (ranked 13th)"},
#     {"subject": "Estonia", "relation": "best results in medal table", "object": "2006 Winter Olympics (ranked 12th)"},
#     {"subject": "Gado-gado", "relation": "widely sold in", "object": "almost every part of Indonesia"},
#     {"subject": "Gado-gado", "relation": "thought to have originated from", "object": "Sundanese dish"},
#     {"subject": "Gado-gado", "relation": "most prevalent in", "object": "Western parts of Java"},
#     {"subject": "Gado-gado", "relation": "has similar version in", "object": "Central and East Java (called pecel)"},
#     {"subject": "Gado-gado", "relation": "served from", "object": "hawkers' carts, stalls, and restaurants"},
#     {"subject": "Gado-gado", "relation": "served in", "object": "Indonesian-style restaurants worldwide"},
#     {"subject": "Union of Utrecht", "relation": "signed by", "object": "northern provinces of the Low Countries"},
#     {"subject": "Union of Utrecht", "relation": "signed in", "object": "1579"},
#     {"subject": "Act of Abjuration", "relation": "declared independence of", "object": "provinces from Philip II"},
#     {"subject": "Act of Abjuration", "relation": "declared in", "object": "1581"},
#     {"subject": "Ramon Magsaysay Award", "relation": "list of winners", "object": "available"},
#     {"subject": "Johan Ferrier", "relation": "served as", "object": "1st President of Suriname"},
#     {"subject": "Johan Ferrier", "relation": "served as", "object": "last governor of Suriname before independence"},
#     {"subject": "Johan Ferrier", "relation": "served as last governor from", "object": "1968 to 1975"},
#     {"subject": "Johan Ferrier", "relation": "served as president from", "object": "25 November 1975 to 13 August 1980"},
#     {"subject": "Kenya", "relation": "is", "object": "presidential representative democratic republic"},
#     {"subject": "Kenya", "relation": "head of state and head of government", "object": "president"},
#     {"subject": "Kenya", "relation": "executive power exercised by", "object": "government"},
#     {"subject": "Kenya", "relation": "legislative power vested in", "object": "government and National Assembly and the Senate"},
#     {"subject": "Kenya", "relation": "judiciary is", "object": "independent of the executive and the legislature"},
#     {"subject": "Kenya", "relation": "had growing concern about", "object": "corruption during Daniel arap Moi's tenure"},
#     {"subject": "Kenya", "relation": "placed", "object": "139th out of 176 in CPI 2012"},
#     {"subject": "Kenya", "relation": "scored", "object": "27/100 in CPI 2012"},
#     {"subject": "Kenya", "relation": "established", "object": "Ethics and Anti-Corruption Commission (EACC)"},
#     {"subject": "East Timor", "relation": "official languages", "object": "Tetum, Portuguese"},
#     {"subject": "East Timor", "relation": "recognized national languages", "object": "15 languages"},
#     {"subject": "East Timor", "relation": "capital and largest city", "object": "Dili"},
#     {"subject": "South Africa", "relation": "declared great white sharks", "object": "legally protected species"},
#     {"subject": "South Africa", "relation": "declared in", "object": "1991"},
#     {"subject": "MESAN", "relation": "won", "object": "347,000 out of 356,000 votes in 1957 election"},
#     {"subject": "MESAN", "relation": "captured", "object": "every legislative seat in 1957 election"},
#     {"subject": "MESAN", "relation": "won", "object": "elections in 1957"},
#     {"subject": "Boganda", "relation": "elected president of", "object": "Grand Council of French Equatorial Africa"},
#     {"subject": "Boganda", "relation": "served as", "object": "prime minister of Central African Republic"},
#     {"subject": "David Dacko", "relation": "became president of", "object": "Central African Republic"},
#     {"subject": "David Dacko", "relation": "forced", "object": "Abel Goumba into exile in France"}
# ]"""
# Define the request payload
question = first_example['extra_info']['question']
sampling_params = {
    "temperature": 0.0,
    "max_tokens": 512
}

# Send the request using the chat.completions.create method
response = client.chat.completions.create(
    model="vllm-tog-rag",  # Model name (matches your server's response)
    messages=[
        {
            "role": "user",
            "content": json.dumps({
                "question": question,
                "triples_string": gen_kg,
                "sampling_params": sampling_params,
            })  # Encode the KGQARequest as a JSON string
        }
    ],
    extra_body={
        "question": question,
        "triples_string": gen_kg,
        "sampling_params": sampling_params,
        "sub_queries": list(decomposed_queries)
    }  # Pass the KGQARequest fields directly in extra_body
)
# Extract and print the answer
answer = response.choices[0].message.content
print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Gold Answer: {first_example['reward_model']['ground_truth']}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")