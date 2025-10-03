TOPIC_PRUNE_PROMPT = """
Given the question: "%s"
And the following topic entities derived from the knowledge graph:
%s

Analyze which of these entities are suitable as starting points for reasoning on the knowledge graph to find information and clues useful for answering the question. Follow these guidelines:
- Select entities that are directly related to the core of the question and can effectively guide the search for relevant information.
- Exclude entities that are too broad, irrelevant, or not specific enough to be useful starting points.
- Provide a JSON-formatted output: {"entities": ["entity1", "entity2", ...]} containing only the suitable entities.
- If no entities are suitable, return an empty JSON object {"entities": []}.

Example:
Question: "What is the capital of France?"
Topic entities: ["France", "capital"]

Analysis: 
"France" is directly related to the question as it points to the country whose capital is being asked. "capital" is too broad and not specific to the query.
Output: {"entities": ["France"]}

Provide the analysis and output in JSON format.
"""

RELATION_PRUNE_PROMPT = """
Task:
1. Carefully review the question provided.
2. For each entity, analyze the list of available relations and select the %s most relevant relations that are likely to provide the most useful information for answering the question.
3. For each selected relation, provide:
   - A score between 0 to 10 reflecting its usefulness in answering the question (10 being the most useful).
   - A brief explanation of why the relation is relevant to the question.
4. Output the results in the following JSON-parsable format:
[
    {
        "entity": "entity1",
        "relations": [
            {"relation": "relation1", "score": X, "explanation": "Explanation for relation1"},
            {"relation": "relation2", "score": Y, "explanation": "Explanation for relation2"}
        ]
    },
    {
        "entity": "entity2",
        "relations": [
            {"relation": "relation1", "score": X, "explanation": "Explanation for relation1"},
            {"relation": "relation2", "score": Y, "explanation": "Explanation for relation2"}
        ]
    }
]

# Input Format:
Question: [The question text]
Entity 1: [The name of entity 1]
Available Relations: [A list of relations for entity 1]
Entity 2: [The name of entity 2]
Available Relations: [A list of relations for entity 2]
...(Continue for additional entities)

# Example Input:
Question: "What is the attitude of Joe Biden towards China?"
Entity 1: "China"
Available Relations: ["alliance", "international relation", "political system", "population"]
Entity 2: "Joe Biden"
Available Relations: ["political position", "presidency", "family", "early life"]

# Example Output:
[
    {
        "entity": "China",
        "relations": [
            {"relation": "alliance", "score": 8, "explanation": "This relation is highly relevant as it provides information about China's alliances."},
            {"relation": "political system", "score": 7, "explanation": "This relation is relevant as it provides information about China's policies."}
        ]
    },
    {
        "entity": "Joe Biden",
        "relations": [
            {"relation": "political position", "score": 10, "explanation": "This relation is highly relevant as it provides information about Joe Biden's stance on international matters."},
            {"relation": "presidency", "score": 2, "explanation": "This relation is slightly relevant as it provides context about Joe Biden's role as president."}
        ]
    }
]

# Instructions:
- Analyze the relations for each entity separately.
- Select only the %s most relevant relations for each entity.
- Provide a JSON-formatted output strictly following the structure shown above.
- If no relations are relevant for an entity, include an empty "relations" list for that entity.
"""

RERANKER_RANK_PATH_PROMPT = 'Given a query, retrieve relevant Knowledge Graph paths that answer the query'

REASONING_PROMPT = """
Given the query: '%s' 
and the Knowledge Graph path: '%s'
Determine if the provided knowledge is sufficient to answer the query.
Perform step-by-step reasoning to evaluate the information, then provide the response in JSON format with the following structure:

{
    "is_answerable": boolean,
    "answer": string
}

- Set 'is_answerable' to true if the knowledge is sufficient to answer the query, false otherwise.\n- If 'is_answerable' is true, provide the answer in the 'answer' field without any explanation. If false, set 'answer' to an empty string.\n\nReturn only the JSON object."
"""

ANSWER_GENERATION_PROMPT = """
As an advanced reading comprehension assistant, your task is to analyze extracted information and corresponding questions meticulously. If the knowledge graph information is not enough, you can use your own knowledge to answer the question.
Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. 
Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.
"""
FEW_SHOT_EXAMPLE = [{"role": "user", "content": """
Given the query: 'Who painted the Mona Lisa?'
and the Knowledge Graph path: '[{'subject': 'Leonardo da Vinci', 'relation': 'painted', 'object': 'Mona Lisa'}]'
Perform step-by-step reasoning to evaluate the information, then provide the response in JSON format with the following structure:
{
"answer": string
}

Provide the answer in the 'answer' field without any explanation, using your best effort to answer the query based on the provided knowledge.
    """},
    {"role": "assistant", "content": """
{
  "answer": "Leonardo da Vinci"
}
    """},
    
    # Example 2
    {"role": "user", "content": """
Given the query: 'Where is the Mona Lisa located?'
and the Knowledge Graph path: '[{'subject': 'Mona Lisa', 'relation': 'located in', 'object': 'Louvre Museum'}, {'subject': 'Louvre Museum', 'relation': 'located in', 'object': 'Paris'}]'
Perform step-by-step reasoning to evaluate the information, then provide the response in JSON format with the following structure:
{
"answer": string
}

Provide the answer in the 'answer' field without any explanation, using your best effort to answer the query based on the provided knowledge.
    """},
    {"role": "assistant", "content": """
{
  "answer": "Louvre Museum"
}
    """},]