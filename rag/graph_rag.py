from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record for record in result]

def generate_answer_with_graph_context(question):
    context_data = conn.run_query(f"""
    MATCH (d:Document)-[:RELATED_TO]->(c)
    WHERE d.name = '{question}'
    RETURN c.content as content
    """)
    context = " ".join([record['content'] for record in context_data])
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(prompt=prompt, engine="davinci", max_tokens=50)
    return response.choices[0].text.strip()

# Example usage
conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "password")
results = conn.run_query("MATCH (n) RETURN n LIMIT 10")
for record in results:
    print(record)
conn.close()