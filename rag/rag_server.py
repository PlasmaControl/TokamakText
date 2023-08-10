from flask import Flask, render_template_string, request
from rag import retrieve, rag_answer_question

app = Flask(__name__)

'''
def retrieve(question):
    # Your retrieval code here
    return {"key": "value"}

def rag(question, results):
    # Your RAG code here
    return "Generated Answer"
'''

@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    retrieved_results = ""
    generated_answer = ""

    if request.method == "POST":
        question = request.form.get("question")
        results = retrieve(question)
        retrieved_results = "\n".join([f"{key}: {value}" for key, value in results.items()])
        generated_answer = rag_answer_question(question, results)

    split_response = generated_answer.split("ANSWER:")
    thinking = split_response[0].strip()
    if len(split_response) == 1:
        answer = ""
    else:
        answer = split_response[1].strip()

    return render_template_string("""
    <form method="post">
        <label for="question">Ask a Question:</label>
        <textarea id="question" name="question" cols="40" rows="5" required></textarea>
        <input type="submit" value="Submit">
    </form>
    <h3>Retrieved Results:</h3>
    <pre style="white-space: pre-wrap;">{{ retrieved_results }}</pre>
    <h3>Thinking:</h3>
    <pre style="white-space: pre-wrap;">{{ thinking }}</pre>
    <h3>Generated Answer:</h3>
    <pre style="white-space: pre-wrap;">{{ answer }}</pre>
    """, question=question, retrieved_results=retrieved_results, thinking=thinking, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)

