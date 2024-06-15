from flask import Flask, render_template_string, request
from rag import retrieve, rag_answer_question
from llm_interface import get_llm_interface
import click
from dotenv import load_dotenv

@click.command()
@click.option('--llm_type', type=click.Choice(['openai', 'huggingface']), default="openai", help='The model to use for generating answers.')
def main(llm_type):
    """
    Initialize and run the Flask application using a specified language model.

    Parameters
    ----------
    llm_type : str
        Type of language model to use, either 'openai' or 'huggingface'.
    """
    llm_interface = get_llm_interface(llm_type)
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        """
        Handle the index route which accepts both GET and POST requests.
        Returns a form and displays results of question processing on POST.
        
        Returns
        -------
        str
            Rendered HTML string containing a form and, if applicable, the results of processing a question.
        """
        question = ""
        retrieved_results = ""
        generated_answer = ""
        display_results = False

        if request.method == "POST":
            question = request.form.get("question")
            results = retrieve(question)
            retrieved_results = "\n".join([f"{key}: {value}" for key, value in results.items()])
            generated_answer = rag_answer_question(question, results, llm_interface)
            display_results = True

        split_response = generated_answer.split("ANSWER:")
        thinking = split_response[0].strip()
        answer = split_response[1].strip() if len(split_response) > 1 else ""

        return render_template_string(
        """
        <form method="post">
            <label for="question">Ask {{ llm_type }} a Question:</label>
            <textarea id="question" name="question" cols="40" rows="5" required></textarea>
            <input type="submit" value="Submit">
        </form>
        {% if display_results %}
            <h3>Model Name:</h3>
            <pre>{{ model_name }}</pre>
            <h3>Question:</h3>
            <pre>{{ question }}</pre>
            <h3>Thinking:</h3>
            <pre>{{ thinking }}</pre>
            <h3>Generated Answer:</h3>
            <pre>{{ answer }}</pre>
            <h3>Retrieved Results:</h3>
            <pre>{{ retrieved_results }}</pre>
        {% endif %}
        """, question=question, 
            retrieved_results=retrieved_results, 
            thinking=thinking, answer=answer, 
            display_results=display_results, 
            model_name=llm_interface.model_name, llm_type=llm_type)

    app.run(debug=True)

if __name__ == "__main__":
    load_dotenv()
    main()