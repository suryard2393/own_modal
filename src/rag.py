import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from calculator import calculate

# ── Alias table ────────────────────────────────────────────────────────────────
# Maps a normalised word to its canonical form.
# Applied word-by-word; only exact whole-word matches fire.
# Do NOT use broad fuzzy matching on every query — add explicit entries here.
_ALIASES = {
    "tramp":      "trump",
    "trum":       "trump",
    "trumpp":     "trump",
    "phyton":     "python",
    "pyhton":     "python",
    "pythn":      "python",
    "jave":       "java",
    "jaava":      "java",
    "tranformer": "transformer",
    "transfomer": "transformer",
}

# Regex that matches a run of spaced-out single letters anywhere in a string,
# e.g. "j a v a" inside "what is j a v a?" -> "java"
# Requires at least 2 spaced letters to avoid collapsing normal short words.
_SPACED_LETTERS = re.compile(r"(?<![a-z])([a-z])( [a-z]){1,}(?![a-z])")


def _collapse_spaced(m):
    """Remove spaces from a spaced-letter match: 'j a v a' -> 'java'."""
    return m.group(0).replace(" ", "")


def _normalize(text):
    """
    Light, transparent normalization applied to every query before retrieval.

    Step 1 - Collapse runs of 3+ identical consecutive characters down to 1.
             Single pass only, so legitimate double letters (ll, ee, oo ...)
             are never touched.
             "javaaaaaa" -> "java"    "hello" -> "hello"    "hellooo" -> "hello"

    Step 2 - Collapse spaced-out single-letter runs anywhere in the query.
             "j a v a" -> "java"    "what is j a v a?" -> "what is java?"
             Requires at least 2 consecutive spaced letters to fire.

    Step 3 - Apply the alias table word-by-word (whole-word matches only).
             "tramp" -> "trump"
    """
    t = text.strip().lower()

    # Step 1: 3+ identical chars in a row -> 1  (single pass, preserves doubles)
    t = re.sub(r"(.)\1{2,}", r"\1", t)

    # Step 2: spaced-out single letters -> joined token (works inside sentences)
    t = _SPACED_LETTERS.sub(_collapse_spaced, t)

    # Step 3: alias substitution (strip trailing punctuation before lookup)
    def _alias(word):
        stripped = word.rstrip("?!.,;:")
        suffix = word[len(stripped):]
        return _ALIASES.get(stripped, stripped) + suffix

    words = t.split()
    words = [_alias(w) for w in words]
    t = " ".join(words)

    return t


# ── Knowledge base ─────────────────────────────────────────────────────────────
KNOWLEDGE = [
    ("What is AI?", "AI is a field of computing focused on systems that perform tasks associated with human intelligence."),
    ("What is a computer?", "A computer is a machine that processes data according to instructions."),
    ("What is Python?", "Python is a programming language known for its readable syntax."),
    ("What is Python used for?", "Python is used for automation, data analysis, machine learning, web development, and many other tasks."),
    ("What is the internet?", "The internet is a global network connecting computers and other devices."),
    ("What is a language model?", "A language model learns patterns in text and predicts what text is likely to come next."),
    ("How does a language model learn?", "A language model adjusts its weights during training to reduce errors in predicting the next token."),
    ("What is a large language model?", "A large language model is a neural network trained on large amounts of text to predict and generate language."),
    ("What is machine learning?", "Machine learning uses data to train systems to identify patterns and make predictions."),
    ("What is deep learning?", "Deep learning is machine learning that uses neural networks with multiple layers."),
    ("What is a neural network?", "A neural network is a computing model made of connected numerical units that learn patterns."),
    ("What is a Transformer?", "A Transformer is a neural network architecture that uses attention to process sequences."),
    ("What is attention in a Transformer?", "Attention lets a model weigh information from different positions in a sequence."),
    ("What is training?", "Training adjusts a model's numerical parameters to reduce prediction errors."),
    ("What is the difference between training and inference?", "Training adjusts a model's parameters using data. Inference uses the trained parameters to make predictions."),
    ("What are model weights?", "Model weights are learned numerical parameters that influence a neural network's predictions."),
    ("What is a token?", "A token is a unit of text processed by a language model. It can be a character, word, or word part."),
    ("What is a tokenization algorithm?", "A tokenizer converts text into units that a model can process."),
    ("What is a subword tokenizer?", "A subword tokenizer divides text into frequently occurring pieces that may be smaller than words."),
    ("What is a character-level language model?", "A character-level language model predicts text one character at a time."),
    ("What is a dataset?", "A dataset is a collection of examples used for training or evaluating a model."),
    ("What is a training dataset?", "A training dataset is the collection of examples used to optimize a model's parameters."),
    ("What is a test dataset?", "A test dataset evaluates a trained model on examples that were not used to train it."),
    ("What is overfitting?", "Overfitting occurs when a model learns its training examples too specifically and performs poorly on new examples."),
    ("What is an epoch?", "An epoch is one complete pass through a training dataset."),
    ("What is a parameter in a neural network?", "A parameter is a learned value, such as a weight or bias, adjusted during training."),
    ("What is a loss function?", "A loss function measures how far a model's predictions are from the target values."),
    ("What is gradient descent?", "Gradient descent updates model parameters in a direction intended to reduce the loss."),
    ("What is backpropagation?", "Backpropagation calculates gradients used to update neural network parameters."),
    ("What is an embedding?", "An embedding represents an item, such as a token, as a vector of numbers."),
    ("What is an algorithm?", "An algorithm is a sequence of steps designed to solve a problem or perform a computation."),
    ("What is data?", "Data is information that can be stored and processed."),
    ("What is RAG?", "Retrieval-Augmented Generation retrieves relevant external information and supplies it as context for generating an answer."),
    ("Does RAG train model weights?", "No. Standard RAG retrieves information at inference time without changing the model's learned weights."),
    ("Can a model learn from new documents through RAG?", "A RAG system can use new documents at query time after they are added to its searchable knowledge base; model retraining is not required."),
    ("What is a vector database?", "A vector database stores vector representations and supports similarity search."),
    ("What is semantic search?", "Semantic search attempts to find information related in meaning, rather than relying only on exact word matches."),
    ("What is TF-IDF?", "TF-IDF weights words according to how informative they are in a document relative to a collection of documents."),
    ("What is hallucination in AI?", "An AI hallucination is an output that presents incorrect or unsupported information as if it were true."),
    ("What is an API?", "An API is an interface that allows software components to communicate."),
    ("What is PyTorch?", "PyTorch is a machine learning library that provides tensors, automatic differentiation, and neural network tools."),
    ("What is a GPU?", "A GPU is a processor designed to perform many operations in parallel and is widely used for neural network training."),
    ("What is RAM?", "RAM is temporary working memory used by a computer while programs are running."),
    ("What is the CPU?", "The CPU executes instructions and coordinates many general-purpose computing operations."),
    ("What is an operating system?", "An operating system manages computer hardware and provides services to applications."),
    ("What is the difference between AI and machine learning?", "AI is the broader field of intelligent systems, while machine learning is one approach within AI."),
    ("Can a small language model answer every question?", "No. Its capabilities are limited by its architecture, training, context, and available information."),
    ("What is the Sun?", "The Sun is the star at the center of our solar system."),
    ("What is Earth?", "Earth is the planet where we live."),
    ("What is the solar system?", "The solar system consists of the Sun and the objects gravitationally bound to it, including planets, moons, asteroids, and comets."),
    ("What is photosynthesis?", "Photosynthesis is the process by which plants, algae, and some bacteria use light energy to make energy-rich compounds."),
    ("What is gravity?", "Gravity is the interaction that attracts objects with mass toward one another."),
    ("What is electricity?", "Electricity involves the presence or movement of electric charge."),
    ("What is water?", "Water is a chemical compound made of hydrogen and oxygen."),
    ("Why is the sky blue?", "The sky often appears blue because air scatters shorter blue wavelengths of sunlight more strongly than longer red wavelengths."),
    ("What causes rain?", "Rain forms when water droplets in clouds grow large enough to fall to the ground."),
    ("What is the water cycle?", "The water cycle describes the movement of water through evaporation, condensation, precipitation, and related processes."),
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of Japan?", "Tokyo is the capital of Japan."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("How many hours are in a day?", "There are twenty-four hours in a day."),
    ("What is a variable in programming?", "A variable is a named reference used to store or represent a value in a program."),
    ("What is a function in programming?", "A function is a reusable block of code that performs a defined task."),
    ("What is a bug in software?", "A bug is an error or defect that causes software to behave incorrectly."),
    ("What is Java?", "Java is a general-purpose programming language commonly used for enterprise applications, backend systems, and Android development."),
    ("What is the meaning of Java?", "Java is the name of a programming language. The word can also refer to the Indonesian island of Java."),
    ("Who is Trump?", "Donald Trump is an American politician and businessman who has served as president of the United States."),
    ("Who is Donald Trump?", "Donald Trump is an American politician and businessman who has served as president of the United States."),
    ("What is memory in a chatbot?", "Memory can store information from earlier messages so a chatbot can use it later."),
    ("Who created you?", "I am a small language model trained from examples supplied by my developer."),
    ("What is your name?", "I am MiniGPT, a small language model built as a local AI project. I am not a commercial product and I do not have a human identity."),
    ("What can you do?", "I can generate text based on patterns learned during training, but my abilities are limited."),
    ("Hello", "Hello! What would you like to ask?"),
    ("Hi", "Hi! Ask me a question."),
    ("How are you?", "I am a small language model ready to respond to your message."),
    ("Thank you", "You are welcome."),
]

questions = [q for q, _ in KNOWLEDGE]
answers   = [a for _, a in KNOWLEDGE]

# No stop_words: our KB questions are short and stop-word removal drops key
# subject nouns (e.g. "name", "trump", "java") leaving nnz=0.
_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
_question_vectors = _vectorizer.fit_transform(questions)


def retrieve(user_question, threshold=0.25):
    # Calculator takes priority before any text processing.
    calc = calculate(user_question)
    if calc is not None:
        return {"answer": calc, "matched": "calculator", "score": None}

    normalised = _normalize(user_question)

    query_vector = _vectorizer.transform([normalised])
    if query_vector.nnz == 0:
        return {
            "answer": "I don't have enough information on that yet.",
            "matched": None,
            "score": 0.0,
        }

    scores = cosine_similarity(query_vector, _question_vectors).flatten()

    # Suppress KB entries whose numeric content differs from the query.
    query_numbers = re.findall(r"\d+(?:\.\d+)?", normalised)
    if query_numbers:
        for i, q in enumerate(questions):
            if re.findall(r"\d+(?:\.\d+)?", q) != query_numbers:
                scores[i] = 0.0

    best       = int(scores.argmax())
    best_score = float(scores[best])

    if best_score < threshold:
        return {
            "answer": "I don't have enough information on that yet.",
            "matched": None,
            "score": best_score,
        }

    return {"answer": answers[best], "matched": questions[best], "score": best_score}


if __name__ == "__main__":
    print("RAG chatbot ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Bot: Goodbye.")
            break
        result = retrieve(q)
        print("Bot:", result["answer"])
        if result["matched"]:
            score_str = (
                f"  score: {result['score']:.3f}"
                if result["score"] is not None
                else ""
            )
            print(f"  (matched: {result['matched']}{score_str})")
        print()
