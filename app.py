from flask import Flask, render_template
from generate_poem import create_poem

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('main.html')


@app.route('/result/<section>')
def result(section):
    return {'data': create_poem(section), 'word': section}


if __name__ == "__main__":
    app.run(debug=True)
